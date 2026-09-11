# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''

import sys
import os
import time
import json
import traceback
from collections import OrderedDict
import math
import random
import torch
import torch.nn as nn

# Training-only flow teacher. Inputs concatenate previous latent (D),
# intermediate future block (4D), scalar flow time (1), and controls (E).
# Output is a 4D flow velocity, not a directly decoded skeletal pose.
class DenoiserNetwork(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_num=2048, layer_num=12):
        
        super().__init__()
        
        layers = []
        
        layers.append(nn.Sequential(
            nn.Linear(input_size, hidden_num),
            nn.LayerNorm([hidden_num]),
            nn.GELU()))
        
        for i in range(layer_num - 1):
            
            layers.append(nn.Sequential(
                nn.Linear(hidden_num, hidden_num),
                nn.LayerNorm([hidden_num]),
                nn.GELU()))

        layers.append(nn.Linear(hidden_num, output_size))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        
        h = self.layers[0](x)
        
        # Hidden blocks preserve width so their outputs can be added residually.
        # The first projection and final output projection are not residual blocks.
        for i in range(1, len(self.layers) - 1):
            h = h + self.layers[i](h)
        
        return self.layers[len(self.layers) - 1](h)


# Reading guide: stage 1 trains the control encoder and flow teacher.
# Stage 2 freezes their computation and distills rollouts into three LODs.
# UE runs one LOD at runtime, followed by the autoencoder decoder.
# Only explanatory comments were added in this copy.
# Notation: B = batch size, D = pose latent width, E = encoded control width.
def train_controller():

    """ Load Config """
    if len(sys.argv) != 2:
        raise Exception('Wrong number of arguments to training script')
    
    # Get the config file path from the command line arguments and load it
    # UE launches this process with a JSON path. The JSON describes the run;
    # large training arrays and network snapshots live in local shared memory.
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f, object_pairs_hook=OrderedDict)
    
    # We need to append some things to the site-packages so we can load NNE models in python
    sys.path.append(config['SitePackagesPath'])    
    sys.path.append(config['EnginePath'] + 'Plugins/Experimental/NNERuntimeBasicCpu/Content/Python/')
    sys.path.append(config['EnginePath'] + 'Plugins/Experimental/LearningCore/Content/Python/')
    
    """ Imports from site-packages """
    import numpy as np
    import torch
    import torch.nn as nn
    from nne_runtime_basic_cpu_pytorch import NeuralNetwork
    from learning_core.communicators.shared_memory import SharedMemory
    from learning_core.train_common import (
        load_snapshot, save_snapshot, 
        schema_noise_mask_observation,
        schema_annotate_normalization_observation,
        schema_write_norm_to_network)
        
    use_tensorboard = True
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print('Failed to import TensorBoard. Please add manually to site-packages.')
        sys.stdout.flush()
        use_tensorboard = False

    """ Settings """
    training_name = config['TaskName']
    timestamp = config['TimeStamp']
    training_identifier = training_name + '_' + timestamp
    
    range_num = int(config['RangeNum'])
    database_total_frame_num = int(config['DatabaseTotalFrameNum'])
    control_total_frame_num = int(config['ControlTotalFrameNum'])
    
    dt = config['DeltaTime']
    control_vector_size = int(config['ControlVectorSize'])
    encoded_control_vector_size = int(config['EncodedControlVectorSize'])
    pose_encoding_size = int(config['PoseEncodingSize'])
    control_schema = config['ControlSchema']

    niterations_denoise = int(config['IterationNum'])
    niterations_distill = int(config['IterationDistillNum'])
    
    lr_warmup = int(config['WarmupIterations']) # for learning rate
    lr = config['LearningRate']
    batchsize = int(config['BatchSize'])
    seed = int(config['Seed'])
    device = { 'GPU': 'cuda', 'CPU': 'cpu' }[config['Device']] if torch.cuda.is_available() else 'cpu'
    steps = config['DenoiserSteps']
    denoiser_hidden_num = config['DenoiserHiddenUnitNum']
    denoiser_layer_num = config['DenoiserLayerNum']

    pose_noise_scale = config['PoseNoiseScale']
    control_noise_scale = config['ControlNoiseScale']
    random_pose_sample_rate = config['RandomPoseSampleRate']
    normalized_pose_stds = torch.as_tensor(config['NormalizedPoseStds'], device=device, dtype=torch.float32)
    
    window = 8 + 1

    snapshots_dir = config['IntermediatePath'] + "/" + training_identifier
    if not os.path.exists(snapshots_dir):
        os.mkdir(snapshots_dir)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    
    """ Load Data """
    print('Loading Data...')

    # Attach to a named memory region created by UE, using its exact shape and dtype.
    # np.frombuffer creates a view of those bytes rather than deserializing a dataset.
    # Keep the returned handle alive while its NumPy view is in use.
    def shared_memory_map_array(guid, shape, dtype):
        size = np.asarray(shape, dtype=np.int64).prod() * np.dtype(dtype).itemsize
        # create=False opens existing UE memory; it does not allocate a new dataset.
        if size > 0:
            handle = SharedMemory(guid, create=False, size=size)
            assert handle is not None
            array = np.frombuffer(handle.buf, dtype=dtype, count=np.asarray(shape, dtype=np.int64).prod()).reshape(shape)
            return handle, array
        else:
            return None, np.empty(shape, dtype=dtype)

    # IPC protocol: these integers are indexes into the shared int32 control array,
    # not training hyperparameters. UE requests exit through slot 0; Python clears it.
    # Python writes iteration/loss slots for UE to display. Network-ready slots use
    # a handshake: Python writes snapshot bytes, sets 1, and UE loads then clears 0.
    # Network weights and dataset arrays are stored in separate memory regions.
    train_control_exit = 0
    train_control_training_iteration = 1
    train_control_distill_iteration = 2
    train_control_training_loss = 3
    train_control_distill_loss = 4
    train_control_control_encoder_network_update = 5
    train_control_lod_network_update = 6
    # Seven slots: exit; teacher/student iterations; teacher/student losses;
    # control-encoder ready; all-three-LOD ready. This is the array length.
    train_control_num = 7
    control_hndl, control = shared_memory_map_array(config['ControlGuid'], [train_control_num], np.int32)
    
    # Network data sizes
    controller_byte_num = int(config['ControllerByteNum'])
    lod0_byte_num = int(config['LOD0ByteNum'])
    lod1_byte_num = int(config['LOD1ByteNum'])
    lod2_byte_num = int(config['LOD2ByteNum'])

    # Load all network data arrays
    controller_data_hndl, controller_data = shared_memory_map_array(config['ControllerGuid'], [controller_byte_num], np.uint8)
    lod0_data_hndl, lod0_data = shared_memory_map_array(config['LOD0Guid'], [lod0_byte_num], np.uint8)
    lod1_data_hndl, lod1_data = shared_memory_map_array(config['LOD1Guid'], [lod1_byte_num], np.uint8)
    lod2_data_hndl, lod2_data = shared_memory_map_array(config['LOD2Guid'], [lod2_byte_num], np.uint8)

    # Load data arrays
    # Paired range offsets align controls and encoded poses even when their
    # full arrays have different frame counts. A shared range length bounds both.
    range_control_starts_hndl, range_control_starts = shared_memory_map_array(config['RangeControlStartsGuid'], [range_num], np.int32)
    range_encoded_starts_hndl, range_encoded_starts = shared_memory_map_array(config['RangeEncodedStartsGuid'], [range_num], np.int32)
    range_lens_hndl, range_lens = shared_memory_map_array(config['RangeLengthsGuid'], [range_num], np.int32)
    
    # C contains behavior control objects flattened using the observation schema.
    # X contains database poses encoded by the already trained autoencoder, then
    # normalized in UE using per-dimension means and one shared scale.
    # Neither autoencoder network is loaded or optimized by this training script.
    Chndl, C = shared_memory_map_array(config['ControlVectorsGuid'], [control_total_frame_num, control_vector_size], np.float32)
    Xhndl, X = shared_memory_map_array(config['EncodedVectorsGuid'], [database_total_frame_num, pose_encoding_size], np.float32)
    
    """ Load Networks """
    print('Loading Networks...')

    # Despite the name, this is only the control encoder: controls -> E features.
    # Its schema-dependent architecture and initial weights come from UE.
    controller_network = NeuralNetwork(device=device)
    load_snapshot(controller_data, controller_network)
    print("Controller Network: \n", controller_network)
    
    print('Performing Normalization...')
    
    # Fit schema-aware control normalization from C and embed it in the encoder.
    # This is separate from the latent-pose normalization already applied to X.
    schema_annotate_normalization_observation(control_schema, C, 0.001)
    schema_write_norm_to_network(control_schema, controller_network.model)
    
    denoiser_network = DenoiserNetwork(
        input_size=5*pose_encoding_size+1+encoded_control_vector_size, 
        output_size=4*pose_encoding_size,
        hidden_num=denoiser_hidden_num, 
        layer_num=denoiser_layer_num).to(device)
    print("Denoiser Network: \n", denoiser_network)

    # Each LOD maps [previous pose D, source noise 4D, controls E] -> future 4D.
    # These are three alternative-capacity students with the same interface.
    # They have no flow-time input and generate a block in one forward pass.
    lod0_network = NeuralNetwork(device=device)
    lod1_network = NeuralNetwork(device=device)
    lod2_network = NeuralNetwork(device=device)
    load_snapshot(lod0_data, lod0_network)
    load_snapshot(lod1_data, lod1_network)
    load_snapshot(lod2_data, lod2_network)
    print("LOD0 Network: \n", lod0_network)

    print('Creating Optimizer...')
    # Stage 1 updates teacher and control encoder together. optimizer_lod below
    # updates all three students separately from that first parameter group.
    optimizer = torch.optim.AdamW(list(denoiser_network.parameters()) + list(controller_network.parameters()), lr=lr, amsgrad=True)
    # The multiplier combines warmup with linear decay over the iteration budget.
    # scheduler.step() follows each optimizer update; iterations are not epochs.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: np.minimum(i / lr_warmup, 1.0) * (1.0 - np.clip(float(i) / niterations_denoise, 0.0, 1.0)))

    optimizer_lod = torch.optim.AdamW(list(lod0_network.parameters()) + list(lod1_network.parameters()) + list(lod2_network.parameters()),  lr=lr, amsgrad=True)
    scheduler_lod = torch.optim.lr_scheduler.LambdaLR(optimizer_lod, lr_lambda=lambda i: np.minimum(i / lr_warmup, 1.0) * (1.0 - np.clip(float(i) / niterations_distill, 0.0, 1.0)))

    print('Creating Batches...')
    # Build aligned nine-frame windows without crossing animation-range boundaries.
    # Separate index tables preserve the mapping from C rows to X rows.
    control_window_indices = []
    encoded_window_indices = []
    for ri in range(len(range_lens)):
        if range_lens[ri] >= window:
            for fi in range(range_lens[ri] - window + 1):
                control_window_indices.append(range_control_starts[ri] + fi + np.arange(window))
                encoded_window_indices.append(range_encoded_starts[ri] + fi + np.arange(window))
    control_window_indices = torch.as_tensor(np.array(control_window_indices, dtype=np.int64))
    encoded_window_indices = torch.as_tensor(np.array(encoded_window_indices, dtype=np.int64))

    # Save config
    with open(snapshots_dir + '/config.json', 'w') as f:
        json.dump(config, f, indent=4)

    if use_tensorboard:
        writer = SummaryWriter(log_dir=snapshots_dir, max_queue=1000)
    
    """ Pin Training Data """
    
    # Pinned host copies hold the full dataset; sampled batches move to device.
    X = torch.from_numpy(X).pin_memory()
    C = torch.from_numpy(C).pin_memory()
    
    """ Train Flow-Matching """
    
    # A generated block predicts horizons 1, 2, 4, and 8 relative to frame 0.
    # These are frame offsets, independent of which LOD is selected at runtime.
    output_frames_np = np.array([1, 2, 4, 8])
    output_frames = torch.as_tensor(output_frames_np, device=device, dtype=torch.long)

    print('Training Flow-Matching...')
    
    rolling_avg_loss = None    
    exiting = False
    
    i = 0
    while i < niterations_denoise:
        
        if control[train_control_exit]:
            control[train_control_exit] = 0
            exiting = True
            print('Exiting...')
            break
        
        control[train_control_training_iteration] = i

        # Sample windows of pose vectors and control vectors to train over
        # Teacher training uses nine ground-truth poses but only the first control.
        # The window provides frame 0 conditioning and four selected future targets.
        batch_indices = torch.randint(0, len(encoded_window_indices), size=[batchsize], dtype=torch.long)
        Xgnd = X[encoded_window_indices[batch_indices]].to(device, non_blocking=True) # (batchsize, window, pose_encoding_size)
        Cgnd = C[control_window_indices[batch_indices][:,0]].to(device, non_blocking=True) # (batchsize, control_vector_size)

        # Sample random frames (we will sometimes use this as the previous pose)
        random_indices = torch.randint(0, len(X), size=[batchsize], dtype=torch.long)
        Xrnd = X[random_indices].to(device, non_blocking=True) # (batchsize, pose_encoding_size)
        
        # Sample the initial noise, scaled by normalized_pose_stds
        # Sample source noise in the normalized latent space, scaled per dimension.
        # The four noise vectors correspond to the four output horizons.
        Xsrc = (normalized_pose_stds * torch.clip(torch.randn([batchsize, 4, pose_encoding_size], dtype=torch.float32, device=device), min=-3, max=3)).reshape([batchsize, 4 * pose_encoding_size])
        
        # Sample a mask for when to replace the previous pose with a random pose
        Xnoised_mask = torch.rand([batchsize, 1], dtype=torch.float32, device=device) < random_pose_sample_rate
        
        # Sample the noise to add to the previous pose
        Xnoised_add = torch.rand([batchsize, 1], dtype=torch.float32, device=device) * pose_noise_scale * normalized_pose_stds * torch.clip(torch.randn([batchsize, pose_encoding_size], device=device), min=-3, max=3)
        
        # Sample the noise to add to the control vectors
        Cnoise_add = torch.rand([batchsize, 1], dtype=torch.float32, device=device) * control_noise_scale * torch.clip(torch.randn([batchsize, control_vector_size], device=device), min=-3, max=3)

        # Sample the alpha value of the flow
        alpha = torch.rand([batchsize, 1], dtype=torch.float32, device=device)

        # Compute the noised version of the control vector
        Cnoised = Cgnd + schema_noise_mask_observation(control_schema, Cgnd, use_scales=True) * Cnoise_add

        # Compute the noised previous pose using either Xnoised_add or the random pose Xrnd
        # In stage 1, RandomPoseSampleRate is the chance of replacing frame 0 with
        # a random dataset pose. Add perturbation to make conditioning more robust.
        Xnoised = torch.where(Xnoised_mask, Xrnd, Xgnd[:,0]) + Xnoised_add
        
        # Compute the output pose vectors using output_frames
        Xout = Xgnd[:,output_frames].reshape([batchsize, -1])
        
        # Linearly interpolate using alpha between the ground truth and the sampled initial noise
        # Interpolate along the straight path from sampled noise to dataset futures.
        # alpha is a flow coordinate, not the animation frame offset or elapsed time.
        Xalpha = alpha * Xout + (1.0 - alpha) * Xsrc

        # Compute flow matching predicted velocity
        Xpred = denoiser_network(torch.cat([Xnoised, Xalpha, alpha, controller_network(Cnoised)], dim=1))
        
        # Compute flow matching loss
        # The analytic path velocity is Xout - Xsrc. MSE trains the teacher to
        # predict this velocity, with gradients also flowing into the control encoder.
        loss = torch.mean((Xpred - (Xout - Xsrc))**2)
        
        # Update Weights
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        scheduler.step()
        
        # Output Stats
        
        loss_item = loss.item()
        
        if rolling_avg_loss is None:
            rolling_avg_loss = loss_item
        else:
            rolling_avg_loss = rolling_avg_loss * 0.99 + loss_item * 0.01
        
        if loss_item > 10.0 * rolling_avg_loss:
            print(loss_item, rolling_avg_loss)
        
        control[train_control_training_loss] = int(rolling_avg_loss * 100000)
        
        if i < 100 or i % 10000 == 0:
            print(('Iter: %7i Loss: %7.5f' % (i, rolling_avg_loss)))
        
        if use_tensorboard:
            writer.add_scalar('loss/flow', loss_item, i)
            writer.add_scalar('lr/flow', scheduler.get_last_lr()[0], i)
        
        # Save Networks
        
        if i % 1000 == 0:
            
            if not control[train_control_control_encoder_network_update]:
                save_snapshot(controller_data, controller_network)
                control[train_control_control_encoder_network_update] = 1
        
        i += 1
        
    # Return the final stage-1 control encoder after UE consumes its prior snapshot.
    # The teacher is not published to UE; it supplies targets for distillation.
    while control[train_control_control_encoder_network_update]: time.sleep(0.01)
    save_snapshot(controller_data, controller_network)
    control[train_control_control_encoder_network_update] = 1

    """ Train LODs """
    
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) / steps)[:,None,None].tile([1,batchsize,1])

    print('Training LODs...')
    
    rolling_avg_loss = None    
    
    i = 0
    while not exiting and i < niterations_distill:
        
        if control[train_control_exit]:
            control[train_control_exit] = 0
            print('Exiting...')
            break
        
        control[train_control_distill_iteration] = i

        # Random indices to use for initial pose
        random_indices = torch.randint(0, len(encoded_window_indices), size=[batchsize], dtype=torch.long)
        
        # Sample indices to use for control vectors
        control_indices = torch.randint(0, len(encoded_window_indices), size=[batchsize], dtype=torch.long)
        
        # For initial pose either use random indices or those from the control vectors
        # Important: here the probability selects a MATCHING pose/control window.
        # Otherwise use an independent pose window, unlike the stage-1 replacement test.
        pose_indices = torch.where(torch.rand([batchsize], dtype=torch.float32) < random_pose_sample_rate, control_indices, random_indices)
        
        # Sample pose vectors and control vectors
        Xgnd = X[encoded_window_indices[pose_indices][:,0]].to(device, non_blocking=True) # (batchsize, pose_encoding_size)
        Cgnd = C[control_window_indices[control_indices]].to(device, non_blocking=True) # (batchsize, window, control_vector_size)
        
        # Compute the initial noise for each frame in the rollout
        # Sample source noise in the normalized latent space, scaled per dimension.
        # The four noise vectors correspond to the four output horizons.
        Xsrc = (normalized_pose_stds * torch.clip(torch.randn([batchsize, window, 4, pose_encoding_size], dtype=torch.float32, device=device), min=-3, max=3)).reshape([batchsize, window, 4 * pose_encoding_size])
        
        # Compute the noise to add to the inital frame
        Xnoised_add = torch.rand([batchsize, 1], dtype=torch.float32, device=device) * pose_noise_scale * normalized_pose_stds * torch.clip(torch.randn([batchsize, pose_encoding_size], dtype=torch.float32, device=device), min=-3, max=3) 
        
        # Compute the noise to add to all the control vectors
        Cnoise_add = torch.rand([batchsize, 1, 1], dtype=torch.float32, device=device) * control_noise_scale * torch.clip(torch.randn([batchsize, window, control_vector_size], dtype=torch.float32, device=device), min=-3, max=3)
        
        # Prepare output lists
        Xout_targ = []
        Xout_lod0 = []
        Xout_lod1 = []
        Xout_lod2 = []
        
        # Choise how many frames to subsample. Most of the time we do the rollout one frame at a time
        # Choose one horizon as the feedback stride for the entire sampled rollout.
        # Every network still predicts all four horizons at each visited position.
        output_choice = np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])
        frame_skip = output_frames_np[output_choice]
        
        # Compute the inital pose
        Xnoised = Xgnd + Xnoised_add    

        # Add noise and encode all of the control vectors
        Cnoise_mask = schema_noise_mask_observation(control_schema, Cgnd.reshape([batchsize * window, control_vector_size]), use_scales=True).reshape([batchsize, window, control_vector_size])
        # Control features are fixed during distillation: no encoder gradients.
        with torch.no_grad(): 
            Cenc = controller_network((Cgnd + Cnoise_mask * Cnoise_add).reshape([batchsize * window, control_vector_size])).reshape([batchsize, window, encoded_control_vector_size])
        
        # Set the current pose to the initial pose
        # Teacher and students share the initial pose but maintain separate states.
        # Later conditioning poses come from each network's own previous prediction.
        Xtarg = Xnoised.clone()
        Xlod0 = Xnoised.clone()
        Xlod1 = Xnoised.clone()
        Xlod2 = Xnoised.clone()
        
        # Perform rollout
        for ii, wi in enumerate(range(0, window, frame_skip)):
            
            # Result from flow network
            with torch.no_grad(): 
                Xpred_targ = Xsrc[:,wi].clone()
                # Explicit Euler integration: start from source noise and advance flow time
                # in steps of 1/steps. Pose and control conditioning stay fixed in this loop.
                # The final integrated block is the student target, not dataset ground truth.
                for step in range(steps):    
                    Xpred_targ += (1 / steps) * denoiser_network(torch.cat(
                        [Xtarg, Xpred_targ, alphas[step], Cenc[:,wi]], dim=1))
            
            # Result from LODs
            # Each student gets its own pose state, but identical source noise and controls.
            # One student evaluation approximates the entire teacher integration.
            Xpred_lod0 = lod0_network(torch.cat([Xlod0, Xsrc[:,wi], Cenc[:,wi]], dim=1))
            Xpred_lod1 = lod1_network(torch.cat([Xlod1, Xsrc[:,wi], Cenc[:,wi]], dim=1))
            Xpred_lod2 = lod2_network(torch.cat([Xlod2, Xsrc[:,wi], Cenc[:,wi]], dim=1))
            
            Xout_targ.append(Xpred_targ)
            Xout_lod0.append(Xpred_lod0)
            Xout_lod1.append(Xpred_lod1)
            Xout_lod2.append(Xpred_lod2)
            
            # Next frame is now based on output_choice
            # Feed the selected horizon back into each network's own rollout state.
            # Student outputs are not detached: gradients propagate through the rollout.
            Xtarg = Xpred_targ.reshape([batchsize, 4, -1])[:,output_choice]
            Xlod0 = Xpred_lod0.reshape([batchsize, 4, -1])[:,output_choice]
            Xlod1 = Xpred_lod1.reshape([batchsize, 4, -1])[:,output_choice]
            Xlod2 = Xpred_lod2.reshape([batchsize, 4, -1])[:,output_choice]
        
        # Concatenate together
        # Stack to [B, rollout positions, 4D]. Late targets can extend beyond the
        # nine-frame dataset window because the teacher generates their future poses.
        Xout_targ = torch.stack(Xout_targ, dim=1)
        Xout_lod0 = torch.stack(Xout_lod0, dim=1)
        Xout_lod1 = torch.stack(Xout_lod1, dim=1)
        Xout_lod2 = torch.stack(Xout_lod2, dim=1)

        # l1 loss on LOD estimations
        loss_lod0 = 0.1 * torch.mean(torch.abs(Xout_targ - Xout_lod0))
        loss_lod1 = 0.1 * torch.mean(torch.abs(Xout_targ - Xout_lod1))
        loss_lod2 = 0.1 * torch.mean(torch.abs(Xout_targ - Xout_lod2))
        
        # Compute pose vector deltas
        # Differences run across rollout positions, not horizons inside one block.
        # Divide by stride * dt because successive positions are that far apart.
        Xtarg_vel = (Xout_targ[:,1:] - Xout_targ[:,:-1]) / (frame_skip * dt)
        Xlod0_vel = (Xout_lod0[:,1:] - Xout_lod0[:,:-1]) / (frame_skip * dt)
        Xlod1_vel = (Xout_lod1[:,1:] - Xout_lod1[:,:-1]) / (frame_skip * dt)
        Xlod2_vel = (Xout_lod2[:,1:] - Xout_lod2[:,:-1]) / (frame_skip * dt)
        
        # l1 loss on velocity of LOD estimations
        loss_vel_lod0 = 0.002 * torch.mean(torch.abs(Xtarg_vel - Xlod0_vel))
        loss_vel_lod1 = 0.002 * torch.mean(torch.abs(Xtarg_vel - Xlod1_vel))
        loss_vel_lod2 = 0.002 * torch.mean(torch.abs(Xtarg_vel - Xlod2_vel))
        
        # Sum all losses
        # Average pose and temporal L1 losses for all three students: six terms total.
        loss_lod = (loss_lod0 + loss_lod1 + loss_lod2 + loss_vel_lod0 + loss_vel_lod1 + loss_vel_lod2) / 6
        
        # Update Weights
        optimizer_lod.zero_grad()
        loss_lod.backward()        
        optimizer_lod.step()
        scheduler_lod.step()

        # Output Stats
        
        loss_lod_item = loss_lod.item()
        
        if rolling_avg_loss is None:
            rolling_avg_loss = loss_lod_item
        else:
            rolling_avg_loss = rolling_avg_loss * 0.99 + loss_lod_item * 0.01
        
        if loss_lod_item > 10.0 * rolling_avg_loss:
            print(loss_lod_item, rolling_avg_loss)
        
        control[train_control_distill_loss] = int(rolling_avg_loss * 100000)
        
        if i < 100 or i % 10000 == 0:
            print(('Iter: %7i Loss: %7.5f' % (i, rolling_avg_loss)))

        if use_tensorboard:
            writer.add_scalar('loss/lod0', loss_lod0.item(), i)
            writer.add_scalar('loss/lod1', loss_lod1.item(), i)
            writer.add_scalar('loss/lod2', loss_lod2.item(), i)
            writer.add_scalar('loss/vel_lod0', loss_vel_lod0.item(), i)
            writer.add_scalar('loss/vel_lod1', loss_vel_lod1.item(), i)
            writer.add_scalar('loss/vel_lod2', loss_vel_lod2.item(), i)
            writer.add_scalar('loss/lod', loss_lod_item, i)
            writer.add_scalar('lr/lod', scheduler_lod.get_last_lr()[0], i)

        # Save Networks
        
        if i % 1000 == 0:
        
            # All three snapshots form one publication group: write every buffer first,
            # then raise the shared ready flag so UE can load the complete group.
            if not control[train_control_lod_network_update]:
                save_snapshot(lod0_data, lod0_network)
                save_snapshot(lod1_data, lod1_network)
                save_snapshot(lod2_data, lod2_network)
                control[train_control_lod_network_update] = 1

        # Update Iteration

        i += 1
        
    
    control[train_control_training_iteration] = 0
    control[train_control_distill_iteration] = 0

    # Wait for any previous LOD publication, then publish final student snapshots.
    # This wait does not poll cancellation, and the newly published group is not
    # explicitly acknowledged before return. The original behavior is preserved.
    while control[train_control_lod_network_update]: time.sleep(0.01)
    save_snapshot(lod0_data, lod0_network)
    save_snapshot(lod1_data, lod1_network)
    save_snapshot(lod2_data, lod2_network)
    control[train_control_lod_network_update] = 1

    if use_tensorboard:
        writer.flush()
        writer.close()

if __name__ == '__main__':
    train_controller()