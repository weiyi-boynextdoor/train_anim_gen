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

def train_autoencoder():

    """ Load Config """

    if len(sys.argv) != 2:
        raise Exception('Wrong number of arguments to training script')
    
    config_file = sys.argv[1]
    
    with open(config_file) as f:
        config = json.load(f, object_pairs_hook=OrderedDict)
        
    print(json.dumps(config, indent=4))
    sys.stdout.flush()
    
    sys.path.append(config['SitePackagesPath'])
    # TODO: Work out how to make this a little more robust    
    sys.path.append(config['EnginePath'] + 'Plugins/Experimental/NNERuntimeBasicCpu/Content/Python/')
    sys.path.append(config['EnginePath'] + 'Plugins/Experimental/LearningCore/Content/Python/')
    
    """ Imports from site-packages """
    
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from nne_runtime_basic_cpu_pytorch import NeuralNetwork, SparseMixtureOfExperts, LipschiztLinear
    from learning_core.communicators.shared_memory import SharedMemory
    from learning_core.train_common import load_snapshot, save_snapshot
    
    use_tensorboard = True

    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            print('Failed to import TensorBoard. Please add manually to site-packages.')
            sys.stdout.flush()
            use_tensorboard = False
    
    """ Settings """
    
    training_name = config['TaskName']
    trainer_type = config['TrainerType']
    timestamp = config['TimeStamp']
    training_identifier = training_name + '_' + trainer_type + '_' + timestamp
    
    range_num = int(config['RangeNum'])
    total_frame_num = int(config['TotalFrameNum'])
    pose_vector_size = int(config['PoseVectorSize'])
    encoder_byte_num = int(config['EncoderByteNum'])
    decoder_byte_num = int(config['DecoderByteNum'])
    dt = config['DeltaTime']
    encoding_size = config['EncodingSize']
    
    niterations = int(config['IterationNum'])
    lr_warmup = int(config['WarmupIterations'])
    lr = config['LearningRate']
    batchsize = int(config['BatchSize'])
    window = 2
    seed = int(config['Seed'])
    device = { 'GPU': 'cuda', 'CPU': 'cpu' }[config['Device']] if torch.cuda.is_available() else 'cpu'
    use_tensorboard = use_tensorboard and config['EnableTensorboard']
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    """ Load Data """
    
    print('Loading Data...')
    sys.stdout.flush()

    def shared_memory_map_array(guid, shape, dtype):
        size = np.asarray(shape, dtype=np.int64).prod() * np.dtype(dtype).itemsize
        if size > 0:
            handle = SharedMemory(guid, create=False, size=size)
            assert handle is not None
            array = np.frombuffer(handle.buf, dtype=dtype, count=np.asarray(shape, dtype=np.int64).prod()).reshape(shape)
            return handle, array
        else:
            return None, np.empty(shape, dtype=dtype)

    train_control_exit = 0
    train_control_iteration	= 1
    train_control_loss = 2
    train_control_encoder_network_update = 3
    train_control_decoder_network_update = 4
    train_control_num = 5

    control_hndl, control = shared_memory_map_array(config['ControlGuid'], [train_control_num], np.int32)
    
    encoder_data_hndl, encoder_data = shared_memory_map_array(config['EncoderGuid'], [encoder_byte_num], np.uint8)
    decoder_data_hndl, decoder_data = shared_memory_map_array(config['DecoderGuid'], [decoder_byte_num], np.uint8)
    
    range_starts_hndl, range_starts = shared_memory_map_array(config['RangeStartsGuid'], [range_num], np.int32)
    range_lens_hndl, range_lens = shared_memory_map_array(config['RangeLengthsGuid'], [range_num], np.int32)
    range_ends = range_starts + range_lens

    Xhndl, X = shared_memory_map_array(config['PoseVectorsGuid'], [total_frame_num, pose_vector_size], np.float32)
    Xwei_hndl, Xwei = shared_memory_map_array(config['PoseVectorWeightsGuid'], [pose_vector_size], np.float32)
    Xwei = torch.as_tensor(Xwei, device=device)
    
    """ Load Networks """
    
    print('Loading Networks...')
    sys.stdout.flush()
    
    encoder_network = NeuralNetwork(device=device)
    load_snapshot(encoder_data, encoder_network)

    decoder_network = NeuralNetwork(device=device)
    load_snapshot(decoder_data, decoder_network)
    
    """ Freeze final denormalize layer """
    
    decoder_network.model[-1].mean.requires_grad = False
    decoder_network.model[-1].std.requires_grad = False
    
    """ Optimizer """
    
    print('Creating Optimizer...')
    sys.stdout.flush()
    
    # We reduce the betas and weight_decay to try and avoid mode collapse during training. This is particularly important when 
    # we have attributes that 99% of the time have one value, but occasionally change. In this case the network struggles to 
    # accurately predict the rare case where the attribute value is different if momentum is too high.

    optimizer = torch.optim.AdamW(
        list(encoder_network.parameters()) + list(decoder_network.parameters()),
        lr=lr,
        betas=(0.5, 0.9),
        amsgrad=True,
        weight_decay=0.0)    
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: np.minimum(i / lr_warmup, 1.0) * (1.0 - i / niterations))
    
    """ Create Batches """
    
    print('Creating Batches...')
    sys.stdout.flush()
    
    window_indices = []
    
    for ri in range(len(range_starts)):
        if range_lens[ri] >= window:
            for fi in range(range_lens[ri] - window + 1):
                window_indices.append(range_starts[ri] + fi + np.arange(window))
            
    window_indices = torch.as_tensor(np.array(window_indices, dtype=np.int64))
    
    """ Pin Training Data """
    
    X = torch.from_numpy(X).pin_memory()
    
    """ Train """
    
    print('Training...')
    sys.stdout.flush()
    
    if use_tensorboard:
        writer = SummaryWriter(log_dir=config['IntermediatePath'] + "/" + training_identifier, max_queue=1000)
    
    rolling_avg_loss = None
    
    for i in range(niterations):
        
        if control[train_control_exit]:
            control[train_control_exit] = 0
            print('Exiting...')
            break
        
        control[train_control_iteration] = i
        
        batch = window_indices[torch.randint(0, len(window_indices), size=[batchsize], dtype=torch.long)]
        
        Xgnd = X[batch].to(device, non_blocking=True)
        Z = encoder_network(Xgnd.reshape([batchsize * window, -1])).reshape([batchsize, window, -1])
        Xrep = decoder_network(Z.reshape([batchsize * window, -1])).reshape([batchsize, window, -1])
        
        loss_val = 0.2 * torch.mean(Xwei * torch.abs(Xgnd - Xrep))
        loss_del = 0.01 * torch.mean(Xwei * torch.abs((Xgnd[:,1:] - Xgnd[:,:-1]) / dt - (Xrep[:,1:] - Xrep[:,:-1]) / dt))
        loss_reg_l1 = 0.001 * torch.mean(torch.abs(Z))
        loss_reg_diff = 0.001 * torch.mean(torch.abs((Z[:,1:] - Z[:,:-1]) / dt))
        loss = loss_val + loss_del + loss_reg_l1 + loss_reg_diff
        
        optimizer.zero_grad()
        loss.backward()
        
        if use_tensorboard:
            enc_grad_norm = torch.cat([p.grad.detach().flatten() for p in encoder_network.parameters() if p.grad is not None]).norm()
            dec_grad_norm = torch.cat([p.grad.detach().flatten() for p in decoder_network.parameters() if p.grad is not None]).norm()
        
        # torch.nn.utils.clip_grad_norm_(encoder_network.parameters(), 0.1)
        # torch.nn.utils.clip_grad_norm_(decoder_network.parameters(), 0.1)
        
        optimizer.step()
        scheduler.step()
        
        loss_item = loss.item()
        
        if use_tensorboard:
            writer.add_scalar('latent/mean', Z.mean().item(), i)
            writer.add_scalar('latent/abs_mean', torch.abs(Z).mean().item(), i)
            writer.add_scalar('latent/std', Z.std().item(), i)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], i)
            writer.add_scalar('loss/loss', loss_item, i)
            writer.add_scalar('loss/val', loss_val.item(), i)
            writer.add_scalar('loss/del', loss_del.item(), i)
            writer.add_scalar('loss/reg_l1', loss_reg_l1.item(), i)
            writer.add_scalar('loss/reg_diff', loss_reg_diff.item(), i)
            writer.add_scalar('grad/enc', enc_grad_norm.item(), i)
            writer.add_scalar('grad/dec', dec_grad_norm.item(), i)
            
        if rolling_avg_loss is None:
            rolling_avg_loss = loss_item
        else:
            rolling_avg_loss = rolling_avg_loss * 0.99 + loss_item * 0.01
               
        control[train_control_loss] = int(rolling_avg_loss * 100000)

        if i < 100 or i % 100 == 0: 
        
            print('Iter: %7i Loss: %7.5f' % (i, rolling_avg_loss))
            sys.stdout.flush()
        
        if i % 1000 == 0: 
            
            if not control[train_control_encoder_network_update]:
                save_snapshot(encoder_data, encoder_network)
                control[train_control_encoder_network_update] = 1
                
            if not control[train_control_decoder_network_update]:
                save_snapshot(decoder_data, decoder_network)
                control[train_control_decoder_network_update] = 1
        
    control[train_control_iteration] = 0
    
    print('Sending Final Networks...')
    sys.stdout.flush()

    while control[train_control_encoder_network_update]: time.sleep(0.01)
    save_snapshot(encoder_data, encoder_network)
    control[train_control_encoder_network_update] = 1
        
    while control[train_control_decoder_network_update]: time.sleep(0.01)
    save_snapshot(decoder_data, decoder_network)
    control[train_control_decoder_network_update] = 1


if __name__ == '__main__':
    train_autoencoder()
