# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''

# Local checkpointing modifications are maintained by the train_anim_gen project.

import sys
import os
import time
import json
import traceback
import hashlib
from datetime import datetime, timezone
from collections import OrderedDict
from pathlib import Path
import math
import random
import torch
import torch.nn as nn

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
        
        for i in range(1, len(self.layers) - 1):
            h = h + self.layers[i](h)
        
        return self.layers[len(self.layers) - 1](h)


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 300.0


def _safe_path_component(value):
    """Return a stable filename component without allowing path traversal."""
    safe_value = ''.join(character if character.isalnum() or character in '-_.' else '_' for character in str(value))
    safe_value = safe_value.strip(' .')
    return safe_value or 'unnamed'


def _atomic_write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with open(temporary_path, 'w', encoding='utf-8') as output_file:
        json.dump(value, output_file, indent=4, sort_keys=True)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        while True:
            chunk = input_file.read(CHECKPOINT_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _update_digest_with_array(digest, name, array):
    digest.update(name.encode('utf-8'))
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(json.dumps(list(array.shape), separators=(',', ':')).encode('ascii'))

    byte_view = memoryview(array)
    if not byte_view.contiguous:
        raise RuntimeError('Checkpoint fingerprint input must be contiguous')
    byte_view = byte_view.cast('B')

    for offset in range(0, byte_view.nbytes, CHECKPOINT_CHUNK_SIZE):
        digest.update(byte_view[offset:offset + CHECKPOINT_CHUNK_SIZE])


class ControllerCheckpointStore:
    """Persist and restore trusted local controller training state."""

    def __init__(self, config, device, fingerprint_arrays, save_snapshot_to_file):
        self.device = device
        self.save_snapshot_to_file = save_snapshot_to_file
        self.resume_mode = os.environ.get('ANIMGEN_CHECKPOINT_RESUME', 'auto').strip().lower()
        if self.resume_mode not in {'auto', 'never', 'required'}:
            raise RuntimeError('ANIMGEN_CHECKPOINT_RESUME must be auto, never, or required')

        interval_value = os.environ.get(
            'ANIMGEN_CHECKPOINT_INTERVAL_SECONDS',
            str(DEFAULT_CHECKPOINT_INTERVAL_SECONDS))
        self.interval_seconds = float(interval_value)
        if self.interval_seconds < 0.0:
            raise RuntimeError('ANIMGEN_CHECKPOINT_INTERVAL_SECONDS must not be negative')

        print('Computing controller checkpoint compatibility fingerprint...')
        sys.stdout.flush()
        self.fingerprint = self._make_fingerprint(config, device, fingerprint_arrays)

        configured_root = os.environ.get('ANIMGEN_CHECKPOINT_ROOT')
        if configured_root:
            checkpoint_root = Path(configured_root).expanduser()
        else:
            intermediate_path = Path(config['IntermediatePath'])
            if intermediate_path.name.lower() == 'animgen' and intermediate_path.parent.name.lower() == 'intermediate':
                checkpoint_root = intermediate_path.parent.parent / 'Saved' / 'AnimGen' / 'Checkpoints'
            else:
                checkpoint_root = intermediate_path / 'Checkpoints'

        task_name = _safe_path_component(config['TaskName'])
        self.directory = checkpoint_root / task_name / self.fingerprint[:16]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / 'latest.pth'
        self.previous_path = self.directory / 'previous.pth'
        self.manifest_path = self.directory / 'latest.manifest.json'
        self.last_save_time = time.monotonic()

    @staticmethod
    def _make_fingerprint(config, device, fingerprint_arrays):
        excluded_fields = {
            'TimeStamp',
            'EnginePath',
            'SitePackagesPath',
            'IntermediatePath',
            'EnableTensorboard',
        }
        stable_config = {
            key: value
            for key, value in config.items()
            if key not in excluded_fields and not key.endswith('Guid')
        }

        digest = hashlib.sha256()
        digest.update(b'animgen-controller-checkpoint-v1')
        digest.update(str(device).encode('ascii'))
        digest.update(json.dumps(stable_config, sort_keys=True, separators=(',', ':')).encode('utf-8'))

        for name, array in fingerprint_arrays.items():
            _update_digest_with_array(digest, name, array)

        return digest.hexdigest()

    def restore(self, models, optimizers, schedulers):
        if self.resume_mode == 'never':
            print('Checkpoint resume is disabled; starting a new training run.')
            return None

        if not self.latest_path.exists():
            if self.resume_mode == 'required':
                raise RuntimeError('A compatible controller checkpoint is required but was not found')
            print('No compatible controller checkpoint was found; starting a new training run.')
            return None

        # This file is local training state and must never be loaded from an untrusted source.
        # Load through CPU so CPU and CUDA RNG byte tensors retain the device expected by
        # their restoration APIs. Model and optimizer loading moves parameter state as needed.
        checkpoint = torch.load(self.latest_path, map_location='cpu', weights_only=False)
        if checkpoint.get('format_version') != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError('Unsupported controller checkpoint format version')
        if checkpoint.get('fingerprint') != self.fingerprint:
            raise RuntimeError('Controller checkpoint compatibility fingerprint mismatch')

        for name, model in models.items():
            model.load_state_dict(checkpoint['models'][name], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint['optimizers'][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(checkpoint['schedulers'][name])

        random.setstate(checkpoint['rng']['python'])
        import numpy as np
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        if torch.cuda.is_available() and checkpoint['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])

        stage = checkpoint['progress']['stage']
        next_iteration = int(checkpoint['progress']['next_iteration'])
        print('Resuming controller training at stage %s, iteration %d.' % (stage, next_iteration))
        sys.stdout.flush()
        self.last_save_time = time.monotonic()
        return checkpoint['progress']

    def is_due(self):
        return self.interval_seconds > 0.0 and time.monotonic() - self.last_save_time >= self.interval_seconds

    def save(self, stage, next_iteration, rolling_avg_loss, models, optimizers, schedulers, ue_networks):
        import numpy as np

        checkpoint = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'fingerprint': self.fingerprint,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'progress': {
                'stage': stage,
                'next_iteration': int(next_iteration),
                'rolling_avg_loss': rolling_avg_loss,
            },
            'models': {name: model.state_dict() for name, model in models.items()},
            'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            'schedulers': {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
            'rng': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        temporary_checkpoint = self.latest_path.with_suffix('.pth.tmp')
        with open(temporary_checkpoint, 'wb') as checkpoint_file:
            torch.save(checkpoint, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        if self.latest_path.exists():
            os.replace(self.latest_path, self.previous_path)
        os.replace(temporary_checkpoint, self.latest_path)

        artifacts = {}
        for name, network in ue_networks.items():
            artifact_path = self.directory / (name + '.bin')
            temporary_artifact_path = artifact_path.with_suffix('.bin.tmp')
            self.save_snapshot_to_file(network, temporary_artifact_path)
            os.replace(temporary_artifact_path, artifact_path)
            artifacts[name] = {
                'file': artifact_path.name,
                'byte_length': artifact_path.stat().st_size,
                'sha256': _sha256_file(artifact_path),
            }

        manifest = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'fingerprint': self.fingerprint,
            'created_at_utc': checkpoint['created_at_utc'],
            'stage': stage,
            'next_iteration': int(next_iteration),
            'checkpoint': {
                'file': self.latest_path.name,
                'byte_length': self.latest_path.stat().st_size,
                'sha256': _sha256_file(self.latest_path),
            },
            'artifacts': artifacts,
        }
        _atomic_write_json(self.manifest_path, manifest)
        self.last_save_time = time.monotonic()
        print('Saved controller checkpoint at stage %s, iteration %d.' % (stage, next_iteration))
        sys.stdout.flush()


def train_controller():

    """ Load Config """
    if len(sys.argv) != 2:
        raise Exception('Wrong number of arguments to training script')
    
    # Get the config file path from the command line arguments and load it
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
        load_snapshot, save_snapshot, save_snapshot_to_file,
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
    train_control_training_iteration = 1
    train_control_distill_iteration = 2
    train_control_training_loss = 3
    train_control_distill_loss = 4
    train_control_control_encoder_network_update = 5
    train_control_lod_network_update = 6
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
    range_control_starts_hndl, range_control_starts = shared_memory_map_array(config['RangeControlStartsGuid'], [range_num], np.int32)
    range_encoded_starts_hndl, range_encoded_starts = shared_memory_map_array(config['RangeEncodedStartsGuid'], [range_num], np.int32)
    range_lens_hndl, range_lens = shared_memory_map_array(config['RangeLengthsGuid'], [range_num], np.int32)
    
    Chndl, C = shared_memory_map_array(config['ControlVectorsGuid'], [control_total_frame_num, control_vector_size], np.float32)
    Xhndl, X = shared_memory_map_array(config['EncodedVectorsGuid'], [database_total_frame_num, pose_encoding_size], np.float32)
    
    """ Load Networks """
    print('Loading Networks...')

    controller_network = NeuralNetwork(device=device)
    load_snapshot(controller_data, controller_network)
    print("Controller Network: \n", controller_network)
    
    print('Performing Normalization...')
    
    schema_annotate_normalization_observation(control_schema, C, 0.001)
    schema_write_norm_to_network(control_schema, controller_network.model)
    
    denoiser_network = DenoiserNetwork(
        input_size=5*pose_encoding_size+1+encoded_control_vector_size, 
        output_size=4*pose_encoding_size,
        hidden_num=denoiser_hidden_num, 
        layer_num=denoiser_layer_num).to(device)
    print("Denoiser Network: \n", denoiser_network)

    lod0_network = NeuralNetwork(device=device)
    lod1_network = NeuralNetwork(device=device)
    lod2_network = NeuralNetwork(device=device)
    load_snapshot(lod0_data, lod0_network)
    load_snapshot(lod1_data, lod1_network)
    load_snapshot(lod2_data, lod2_network)
    print("LOD0 Network: \n", lod0_network)

    print('Creating Optimizer...')
    optimizer = torch.optim.AdamW(list(denoiser_network.parameters()) + list(controller_network.parameters()), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: np.minimum(i / lr_warmup, 1.0) * (1.0 - np.clip(float(i) / niterations_denoise, 0.0, 1.0)))

    optimizer_lod = torch.optim.AdamW(list(lod0_network.parameters()) + list(lod1_network.parameters()) + list(lod2_network.parameters()),  lr=lr, amsgrad=True)
    scheduler_lod = torch.optim.lr_scheduler.LambdaLR(optimizer_lod, lr_lambda=lambda i: np.minimum(i / lr_warmup, 1.0) * (1.0 - np.clip(float(i) / niterations_distill, 0.0, 1.0)))

    checkpoint_models = {
        'controller': controller_network,
        'denoiser': denoiser_network,
        'lod0': lod0_network,
        'lod1': lod1_network,
        'lod2': lod2_network,
    }
    checkpoint_optimizers = {
        'flow': optimizer,
        'distillation': optimizer_lod,
    }
    checkpoint_schedulers = {
        'flow': scheduler,
        'distillation': scheduler_lod,
    }
    checkpoint_ue_networks = {
        'controller': controller_network,
        'lod0': lod0_network,
        'lod1': lod1_network,
        'lod2': lod2_network,
    }
    checkpoint_fingerprint_arrays = OrderedDict([
        ('range_control_starts', range_control_starts),
        ('range_encoded_starts', range_encoded_starts),
        ('range_lengths', range_lens),
        ('control_vectors', C),
        ('encoded_vectors', X),
        ('initial_controller_snapshot', controller_data),
        ('initial_lod0_snapshot', lod0_data),
        ('initial_lod1_snapshot', lod1_data),
        ('initial_lod2_snapshot', lod2_data),
    ])
    checkpoint_store = ControllerCheckpointStore(
        config,
        device,
        checkpoint_fingerprint_arrays,
        save_snapshot_to_file)
    resume_progress = checkpoint_store.restore(
        checkpoint_models,
        checkpoint_optimizers,
        checkpoint_schedulers)

    resume_stage = resume_progress['stage'] if resume_progress is not None else 'flow_matching'
    resume_iteration = int(resume_progress['next_iteration']) if resume_progress is not None else 0
    resume_loss = resume_progress['rolling_avg_loss'] if resume_progress is not None else None

    if resume_stage not in {'flow_matching', 'distillation', 'completed'}:
        raise RuntimeError('Unknown controller checkpoint stage')
    if resume_stage == 'flow_matching' and not 0 <= resume_iteration <= niterations_denoise:
        raise RuntimeError('Flow-matching checkpoint iteration is outside the configured range')
    if resume_stage == 'distillation' and not 0 <= resume_iteration <= niterations_distill:
        raise RuntimeError('Distillation checkpoint iteration is outside the configured range')

    flow_start_iteration = resume_iteration if resume_stage == 'flow_matching' else niterations_denoise
    if resume_stage == 'distillation':
        distill_start_iteration = resume_iteration
    elif resume_stage == 'completed':
        distill_start_iteration = niterations_distill
    else:
        distill_start_iteration = 0

    print('Creating Batches...')
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
    
    X = torch.from_numpy(X).pin_memory()
    C = torch.from_numpy(C).pin_memory()
    
    """ Train Flow-Matching """
    
    output_frames_np = np.array([1, 2, 4, 8])
    output_frames = torch.as_tensor(output_frames_np, device=device, dtype=torch.long)

    print('Training Flow-Matching...')
    
    rolling_avg_loss = resume_loss if resume_stage == 'flow_matching' else None
    exiting = False
    
    i = flow_start_iteration
    while i < niterations_denoise:
        
        if control[train_control_exit]:
            control[train_control_exit] = 0
            exiting = True
            print('Exiting...')
            break
        
        control[train_control_training_iteration] = i

        # Sample windows of pose vectors and control vectors to train over
        batch_indices = torch.randint(0, len(encoded_window_indices), size=[batchsize], dtype=torch.long)
        Xgnd = X[encoded_window_indices[batch_indices]].to(device, non_blocking=True) # (batchsize, window, pose_encoding_size)
        Cgnd = C[control_window_indices[batch_indices][:,0]].to(device, non_blocking=True) # (batchsize, control_vector_size)

        # Sample random frames (we will sometimes use this as the previous pose)
        random_indices = torch.randint(0, len(X), size=[batchsize], dtype=torch.long)
        Xrnd = X[random_indices].to(device, non_blocking=True) # (batchsize, pose_encoding_size)
        
        # Sample the initial noise, scaled by normalized_pose_stds
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
        Xnoised = torch.where(Xnoised_mask, Xrnd, Xgnd[:,0]) + Xnoised_add
        
        # Compute the output pose vectors using output_frames
        Xout = Xgnd[:,output_frames].reshape([batchsize, -1])
        
        # Linearly interpolate using alpha between the ground truth and the sampled initial noise
        Xalpha = alpha * Xout + (1.0 - alpha) * Xsrc

        # Compute flow matching predicted velocity
        Xpred = denoiser_network(torch.cat([Xnoised, Xalpha, alpha, controller_network(Cnoised)], dim=1))
        
        # Compute flow matching loss
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

        if checkpoint_store.is_due():
            checkpoint_store.save(
                'flow_matching',
                i,
                rolling_avg_loss,
                checkpoint_models,
                checkpoint_optimizers,
                checkpoint_schedulers,
                checkpoint_ue_networks)

    if not exiting and resume_stage == 'flow_matching':
        checkpoint_store.save(
            'distillation',
            0,
            None,
            checkpoint_models,
            checkpoint_optimizers,
            checkpoint_schedulers,
            checkpoint_ue_networks)
        
    while control[train_control_control_encoder_network_update]: time.sleep(0.01)
    save_snapshot(controller_data, controller_network)
    control[train_control_control_encoder_network_update] = 1

    """ Train LODs """
    
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) / steps)[:,None,None].tile([1,batchsize,1])

    print('Training LODs...')
    
    rolling_avg_loss = resume_loss if resume_stage == 'distillation' else None
    
    i = distill_start_iteration
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
        pose_indices = torch.where(torch.rand([batchsize], dtype=torch.float32) < random_pose_sample_rate, control_indices, random_indices)
        
        # Sample pose vectors and control vectors
        Xgnd = X[encoded_window_indices[pose_indices][:,0]].to(device, non_blocking=True) # (batchsize, pose_encoding_size)
        Cgnd = C[control_window_indices[control_indices]].to(device, non_blocking=True) # (batchsize, window, control_vector_size)
        
        # Compute the initial noise for each frame in the rollout
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
        output_choice = np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])
        frame_skip = output_frames_np[output_choice]
        
        # Compute the inital pose
        Xnoised = Xgnd + Xnoised_add    

        # Add noise and encode all of the control vectors
        Cnoise_mask = schema_noise_mask_observation(control_schema, Cgnd.reshape([batchsize * window, control_vector_size]), use_scales=True).reshape([batchsize, window, control_vector_size])
        with torch.no_grad(): 
            Cenc = controller_network((Cgnd + Cnoise_mask * Cnoise_add).reshape([batchsize * window, control_vector_size])).reshape([batchsize, window, encoded_control_vector_size])
        
        # Set the current pose to the initial pose
        Xtarg = Xnoised.clone()
        Xlod0 = Xnoised.clone()
        Xlod1 = Xnoised.clone()
        Xlod2 = Xnoised.clone()
        
        # Perform rollout
        for ii, wi in enumerate(range(0, window, frame_skip)):
            
            # Result from flow network
            with torch.no_grad(): 
                Xpred_targ = Xsrc[:,wi].clone()
                for step in range(steps):    
                    Xpred_targ += (1 / steps) * denoiser_network(torch.cat(
                        [Xtarg, Xpred_targ, alphas[step], Cenc[:,wi]], dim=1))
            
            # Result from LODs
            Xpred_lod0 = lod0_network(torch.cat([Xlod0, Xsrc[:,wi], Cenc[:,wi]], dim=1))
            Xpred_lod1 = lod1_network(torch.cat([Xlod1, Xsrc[:,wi], Cenc[:,wi]], dim=1))
            Xpred_lod2 = lod2_network(torch.cat([Xlod2, Xsrc[:,wi], Cenc[:,wi]], dim=1))
            
            Xout_targ.append(Xpred_targ)
            Xout_lod0.append(Xpred_lod0)
            Xout_lod1.append(Xpred_lod1)
            Xout_lod2.append(Xpred_lod2)
            
            # Next frame is now based on output_choice
            Xtarg = Xpred_targ.reshape([batchsize, 4, -1])[:,output_choice]
            Xlod0 = Xpred_lod0.reshape([batchsize, 4, -1])[:,output_choice]
            Xlod1 = Xpred_lod1.reshape([batchsize, 4, -1])[:,output_choice]
            Xlod2 = Xpred_lod2.reshape([batchsize, 4, -1])[:,output_choice]
        
        # Concatenate together
        Xout_targ = torch.stack(Xout_targ, dim=1)
        Xout_lod0 = torch.stack(Xout_lod0, dim=1)
        Xout_lod1 = torch.stack(Xout_lod1, dim=1)
        Xout_lod2 = torch.stack(Xout_lod2, dim=1)

        # l1 loss on LOD estimations
        loss_lod0 = 0.1 * torch.mean(torch.abs(Xout_targ - Xout_lod0))
        loss_lod1 = 0.1 * torch.mean(torch.abs(Xout_targ - Xout_lod1))
        loss_lod2 = 0.1 * torch.mean(torch.abs(Xout_targ - Xout_lod2))
        
        # Compute pose vector deltas
        Xtarg_vel = (Xout_targ[:,1:] - Xout_targ[:,:-1]) / (frame_skip * dt)
        Xlod0_vel = (Xout_lod0[:,1:] - Xout_lod0[:,:-1]) / (frame_skip * dt)
        Xlod1_vel = (Xout_lod1[:,1:] - Xout_lod1[:,:-1]) / (frame_skip * dt)
        Xlod2_vel = (Xout_lod2[:,1:] - Xout_lod2[:,:-1]) / (frame_skip * dt)
        
        # l1 loss on velocity of LOD estimations
        loss_vel_lod0 = 0.002 * torch.mean(torch.abs(Xtarg_vel - Xlod0_vel))
        loss_vel_lod1 = 0.002 * torch.mean(torch.abs(Xtarg_vel - Xlod1_vel))
        loss_vel_lod2 = 0.002 * torch.mean(torch.abs(Xtarg_vel - Xlod2_vel))
        
        # Sum all losses
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
        
            if not control[train_control_lod_network_update]:
                save_snapshot(lod0_data, lod0_network)
                save_snapshot(lod1_data, lod1_network)
                save_snapshot(lod2_data, lod2_network)
                control[train_control_lod_network_update] = 1

        # Update Iteration

        i += 1

        if checkpoint_store.is_due():
            checkpoint_store.save(
                'distillation',
                i,
                rolling_avg_loss,
                checkpoint_models,
                checkpoint_optimizers,
                checkpoint_schedulers,
                checkpoint_ue_networks)
        
    
    control[train_control_training_iteration] = 0
    control[train_control_distill_iteration] = 0

    while control[train_control_lod_network_update]: time.sleep(0.01)
    save_snapshot(lod0_data, lod0_network)
    save_snapshot(lod1_data, lod1_network)
    save_snapshot(lod2_data, lod2_network)
    control[train_control_lod_network_update] = 1

    if not exiting and resume_stage != 'completed':
        checkpoint_store.save(
            'completed',
            0,
            rolling_avg_loss,
            checkpoint_models,
            checkpoint_optimizers,
            checkpoint_schedulers,
            checkpoint_ue_networks)

    if not exiting:
        print('Waiting for Unreal Engine to acknowledge the final networks...')
        sys.stdout.flush()
        while (control[train_control_control_encoder_network_update] or
               control[train_control_lod_network_update]):
            if control[train_control_exit]:
                control[train_control_exit] = 0
                break
            time.sleep(0.01)

    if use_tensorboard:
        writer.flush()
        writer.close()

if __name__ == '__main__':
    train_controller()
