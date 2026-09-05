# -*- coding: utf-8 -*-
"""
Copyright Epic Games, Inc. All Rights Reserved.
"""

# Local checkpointing modifications are maintained by the train_anim_gen project.

import sys
import os
import time
import json
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

        layers.append(
            nn.Sequential(
                nn.Linear(input_size, hidden_num), nn.LayerNorm([hidden_num]), nn.GELU()
            )
        )

        for i in range(layer_num - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_num, hidden_num),
                    nn.LayerNorm([hidden_num]),
                    nn.GELU(),
                )
            )

        layers.append(nn.Linear(hidden_num, output_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = self.layers[0](x)

        for i in range(1, len(self.layers) - 1):
            h = h + self.layers[i](h)

        return self.layers[len(self.layers) - 1](h)


CHECKPOINT_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 300.0


def _safe_path_component(value):
    """Return a stable filename component without allowing path traversal."""
    safe_value = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    )
    safe_value = safe_value.strip(" .")
    return safe_value or "unnamed"


def _atomic_write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=4, sort_keys=True)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            chunk = input_file.read(CHECKPOINT_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _update_digest_with_array(digest, name, array):
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))

    byte_view = memoryview(array)
    if not byte_view.contiguous:
        raise RuntimeError("Checkpoint fingerprint input must be contiguous")
    byte_view = byte_view.cast("B")

    for offset in range(0, byte_view.nbytes, CHECKPOINT_CHUNK_SIZE):
        digest.update(byte_view[offset : offset + CHECKPOINT_CHUNK_SIZE])


def _model_signature(model: nn.Module) -> dict:
    """Describe topology and layer settings without hashing learned weights."""
    modules = {}
    for name, module in model.named_modules():
        settings = {
            key: value
            for key, value in vars(module).items()
            if not key.startswith("_")
            and key != "training"
            and isinstance(value, (str, int, float, bool, tuple, list, type(None)))
        }
        modules[name] = {"type": type(module).__qualname__, "settings": settings}
    return {
        "modules": modules,
        "tensors": {
            name: [list(value.shape), str(value.dtype)]
            for name, value in model.state_dict().items()
        },
    }


def _model_weights_digest(models: dict) -> str:
    digest = hashlib.sha256()
    for name, model in sorted(models.items()):
        for key, value in sorted(model.state_dict().items()):
            _update_digest_with_array(
                digest, name + "." + key, value.detach().cpu().contiguous().numpy()
            )
    return digest.hexdigest()


def _data_fingerprint(arrays: dict) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        _update_digest_with_array(digest, name, array)
    return digest.hexdigest()


def _stage_fingerprint(
    config: dict, device: str, name: str, models: dict, dependencies: dict
) -> str:
    excluded = {
        "TaskName",
        "TimeStamp",
        "EnginePath",
        "SitePackagesPath",
        "IntermediatePath",
        "EnableTensorboard",
        "IterationNum",
        "IterationDistillNum",
        "RangeSequenceNames",
    }
    # LOD architecture is provided by the loaded network, not by its byte count.
    settings = {
        key: value
        for key, value in config.items()
        if key not in excluded
        and not key.endswith("Guid")
        and not key.startswith("LOD")
        and not (name == "teacher" and key == "DenoiserSteps")
    }
    payload = {
        "stage": name,
        "device": str(device),
        "settings": settings,
        "dependencies": dependencies,
        "networks": {key: _model_signature(model) for key, model in models.items()},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class TrainingSchedule:
    """Persist a decay segment; an increased budget starts a new segment."""

    def __init__(
        self,
        optimizer,
        learning_rate: float,
        warmup: int,
        target: int,
        completed: int = 0,
        state: dict | None = None,
    ):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.warmup = max(1, warmup)
        self.completed = completed
        self.start = int(state["start"]) if state else 0
        self.target = int(state["target"]) if state else target
        if target > self.target:
            self.start = completed
            self.target = target
        self._apply()

    def _apply(self) -> None:
        offset = self.completed - self.start
        span = max(1, self.target - self.start)
        factor = min((offset + 1) / self.warmup, 1.0) * max(1.0 - offset / span, 0.0)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate * factor

    def step(self) -> None:
        self.completed += 1
        self._apply()

    def get_last_lr(self) -> list:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"start": self.start, "target": self.target}


class ControllerCheckpointStore:
    """One dependency-scoped training state, with an independent optimizer."""

    def __init__(
        self,
        config: dict,
        device: str,
        name: str,
        models: dict,
        dependencies: dict,
        target: int,
        save_snapshot_to_file,
        ue_networks: dict,
    ):
        self.name = name
        self.models = models
        self.dependencies = dependencies
        self.target = target
        self.config = config
        self.ue_networks = ue_networks
        self.save_snapshot_to_file = save_snapshot_to_file
        self.resume_mode = (
            os.environ.get("ANIMGEN_CHECKPOINT_RESUME", "auto").strip().lower()
        )
        if self.resume_mode not in {"auto", "never", "required"}:
            raise RuntimeError(
                "ANIMGEN_CHECKPOINT_RESUME must be auto, never, or required"
            )
        self.interval_seconds = float(
            os.environ.get(
                "ANIMGEN_CHECKPOINT_INTERVAL_SECONDS",
                str(DEFAULT_CHECKPOINT_INTERVAL_SECONDS),
            )
        )
        if not math.isfinite(self.interval_seconds) or self.interval_seconds < 0:
            raise RuntimeError(
                "ANIMGEN_CHECKPOINT_INTERVAL_SECONDS must be finite and nonnegative"
            )
        self.fingerprint = _stage_fingerprint(
            config, device, name, models, dependencies
        )
        configured_root = os.environ.get("ANIMGEN_CHECKPOINT_ROOT")
        intermediate = Path(config["IntermediatePath"])
        if configured_root:
            root = Path(configured_root).expanduser()
        elif (
            intermediate.name.lower() == "animgen"
            and intermediate.parent.name.lower() == "intermediate"
        ):
            root = intermediate.parent.parent / "Saved" / "AnimGen" / "Checkpoints"
        else:
            root = intermediate / "Checkpoints"
        task_directory = root / _safe_path_component(config["TaskName"])
        self.directory = task_directory / name / self.fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / "latest.pth"
        self.previous_path = self.directory / "previous.pth"
        self.manifest_path = self.directory / "latest.manifest.json"
        self.optimizer = torch.optim.AdamW(
            [
                parameter
                for model in models.values()
                for parameter in model.parameters()
            ],
            lr=config["LearningRate"],
            amsgrad=True,
        )
        self.completed = 0
        self.rolling_avg_loss = None
        schedule_state = None
        self.last_saved_iteration = None
        if self.resume_mode != "never" and self.latest_path.exists():
            # Trusted local state only; this is not a network transport format.
            checkpoint = torch.load(
                self.latest_path, map_location="cpu", weights_only=False
            )
            if checkpoint.get("fingerprint") != self.fingerprint:
                raise RuntimeError(
                    "Controller checkpoint compatibility fingerprint mismatch"
                )
            if (
                checkpoint.get("dependencies") != dependencies
                or checkpoint.get("stage") != name
            ):
                raise RuntimeError("Controller checkpoint dependency mismatch")
            self.completed = int(checkpoint["progress"]["next_iteration"])
            self.last_saved_iteration = self.completed
            if self.completed < 0:
                raise RuntimeError(
                    "Controller checkpoint iteration must be nonnegative"
                )
            for key, model in models.items():
                model.load_state_dict(checkpoint["models"][key], strict=True)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            schedule_state = checkpoint["schedule"]
            self.rolling_avg_loss = checkpoint["progress"]["rolling_avg_loss"]
            random.setstate(checkpoint["rng"]["python"])
            import numpy as np

            np.random.set_state(checkpoint["rng"]["numpy"])
            torch.set_rng_state(checkpoint["rng"]["torch"])
            if torch.cuda.is_available() and checkpoint["rng"]["cuda"] is not None:
                torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
            print(
                "Resuming %s at iteration %d (target %d)."
                % (name, self.completed, target)
            )
        elif self.resume_mode == "required":
            raise RuntimeError(
                "A compatible controller checkpoint is required but was not found"
            )
        else:
            print(
                "Starting %s from iteration 0: no compatible checkpoint or resume disabled."
                % name
            )
        self.scheduler = TrainingSchedule(
            self.optimizer,
            config["LearningRate"],
            int(config["WarmupIterations"]),
            target,
            self.completed,
            schedule_state,
        )
        self.last_save_time = time.monotonic()
        sys.stdout.flush()

    def is_due(self) -> bool:
        return (
            self.interval_seconds > 0
            and time.monotonic() - self.last_save_time >= self.interval_seconds
        )

    def save(self) -> None:
        import numpy as np

        checkpoint = {
            "fingerprint": self.fingerprint,
            "dependencies": self.dependencies,
            "stage": self.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "progress": {
                "next_iteration": self.completed,
                "rolling_avg_loss": self.rolling_avg_loss,
            },
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optimizer": self.optimizer.state_dict(),
            "schedule": self.scheduler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
        }
        temporary = self.latest_path.with_suffix(".pth.tmp")
        with open(temporary, "wb") as output:
            torch.save(checkpoint, output)
            output.flush()
            os.fsync(output.fileno())
        if self.latest_path.exists():
            os.replace(self.latest_path, self.previous_path)
        os.replace(temporary, self.latest_path)
        artifacts = {}
        for name, network in self.ue_networks.items():
            path = self.directory / (name + ".bin")
            temporary = path.with_suffix(".bin.tmp")
            self.save_snapshot_to_file(network, temporary)
            os.replace(temporary, path)
            artifacts[name] = {
                "file": path.name,
                "byte_length": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        _atomic_write_json(
            self.manifest_path,
            {
                "stage": self.name,
                "fingerprint": self.fingerprint,
                "dependencies": self.dependencies,
                "created_at_utc": checkpoint["created_at_utc"],
                "next_iteration": self.completed,
                "target_iteration": self.target,
                "checkpoint": {
                    "file": "latest.pth",
                    "byte_length": self.latest_path.stat().st_size,
                    "sha256": _sha256_file(self.latest_path),
                },
                "artifacts": artifacts,
            },
        )
        self.last_save_time = time.monotonic()
        self.last_saved_iteration = self.completed
        print(
            "Saved controller checkpoint at stage %s, iteration %d."
            % (self.name, self.completed)
        )
        sys.stdout.flush()


def train_controller():
    """Load Config"""
    if len(sys.argv) != 2:
        raise Exception("Wrong number of arguments to training script")

    # Get the config file path from the command line arguments and load it
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f, object_pairs_hook=OrderedDict)

    # We need to append some things to the site-packages so we can load NNE models in python
    sys.path.append(config["SitePackagesPath"])
    sys.path.append(
        config["EnginePath"] + "Plugins/Experimental/NNERuntimeBasicCpu/Content/Python/"
    )
    sys.path.append(
        config["EnginePath"] + "Plugins/Experimental/LearningCore/Content/Python/"
    )

    """ Imports from site-packages """
    import numpy as np
    import torch
    from nne_runtime_basic_cpu_pytorch import NeuralNetwork
    from learning_core.communicators.shared_memory import SharedMemory
    from learning_core.train_common import (
        load_snapshot,
        save_snapshot,
        save_snapshot_to_file,
        schema_noise_mask_observation,
        schema_annotate_normalization_observation,
        schema_write_norm_to_network,
    )

    use_tensorboard = True
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("Failed to import TensorBoard. Please add manually to site-packages.")
        sys.stdout.flush()
        use_tensorboard = False

    """ Settings """
    training_name = config["TaskName"]
    timestamp = config["TimeStamp"]
    training_identifier = training_name + "_" + timestamp

    range_num = int(config["RangeNum"])
    database_total_frame_num = int(config["DatabaseTotalFrameNum"])
    control_total_frame_num = int(config["ControlTotalFrameNum"])

    dt = config["DeltaTime"]
    control_vector_size = int(config["ControlVectorSize"])
    encoded_control_vector_size = int(config["EncodedControlVectorSize"])
    pose_encoding_size = int(config["PoseEncodingSize"])
    control_schema = config["ControlSchema"]

    niterations_denoise = int(config["IterationNum"])
    niterations_distill = int(config["IterationDistillNum"])

    batchsize = int(config["BatchSize"])
    seed = int(config["Seed"])
    device = (
        {"GPU": "cuda", "CPU": "cpu"}[config["Device"]]
        if torch.cuda.is_available()
        else "cpu"
    )
    steps = int(config["DenoiserSteps"])
    if steps <= 0 or niterations_denoise < 0 or niterations_distill < 0:
        raise RuntimeError(
            "Training budgets must be nonnegative and DenoiserSteps must be positive"
        )
    denoiser_hidden_num = config["DenoiserHiddenUnitNum"]
    denoiser_layer_num = config["DenoiserLayerNum"]

    pose_noise_scale = config["PoseNoiseScale"]
    control_noise_scale = config["ControlNoiseScale"]
    random_pose_sample_rate = config["RandomPoseSampleRate"]
    normalized_pose_stds = torch.as_tensor(
        config["NormalizedPoseStds"], device=device, dtype=torch.float32
    )

    window = 8 + 1

    snapshots_dir = config["IntermediatePath"] + "/" + training_identifier
    if not os.path.exists(snapshots_dir):
        os.mkdir(snapshots_dir)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    """ Load Data """
    print("Loading Data...")

    def shared_memory_map_array(guid, shape, dtype):
        size = np.asarray(shape, dtype=np.int64).prod() * np.dtype(dtype).itemsize
        if size > 0:
            handle = SharedMemory(guid, create=False, size=size)
            assert handle is not None
            array = np.frombuffer(
                handle.buf, dtype=dtype, count=np.asarray(shape, dtype=np.int64).prod()
            ).reshape(shape)
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
    control_hndl, control = shared_memory_map_array(
        config["ControlGuid"], [train_control_num], np.int32
    )

    # Network data sizes
    controller_byte_num = int(config["ControllerByteNum"])
    lod0_byte_num = int(config["LOD0ByteNum"])
    lod1_byte_num = int(config["LOD1ByteNum"])
    lod2_byte_num = int(config["LOD2ByteNum"])

    # Load all network data arrays
    controller_data_hndl, controller_data = shared_memory_map_array(
        config["ControllerGuid"], [controller_byte_num], np.uint8
    )
    lod0_data_hndl, lod0_data = shared_memory_map_array(
        config["LOD0Guid"], [lod0_byte_num], np.uint8
    )
    lod1_data_hndl, lod1_data = shared_memory_map_array(
        config["LOD1Guid"], [lod1_byte_num], np.uint8
    )
    lod2_data_hndl, lod2_data = shared_memory_map_array(
        config["LOD2Guid"], [lod2_byte_num], np.uint8
    )

    # Load data arrays
    range_control_starts_hndl, range_control_starts = shared_memory_map_array(
        config["RangeControlStartsGuid"], [range_num], np.int32
    )
    range_encoded_starts_hndl, range_encoded_starts = shared_memory_map_array(
        config["RangeEncodedStartsGuid"], [range_num], np.int32
    )
    range_lens_hndl, range_lens = shared_memory_map_array(
        config["RangeLengthsGuid"], [range_num], np.int32
    )

    Chndl, C = shared_memory_map_array(
        config["ControlVectorsGuid"],
        [control_total_frame_num, control_vector_size],
        np.float32,
    )
    Xhndl, X = shared_memory_map_array(
        config["EncodedVectorsGuid"],
        [database_total_frame_num, pose_encoding_size],
        np.float32,
    )

    """ Load Networks """
    print("Loading Networks...")

    controller_network = NeuralNetwork(device=device)
    load_snapshot(controller_data, controller_network)
    print("Controller Network: \n", controller_network)

    print("Performing Normalization...")

    schema_annotate_normalization_observation(control_schema, C, 0.001)
    schema_write_norm_to_network(control_schema, controller_network.model)

    denoiser_network = DenoiserNetwork(
        input_size=5 * pose_encoding_size + 1 + encoded_control_vector_size,
        output_size=4 * pose_encoding_size,
        hidden_num=denoiser_hidden_num,
        layer_num=denoiser_layer_num,
    ).to(device)
    print("Denoiser Network: \n", denoiser_network)

    lod0_network = NeuralNetwork(device=device)
    lod1_network = NeuralNetwork(device=device)
    lod2_network = NeuralNetwork(device=device)
    load_snapshot(lod0_data, lod0_network)
    load_snapshot(lod1_data, lod1_network)
    load_snapshot(lod2_data, lod2_network)
    print("LOD0 Network: \n", lod0_network)

    print("Computing controller checkpoint data dependency...")
    sys.stdout.flush()
    data_dependency = _data_fingerprint(
        {
            "range_control_starts": range_control_starts,
            "range_encoded_starts": range_encoded_starts,
            "range_lengths": range_lens,
            "control_vectors": C,
            "encoded_vectors": X,
        }
    )
    teacher_models = {"controller": controller_network, "denoiser": denoiser_network}
    teacher_store = ControllerCheckpointStore(
        config,
        device,
        "teacher",
        teacher_models,
        {"data": data_dependency},
        niterations_denoise,
        save_snapshot_to_file,
        {"controller": controller_network},
    )
    optimizer = teacher_store.optimizer
    scheduler = teacher_store.scheduler
    flow_start_iteration = teacher_store.completed

    print("Creating Batches...")
    control_window_indices = []
    encoded_window_indices = []
    for ri in range(len(range_lens)):
        if range_lens[ri] >= window:
            for fi in range(range_lens[ri] - window + 1):
                control_window_indices.append(
                    range_control_starts[ri] + fi + np.arange(window)
                )
                encoded_window_indices.append(
                    range_encoded_starts[ri] + fi + np.arange(window)
                )
    control_window_indices = torch.as_tensor(
        np.array(control_window_indices, dtype=np.int64)
    )
    encoded_window_indices = torch.as_tensor(
        np.array(encoded_window_indices, dtype=np.int64)
    )

    # Save config
    with open(snapshots_dir + "/config.json", "w") as f:
        json.dump(config, f, indent=4)

    if use_tensorboard:
        writer = SummaryWriter(log_dir=snapshots_dir, max_queue=1000)

    """ Pin Training Data """

    X = torch.from_numpy(X).pin_memory()
    C = torch.from_numpy(C).pin_memory()

    """ Train Flow-Matching """

    output_frames_np = np.array([1, 2, 4, 8])
    output_frames = torch.as_tensor(output_frames_np, device=device, dtype=torch.long)

    print("Training Flow-Matching...")

    rolling_avg_loss = teacher_store.rolling_avg_loss
    exiting = False

    i = flow_start_iteration
    while i < niterations_denoise:
        if control[train_control_exit]:
            control[train_control_exit] = 0
            exiting = True
            print("Exiting...")
            break

        control[train_control_training_iteration] = i

        # Sample windows of pose vectors and control vectors to train over
        batch_indices = torch.randint(
            0, len(encoded_window_indices), size=[batchsize], dtype=torch.long
        )
        Xgnd = X[encoded_window_indices[batch_indices]].to(
            device, non_blocking=True
        )  # (batchsize, window, pose_encoding_size)
        Cgnd = C[control_window_indices[batch_indices][:, 0]].to(
            device, non_blocking=True
        )  # (batchsize, control_vector_size)

        # Sample random frames (we will sometimes use this as the previous pose)
        random_indices = torch.randint(0, len(X), size=[batchsize], dtype=torch.long)
        Xrnd = X[random_indices].to(
            device, non_blocking=True
        )  # (batchsize, pose_encoding_size)

        # Sample the initial noise, scaled by normalized_pose_stds
        Xsrc = (
            normalized_pose_stds
            * torch.clip(
                torch.randn(
                    [batchsize, 4, pose_encoding_size],
                    dtype=torch.float32,
                    device=device,
                ),
                min=-3,
                max=3,
            )
        ).reshape([batchsize, 4 * pose_encoding_size])

        # Sample a mask for when to replace the previous pose with a random pose
        Xnoised_mask = (
            torch.rand([batchsize, 1], dtype=torch.float32, device=device)
            < random_pose_sample_rate
        )

        # Sample the noise to add to the previous pose
        Xnoised_add = (
            torch.rand([batchsize, 1], dtype=torch.float32, device=device)
            * pose_noise_scale
            * normalized_pose_stds
            * torch.clip(
                torch.randn([batchsize, pose_encoding_size], device=device),
                min=-3,
                max=3,
            )
        )

        # Sample the noise to add to the control vectors
        Cnoise_add = (
            torch.rand([batchsize, 1], dtype=torch.float32, device=device)
            * control_noise_scale
            * torch.clip(
                torch.randn([batchsize, control_vector_size], device=device),
                min=-3,
                max=3,
            )
        )

        # Sample the alpha value of the flow
        alpha = torch.rand([batchsize, 1], dtype=torch.float32, device=device)

        # Compute the noised version of the control vector
        Cnoised = (
            Cgnd
            + schema_noise_mask_observation(control_schema, Cgnd, use_scales=True)
            * Cnoise_add
        )

        # Compute the noised previous pose using either Xnoised_add or the random pose Xrnd
        Xnoised = torch.where(Xnoised_mask, Xrnd, Xgnd[:, 0]) + Xnoised_add

        # Compute the output pose vectors using output_frames
        Xout = Xgnd[:, output_frames].reshape([batchsize, -1])

        # Linearly interpolate using alpha between the ground truth and the sampled initial noise
        Xalpha = alpha * Xout + (1.0 - alpha) * Xsrc

        # Compute flow matching predicted velocity
        Xpred = denoiser_network(
            torch.cat([Xnoised, Xalpha, alpha, controller_network(Cnoised)], dim=1)
        )

        # Compute flow matching loss
        loss = torch.mean((Xpred - (Xout - Xsrc)) ** 2)

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
            print(("Iter: %7i Loss: %7.5f" % (i, rolling_avg_loss)))

        if use_tensorboard:
            writer.add_scalar("loss/flow", loss_item, i)
            writer.add_scalar("lr/flow", scheduler.get_last_lr()[0], i)

        # Save Networks

        if i % 1000 == 0:
            if not control[train_control_control_encoder_network_update]:
                save_snapshot(controller_data, controller_network)
                control[train_control_control_encoder_network_update] = 1

        i += 1

        teacher_store.completed = i
        teacher_store.rolling_avg_loss = rolling_avg_loss
        if teacher_store.is_due() and not control[train_control_exit]:
            teacher_store.save()

    if (
        not exiting
        and not control[train_control_exit]
        and teacher_store.completed != teacher_store.last_saved_iteration
    ):
        teacher_store.save()

    while control[train_control_control_encoder_network_update]:
        time.sleep(0.01)
    save_snapshot(controller_data, controller_network)
    control[train_control_control_encoder_network_update] = 1

    """ Train LODs """

    alphas = (torch.arange(steps, device=device, dtype=torch.float32) / steps)[
        :, None, None
    ].tile([1, batchsize, 1])

    print("Training LODs...")

    lod_networks = [lod0_network, lod1_network, lod2_network]
    lod_stores = []
    if not exiting and not control[train_control_exit]:
        teacher_dependency = {
            "teacher_fingerprint": teacher_store.fingerprint,
            "teacher_weights": _model_weights_digest(teacher_models),
        }
        for index, network in enumerate(lod_networks):
            name = "lod%d" % index
            lod_stores.append(
                ControllerCheckpointStore(
                    config,
                    device,
                    name,
                    {name: network},
                    teacher_dependency,
                    niterations_distill,
                    save_snapshot_to_file,
                    {name: network},
                )
            )
    else:
        exiting = True
    rolling_avg_loss = None
    i = min((store.completed for store in lod_stores), default=niterations_distill)
    while not exiting and i < niterations_distill:
        if control[train_control_exit]:
            control[train_control_exit] = 0
            print("Exiting...")
            break

        control[train_control_distill_iteration] = i
        active_lods = [
            store.completed == i and i < store.target for store in lod_stores
        ]
        for network, active in zip(lod_networks, active_lods):
            network.requires_grad_(active)

        # Random indices to use for initial pose
        random_indices = torch.randint(
            0, len(encoded_window_indices), size=[batchsize], dtype=torch.long
        )

        # Sample indices to use for control vectors
        control_indices = torch.randint(
            0, len(encoded_window_indices), size=[batchsize], dtype=torch.long
        )

        # For initial pose either use random indices or those from the control vectors
        pose_indices = torch.where(
            torch.rand([batchsize], dtype=torch.float32) < random_pose_sample_rate,
            control_indices,
            random_indices,
        )

        # Sample pose vectors and control vectors
        Xgnd = X[encoded_window_indices[pose_indices][:, 0]].to(
            device, non_blocking=True
        )  # (batchsize, pose_encoding_size)
        Cgnd = C[control_window_indices[control_indices]].to(
            device, non_blocking=True
        )  # (batchsize, window, control_vector_size)

        # Compute the initial noise for each frame in the rollout
        Xsrc = (
            normalized_pose_stds
            * torch.clip(
                torch.randn(
                    [batchsize, window, 4, pose_encoding_size],
                    dtype=torch.float32,
                    device=device,
                ),
                min=-3,
                max=3,
            )
        ).reshape([batchsize, window, 4 * pose_encoding_size])

        # Compute the noise to add to the inital frame
        Xnoised_add = (
            torch.rand([batchsize, 1], dtype=torch.float32, device=device)
            * pose_noise_scale
            * normalized_pose_stds
            * torch.clip(
                torch.randn(
                    [batchsize, pose_encoding_size], dtype=torch.float32, device=device
                ),
                min=-3,
                max=3,
            )
        )

        # Compute the noise to add to all the control vectors
        Cnoise_add = (
            torch.rand([batchsize, 1, 1], dtype=torch.float32, device=device)
            * control_noise_scale
            * torch.clip(
                torch.randn(
                    [batchsize, window, control_vector_size],
                    dtype=torch.float32,
                    device=device,
                ),
                min=-3,
                max=3,
            )
        )

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
        Cnoise_mask = schema_noise_mask_observation(
            control_schema,
            Cgnd.reshape([batchsize * window, control_vector_size]),
            use_scales=True,
        ).reshape([batchsize, window, control_vector_size])
        with torch.no_grad():
            Cenc = controller_network(
                (Cgnd + Cnoise_mask * Cnoise_add).reshape(
                    [batchsize * window, control_vector_size]
                )
            ).reshape([batchsize, window, encoded_control_vector_size])

        # Set the current pose to the initial pose
        Xtarg = Xnoised.clone()
        Xlod0 = Xnoised.clone()
        Xlod1 = Xnoised.clone()
        Xlod2 = Xnoised.clone()

        # Perform rollout
        for ii, wi in enumerate(range(0, window, frame_skip)):
            # Result from flow network
            with torch.no_grad():
                Xpred_targ = Xsrc[:, wi].clone()
                for step in range(steps):
                    Xpred_targ += (1 / steps) * denoiser_network(
                        torch.cat([Xtarg, Xpred_targ, alphas[step], Cenc[:, wi]], dim=1)
                    )

            # Result from LODs
            Xpred_lod0 = lod0_network(
                torch.cat([Xlod0, Xsrc[:, wi], Cenc[:, wi]], dim=1)
            )
            Xpred_lod1 = lod1_network(
                torch.cat([Xlod1, Xsrc[:, wi], Cenc[:, wi]], dim=1)
            )
            Xpred_lod2 = lod2_network(
                torch.cat([Xlod2, Xsrc[:, wi], Cenc[:, wi]], dim=1)
            )

            Xout_targ.append(Xpred_targ)
            Xout_lod0.append(Xpred_lod0)
            Xout_lod1.append(Xpred_lod1)
            Xout_lod2.append(Xpred_lod2)

            # Next frame is now based on output_choice
            Xtarg = Xpred_targ.reshape([batchsize, 4, -1])[:, output_choice]
            Xlod0 = Xpred_lod0.reshape([batchsize, 4, -1])[:, output_choice]
            Xlod1 = Xpred_lod1.reshape([batchsize, 4, -1])[:, output_choice]
            Xlod2 = Xpred_lod2.reshape([batchsize, 4, -1])[:, output_choice]

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
        Xtarg_vel = (Xout_targ[:, 1:] - Xout_targ[:, :-1]) / (frame_skip * dt)
        Xlod0_vel = (Xout_lod0[:, 1:] - Xout_lod0[:, :-1]) / (frame_skip * dt)
        Xlod1_vel = (Xout_lod1[:, 1:] - Xout_lod1[:, :-1]) / (frame_skip * dt)
        Xlod2_vel = (Xout_lod2[:, 1:] - Xout_lod2[:, :-1]) / (frame_skip * dt)

        # l1 loss on velocity of LOD estimations
        loss_vel_lod0 = 0.002 * torch.mean(torch.abs(Xtarg_vel - Xlod0_vel))
        loss_vel_lod1 = 0.002 * torch.mean(torch.abs(Xtarg_vel - Xlod1_vel))
        loss_vel_lod2 = 0.002 * torch.mean(torch.abs(Xtarg_vel - Xlod2_vel))

        # Keep the original per-network gradient scale while updating only pending LODs.
        lod_losses = [
            (loss_lod0 + loss_vel_lod0) / 6,
            (loss_lod1 + loss_vel_lod1) / 6,
            (loss_lod2 + loss_vel_lod2) / 6,
        ]
        loss_lod = sum(loss for loss, active in zip(lod_losses, active_lods) if active)
        for store, active in zip(lod_stores, active_lods):
            if active:
                store.optimizer.zero_grad()
        loss_lod.backward()
        for store, loss, active in zip(lod_stores, lod_losses, active_lods):
            if active:
                store.optimizer.step()
                store.scheduler.step()
                store.completed += 1
                value = loss.item()
                store.rolling_avg_loss = (
                    value
                    if store.rolling_avg_loss is None
                    else store.rolling_avg_loss * 0.99 + value * 0.01
                )

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
            print(("Iter: %7i Loss: %7.5f" % (i, rolling_avg_loss)))

        if use_tensorboard:
            writer.add_scalar("loss/lod0", loss_lod0.item(), i)
            writer.add_scalar("loss/lod1", loss_lod1.item(), i)
            writer.add_scalar("loss/lod2", loss_lod2.item(), i)
            writer.add_scalar("loss/vel_lod0", loss_vel_lod0.item(), i)
            writer.add_scalar("loss/vel_lod1", loss_vel_lod1.item(), i)
            writer.add_scalar("loss/vel_lod2", loss_vel_lod2.item(), i)
            writer.add_scalar("loss/lod", loss_lod_item, i)
            for index, store in enumerate(lod_stores):
                writer.add_scalar(
                    "lr/lod%d" % index, store.scheduler.get_last_lr()[0], i
                )

        # Save Networks

        if i % 1000 == 0:
            if not control[train_control_lod_network_update]:
                save_snapshot(lod0_data, lod0_network)
                save_snapshot(lod1_data, lod1_network)
                save_snapshot(lod2_data, lod2_network)
                control[train_control_lod_network_update] = 1

        # Update Iteration

        i += 1

        for store, active in zip(lod_stores, active_lods):
            if active and store.is_due() and not control[train_control_exit]:
                store.save()

    control[train_control_training_iteration] = 0
    control[train_control_distill_iteration] = 0

    while control[train_control_lod_network_update]:
        time.sleep(0.01)
    save_snapshot(lod0_data, lod0_network)
    save_snapshot(lod1_data, lod1_network)
    save_snapshot(lod2_data, lod2_network)
    control[train_control_lod_network_update] = 1

    if not exiting:
        for store in lod_stores:
            if control[train_control_exit]:
                break
            if store.completed != store.last_saved_iteration:
                store.save()

    if not exiting:
        print("Waiting for Unreal Engine to acknowledge the final networks...")
        sys.stdout.flush()
        while (
            control[train_control_control_encoder_network_update]
            or control[train_control_lod_network_update]
        ):
            if control[train_control_exit]:
                control[train_control_exit] = 0
                break
            time.sleep(0.01)

    if use_tensorboard:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    train_controller()
