# Local Controller Checkpoint Trainer

`train_controller.py` is a drop-in local replacement for Unreal Engine AnimGen's controller training entry point. It preserves the original command-line and shared-memory contracts while adding periodic, resumable local checkpoints.

The file currently targets the AnimGen script inspected at Unreal Engine revision `a899359c3efb`.

## Installation

Close any active AnimGen training process. Back up the Unreal Engine file below, then copy this directory's `train_controller.py` over it:

```text
Engine/Plugins/Experimental/Animation/AnimGen/Content/Python/train_controller.py
```

Do not replace `train_autoencoder.py`.

## Default behavior

- The trainer computes a compatibility fingerprint from the effective configuration, input arrays, and initial Unreal network snapshots.
- It looks for a compatible `latest.pth` and resumes it automatically.
- If no compatible checkpoint exists, it starts a new run.
- It saves every ten minutes, at the transition from flow matching to distillation, and after successful completion.
- Each save contains full PyTorch training state and UE-compatible snapshots for the controller and three LOD networks.
- It keeps the immediately preceding PyTorch checkpoint as `previous.pth`.
- It waits for Unreal Engine to acknowledge final shared-memory snapshots before exiting.
- It does not start a new save after an Unreal cancellation request is observed.

The default checkpoint directory is:

```text
<Project>/Saved/AnimGen/Checkpoints/<TaskName>/<FingerprintPrefix>/
```

## Environment variables

### `ANIMGEN_CHECKPOINT_RESUME`

Controls startup behavior:

- `auto` restores a compatible checkpoint when present and otherwise starts a new run. This is the default.
- `never` ignores existing checkpoints and starts a new run. New checkpoints are still written.
- `required` fails if a compatible checkpoint is not present.

### `ANIMGEN_CHECKPOINT_INTERVAL_SECONDS`

Sets the wall-clock interval between periodic checkpoints. The default is `600`. Set it to `0` to disable periodic saves; stage-transition and successful-completion saves still occur.

### `ANIMGEN_CHECKPOINT_ROOT`

Overrides the checkpoint root directory. The task name and compatibility fingerprint subdirectories are still appended.

Environment variables must be set before Unreal Editor starts so the training child process inherits them.

## Checkpoint contents

`latest.pth` contains:

- Controller, Denoiser, LOD0, LOD1, and LOD2 model states.
- Flow-matching and distillation optimizer states.
- Both scheduler states.
- Python, NumPy, PyTorch, and CUDA random number generator states.
- Current training stage, next iteration, rolling loss, and compatibility fingerprint.

The same directory also contains:

- `controller.bin`
- `lod0.bin`
- `lod1.bin`
- `lod2.bin`
- `latest.manifest.json`

The `.bin` files use Unreal's native learning-network snapshot format. The manifest records their lengths and SHA-256 digests.

## Compatibility behavior

Changing training inputs, schema, network initialization, architecture settings, hyperparameters, requested device, or relevant dimensions produces a different fingerprint and therefore a separate checkpoint directory.

Resume is exact rather than a warm start. The total iteration counts and learning-rate schedule must remain identical. To deliberately train from scratch with the same inputs, set `ANIMGEN_CHECKPOINT_RESUME=never` before launching Unreal Editor.

PyTorch `.pth` files use PyTorch serialization and must be treated as trusted local files. Do not download a checkpoint from an untrusted source or use this format as the future remote-training wire format.

## Operational notes

Controller checkpoints can be large because AdamW maintains additional tensors for each trained parameter. Atomic save also temporarily requires enough free space for both the previous checkpoint and the new temporary file.

If Unreal terminates Python while a periodic save is in progress, the temporary file may remain, but `latest.pth` or `previous.pth` remains the recovery source. Temporary files are never selected automatically for resume.
