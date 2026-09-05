# Local Controller Checkpoint Trainer

`train_controller_checkpoint.py` is a drop-in local replacement for the AnimGen controller training entry point inspected at Unreal Engine revision `a899359c3efb`. It preserves the command-line, snapshot format, and shared-memory indexes. The original script is [train_controller.py](../reference/train_controller.py).

## Installation

Back up the installed `Engine/Plugins/Experimental/Animation/AnimGen/Content/Python/train_controller.py`, then copy this script to that location with the destination filename `train_controller.py`. No additional Python helper files are required. Installing a new script does not change an already running training process.

## Dependency-based resume

Training state is split into four independently resumable groups:

- `teacher`: denoiser and control encoder together, with their joint optimizer.
- `lod0`, `lod1`, `lod2`: one network and optimizer per group.

Each group stores its own completed iteration count, learning-rate schedule, rolling loss, and random number generator state. Iterations are minibatch updates, not dataset epochs. The default `auto` mode restores a matching group; otherwise that group starts from the network initialization supplied by Unreal (or the newly initialized denoiser). A new optimizer is used for an incompatible group.

| Change | Result |
| --- | --- |
| Increase `IterationNum` | Continue the teacher from its saved count and optimizer state. |
| Increase `IterationDistillNum` | Skip an already complete teacher; continue each LOD from its saved count. |
| Reduce a target below the saved count | Skip that group; never rewind saved progress. |
| Change one LOD architecture | Retrain that LOD; preserve the teacher and other compatible LODs. |
| Change denoiser or control encoder architecture | Retrain the teacher; students depend on the resulting teacher version. |
| Change teacher weights through additional training | Select new student checkpoints, or train students anew if none match. |
| Change `DenoiserSteps` | Preserve the teacher; invalidate student checkpoints. |
| Change encoded data, control data, ranges, normalization, schema, or shared training hyperparameters | Invalidate the affected dependency chain. |

Teacher compatibility hashes the input arrays, effective device, relevant configuration, and network topology. Both total iteration budgets are excluded. LOD byte counts and sampling steps are excluded from the teacher identity. Student identity includes its own topology and a digest of the actual denoiser and control encoder weights, as well as the teacher compatibility fingerprint and distillation configuration. Changes to one LOD do not enter another LOD's identity.

Network topology includes layer types, scalar/list settings, parameter shapes, and dtypes. Incoming learned controller/LOD weights are deliberately not an identity: Unreal can send previously published weights back on a later run. A compatible checkpoint takes precedence over those incoming weights. Use `ANIMGEN_CHECKPOINT_RESUME=never` or remove the applicable checkpoint directory to deliberately replace a compatible trained network with different incoming weights. Unknown non-LOD configuration fields conservatively participate in compatibility; LOD architecture is derived from the loaded network, not its byte count.

No autoencoder weight fingerprint is requested from Unreal. Its encoded pose data and the normalization configuration are still hashed. If an autoencoder change is not reflected in those inputs (for example, stale encoded data), manually remove the task's checkpoint directory. The current C++ controller configuration does not expose autoencoder weights or a weight digest.

## Learning-rate extension

With an unchanged target, restore the optimizer and the saved learning-rate segment. Increasing a target preserves weights, optimizer moments, and cumulative iterations, but starts a new warmup/decay segment at the saved count, ending at the new target. This avoids resuming with the old terminal zero learning rate. It is not equivalent to having trained with the larger budget from the beginning. `WarmupIterations` remains a compatibility setting; the first update in a segment uses a positive warmup learning rate.

A teacher extension changes its actual weights, so its former students are not counted as already trained against the new teacher. The script does not silently warm-start them from an incompatible student checkpoint.

LODs with different saved counts share teacher rollouts. Only students whose saved count matches the current loop index are updated; more advanced or complete students remain unchanged. Per-network gradient scaling is preserved from the original three-student loss. When the active student set changes, the random sampling stream need not match a historical joint run bit for bit.

## Files and saving

The default directory is:

```text
<Project>/Saved/AnimGen/Checkpoints/<TaskName>/
  teacher/<DependencyFingerprint>/
  lod0/<DependencyFingerprint>/
  lod1/<DependencyFingerprint>/
  lod2/<DependencyFingerprint>/
```

Each leaf contains `latest.pth`, `latest.manifest.json`, and its UE snapshots (`controller.bin` for the teacher, or the corresponding `lodN.bin`). `latest.pth` contains full local training state; `previous.pth` retains the immediately preceding save. The manifest exposes the dependency digests, saved count, target at save time, artifact sizes, and SHA-256 checksums. Denoiser weights are in the teacher's `.pth` file, not a UE runtime snapshot.

The default save interval is **300 seconds**. The teacher is also saved after its training phase, and students after successful completion, when their state has changed. A periodic save is checked after an iteration. No new save starts after an observed cancellation. Final runtime snapshots still use Unreal's existing publication handshake. A fully resumed run with no updates does not rewrite its large checkpoint files.

Writes use a temporary file and replacement, retaining the previous checkpoint. Temporary files and `previous.pth` are not selected automatically for resume. Checkpoints can be large; atomic replacement temporarily requires additional disk space. Checkpoint files are trusted local PyTorch serialization, not a remote transport format.


## Environment variables

Set these before launching Unreal Editor so its child process inherits them:

- `ANIMGEN_CHECKPOINT_RESUME`: `auto` (default), `never` (start all groups fresh), or `required` (fail if any needed group has no compatible checkpoint). Under `required`, a teacher extension may leave no matching student checkpoints and fail at the student phase; use `auto` for selective retraining.
- `ANIMGEN_CHECKPOINT_INTERVAL_SECONDS`: default `300`; `0` disables periodic saves but retains phase-end saves.
- `ANIMGEN_CHECKPOINT_ROOT`: overrides the root before `<TaskName>/...` is appended.

## Validation

```powershell
python -m unittest discover -s animgen_checkpoint/tests -q
python -m ruff check animgen_checkpoint
python -m ruff format --check animgen_checkpoint
```

Tests run on CPU without Unreal, shared-memory services, or a GPU. They cover dependency invalidation, independent LOD progress, budget extensions, restored optimizer/schedule state, and the actual training loops with synthetic data and a fake Unreal transport. They do not establish end-to-end compatibility with a live Unreal editor or golden UE snapshots.
