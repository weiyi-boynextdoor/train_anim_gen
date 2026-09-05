# Remote Training Architecture for Unreal Engine AnimGen

Status: proposed design; no client or server implementation exists in this repository yet.

The repository now also contains an implemented, local-only checkpoint variant at `animgen_checkpoint/train_controller_checkpoint.py`. To install it, copy it over Unreal Engine's `Engine/Plugins/Experimental/Animation/AnimGen/Content/Python/train_controller.py`, keeping the destination filename `train_controller.py`. The original UE 5.8 scripts copied into `reference/` remain comparison references and should not be installed from there. The checkpoint script is an intermediate reliability improvement and is not the remote client/server implementation described below. Its usage and behavior are documented in `animgen_checkpoint/README.md`.

The local checkpoint implementation now uses format v2: a joint denoiser/control-encoder teacher checkpoint and three independent LOD checkpoints. Total iteration budgets do not affect compatibility. Students depend on the actual teacher weights, while teacher identity excludes LOD architecture and sampling steps. Autoencoder compatibility is inferred from encoded input data, not an encoder weight digest. Increasing a budget retains optimizer state and starts an extended learning-rate segment. Legacy v1 checkpoints are retained but not automatically migrated. This local format change does not change the proposed remote protocol or Unreal's snapshot/shared-memory contracts.

## Executive decision

The proposed client/server split is feasible and is a good way to avoid running long GPU workloads on an Unreal Editor workstation. The safest first integration is to replace the two Python entry points that Unreal launches with thin bridge clients while leaving the C++ editor code unchanged.

The proposal should be adjusted in one important way: do not split each upstream script into a network-aware copy on both sides. Instead, create a shared remote-job client and two small Unreal adapters, plus two server-side training workers. This keeps Unreal shared memory local, makes the wire protocol explicit, and isolates the model code from transport concerns.

This recommendation is based on inspection of Unreal Engine source revision `a899359c3efb`. AnimGen is experimental, so all assumptions must be revalidated when the engine revision changes.

## What Unreal requires today

Unreal launches each Python entry point as a child process with exactly one JSON configuration path:

- `train_autoencoder.py <config.json>`
- `train_controller.py <config.json>`

The JSON does not contain the training tensors or network snapshots. It contains dimensions, hyperparameters, local engine paths, and GUIDs for shared-memory regions allocated by the editor. The Python process maps those regions and communicates with Unreal through fixed integer control arrays.

### Autoencoder contract

Inputs read from shared memory:

| Name | Dtype | Shape |
| --- | --- | --- |
| Range starts | `int32` | `[RangeNum]` |
| Range lengths | `int32` | `[RangeNum]` |
| Pose vectors | `float32` | `[TotalFrameNum, PoseVectorSize]` |
| Pose vector weights | `float32` | `[PoseVectorSize]` |
| Initial encoder snapshot | `uint8` | `[EncoderByteNum]` |
| Initial decoder snapshot | `uint8` | `[DecoderByteNum]` |

Control array indexes:

| Index | Meaning |
| --- | --- |
| 0 | Client observes a local exit request and clears it |
| 1 | Training iteration |
| 2 | Rolling loss multiplied by 100,000 and converted to `int32` |
| 3 | Encoder snapshot is ready for Unreal to consume |
| 4 | Decoder snapshot is ready for Unreal to consume |

### Controller contract

Inputs read from shared memory:

| Name | Dtype | Shape |
| --- | --- | --- |
| Range control starts | `int32` | `[RangeNum]` |
| Range encoded starts | `int32` | `[RangeNum]` |
| Range lengths | `int32` | `[RangeNum]` |
| Control vectors | `float32` | `[ControlTotalFrameNum, ControlVectorSize]` |
| Encoded pose vectors | `float32` | `[DatabaseTotalFrameNum, PoseEncodingSize]` |
| Initial control encoder snapshot | `uint8` | `[ControllerByteNum]` |
| Initial LOD 0 snapshot | `uint8` | `[LOD0ByteNum]` |
| Initial LOD 1 snapshot | `uint8` | `[LOD1ByteNum]` |
| Initial LOD 2 snapshot | `uint8` | `[LOD2ByteNum]` |

The JSON also carries `ControlSchema` and `NormalizedPoseStds`.

Control array indexes:

| Index | Meaning |
| --- | --- |
| 0 | Client observes a local exit request and clears it |
| 1 | Flow-matching training iteration |
| 2 | Distillation iteration |
| 3 | Flow-matching loss multiplied by 100,000 and converted to `int32` |
| 4 | Distillation loss multiplied by 100,000 and converted to `int32` |
| 5 | Control encoder snapshot is ready for Unreal to consume |
| 6 | All three LOD snapshots are ready for Unreal to consume |

Snapshot publication is a handshake. A producer writes the complete snapshot bytes and then sets its update flag to one. It must not overwrite that region again until Unreal loads the snapshot and clears the flag. Final publication must follow the same ordering.

The local child process remaining alive is also Unreal's current job-liveness signal. When the process exits, the editor tears down all associated shared memory. The bridge client must therefore remain alive for the remote job and must finish every final download and publication before it exits.

## Recommended architecture

```text
Unreal Editor
  -> existing C++ trainer and shared memory
  -> thin Python adapter with the original entry-point filename
  -> shared remote client library
  -> HTTPS control API and artifact storage
  -> queued version-matched GPU worker
  -> result artifacts
  -> Python adapter validates and publishes snapshots
  -> Unreal assets consume the snapshots
```

### Client responsibilities

The local adapter should:

1. Parse and validate Unreal's JSON configuration.
2. Map all required shared-memory regions.
3. Copy each input into an immutable submission package before upload. Never upload while Unreal can still mutate the source region.
4. Submit the package with an idempotency key.
5. Poll job state and write iterations and losses into the local control array.
6. Detect Unreal's exit flag, clear it, issue an idempotent remote cancellation request, and exit promptly.
7. Download optional checkpoints and the final artifacts.
8. Validate artifact name, exact byte length, digest, protocol version, and compatibility fingerprint.
9. Publish snapshots with the existing update-flag handshake.
10. Exit with zero only after all required final snapshots have been acknowledged by Unreal.

The adapter should not import PyTorch or contain model code. Keeping its dependencies close to the Python standard library reduces conflicts with Unreal's managed Python environment.

### Service responsibilities

The control service should authenticate requests, validate manifests, allocate job IDs, enforce idempotency, persist job state, issue artifact upload/download locations, schedule compatible workers, and expose cancellation and progress. It should not execute GPU training in the API process.

Recommended API surface for version 1:

| Method and path | Purpose |
| --- | --- |
| `POST /v1/jobs` | Create or return an idempotent job |
| `POST /v1/jobs/{job_id}/inputs:complete` | Confirm all input blobs are uploaded and start validation |
| `GET /v1/jobs/{job_id}` | Poll state, progress, metrics, and artifact metadata |
| `POST /v1/jobs/{job_id}:cancel` | Request cancellation idempotently |
| `GET /v1/jobs/{job_id}/artifacts/{name}` | Obtain or redirect to a result artifact |

Large tensor data should use object storage or a resumable blob endpoint rather than a single large JSON or multipart API request. Polling is recommended for the first version because it works through ordinary proxies and is sufficient for iteration/loss updates. Server-sent events can be added later without changing job semantics.

### Worker responsibilities

Workers should be selected by an immutable compatibility key and should:

1. Download inputs into an isolated job directory.
2. Verify every length and SHA-256 digest before allocating large arrays or GPU memory.
3. Reconstruct arrays from non-executable formats.
4. Use the AnimGen model and snapshot helpers matching the submitted Unreal revision.
5. Emit structured progress and heartbeat events.
6. Check for cancellation at least once per iteration and during long transfer/checkpoint operations.
7. Write artifacts to temporary names, verify them, and publish them atomically.
8. Upload final snapshots and a result manifest before marking the job successful.

Do not pass client `EnginePath`, `SitePackagesPath`, `IntermediatePath`, or shared-memory GUIDs to a worker. The worker image supplies its own compatible Python modules from `NNERuntimeBasicCpu` and `LearningCore`.

## Submission package

Use a manifest plus one file per logical array or snapshot. `.npy` is suitable for tensors because it records dtype and shape without executable deserialization. Use raw `.bin` files for Unreal network snapshots, whose exact bytes and lengths must be preserved. A tar container is optional; compression should be configurable because dense floating-point arrays may cost substantial CPU time for limited size reduction.

Example manifest outline:

```json
{
  "protocol_version": 1,
  "trainer_type": "autoencoder",
  "client_job_id": "018f...",
  "compatibility": {
    "unreal_revision": "a899359c3efb",
    "animgen_fingerprint": "sha256:...",
    "worker_key": "ue-a899359c3efb-animgen-v1"
  },
  "training_config": {
    "iteration_num": 100000,
    "learning_rate": 0.0001,
    "batch_size": 256,
    "seed": 1234,
    "requested_device": "gpu"
  },
  "blobs": [
    {
      "name": "pose_vectors",
      "file": "pose_vectors.npy",
      "dtype": "float32",
      "shape": [120000, 384],
      "byte_order": "little",
      "byte_length": 184320000,
      "sha256": "..."
    }
  ]
}
```

The real manifest must preserve all upstream hyperparameters needed by the relevant trainer. JSON field conversion should happen in one tested adapter layer; server model code should receive a typed internal configuration.

## Job lifecycle

Use explicit monotonic states:

```text
CREATED -> UPLOADING -> QUEUED -> RUNNING -> FINALIZING -> SUCCEEDED
                                  |            |
                                  +----------> FAILED
                                  +----------> CANCEL_REQUESTED -> CANCELLED
```

Each status response should include a monotonically increasing event sequence, current stage, iteration, maximum iterations, rolling loss, heartbeat time, and a structured error when applicable. The client should ignore older event sequences after retries or reconnects.

A successful result is immutable. Retrying a request with the same idempotency key and identical manifest digest returns the same job. Reusing the key with different content is an error.

## Failure behavior

- Uploads and downloads must be resumable or safely retryable with bounded exponential backoff and jitter.
- A connection loss must not create a second training job.
- If the client cannot contact the service, it should report the error, leave the last Unreal-accepted snapshots intact, and exit nonzero.
- If Unreal requests cancellation, the client should send the request with a short timeout and then exit even if the service is unavailable. Unreal currently gives its child process approximately five seconds before forced termination.
- The service should classify validation, compatibility, capacity, training, cancellation, and internal failures separately.
- The client must never publish a partial or incompatible snapshot. For the controller, all LOD files must validate before any LOD update flag is set.

The current C++ integration primarily observes child-process liveness and shared-memory values. It does not provide a rich remote failure UI. The PoC can rely on redirected process logs and a nonzero exit, but a production integration should add a small C++ change or plugin setting to display structured bridge errors and configure the endpoint and credentials.

## Security and operational requirements

- Require TLS and short-lived credentials or an authenticated local credential provider.
- Store secrets outside Unreal-generated JSON and outside this repository.
- Redact signed URLs, authorization headers, local paths, and dataset contents from logs.
- Enforce maximum blob sizes, tensor ranks, dimensions, total expanded bytes, runtime, and GPU quota before allocation.
- Never deserialize pickle, arbitrary Python objects, or user-controlled code.
- Isolate jobs and prevent archive path traversal if container archives are accepted.
- Encrypt artifacts at rest, define retention, and support explicit deletion.
- Treat animation and control data as potentially sensitive project assets.
- Confirm that any server image or distributed package complies with Unreal Engine licensing; do not publish Epic source or derived proprietary modules in a public image.

## Repository layout to implement next

```text
src/animgen_remote/
  client/
    autoencoder_adapter.py
    controller_adapter.py
    bridge.py
    shared_memory.py
    publisher.py
  protocol/
    manifest.py
    models.py
    validation.py
  server/
    api.py
    jobs.py
    storage.py
  workers/
    autoencoder.py
    controller.py
    runtime.py
ue_wrappers/
  train_autoencoder.py
  train_controller.py
tests/
  unit/
  integration/
  golden/
tools/
  install_ue_wrappers.py
  verify_ue_compatibility.py
```

The two files under `ue_wrappers/` should be tiny executable shims. An installer should back up or verify the upstream files, install the shims deliberately, and record their source hashes. Development must not modify the external Unreal tree as an incidental side effect of tests or imports.

## Delivery plan

### Phase 1: Compatibility capture

- Hash the two upstream scripts and the relevant `LearningCore` and `NNERuntimeBasicCpu` Python helpers.
- Capture tiny autoencoder and controller input packages from Unreal.
- Run the original scripts locally and save golden final snapshots and progress behavior.
- Verify whether snapshots produced on the intended server OS and Python/PyTorch stack load identically in Unreal.

This phase is mandatory because the snapshot format and helper modules are engine-specific and are not a stable public wire format.

### Phase 2: Local split without networking

- Implement adapters, manifests, validation, and workers.
- Run the worker as a separate local process using files rather than shared memory.
- Prove that the adapter republishes final and periodic snapshots through the exact Unreal handshake.
- Prove cancellation completes within the editor timeout.

This isolates serialization and lifecycle bugs before network behavior is introduced.

### Phase 3: Single-node remote PoC

- Add the authenticated job API and artifact store.
- Support one worker compatibility key and one GPU queue.
- Implement idempotent submission, polling, heartbeat timeout, cancellation, checksums, and final artifacts.
- Keep checkpoint download optional; final-only download is adequate for the first end-to-end test.

### Phase 4: Production hardening

- Add resumable transfers, quotas, retention, observability, worker autoscaling, endpoint configuration, structured Unreal UI errors, and a compatibility matrix.
- Add rolling deployment tests that prevent an incompatible worker from accepting a job.
- Add recovery behavior for editor crashes, client restarts, and orphaned jobs.

## Acceptance criteria for the PoC

The design is validated only when all of the following are demonstrated for both trainer types:

- The same captured input can run through the original local trainer and the split worker.
- Every returned snapshot has the expected byte length and loads successfully through Unreal's `LoadFromSnapshot` path.
- Unreal receives live iteration and loss updates.
- Unreal receives the required final networks before the client exits.
- Cancelling in the editor makes the local client exit within five seconds and eventually cancels or safely abandons the remote job.
- Network retry does not duplicate a job or publish a partial snapshot.
- A mismatched engine/plugin fingerprint is rejected before GPU work begins.
- No local paths, shared-memory GUIDs, credentials, or raw data are written to normal service logs.

## Overall assessment

The original idea is technically sound as an integration strategy, with medium implementation risk. The primary risk is not HTTP communication; it is faithfully bridging Unreal's shared-memory lifecycle and engine-specific snapshot format across a versioned remote boundary.

Replacing the two Python entry points is appropriate for a controlled PoC and avoids an initial C++ rebuild. For maintainability, install generated thin wrappers rather than manually editing the Unreal files, keep the server trainers transport-independent, and plan a small native plugin integration once endpoint configuration and actionable editor-side errors are required.
