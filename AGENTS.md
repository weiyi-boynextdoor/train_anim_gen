# Repository Instructions

## Purpose

This repository implements remote deep-learning training for Unreal Engine's experimental AnimGen plugin. The Unreal Engine source tree is an external reference, not part of this repository.

## Language

- Write all source code, code comments, identifiers, user-facing strings, commit-ready documentation, and examples in English.
- Keep protocol field names and error messages stable once published. Treat changes to them as compatibility changes.

## Unreal Engine source reference

- Read the local Unreal Engine source root from `.ue_source_folder`. Trim surrounding whitespace before using it.
- `.ue_source_folder` is machine-local and must remain ignored by Git.
- Treat the referenced Unreal Engine tree as read-only unless the user explicitly asks to install or update the bridge scripts there.
- Never commit Unreal Engine source or proprietary Epic code into this repository. Reimplement only the minimal integration boundary needed by this project.
- The current integration targets these upstream entry points:
  - `Engine/Plugins/Experimental/Animation/AnimGen/Content/Python/train_autoencoder.py`
  - `Engine/Plugins/Experimental/Animation/AnimGen/Content/Python/train_controller.py`
- Before changing bridge behavior, compare the upstream scripts and their C++ callers with the assumptions in `docs/remote-training-architecture.md`.

## Architecture boundaries

- Keep the Unreal-facing client thin. It owns shared-memory mapping, package upload, status polling, cancellation forwarding, artifact download, and shared-memory publication.
- Keep model construction, normalization, optimization, checkpointing, and distillation in server-side workers.
- Do not let server-side training code depend on Unreal shared-memory GUIDs or client-local paths.
- Exchange immutable, versioned manifests and blobs. Do not use Python pickle for network or dataset transport.
- Preserve Unreal's exact snapshot byte format and byte count when publishing trained networks back to shared memory.
- Preserve the control-array indexes documented for each trainer. Changing them requires a coordinated Unreal-side change.
- Keep autoencoder and controller job types separate at the API boundary, while sharing transport, validation, job lifecycle, and artifact code.

## Compatibility and safety

- Include protocol version, trainer type, Unreal source revision or build identifier, plugin fingerprint, tensor dtype, shape, byte order, byte length, and SHA-256 digest in every submitted job manifest.
- Reject unknown protocol versions, incompatible worker images, unexpected artifact names, size mismatches, and checksum mismatches before training or publication.
- Remove `EnginePath`, `SitePackagesPath`, `IntermediatePath`, and shared-memory GUIDs from the server-visible training configuration.
- Never log credentials, signed URLs, complete control schemas, raw training arrays, or local absolute paths.
- Use TLS and authenticated requests. Make submission and cancellation idempotent.
- A client cancellation must not wait for the remote worker to stop. Notify the service, clean up locally, and exit within Unreal's five-second subprocess shutdown window.

## Development workflow

- Prefer small modules with typed public interfaces and explicit exceptions.
- Keep the transport interface injectable so unit tests do not require Unreal Engine, a GPU, or a live server.
- Add unit tests for manifest validation, package round trips, checksums, job-state transitions, cancellation, retry behavior, and snapshot publication handshakes.
- Add golden compatibility tests using small snapshots produced by the exact supported Unreal revision before claiming end-to-end support.
- Run formatting, static checks, and the smallest relevant test suite after each change. Document any test that cannot run locally.
- Do not silently fall back from remote training to local training. Surface a clear error and preserve the last network already accepted by Unreal.

## Documentation

- Keep `docs/remote-training-architecture.md` aligned with implementation and protocol changes.
- Record protocol-breaking changes and supported Unreal revisions explicitly.
- Clearly distinguish implemented behavior from proposed behavior.
