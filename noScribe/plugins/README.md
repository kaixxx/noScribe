# noScribe backend plugins

This document is the implementation guide for contributors and coding agents
that add or change an inference backend. Read it together with `manifest.py`,
`registry.py`, and `protocol.py` before modifying a plugin.

## Directory layout

Shared infrastructure lives directly in this package. Every backend shipped
with noScribe has its own importable subpackage:

```text
plugins/
├── factory.py
├── manifest.py
├── protocol.py
├── registry.py
├── local_whisper/
│   ├── __init__.py
│   ├── plugin.py
│   └── backend.json
└── local_pyannote/
    ├── __init__.py
    ├── plugin.py
    └── backend.json
```

Python package names use underscores. Public backend IDs use hyphens. For
example, package `local_whisper` declares backend ID `local-whisper`.

Future downloaded plugins will live outside the source tree in the user's
noScribe configuration directory. Their manifests use the same format, but
their discovery and signature verification belong to the future plugin
manager, not to `BackendRegistry`.

## Responsibilities

- `manifest.py` parses and validates versioned `backend.json` files.
- `protocol.py` owns the transport-neutral request and event envelopes.
- `registry.py` stores plugins, publishes their model catalogue, routes a
  request by `ModelRef.backend_id`, and forwards cancellation.
- `factory.py` constructs only the plugins shipped with noScribe. Plugin
  behavior must not be implemented there.
- A plugin's `plugin.py` adapts its worker or remote driver to the registry
  interface and returns `ModelDescriptor` objects.
- Hardware and dependency details stay inside the providing backend. They are
  not job parameters and must not leak into the GUI.

The GUI must build model choices from `BackendRegistry.model_options()`. It
must not contain a hard-coded list of backend IDs or model IDs. Qualified
model references remain the stored and internal values. Presentation labels
hide local backend names unless two local models have the same display name;
remote models always include the remote profile name in parentheses.

Backend plugins are addressed by the backend part of a qualified model
reference such as `local-whisper:precise`. Every plugin has a versioned
`backend.json` manifest and advertises its models at runtime.
The optional manifest-level `engine` describes single-engine plugins; a remote
or external plugin may instead advertise models from several engines.

## Execution types

- `builtin`: adapter code and dependencies are part of the noScribe build.
- `external`: an independently packaged worker executable. The manifest's
  `command` is resolved relative to the installed plugin directory.
- `remote`: a configuration package using a driver bundled with noScribe.
  The first supported driver will be `noscribe-http-v1`.

Whisper and Pyannote currently use `builtin`, so they continue to share the
same PyInstaller runtime and dependencies. The execution type is packaging
metadata; all plugins are exposed through the same registry interface.

## Protocol version 1

External workers will use JSON Lines over standard input and output. A request
has this envelope:

```json
{"protocol_version":1,"id":"job-1","method":"transcribe","params":{}}
```

Supported methods are `transcribe`, `diarize`, `cancel`, and `health`. Workers
respond with the event shapes already used by the built-in multiprocessing
workers: `log`, `status`, `progress`, `segment`, and `result`. A `result` event
always contains an `ok` flag. Only one inference request is active per worker;
therefore the current event envelope does not need to repeat the request ID.

Manifest and protocol versions are checked before a plugin is registered.
Downloading, signature verification, and discovery of an installed-plugin
directory will be added with the plugin manager; they are deliberately not
part of the registry itself.

## Adding a bundled plugin

1. Create an importable directory such as `local_voxtral/`.
2. Add `__init__.py`, `plugin.py`, and `backend.json`.
3. Give the manifest a unique, stable backend ID without a colon.
4. Implement `list_models()`, `cancel()`, and `close()`. Implement
   `transcribe()` and/or `diarize()` according to the advertised capabilities.
5. Return qualified `ModelRef` values whose backend ID equals the manifest ID.
6. Add construction to `factory.py`; do not add backend-specific branches to
   the GUI or registry.
7. Use message IDs plus structured parameters for user-facing status. Do not
   translate messages in a worker.
8. Keep worker imports light. Windows and macOS multiprocessing use `spawn`, so
   heavy libraries must be imported inside the worker entry point.
9. Add manifest, catalogue, routing, cancellation, and error tests.
10. Run the complete test suite and verify all PyInstaller specs.

All PyInstaller specs collect `**/backend.json` from `noScribe.plugins`
automatically. A plugin with additional data or native libraries still needs
explicit packaging rules.

## Compatibility rules

- Increment the protocol version only for an incompatible wire-format change.
- Treat backend IDs and model IDs as persistent identifiers once released.
- Add capabilities rather than testing concrete backend names in consumers.
- Word-level timestamps are optional result data even when requested by a
  backend; consumers must continue to handle an empty word list.
- A plugin must reject unsupported operations or options explicitly rather
  than silently routing them to another backend.
