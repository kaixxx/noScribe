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

## Remote backend profiles

Remote server connections are configured independently from `config.yml`.
Put one YAML file per connection into the `backends/` directory below the
noScribe user configuration directory:

```text
backends/
|-- ifs-server.yml
`-- test-server.yml
```

Each file uses the versioned remote-profile schema:

```yaml
schema_version: 1
id: ifs-server
name: IfS-Server
driver: noscribe-http-v1
enabled: true
url: https://noscribe.example.org
api_key: secret
```

The `id` is the stable internal backend ID and must be unique across all
profiles. The `name` is shown in the GUI after every model offered by this
profile. URLs must use HTTPS and must not contain credentials; plain HTTP is
accepted only for loopback development servers. API keys are currently stored
as plain text, so profile files must be protected like other credentials.

`remote_profiles.py` creates the directory and loads both `.yml` and `.yaml`
files. Invalid files are reported individually and do not prevent other
profiles from loading. A valid profile configures a connection; the driver is
still responsible for obtaining the server's model catalogue and creating a
remote plugin for the registry.

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
  The bundled `noscribe-http-v1` driver is implemented in `remote_http/`.

Whisper and Pyannote currently use `builtin`, so they continue to share the
same PyInstaller runtime and dependencies. The execution type is packaging
metadata; all plugins are exposed through the same registry interface.

## noScribe HTTP protocol version 1

The bundled remote driver authenticates every request with
`Authorization: Bearer <API_KEY>`. On startup it requests `GET /v1/models`.
The response identifies the protocol and advertises model capabilities:

```json
{
  "protocol_version": 1,
  "server_version": "0.1.0",
  "data": [
    {
      "id": "precise",
      "name": "precise",
      "engine": "faster-whisper",
      "capabilities": ["transcription"]
    }
  ]
}
```

Servers that advertise the `queued_workflows` feature use a two-phase job
protocol. `POST /v1/audio/jobs` reserves an atomic list of transcription and/or
diarization tasks without uploading audio. The client polls the returned job
with its short-lived capability token. Only after the state changes to
`ready_for_upload` does it send the raw Opus body to the job's `/audio`
endpoint. One workflow therefore uploads its recording exactly once.

The response is an `application/x-ndjson` stream. Worker events include a
`task_index` and `operation`; every task ends with `task_result`, and the whole
workflow ends with one `result` event. Closing the response and sending
`DELETE /v1/audio/jobs/{job_id}` cancels a job. Servers without the feature
continue to use the legacy multipart `/v1/audio/transcriptions` and
`/v1/audio/diarizations` endpoints.

Before any remote request, the desktop pipeline extracts and converts the
selected recording range to mono Opus at 32 kbit/s. The driver refuses other
file extensions, preventing the existing WAV working file or an original
video from being uploaded accidentally. When a selected remote profile also
offers diarization, noScribe automatically uses that profile's diarization
model; otherwise it falls back to local Pyannote.

### Development dummy server

The repository includes a dependency-free dummy server for testing queued
workflows without inference models or a GPU:

```powershell
conda run -n noScribe_0_6_non_cuda python tools/noscribe_dummy_server.py
```

The server listens only on `127.0.0.1:8765` by default and prints a complete
profile on startup. Save that profile as, for example,
`backends/dummy-server.yml` below the noScribe user configuration directory,
then restart noScribe. The model menu will contain
`Dummy transcription (Dummy-Server)`. The server accepts the generated Opus
upload in memory, streams fixed transcription and diarization events, and
does not write uploaded data to disk. Stop it with Ctrl+C.

Useful options are `--port`, `--token`, and `--delay`. The dummy server uses
plain HTTP deliberately and is intended only for loopback development.

## External worker protocol version 1

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
