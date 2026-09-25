# noScribe inference server

The server reuses the same `local-whisper` and `local-pyannote` plugins as the
desktop application. It accepts atomic workflows containing transcription,
diarization, or both. A single execution slot serializes all workflows on CPU
and GPU alike.

## Trust boundary

The worker has no users, API keys, quotas, or result store. It must listen only
on `127.0.0.1` or `::1`. An authenticated reverse proxy is the only public
service and forwards accepted requests to the worker. The bearer token in a
desktop remote profile is meant for that proxy; the worker deliberately
ignores it.

Job IDs do not grant access. Every reservation returns a random 256-bit
capability token, sent only in the `X-noScribe-Job-Token` header. It authorizes
status, upload, and cancellation for that job and disappears shortly after the
job finishes.

## Installation and startup

The HTTP server dependencies are included in noScribe's normal platform
requirements. Install the appropriate requirements file for the host. For a
Linux server:

```bash
python -m pip install -r environments/requirements_linux.txt
cp noscribe-server.example.yml /etc/noscribe/server.yml
python -m noScribe.server --config /etc/noscribe/server.yml
```

On Windows, use `requirements_win_cuda.txt` or `requirements_win_cpu.txt`
instead. No separate server requirements installation is needed.

The configured Whisper directory contains one subdirectory per model, each
with a `model.bin`. `force_cpu: true` applies to both Whisper and Pyannote. If
CUDA is unavailable, their existing workers fall back to CPU automatically.

## Queue and upload protocol

Reserve a job without audio:

```http
POST /v1/audio/jobs
Content-Type: application/json

{
  "tasks": [
    {
      "type": "diarization",
      "model": "local-pyannote/default",
      "options": {"num_speakers": 2}
    },
    {
      "type": "transcription",
      "model": "local-whisper/precise",
      "options": {"language": "de"}
    }
  ],
  "audio": {"filename": "interview.flac", "size": 12345678}
}
```

The response contains `job_id`, `job_token`, `state`, and `position`. Poll with
the token in `X-noScribe-Job-Token`. Once the state is `ready_for_upload`, send
the FLAC bytes as the raw request body:

```http
POST /v1/audio/jobs/{job_id}/audio
X-noScribe-Job-Token: <job_token>
Content-Type: audio/flac
Content-Length: 12345678
```

The NDJSON response contains progress and segment events, one `task_result`
per operation, and a final `result`. There is no endpoint for downloading a
previous result.

For simple integrations, the server also exposes the OpenAI-shaped paths
`POST /v1/audio/transcriptions` and `POST /v1/audio/diarizations`. They accept
one raw FLAC request body and query parameters such as `model` and
`language`, and return the same NDJSON event stream. These convenience routes
do not queue: they return HTTP 429 with `Retry-After` when the execution slot is
occupied. They deliberately do not claim full OpenAI multipart compatibility;
clients needing reliable waiting and combined processing should use the job
protocol.

At most 50 reservations wait by default. A ready client has 30 seconds to
begin its upload. Queued reservations must keep polling and expire when
abandoned. Audio is not accepted while a reservation waits.

## Zero-retention controls

Production configuration should use a size-limited tmpfs such as `/run` and
set `require_tmpfs: true`. Uploaded FLAC and derived WAV files receive random
private names, are deleted in all normal completion and error paths, and stale
files with those generated names are removed at startup. Terminal job metadata
is removed after a short TTL.

The server verifies actual byte count, FLAC format, decoded duration,
upload time, and processing time. It disables process core dumps when launched
through its module entry point. For the zero-retention claim to include memory
pressure, disable swap on the host or use encrypted swap; tmpfs pages can
otherwise be swapped out by Linux.

The server never logs request bodies, filenames, transcript text, tokens, or
worker tracebacks. Uvicorn access logging is disabled by the supplied entry
point. Apply the same rule to reverse-proxy and monitoring configuration.

## Reverse proxy

`deploy/nginx/noscribe.conf.example` shows the relevant transport settings.
Its `auth_request` target is intentionally a placeholder for the institution's
identity/API-key gateway. Authentication, revocation, rate limits, and audit
policy belong there, not in noScribe.

The proxy and worker limits should match. Nginx rejects bodies over 1 GiB and
passes request and response streams without disk buffering. Its temporary body
directory should also be on tmpfs as a safeguard. Only the proxy port is
exposed by the firewall.

## GPU coordination

The noScribe scheduler guarantees only that no two noScribe workflows execute
at once. It cannot reserve VRAM against an independent LLM process. Parallel
operation therefore needs measured headroom; exclusive operation needs a
shared coordinator understood by both services. `force_cpu` avoids this issue
at the cost of throughput.
