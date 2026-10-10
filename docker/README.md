# noScribe in Docker

`docker/Dockerfile` builds an image that runs noScribe
without its window: audio in, transcript out. It is meant for servers, batch
jobs and machines where installing Python, PyTorch and the models by hand is
not wanted. For desktop use, the regular installers from
[noscribe.de](https://noscribe.de) are the more comfortable choice.

Everything runs locally inside the container. The image contains the Whisper
and pyannote models, so nothing is downloaded at runtime and no audio leaves
the machine. (Only the GUI mode contacts GitHub on start to look for a new
noScribe release, as the desktop app does; `check_for_update` in `config.yml`
turns that off.)

- CPU only; a separate GPU (CUDA) image is coming soon
- amd64 (x86_64) only; see [Apple Silicon and ARM](#apple-silicon-and-arm)
- about 3 GB to download, about 8 GB on disk
- transcription takes roughly 1.5 times the length of the recording on a
  desktop CPU

## Build

Run this in the repository root:

```bash
docker build -f docker/Dockerfile -t noscribe .
```

The build downloads both Whisper models from Hugging Face, pinned to a fixed
revision and verified by SHA-256:

| Model     | Source                                                |
|-----------|-------------------------------------------------------|
| `precise` | `dropbox-dash/faster-whisper-large-v3-turbo` (fp16), the default |
| `fast`    | `mukowaty/faster-whisper-int8` (large-v3-turbo, int8) |

A local `models/` folder is ignored (see `docker/Dockerfile.dockerignore`).

## Usage

Mount the folder containing your audio at `/data`, the working directory
inside the container. The transcript is written next to the audio file:

```bash
docker run --rm -v "$PWD:/data" noscribe interview.mp3 interview.html --language de
```

The output format follows the file extension: `.html`, `.txt` or `.vtt`. All
command line options of noScribe work, for example:

```bash
docker run --rm -v "$PWD:/data" noscribe interview.mp3 interview.txt \
  --model fast --speaker-detection 2 --start 00:01:00 --stop 00:10:00
```

```bash
docker run --rm noscribe --help         # all options
docker run --rm noscribe --help-models  # installed Whisper models
```

`--no-gui` is added automatically.

In PowerShell, write `${PWD}` instead of `$PWD`.

### Apple Silicon and ARM

The image is built for amd64 only. The pinned `docker/requirements.lock`
contains `torchaudio==2.8.0+cpu`, which exists only for x86_64, and
`torchcodec==0.7.0` has no Linux aarch64 wheel. On Apple Silicon and other ARM
machines, build and run it as an amd64 image, which works through emulation
but is slow:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t noscribe .
docker run --rm --platform linux/amd64 -v "$PWD:/data" noscribe interview.mp3 interview.html
```

### File permissions on Linux

The container runs as UID 1000. If your user has a different UID, pass it so
that the transcript belongs to you:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" noscribe interview.mp3 interview.html
```

### Settings and additional models

noScribe keeps its `config.yml` in `/config/noScribe`. Mount a volume there to
keep settings between runs. Additional Whisper models in CTranslate2 format go
into `/config/noScribe/whisper_models/<name>/` and can then be selected with
`--model <name>`:

```bash
docker run --rm -v noscribe-config:/config -v "$PWD:/data" noscribe interview.mp3 interview.html
```

## GUI mode (experimental)

With `gui` as the first argument the regular noScribe window is started
instead. This needs an X server on the host that is passed into the
container. So far it has only been tested on Windows 11 with Docker Desktop,
which provides one through WSLg (PowerShell):

```powershell
docker run --rm -e DISPLAY=:0 `
  -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix `
  -v "C:\Users\you\Documents\Interviews:/data" `
  -v noscribe-config:/config `
  noscribe gui
```

The container has its own file system. The file dialogs inside noScribe only
see what was mounted with `-v`: open audio files from `/data` and save
transcripts there. Anything saved elsewhere is gone when the window closes.
More folders can be mounted under their own names, for example
`-v "D:\Recordings:/recordings"`.

Limitations: the noScribe Editor is not part of the image, and links cannot
open a browser from inside the container.

## Image tags

This repository does not publish images. The naming scheme below is the one
used by the [noscribe-docker](https://github.com/chrtmnn/noscribe-docker)
project (`ghcr.io/chrtmnn/noscribe`), which builds from the same Dockerfile
layout; it is recorded here so that locally built and published images are
named alike.

```
<noScribe version>[-r<N>][-<flavor>]
```

- **noScribe version**: the upstream release the image contains, without a
  leading `v` (`0.7.2`).
- **`-r<N>`**: build revision for that noScribe version, starting at `r1`. It
  increases whenever the image changes while noScribe stays the same: new
  `requirements.lock`, new base image, changed entrypoint, different model
  revision.
- **`-<flavor>`**: image variant. The default (headless, CPU, models included)
  has no suffix.

| Tag                  | Points to                                          | Moves? |
|----------------------|----------------------------------------------------|--------|
| `0.7.2-r1`           | exactly this build, default flavor                 | never  |
| `0.7.2-r1-gui`       | exactly this build, `gui` flavor                   | never  |
| `0.7.2`, `0.7.2-gui` | latest revision for noScribe 0.7.2                 | yes    |
| `latest`, `gui`      | latest revision of the latest noScribe version     | yes    |
| `edge`, `edge-gui`   | current `main`, not a release                      | yes    |

Use a full `X.Y.Z-rN[-flavor]` tag wherever reproducibility matters.

| Flavor   | Content                                           | Status  |
|----------|---------------------------------------------------|---------|
| *(none)* | headless CLI, CPU, both Whisper models            | built   |
| `gui`    | default plus noScribe Editor and GUI dependencies | experimental |
| `cuda`   | NVIDIA GPU build (CUDA PyTorch wheels, own lock file) | planned |
| `slim`   | without Whisper models                            | idea    |

Flavors are not combined (`-cuda-gui`) unless there is a concrete need; every
combination multiplies the build matrix.

A release is a git tag `v<noScribe version>-r<N>` (for example `v0.7.2-r1`);
the image tag is the git tag without the `v`. Branch and pull-request builds
are never pushed.

## Python dependencies

`environments/requirements_linux.txt` leaves most versions open, so two builds
a few weeks apart can end up with different packages. For the image,
`docker/requirements.lock` records the exact package set of the last tested
build. The build installs the requirements constrained to that set and fails
if the result differs, for example after `requirements_linux.txt` changed.

To move to newer versions, or after changing the requirements:

```bash
docker/update-lock.sh
docker build -f docker/Dockerfile -t noscribe .
```

Test the image before committing the new lock file. Pins that must survive
regeneration belong in `docker/constraints.txt`, each with its reason. At the
moment that is `av<19`: PyAV 19 removed an argument that faster-whisper 1.2.1
still passes.

## Not to be confused with the root Dockerfile

The `Dockerfile` in the repository root is unrelated to this image. It is the
build environment that `generate_linux_binary.sh` uses for the PyInstaller
binary. That script also tags its image `noscribe`, so running it replaces the
image built here; rebuild afterwards or use a different tag for one of them.

## Files

| File                             | Purpose                                         |
|----------------------------------|-------------------------------------------------|
| `docker/Dockerfile`              | the image described here                        |
| `docker/Dockerfile.dockerignore` | keeps `.git`, tests and local models out        |
| `docker/entrypoint.sh`           | starts noScribe headless, or the GUI with `gui` |
| `docker/constraints.txt`         | hand-maintained version pins                    |
| `docker/requirements.lock`       | generated package list of the tested image      |
| `docker/update-lock.sh`          | regenerates `requirements.lock`                 |
