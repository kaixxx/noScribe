# syntax=docker/dockerfile:1

# Headless CLI image. Build from the repository root:
#
#   docker build -f docker/Dockerfile -t noscribe .
#   docker build -f docker/Dockerfile --build-arg TORCH_VARIANT=cu128 -t noscribe:cuda .
#
# The Dockerfile in the repository root is unrelated: it is the build
# environment for the Linux binary, used by generate_linux_binary.sh.

ARG PYTHON_VERSION=3.12
ARG DEBIAN_RELEASE=bookworm

# ---------------------------------------------------------------------------
# Stage 1: download the Whisper models (pinned revision + sha256)
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS models

# "precise": faster-whisper large-v3-turbo (fp16). The repo moved from
# mobiuslabsgmbh/ to dropbox-dash/ on Hugging Face; the old name redirects.
ARG PRECISE_REPO=dropbox-dash/faster-whisper-large-v3-turbo
ARG PRECISE_REV=0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf
ARG PRECISE_SHA256=e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da
# "fast": large-v3-turbo, int8 quantization
ARG FAST_REPO=mukowaty/faster-whisper-int8
ARG FAST_REV=db1cdb5990df067cf53670d86241330c7ea454cf
ARG FAST_SUBDIR=faster-whisper-large-v3-turbo-int8
ARG FAST_SHA256=2caaf13921eded3ac31420512b2a253941e3fdf95abcc71c2299a4c836e63862

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    files="config.json model.bin preprocessor_config.json tokenizer.json vocabulary.json"; \
    fetch() { \
      dest="$1"; base="$2"; sha="$3"; \
      mkdir -p "$dest"; \
      for f in $files; do \
        curl -fsSL --retry 5 --retry-delay 5 -o "$dest/$f" "$base/$f"; \
      done; \
      echo "$sha  $dest/model.bin" | sha256sum -c -; \
    }; \
    fetch /models/precise "https://huggingface.co/${PRECISE_REPO}/resolve/${PRECISE_REV}" "${PRECISE_SHA256}"; \
    fetch /models/fast "https://huggingface.co/${FAST_REPO}/resolve/${FAST_REV}/${FAST_SUBDIR}" "${FAST_SHA256}"

# ---------------------------------------------------------------------------
# Stage 2: system packages and Python dependencies
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS deps

ARG TORCH_VERSION=2.8.0
# PyTorch wheel variant: "cpu" (default) or the CUDA 12.8 build "cu128" (the
# `cuda` flavor). It selects the PyTorch package index and the lock file:
# docker/requirements.lock for cpu, docker/requirements-<variant>.lock otherwise.
ARG TORCH_VARIANT=cpu
# USE_LOCK=0 ignores the lock file and resolves fresh versions; only
# docker/update-lock.sh uses that, to regenerate the lock file.
ARG USE_LOCK=1

# libtk8.6: noScribe always imports tkinter/customtkinter, even headless.
# ffmpeg libs: torchaudio's ffmpeg backend (pyannote) and torchcodec.
# fonts-symbola: only for the optional GUI mode; provides the folder icon
# (U+1F4C2) on the file buttons, which DejaVu lacks.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libtk8.6 ffmpeg fonts-symbola \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# PyTorch from its own index first (TORCH_VARIANT), then requirements_linux.txt
# (minus pyinstaller). The CUDA wheels bring cuBLAS and cuDNN as pip packages,
# so no CUDA base image is needed.
# Versions come from three constraint sources:
#   - TORCH_VERSION, so pip cannot swap in a torch build from PyPI
#   - docker/constraints.txt, hand-maintained pins with a reason each
#   - the lock file, the full package set of the last tested image
# The final check fails the build when the installed set no longer matches
# the lock, e.g. after requirements_linux.txt changed. It strips CRs, since a
# Windows checkout with core.autocrlf may hold the lock file with CRLF.
COPY environments/requirements_linux.txt /tmp/pip/requirements.txt
COPY docker/constraints.txt docker/requirements*.lock /tmp/pip/
RUN set -eu; cd /tmp/pip; \
    lock=requirements.lock; \
    [ "${TORCH_VARIANT}" = cpu ] || lock="requirements-${TORCH_VARIANT}.lock"; \
    echo "torch==${TORCH_VERSION}" >> constraints.txt; \
    echo "torchaudio==${TORCH_VERSION}" >> constraints.txt; \
    if [ "${USE_LOCK}" = 1 ]; then \
      [ -f "$lock" ] || { echo "ERROR: no docker/$lock; run docker/update-lock.sh ${TORCH_VARIANT}" >&2; exit 1; }; \
      cat "$lock" >> constraints.txt; \
    fi; \
    pip install --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" -c constraints.txt \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}"; \
    grep -viE '^[[:space:]]*pyinstaller' requirements.txt > requirements-docker.txt; \
    pip install -r requirements-docker.txt -c constraints.txt; \
    if [ "${USE_LOCK}" = 1 ]; then \
      pip freeze --all | grep -v '^pip==' > installed.txt; \
      grep -v '^#' "$lock" | tr -d '\r' | diff -u - installed.txt \
        || { echo "ERROR: installed packages differ from docker/$lock; run docker/update-lock.sh ${TORCH_VARIANT}" >&2; exit 1; }; \
    fi; \
    rm -rf /tmp/pip

# ---------------------------------------------------------------------------
# Stage 3: runtime
# ---------------------------------------------------------------------------
FROM deps

# Local ./models is excluded via Dockerfile.dockerignore, so the models downloaded
# above are the ones in the image.
COPY --from=models /models /opt/noscribe/models
COPY . /opt/noscribe/
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Run as an unprivileged user. /config and $HOME are world-writable and HOME
# is set explicitly, so `docker run --user $(id -u):$(id -g)` works as well
# (customtkinter copies its fonts to ~/.fonts on start).
RUN useradd --create-home --uid 1000 noscribe \
 && mkdir -p /config /data \
 && chmod 1777 /config /home/noscribe \
 && chown noscribe:noscribe /data

# PYTHONPATH: noScribe locates its resources (models, pyannote, trans, img,
# prompts) as top-level packages, and the working directory is /data.
# PYTHONSAFEPATH: keeps `python -m` from putting that working directory, the
# user's mounted folder, in front of it on sys.path.
ENV PYTHONPATH=/opt/noscribe \
    PYTHONSAFEPATH=1 \
    HOME=/home/noscribe \
    XDG_CONFIG_HOME=/config \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HUB_OFFLINE=1

USER noscribe
WORKDIR /data

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--help"]
