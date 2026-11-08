# NVIDIA license documents

These files are unmodified license documents extracted from NVIDIA's original
Python wheels. `SOURCES.json` records the package versions, original paths,
download URLs, published wheel checksums, and document checksums. Extracted
documents were verified against the SHA-256 entries in each wheel's `RECORD`.
Identical documents are stored only once.

| Document | Source components |
| --- | --- |
| `CUDA-12.8-LICENSE.txt` | CUDA Runtime 12.8.90, CUPTI 12.8.90, NVRTC 12.8.93, cuBLAS 12.8.4.1, cuFFT 11.3.3.83, cuFile 1.13.1.3, cuRAND 10.3.9.90, cuSOLVER 11.7.3.90, cuSPARSE 12.5.8.93, nvJitLink 12.8.93 |
| `cuDNN-9-LICENSE.txt` | cuDNN 9.10.2.21 on Linux and Windows; the same document is supplied by cuDNN 9.1.0.70 on Linux |
| `cuDNN-9.1-Windows-LICENSE.txt` | cuDNN 9.1.0.70 on Windows, including its additional third-party notices |
| `cuSPARSELt-0.7.1-LICENSE.txt` | cuSPARSELt 0.7.1 |
| `NCCL-2.27.3-LICENSE.txt` | NCCL 2.27.3 |
| `NVTX-12.8-LICENSE.txt` | NVTX 12.8.90 |

The CUDA packages correspond to `docker/requirements-cu128.lock` and the CUDA
12.8 libraries supplied with PyTorch 2.8.0+cu128. cuDNN 9.1.0.70 is also present
in the existing Windows CTranslate2 4.5.0 installation. Its Windows document
differs from the cuDNN 9.10 document, so both are retained.

## Packaging and updates

All PyInstaller specs use the shared collector in `pyinstaller/collect_licenses.py`.
When its analysis contains NVIDIA CUDA runtime libraries, the collector includes
this directory under `third_party_licenses/nvidia/` in addition to the documents
collected from the installed distributions. The source-based Docker image
includes this directory at `/opt/noscribe/third_party_licenses/nvidia/` through
its existing repository copy. Not every listed library is present in every build.

These are versioned snapshots, not a declaration that their terms apply to all
future versions. When changing CUDA, cuDNN, or the bundled NVIDIA components,
check the license documents in the new original packages and update these
snapshots and their provenance if needed. Keep the original text unchanged,
including supplements and third-party notices; do not substitute a current web
page for the documents supplied with a different version.

The NVIDIA libraries retain their respective licenses. The separate
`CUDA-EXCEPTION.txt` at the repository root describes the limited additional
permission for noScribe; this directory does not grant a project-wide exception
or permissions on behalf of other contributors.
