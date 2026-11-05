# noScribe VOICE Edition — cross-platform build kit (Whisper Small)

This is **source/build infrastructure**, not prebuilt binaries. It must be added to
**your modified noScribe fork**, containing the working opt-in VOICE-style pause
markers. The build jobs package whatever source is committed to the branch.

## What the workflow builds

- Windows x64: a portable ZIP with `noScribe.exe` and its dependencies.
- macOS arm64 (Apple Silicon): a ZIP with `noScribe.app`, unsigned/unnotarized.
- Linux x64: a tar.gz with the `noScribe` executable and its dependencies.

All build jobs download `Systran/faster-whisper-small` (multilingual CTranslate2
model) into `models/small` on the build machine, and create **new** release specs
based on the upstream PyInstaller specs. Standard specs remain untouched. The
builds do not include the usual Precise/Fast models or bundle noScribe Editor.
The official separate editor can open HTML transcripts; automatic Editor launch
from noScribe may not work if the editor isn't installed in the expected place.

## Add to your existing macOS source checkout

From `~/code/noScribe`:

```bash
# First inspect current changes and keep a backup/commit of your working code.
git status --short
git diff --check

# Extract the release kit from your Downloads folder into the repository root.
unzip -o ~/Downloads/noscribe_voice_small_release_kit.zip -d .

# Ensure local downloaded model weights and temporary build output are not committed.
cat >> .git/info/exclude <<'EOF'
/models/small/
/pyinstaller/noScribe_voice_win.spec
/pyinstaller/noScribe_voice_macOS.spec
/pyinstaller/noScribe_voice_linux.spec
EOF

# Review then commit only intended source and build files.
git add noScribe tests .github/workflows/voice-release-builds.yml tools/voice_release/prepare_build.py README_VOICE_RELEASE.md
git status --short
git commit -m "Add VOICE pause option and Whisper Small cross-platform builds"
```

Note: `git add noScribe tests` is appropriate only if your edits are in those
folders. Review `git status` to include any other intentional source changes.
Never commit your large `models/small` weights; the Actions runner downloads them.

## Push to your fork

Inspect your Git remotes with `git remote -v` first. If `origin` is the upstream
repository, fork `kaixxx/noScribe` in your browser and update `origin` to point
to your fork. Then:

```bash
git push -u origin voice-custom-pauses
```

The workflow runs on pushes to branches named `voice-custom-pauses` or
`voice-pause-markers`. GitHub Actions must be enabled in the fork. You can also
run it manually once the workflow appears in GitHub Actions on the default branch.

In **Actions -> Build noScribe VOICE (Whisper Small)**, open a completed run and
download the three artifacts. Each artifact is a GitHub ZIP wrapper containing
the platform release archive.

## Before public release

- All three builds are **unverified until CI runs and a real transcription is
  tested** on each operating system. `--help` only checks basic startup.
- macOS output is for **Apple Silicon**, not an Intel Mac. noScribe's current
  Intel support is limited; you can continue running your source build locally.
- macOS app is **not signed or notarized** and may trigger Gatekeeper warnings.
- Bundled model is Whisper Small, not the Precise/Fast upstream default.
  Performance and segmentation may differ from your large-v3 research baseline.
- Speaker identification requires compatible PyAnnote assets/dependencies.
  Validate it separately from speech transcription.
- The separate noScribe Editor is not bundled here. Users can obtain the
  official editor separately; automatic editor opening may require setup.
- Observe the original noScribe **GPL-3.0** requirements, copyright attribution,
  required source code availability, and all third-party/model licenses. Include
  the upstream license and your patch/source link in the GitHub Release.
- For distribution, create a GitHub Release in your fork and upload tested ZIP/
  tar.gz archives. This workflow intentionally does **not** auto-publish.

## If a build fails

Download or copy the corresponding GitHub Actions job log. Failures from pinned
PyTorch/torchaudio/macOS requirements or PyInstaller hooks are possible; do not
assume the three archives exist before inspecting the results.

Official source: https://github.com/kaixxx/noScribe
Model source: https://huggingface.co/Systran/faster-whisper-small
