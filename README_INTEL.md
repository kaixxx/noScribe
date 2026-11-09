# noScribe VOICE — Intel Mac application build

Add these files to your existing `voice-custom-pauses` branch at their indicated paths. The workflow targets the GitHub-hosted `macos-15-intel` runner, uses Python 3.12, downloads multilingual Faster-Whisper Small, creates an Intel-only CPU build, and uploads `noScribe-VOICE-macOS-Intel-x64-Small.zip`.

This edition is a standalone `.app`. Students do not need Python or Conda. The app bundles its model; downloading the ZIP requires an internet connection, but normal on-device transcription runs locally.

**Differences from the full noScribe release:** Speaker identification and overlap detection via PyAnnote are unavailable in this Intel edition. The speaker dropdown is locked at `none`. The checkbox for VOICE-style pauses is unchanged. The noScribe Editor remains separate and is not bundled. The workflow changes source code only in the temporary GitHub runner checkout; it does *not* patch your Mac's working application, the other platform specs, or the upstream repository.

## Installation in your cloned repo

After extracting this kit, copy its `.github/` and `tools/` folders over the corresponding paths in your Git checkout (merge directories; do not delete existing files). A convenient one-line approach from the repo root is:

```bash
unzip -o ~/Downloads/noScribe-VOICE-Intel-build-kit.zip -d .
```

Then:

```bash
git add .github/workflows/build-macos-intel.yml tools/voice_release/prepare_intel_build.py README_INTEL.md
git diff --cached --check
git commit -m "Add Intel macOS Whisper Small package"
git push origin voice-custom-pauses
```

Open the **Actions** tab and select **Build noScribe VOICE — macOS Intel (Whisper Small)**. On success, download the artifact `noScribe-VOICE-macOS-Intel-x64-Small` (the GitHub artifact itself contains the ZIP archive). Extract the inner ZIP to obtain `noScribe-VOICE.app`.

**First open on student Macs:** The app is unsigned/unnotarized, so macOS Gatekeeper may block the first launch. Try Control-clicking the app in Finder and choosing Open, or use System Settings > Privacy & Security > Open Anyway after attempting launch. Never instruct students to disable Gatekeeper globally. For a smoother truly double-click-first-time install for a whole class, sign and notarize the app using an Apple Developer account.

**Validation still required:** An Action success proves that a Python model can load and that the bundled executable lists the model; it does not prove successful end-to-end audio transcription on students' actual Intel Mac configurations. Test with real audio, check the pause checkbox ON/OFF, and note the oldest macOS version tested before public distribution. The CI runner is macOS 15 Intel, so compatibility with earlier versions is not guaranteed.

**Licensing:** Retain the upstream GPL-3.0 license and publish corresponding source code for your modifications. Check the model and third-party licenses too.
