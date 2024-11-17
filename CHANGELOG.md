# noScribe Changelog

## version 0.5:
- new WebVTT output (video subtitle format), allows also importing noScribe transcripts into EXMARaLDA
- new plain text output (*.txt)
- new automatic update notification on new releases
- improved speaker detection: number of speakers can be defined beforehand
- reduced hallucination and looping by adding a VAD filter
- CUDA support now non-beta
- small fixes with hebrew language setting, chinese UI locale, requirements file, etc.  

## version 0.4.5:
- Windows: beta version to test CUDA support (acceleration with NVIDIA graphics cards) 

## version 0.4.2:
- MacOS: Solves a bug where speaker-detection would become unreliable with MPS-acceleration (a switch to torchaudio 2.1.0 rectified this).
- Windows: no changes

## version 0.4.1:
- Windows: bugfix, rectifies a problem in combination with NVIDIA graphics cards
- macOS: First beta release. Solved a bug with macOS Sonoma where noScribe would not react to the mouse. 

## version 0.4 beta:
- much improved **speaker detection/separation**
- new option to mark **pauses** (sections with no voice activity) in the transkript
- new option to mark **overlapping speech** (experimental)
- new option to include **timestamps** in the transcript
- new **noScribe Editor** app to check and correct transcripts (no MS Word-Macros anymore)
- noScribe now outputs an **HTML-file** which can be opened in every major word editor (MS Word, LibreOffice, OpenOffice...) or QDA-software package
- many changes under the hood to prepare for an upcoming macOS-version and improve reliability and quality of the transcription
- switched from "whisper.cpp" to "faster-whisper" as the basic framework (mainly because of the more precise timestamps)  
- macOS: First alpha release

## version 0.3:
**new:**
- Translations of the user interface into Spanish, French, Italian, Japanese, Portuguese, Russian, Chinese. Thank you, [mlynar-czyk]( https://github.com/mlynar-czyk), for this contribution! Be aware: These translations have been generated with a clever use of chatGPT. Please report any errors that you will find and make – if possible – a pull request with a better translation.
- Added hyperlinks to the main window. You can now open the finished transcript directly by clicking on the filename in the log.
- Improved speaker identification, especially in situations with quick changes (by reducing "max-len" in whisper to 30).  
- Installer now runs without admin rights. You should be able to install noScribe on a computer where you don’t have administrator privileges (i.e., because the machine is managed by the IT-department of your university). Thanks you, [BabyFnord](https://github.com/BabyFnord), for this suggestion!
 
**fixes:**
- To solve the problem described in issue https://github.com/kaixxx/noScribe/issues/2 (transcription failing with error 3221225794 or 3221225501), I have now included a version of whisper.cpp that supports older hardware (non AVX2). NoScribe selects automatically which version to use. Be aware though that using such old hardware will result in a very slow transcription.
- Corrected UTF-8 encoding error that resulted in a failing transcription in some languages (i.e., Japanese, Hungarian). Thank you to the two people reporting this problem via e-mail!
- fixed: Auto save was saving too often during transcription.
- fixed: Play along function in Word sometimes not finding the beginning of the transcription
- fixed: Funny mistake in readme ("sensible data" instead of "sensitive data"). Thanks [TheOnlyWayUp](https://github.com/TheOnlyWayUp)!


## version 0.2b: 
- initial beta release
