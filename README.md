# noScribe 
## Cutting Edge AI Technology for Automated Audio Transcription

**Download current version 0.4:**
- Windows: [https://drive.switch.ch/index.php/s/EIVup04qkSHb54j?path=%2FnoScribe%20vers.%200.4%2FWindows](https://drive.switch.ch/index.php/s/EIVup04qkSHb54j?path=%2FnoScribe%20vers.%200.4%2FWindows)
- macOS (beta version, ported by [gernophil](https://github.com/gernophil)):  
[https://drive.switch.ch/index.php/s/EIVup04qkSHb54j?path=%2FnoScribe%20vers.%200.4%2FmacOS](https://drive.switch.ch/index.php/s/EIVup04qkSHb54j?path=%2FnoScribe%20vers.%200.4%2FmacOS)
> [Please discuss your experiences with version 0.4 here](https://github.com/kaixxx/noScribe/discussions/28) or open an issue if you find errors. Thank you!


## What is noScribe?
- An AI-based software that **transcribes interviews** for qualitative social research or journalistic use
- noScribe is **free and open source** ([GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html))
- It runs **completely local** on your computer. No data is sent to the internet. No cloud, no worries
- It can distinguish different **speakers** and understands 99 languages (more or less, see below)
- It includes a **nice editor** to review, verify and correct the resulting transcript
- It is standing on the shoulders of giants: [Whisper from OpenAI](https://github.com/openai/whisper), [faster-whisper by Guillaume Klein](https://github.com/guillaumekln/faster-whisper) and [pyannote from Hervé Bredin](https://github.com/pyannote/pyannote-audio)

  

![Main window](img/noScribe_main_window.png)
(The transcript is from [this interview](https://www.youtube.com/watch?v=vOwajAbvPzQ&t=2018s) which I did in May 2022 with the Russian sociologist Natalia Savelyeva.)

## Limitations 
- noScribe needs a fairly up-to-date computer, or the transcription will take forever. (Consider letting it run over night on a slower machine.)
- Since it uses sophisticated AI models, the download is quite large – about 3.7 GB
- Poor audio quality will lead to poor transcription results. 
- No automatic transcription is perfect, there will always be some manual revision necessary. Use the included Editor to check your transcripts thouroughly. (See also ["Factors Influencing the Quality"](#factors-influencing-the-quality-of-the-transcription) and ["Known Issues"](#known-issues) below.)

## Why the Name "noScribe"?
The [urban dictionary](https://www.urbandictionary.com/define.php?term=Scribe) defines **scribe** as *"a person whose entire miserable existence has been reduced to academic grunge and pain".* I hope this software will make your academic life a little less painful and grungy, hence the name noScribe :)

## About Me
**Kai Dröge**, PhD in sociology (with a background in computer science), qualitative researcher and teacher, [Lucerne University for Applied Science (Switzerland)](https://www.hslu.ch/de-ch/hochschule-luzern/ueber-uns/personensuche/profile/?pid=823) and [Institute for Social Research, Frankfurt/M. (Germany)](https://www.ifs.uni-frankfurt.de/personendetails/kai-droege.html).

## Usage
### Installation
- **Download** the latest release for your operating system from here: [noScript releases](https://drive.switch.ch/index.php/s/EIVup04qkSHb54j) (SWITCHdrive is a secure data sharing platform for Swiss universities)
- Start the downloaded setup file. This may take a while, be patient.

### Settings
<img align="left" src="img/noScribe_settings.png" width="300">

- Select your **audio file** and a **filename for the transcript.**
- **Start** and **Stop** accept timestamps in the format hh:mm:ss. Use this to limit the transcription to a particular part of the recording. This is especially helpful to test your settings with a small sample before committing to transcribe the whole interview, which may take several hours. Leave Stop empty if you want to transcribe till the end of the audio file.
- **Language:** choose the language of your transcript or leave it on "auto"
- **Quality:** "precise" is the recommended setting and will give you the most accurate transcript. On a slower machine, you can also try the "fast" option. This will be much quicker but requires more manual revision afterwards. 
- **Mark pause**: If enabled, parts of your audio with no voice activity will be marked as pauses. Pauses are transcribed as round brackets with one dot per second inside, i.e., "(..)" for "two seconds pause". Pauses longer then 10 seconds are written out as "(XX seconds pause)" or "(XX minutes pause)". You have the option to mark either pauses of one second and more ("1sec+"), two seconds and more ("2sec+"), or only the longer ones of three seconds and more ("3sec+"). Choose "none" to disable this feature entirely.
- **Speaker detection:** "auto" will use the pyannote AI model to detect different speakers in your audio and structure the transcript accordingly. Setting this to "none" will skip this step and save you about half the time of the whole process. But the resulting transcript will be a continuous text without any information about speaker changes. 
- **Overlapping speech**: If enabled, noScribe tries to mark when two people speak at the same time. The overlapping portion is enclosed in //double slashes//. (This is an experimental feature which may or may not work in some instances.)
- **Timestamps**: If enabled, noScribe will include timestamps in the format [hh:mm:ss] into the transcript at any speaker change or every 60 seconds. I find these timestamps a little distracting, so I have disabled them by default. But they can be helpful in certain applications. Even with disabled timestamps, it is quite easy to find out the audio timecode for a certain segment: Just open the transcript in the noScribe Editor, move the cursor through the text and you will see the corresponding timecode in the bottom right corner of the app. 

### Transcription process
- If you are ready, click the **Start**-button in the bottom left. **Cancel** will abort the process. 
- Be aware that **a one-hour interview can take two to five hours processing time** and will put a heavy load on your machine. Doing this on battery-power is not recommended.
- A **progress bar** at the bottom of the app will show how far you are into the whole process. 
- The **main window** will log progress-messages and errors. It will also show the text of your interview during the last step of the transcription. 
- The transcript will be auto saved every few seconds under the given filename.
- NoScribe produces an HTML-file. This can be opened in every common word editor (including MS Word, LibreOffice). 
- Before working with the transcript though, you should check it with the included editor. There will always be some errors. Click on the filename in the progress window (blue + underlined) to edit the file. 


## noScribeEdit 
The included editor to check the final transcript. 

![The transcript in the noScribe Editor](img/noScribe_Editor.png)

The noScribe Editor is a separate app that can also be run independent from noScribe. It contains some handy features to check your finished transcript for errors and correct them:
- Press **Ctrl + Spacebar** or the **orange button in the toolbar** to hear the audio which corresponds to your current position in the text. 
- The **selection of the text will follow the audio that you hear**. If you want to **make changes,** click anywhere in the text with your mouse or use the arrow keys to move the cursor. The audio will stop, and you can edit the text.
- You can also **stop the audio** by pressing Ctrl + Spacebar again or clicking the orange button.
- If you want to **speed up or slow down the audio**, change the "100%"-field next to the "Play/Pause Audio"-Button to the appropriate speed.
- Use the loupe in the toolbar to **zoom in or out**
- You will find the **most common features of a basic text editor** in the toolbar as well as in the menu at the top (basic text formatting, cut, copy & paste, undo & redo).
- Your typical **hotkeys** will also work (i.e., Ctrl+S for Save). You can see all the hotkeys if you open the menu. As already mentioned, 'Ctrl+Space' is the hotkey you'll use the most as it starts or pauses the audio. 

## Factors Influencing the Quality of the Transcription
- A **good audio recording with clear voices and no ambient noise** is crucial for a high-quality transcription. Investing some effort in the quality of the recording will save you much time in the manual revision process later. 
- Whisper (the AI powering noScribe) understands 99 different languages, but the quality of the transcription varies widely between them. **Spanish, Italian, English, Portuguese and German** are best supported (see [here for more info]( https://github.com/openai/whisper#available-models-and-languages)).
- Whisper handles **dialects** fairly well (i.e., Swiss-German), but the transcript might need more manual work in the revision.

## Known Issues
- The whisper AI can sometimes get **stuck in a loop of repeating text,** especially on longer audio files. If this happens, try to transcribe shorter sections (using the "Start" and "Stop" fields in noScribe), and join them manually.
- **Multilingual audio** is not supported. If the language changes mid interview, whisper will actually try to translate the text, which is usually not what we want.
- **Filler words** like "uhm" and especially **nonverbal expressions** like laughter are often not included in the transcript, although they are usually required for a good qualitative analysis. You must add these elements manually. (The identification of filler words works best if you select the correct language for the transcript, not "auto".) 
- **Speaker identification:** In some recordings, the AI used by noScribe may not be able to tell the voices of certain speakers apart, even if they sound quite different to the human ear. It may also happen that noScribe identifies more speakers in a recording then there actually are. Check the results carefully.
- The whisper AI can sometimes **hallucinate**, especially in silent parts of the recording when it interprets background noise as 'text'. Check your transcripts carefully. 

## Advanced Options
- After the app has run for the first time, you will find a file named **config.yml** in the user config directory (on windows: C:\Users\<username>\AppData\Local\noScribe\noScribe\config.yml). Here, you can change a few **extra settings,** i.e., the language of the user interface.
- **Prompts**: The whisper AI can be initialized with a short text-sequence called prompt (see [here for more info](https://platform.openai.com/docs/guides/speech-to-text/prompting)). This will influence the style of the following transcription. I tried to force the AI to include filler words like "uhm" in the transcription by giving it a prompt containing them (like "Umm, let me think like, hmm."). But this only worked on some occasions (whisper tends to 'forget' the prompt quite quickly). Prompts are language specific and will only be applied if you select a particular language (not 'auto'). You can change or add prompts for other languages in the file "prompt.yml" in the home directory of the app. Please don’t use prompts longer than one sentence since this will mess up the speaker separation.
- Also in the user config directory you will find a folder named **log** with detailed log-files for every transcript (also unfinished ones). This can be helpful in the case of any errors. Be aware though that these files also contain the text of your transcripts which might include sensitive information. 

## Development and Contribution
- I developed noScribe in python 3.9
- If you want to run noScribe directly from the source, I recommend setting up pyannote and all its dependencies first. You **must use my fork of pyannote,** which includes a small modification run on local files only.
- I cannot host the whisper-models on GitHub because they are too large. There is a readme in the models-folder with instructions on how to get them. 
- I am happy to review tests, bug reports and pull requests (if my time allows it)

### Translations
- The noScribe UI has already been translated into many languages (thanks mlynar-czyk).
- Since most of the translations have been created with ChatGPT, there will be problems. Please report any errors that you’ll find and make – if possible – a pull request with a better translation. 
- You will find the language files in the folder "trans". 
- If you change anything in the language files, make sure to follow the conventions of the YAML language.
- If you want to change the language of the user interface, you have to change the value of the "locale" setting in the advanced settings (see above).

### MAC and LINUX support
- [noScribe has been ported to macOS](https://drive.switch.ch/index.php/s/EIVup04qkSHb54j?path=%2FnoScribe%20vers.%200.4b%2FmacOS) (alpha version, please help testing!) 
- If you want to port noScribe to Linux, go ahead! There once was a working version but it is no longer up to date.
- If you make any changes to the code of noScribe, try not to break compatibility with other platforms. I would like to keep one single codebase for all platforms.

## Other Software
If you are interested in open source software for the analysis of qualitative data, take a look at [QualCoder](https://github.com/ccbogel/QualCoder) and [Taguette](https://www.taguette.org/).
