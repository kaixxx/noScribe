---
tags:
  - pyannote
  - pyannote-audio
  - pyannote-audio-pipeline
  - audio
  - voice
  - speech
  - speaker
  - speaker-diarization
  - speaker-change-detection
  - voice-activity-detection
  - overlapped-speech-detection
  - automatic-speech-recognition
license: cc-by-4.0
extra_gated_prompt: "Your input helps us strengthen the pyannote community and improve our open-source offerings. This pipeline is released under the CC-BY-4.0 license and will always remain freely accessible. By providing your details, you agree that we may email you occasionally with important news about pyannote models, invitations to try premium pipelines, and information about specific services designed for researchers and professionals like you."
extra_gated_fields:
  Company/university: text
  Use case:
     type: select
     options: 
       - label: Meeting note taker (automated meeting transcription, action item extraction, and speaker identification in recordings)
         value: meeting
       - label: Conversation AI (chatbots, voice assistants, multi-turn dialogue systems with speaker awareness)
         value: conversation
       - label: CCaaS and customer experience (call center analytics, customer service optimization, and interaction quality monitoring)
         value: ccaas
       - label: Voice agents (AI-powered phone systems, automated customer service, voice-based interactions)
         value: agent
       - label: Media and automated dubbing (content creation, podcast processing, video production, and multilingual media)
         value: dubbing
       - label: Training and development (educational content analysis, corporate training evaluation, and learning assessment tools)
         value: training
       - label: Other
         value: other
---

# `community-1` speaker diarization

This pipeline ingests mono audio sampled at 16kHz and outputs speaker diarization.

- stereo or multi-channel audio files are automatically downmixed to mono by averaging the channels.
- audio files sampled at a different rate are resampled to 16kHz automatically upon loading.

The [main improvements brought by `Community-1`](https://www.pyannote.ai/blog/community-1) are:

- [improved](#benchmark) speaker assignment and counting
- simpler reconciliation with transcription timestamps with [*exclusive*](#exclusive-speaker-diarization) speaker diarization
- easy [offline use](#offline-use) (i.e. without internet connection)
- (optionally) [hosted](https://hf.co/pyannote/speaker-diarization-community-1-cloud) on pyannoteAI cloud


## Setup

1. `pip install pyannote.audio`
2. Accept user conditions
3. Create access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

## Quick start

```python
# download the pipeline from Huggingface
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", 
    token="{huggingface-token}")

# run the pipeline locally on your computer
output = pipeline("audio.wav")

# print the predicted speaker diarization 
for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
```

## Benchmark

Out of the box, `Community-1` is much better than `speaker-diarization-3.1`. 

We report [diarization error rates](http://pyannote.github.io/pyannote-metrics/reference.html#diarization) (in %) on large collection of academic benchmarks (fully automatic processing, no forgiveness collar, nor skipping overlapping speech).

| Benchmark (last updated in 2025-09) | <a href="https://hf.co/pyannote/speaker-diarization-3.1">`legacy` (3.1)</a>| <a href="https://www.pyannote.ai/blog/community-1">`community-1`</a> | <a href="https://www.pyannote.ai/blog/precision-2">`precision-2`</a> | 
| --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------| ------------------------------------------------ |
| [AISHELL-4](https://arxiv.org/abs/2104.03603)                                                                               | 12.2 | 11.7 | 11.4 |
| [AliMeeting](https://www.openslr.org/119/) (channel 1)                                                                      | 24.5 | 20.3 | 15.2 |
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) (IHM)                                                                        | 18.8 | 17.0 | 12.9 |
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) (SDM)                                                                        | 22.7 | 19.9 | 15.6 |
| [AVA-AVD](https://arxiv.org/abs/2111.14448)                                                                                 | 49.7 | 44.6 | 37.1 |
| [CALLHOME](https://catalog.ldc.upenn.edu/LDC2001S97) ([part 2](https://github.com/BUTSpeechFIT/CALLHOME_sublists/issues/1)) | 28.5 | 26.7 | 16.6 |
| [DIHARD 3](https://catalog.ldc.upenn.edu/LDC2022S14) ([full](https://arxiv.org/abs/2012.01477))                             | 21.4 | 20.2 | 14.7 |
| [Ego4D](https://arxiv.org/abs/2110.07058) (dev.)                                                                            | 51.2 | 46.8 | 39.0 |
| [MSDWild](https://github.com/X-LANCE/MSDWILD)                                                                               | 25.4 | 22.8 | 17.3 |
| [RAMC](https://www.openslr.org/123/)                                                                                        | 22.2 | 20.8 | 10.5 |
| [REPERE](https://www.islrn.org/resources/360-758-359-485-0/) (phase2)                                                       | 7.9  |  8.9 |  7.4 |
| [VoxConverse](https://github.com/joonson/voxconverse) (v0.3)                                                                | 11.2 | 11.2 |  8.5 |

`Precision-2` model is even better and can be tested like this:

1. Create an API key on [pyannoteAI dashboard]((https://dashboard.pyannote.ai)) (free credits included)
2. Change one line of code

```diff
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
-     'pyannote/speaker-diarization-community-1', token="{huggingface-token}")
+     'pyannote/speaker-diarization-precision-2', token="{pyannoteAI-api-key}")
diarization = pipeline("audio.wav")  # runs on pyannoteAI servers
```

## Processing on GPU

`pyannote.audio` pipelines run on CPU by default.
You can send them to GPU with the following lines:

```python
import torch
pipeline.to(torch.device("cuda"))
```

## Processing from memory

Pre-loading audio files in memory may result in faster processing:

```python
waveform, sample_rate = torchaudio.load("audio.wav")
output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
```

## Monitoring progress

Hooks are available to monitor the progress of the pipeline:

```python
from pyannote.audio.pipelines.utils.hook import ProgressHook
with ProgressHook() as hook:
    output = pipeline("audio.wav", hook=hook)
```

## Controlling the number of speakers

In case the number of speakers is known in advance, one can use the `num_speakers` option:

```python
output = pipeline("audio.wav", num_speakers=2)
```

One can also provide lower and/or upper bounds on the number of speakers using `min_speakers` and `max_speakers` options:

```python
output = pipeline("audio.wav", min_speakers=2, max_speakers=5)
```

## Exclusive speaker diarization

`Community-1` pretrained pipeline returns a new *exclusive* speaker diarization, on top of the regular speaker diarization, available as `output.exclusive_speaker_diarization`.

This is a feature which is [backported from our latest commercial model](https://www.pyannote.ai/blog/precision-2) that simplifies the reconciliation between fine-grained speaker diarization timestamps and (sometimes not so precise) transcription timestamps.

## Offline use

1. In the terminal, copy the pipeline on disk:

```bash
# make sure git-lfs is installed (https://git-lfs.com)
git lfs install

# create a directory on disk
mkdir /path/to/directory

# when prompted for a password, use an access token with write permissions.
# generate one from your settings: https://huggingface.co/settings/tokens
git clone https://hf.co/pyannote/speaker-diarization-community-1 /path/to/directory/pyannote-speaker-diarization-community-1
```

2. In Python, use the pipeline without internet connection:

```python
# load pipeline from disk (works without internet connection)
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained('/path/to/directory/pyannote-speaker-diarization-community-1')

# run the pipeline locally on your computer
output = pipeline("audio.wav")
```

## Citations

1. Speaker segmentation model

```bibtex
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```

2. Speaker embedding model

```bibtex
@inproceedings{Wang2023,
  title={Wespeaker: A research and production oriented speaker embedding learning toolkit},
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  booktitle={ICASSP 2023, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```


3. Speaker clustering

```bibtex
@article{Landini2022,
  author={Landini, Federico and Profant, J{\'a}n and Diez, Mireia and Burget, Luk{\'a}{\v{s}}},
  title={{Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks}},
  year={2022},
  journal={Computer Speech \& Language},
}
```

## Acknowledgment

Training and tuning made possible thanks to [GENCI](https://www.genci.fr/) on the [**Jean Zay**](http://www.idris.fr/eng/jean-zay/) supercomputer.

