# noScribe - AI-powered Audio Transcription
# Copyright (C) 2025 Kai Dröge
# ported to MAC by Philipp Schneider (gernophil)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import argparse
import os
# In the compiled version (no command line), stdout is None which might lead to errors
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import tkinter as tk
import customtkinter as ctk
from customtkinter.windows.widgets.scaling import CTkScalingBaseClass
from CTkToolTips import CTkToolTip
from tkHyperlinkManager import HyperlinkManager
import webbrowser
from functools import partial
from PIL import Image
import platform
import yaml
import locale
import appdirs
from subprocess import run, Popen, PIPE, STDOUT, DEVNULL
if platform.system() == 'Windows':
    # import torch.cuda # to check with torch.cuda.is_available()
    from subprocess import STARTUPINFO, STARTF_USESHOWWINDOW
#if platform.system() in ("Windows", "Linux"):
#    from ctranslate2 import get_cuda_device_count
#    import torch
import re
if platform.system() == "Darwin": # = MAC
    from subprocess import check_output
    if platform.machine() == "x86_64":
        os.environ['KMP_DUPLICATE_LIB_OK']='True' # prevent OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.
    # import torch.backends.mps # loading torch modules leads to segmentation fault later
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps
import AdvancedHTMLParser
import html
from threading import Thread
from tempfile import TemporaryDirectory
import datetime
from pathlib import Path
if platform.system() in ("Darwin", "Linux"):
    import shlex
if platform.system() == 'Windows':
    import cpufeature
if platform.system() == 'Darwin':
    import Foundation
import logging
import json
import urllib
import multiprocessing as mp
import queue as pyqueue
import gc
import traceback
from enum import Enum
from typing import Optional, List
import time

import utils
import speaker_db

 # Pyinstaller fix, used to open multiple instances on Mac
mp.freeze_support()

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

app_version = '0.7'
app_year = '2025'
app_dir = os.path.abspath(os.path.dirname(__file__))

ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('blue')

default_html = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html >
<head >
<meta charset="UTF-8" />
<meta name="qrichtext" content="1" />
<style type="text/css" >
p, li { white-space: pre-wrap; }
</style>
<style type="text/css" > 
 p { font-size: 0.9em; } 
 .MsoNormal { font-family: "Arial"; font-weight: 400; font-style: normal; font-size: 0.9em; }
 @page WordSection1 {mso-line-numbers-restart: continuous; mso-line-numbers-count-by: 1; mso-line-numbers-start: 1; }
 div.WordSection1 {page:WordSection1;} 
</style>
</head>
<body style="font-family: 'Arial'; font-weight: 400; font-style: normal" >
</body>
</html>"""

languages = {
    "Auto": "auto",
    "Multilingual": "multilingual",
    "Afrikaans": "af",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Belarusian": "be",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese": "zh",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Malay": "ms",
    "Marathi": "mr",
    "Maori": "mi",
    "Nepali": "ne",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tamil": "ta",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "Welsh": "cy",
}

# config
config_dir = appdirs.user_config_dir('noScribe')
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

config_file = os.path.join(config_dir, 'config.yml')

# -----------------------------------------------------------------------
# languages.yml – user-editable language filter
# -----------------------------------------------------------------------
# The file lives in the same folder as noScribe.py (app_dir).
# noScribe reads it but NEVER overwrites it, so comments and formatting
# are always preserved.
#
# On first run the file is created automatically with all languages
# listed and an explanatory header, so users can immediately start
# commenting out the ones they don't need.

_languages_file = os.path.join(app_dir, 'languages.yml')

_LANGUAGES_FILE_HEADER = """\
# noScribe – Transcription Language List
# ----------------------------------------
# Each uncommented line enables that language in the dropdown menu.
# To hide a language, add '#' at the beginning of the line.
# This file is NEVER rewritten by noScribe, so your edits are safe.
#
# Tip: keep only the languages you actually use to shorten the list.
# 'Auto' lets noScribe detect the language automatically (recommended).
# 'Multilingual' is an experimental mode for mixed-language recordings.

"""

def _build_languages_file_content() -> str:
    lines = [_LANGUAGES_FILE_HEADER]
    for name in languages:
        lines.append(f"- {name}\n")
    return "".join(lines)

if not os.path.exists(_languages_file):
    try:
        with open(_languages_file, 'w', encoding='utf-8') as _f:
            _f.write(_build_languages_file_content())
    except Exception:
        pass  # Non-fatal: fall back to full list

# Load and apply the language filter (if the file exists and is readable)
try:
    with open(_languages_file, 'r', encoding='utf-8') as _f:
        _lang_list = yaml.safe_load(_f)
    if isinstance(_lang_list, list) and _lang_list:
        _allowed = {str(x) for x in _lang_list if x is not None}
        _filtered = {k: v for k, v in languages.items() if k in _allowed}
        # 'Auto' must always be present so the dropdown never breaks
        _filtered.setdefault('Auto', 'auto')
        languages = _filtered
except Exception:
    pass  # Non-fatal: keep the full list

try:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        if not config:
            raise # config file is empty (None)        
except: # seems we run it for the first time and there is no config file
    config = {}
    
def get_config(key: str, default) -> str:
    """ Get a config value, set it if it doesn't exist """
    if key not in config:
        config[key] = default
    return config[key]

force_pyannote_cpu = get_config('force_pyannote_cpu', '').lower() == 'true'
force_whisper_cpu = get_config('force_whisper_cpu', '').lower() == 'true'

_CUDA_ERROR_KEYWORDS = (
    'cuda',
    'cublas',
    'cudnn',
    'cufft',
    'device-side assert',
    'invalid device function',
    'nccl',
    'gpu driver',
    'compute capability',
    'hip error',
)

def _is_cuda_error_message(message: str) -> bool:
    if not message:
        return False
    if message.find('(device_cpu)') != -1:
        return False
    lower_message = message.lower()
    return any(keyword in lower_message for keyword in _CUDA_ERROR_KEYWORDS)

def version_higher(version1, version2, subversion_level=99) -> int:
    """Will return 
    1 if version1 is higher
    2 if version2 is higher
    0  if both are equal 
    
    subversion_level: Adjusts how deep suversions are compared. 
                      If subversion_level = 1, "0.7.3" and "0.7.4" will be equal, because the comparison
                      stops after the first level of subversions ("0.7").  
                      Default: 99
    """
    version1_elems = version1.split('.')
    version2_elems = version2.split('.')
    # make both versions the same length
    elem_num = max(len(version1_elems), len(version2_elems))
    while len(version1_elems) < elem_num:
        version1_elems.append('0')
    while len(version1_elems) < elem_num:
        version1_elems.append('0')
    for i in range(elem_num):
        if int(version1_elems[i]) > int(version2_elems[i]):
            return 1
        elif int(version2_elems[i]) > int(version1_elems[i]):
            return 2
        if i >= subversion_level:
            break
    # must be completely equal
    return 0
    
config['app_version'] = app_version

def save_config():
    with open(config_file, 'w') as file:
        yaml.safe_dump(config, file)

save_config()

# locale: setting the language of the UI
# see https://pypi.org/project/python-i18n/
import i18n
from i18n import t
i18n.set('filename_format', '{locale}.{format}')
i18n.load_path.append(os.path.join(app_dir, 'trans'))


def _show_startup_error(message: str) -> None:
    """Show a message box during startup failures if possible."""
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror(title='noScribe', message=message)
    except Exception as tk_error:
        print(f"ERROR: {message}", file=sys.stderr)
    finally:
        if root is not None:
            root.destroy()

try:
    app_locale = config['locale']
except:
    app_locale = 'auto'

if app_locale == 'auto': # read system locale settings
    try:
        if platform.system() == 'Windows':
            app_locale = locale.getdefaultlocale()[0][0:2]
        elif platform.system() == "Darwin": # = MAC
            app_locale = Foundation.NSUserDefaults.standardUserDefaults().stringForKey_('AppleLocale')[0:2]
    except:
        app_locale = 'en'
i18n.set('fallback', 'en')

translation_error = ''
print('\nnoScribe')
try:
    i18n.set('locale', app_locale)
    print(t('app_header'), '\n')
    config['locale'] = app_locale
except Exception as locale_error:
    translation_error = f"Failed to load translation for locale '{app_locale}'. Falling back to English.\n\n"
    if app_locale != 'en':
        try:
            i18n.set('locale', 'en')
            print(t('app_header'), '\n')
            app_locale = 'en'
        except Exception as english_error:
            print("Failed to load English fallback translation.")
            _show_startup_error(
                'NoScribe could not load the English fallback translation and needs to close.'
            )
            raise SystemExit(1) from english_error
    else:
        print("English translation failed to load during startup.")
        _show_startup_error(
            'noScribe could not load the English translation and needs to close.'
        )
        raise SystemExit(1) from locale_error

# determine optimal number of threads for faster-whisper (depending on cpu cores)
if platform.system() == 'Windows':
    number_threads = get_config('threads', cpufeature.CPUFeature["num_physical_cores"])
elif platform.system() == "Linux":
    number_threads = get_config('threads', os.cpu_count() if os.cpu_count() is not None else 4)
elif platform.system() == "Darwin": # = MAC
    if platform.machine() == "arm64":
        cpu_count = int(check_output(["sysctl", "-n", "hw.perflevel0.logicalcpu_max"]))
    elif platform.machine() == "x86_64":
        cpu_count = int(check_output(["sysctl", "-n", "hw.logicalcpu_max"]))
    else:
        raise Exception("Unsupported mac")
    number_threads = get_config('threads', int(cpu_count * 0.75))
else:
    raise Exception('Platform not supported yet.')

# timestamp regex
timestamp_re = re.compile(r'\[\d\d:\d\d:\d\d.\d\d\d --> \d\d:\d\d:\d\d.\d\d\d\]')

# Helper functions

def iter_except(function, exception):
        # Works like builtin 2-argument `iter()`, but stops on `exception`.
        try:
            while True:
                yield function()
        except exception:
            return
        
# Helper for text only output
        
def html_node_to_text(node: AdvancedHTMLParser.AdvancedTag) -> str:
    """
    Recursively get all text from a html node and its children. 
    """
    # For text nodes, return their value directly
    if AdvancedHTMLParser.isTextNode(node): # node.nodeType == node.TEXT_NODE:
        return html.unescape(node)
    # For element nodes, recursively process their children
    elif AdvancedHTMLParser.isTagNode(node):
        text_parts = []
        for child in node.childBlocks:
            text = html_node_to_text(child)
            if text:
                text_parts.append(text)
        # For block-level elements, prepend and append newlines
        if node.tagName.lower() in ['p', 'div', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br']:
            if node.tagName.lower() == 'br':
                return '\n'
            else:
                return '\n' + ''.join(text_parts).strip() + '\n'
        else:
            return ''.join(text_parts)
    else:
        return ''

def html_to_text(parser: AdvancedHTMLParser.AdvancedHTMLParser) -> str:
    return html_node_to_text(parser.body)

# Helper for WebVTT output

def vtt_escape(txt: str) -> str:
    txt = html.escape(txt, quote=False)
    while txt.find('\n\n') > -1:
        txt = txt.replace('\n\n', '\n')
    return txt    


def html_to_webvtt(parser: AdvancedHTMLParser.AdvancedHTMLParser, media_path: str):
    vtt = 'WEBVTT '
    paragraphs = parser.getElementsByTagName('p')
    # The first paragraph contains the title
    vtt += vtt_escape(paragraphs[0].textContent) + '\n\n'
    # Next paragraph contains info about the transcript. Add as a note.
    vtt += vtt_escape('NOTE\n' + html_node_to_text(paragraphs[1])) + '\n\n'
    # Add media source:
    vtt += f'NOTE media: {media_path}\n\n'

    #Add all segments as VTT cues
    segments = parser.getElementsByTagName('a')
    i = 0
    for i in range(len(segments)):
        segment = segments[i]
        name = segment.attributes['name']
        if name is not None:
            name_elems = name.split('_', 4)
            if len(name_elems) > 1 and name_elems[0] == 'ts':
                start = utils.ms_to_webvtt(int(name_elems[1]))
                end = utils.ms_to_webvtt(int(name_elems[2]))
                spkr = name_elems[3]
                txt = vtt_escape(html_node_to_text(segment))
                vtt += f'{i+1}\n{start} --> {end}\n<v {spkr}>{txt.lstrip()}\n\n'
    return vtt

# Transcription Job Management Classes

class JobStatus(Enum):
    WAITING = "waiting"
    AUDIO_CONVERSION = "audio_conversion"
    SPEAKER_IDENTIFICATION = "speaker_identification"
    TRANSCRIPTION = "transcription"
    CANCELING = "canceling"
    CANCELED = "canceled"
    FINISHED = "finished"
    ERROR = "error"

class TranscriptionJob:
    """Represents a single transcription job with all its parameters and status"""
    
    def __init__(self):
        # Status tracking
        self.status: JobStatus = JobStatus.WAITING
        self.error_message: Optional[str] = None
        self.error_tb: Optional[str] = None
        self.created_at: datetime.datetime = datetime.datetime.now()
        self.started_at: Optional[datetime.datetime] = None
        self.finished_at: Optional[datetime.datetime] = None
        
        # Progress tracking
        self.progress: float = 0.0  # Progress from 0.0 to 1.0
        
        # File paths
        self.audio_file: str = ''
        self.transcript_file: str = ''
        # Partial transcript tracking
        self.has_partial_transcript: bool = False
        
        # Time range
        self.start: int = 0  # milliseconds
        self.stop: int = 0   # milliseconds (0 means until end)
        
        # Language and model settings
        self.language_name: str = 'Auto'
        self.whisper_model: str = ''  # path to the model
        
        # Processing options
        self.speaker_detection: str = 'auto'
        self.overlapping: bool = True
        self.timestamps: bool = False
        self.disfluencies: bool = True
        self.pause: int = 0  # index value (0=none, 1=1sec+, etc.)
        
        # Config-based options
        self.whisper_beam_size: int = 1
        self.whisper_temperature: float = 0.0
        self.whisper_compute_type: str = 'default'
        self.timestamp_interval: int = 60_000
        self.timestamp_color: str = '#78909C'
        self.pause_marker: str = '.'
        self.auto_save: bool = True
        self.whisper_xpu: str = 'cpu' 
        self.vad_threshold: float = 0.5
        
        # Derived properties
        self.file_ext: str = ''
    
    def set_running(self):
        """Mark job as running and record start time"""
        self.status = JobStatus.AUDIO_CONVERSION
        self.started_at = datetime.datetime.now()
    
    def set_finished(self):
        """Mark job as finished and record completion time"""
        self.status = JobStatus.FINISHED
        self.finished_at = datetime.datetime.now()
    
    def set_error(self, error_message: str, error_tb: str = ''):
        """Mark job as failed and store error message"""
        self.status = JobStatus.ERROR
        self.error_message = error_message
        self.error_tb = error_tb
        self.finished_at = datetime.datetime.now()

    def set_canceled(self, message: Optional[str] = None):
        """Mark job as canceled by the user"""
        self.status = JobStatus.CANCELED
        self.error_message = message
        self.finished_at = datetime.datetime.now()
    
    def get_duration(self) -> Optional[datetime.timedelta]:
        """Get processing duration if job is completed"""
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    def format_summary(self) -> str:
        """Build a concise, multi-line summary for tooltips.

        Uses localized UI labels where available and simple symbols for booleans.
        """
        lines = []

        def yn(v: bool) -> str:
            return '✓' if bool(v) else '✗'

        # Output file (show basename and format)
        try:
            out_name = os.path.basename(self.transcript_file) if self.transcript_file else ''
            lines.append(f"{t('job_tt_transcript_file')} {out_name}")
        except Exception:
            pass

        # Time range
        try:
            start_ms = getattr(self, 'start', 0) or 0
            stop_ms = getattr(self, 'stop', 0) or 0
            start_txt = utils.ms_to_str(start_ms) if start_ms > 0 else '00:00:00'
            stop_txt = utils.ms_to_str(stop_ms) if stop_ms > 0 else 'end'
            lines.append(f"{t('label_start')} {start_txt}")
            lines.append(f"{t('label_stop')} {stop_txt}")
        except Exception:
            pass

        # Language
        try:
            lines.append(f"{t('label_language')} {self.language_name}")
        except Exception:
            pass

        # Model (display basename if a path)
        try:
            model_disp = os.path.basename(self.whisper_model) if self.whisper_model else ''
            if not model_disp:
                model_disp = str(self.whisper_model)
            lines.append(f"{t('label_whisper_model')} {model_disp}")
        except Exception:
            pass

        # Pause threshold (map int index back to label)
        try:
            pause_opts = ['none', '1sec+', '2sec+', '3sec+']
            pause_disp = pause_opts[self.pause] if isinstance(self.pause, int) and 0 <= self.pause < len(pause_opts) else str(self.pause)
            lines.append(f"{t('label_pause')} {pause_disp}")
        except Exception:
            pass

        # Speaker detection
        try:
            lines.append(f"{t('label_speaker')} {self.speaker_detection}")
        except Exception:
            pass

        # Overlapping speech
        try:
            lines.append(f"{t('label_overlapping')} {yn(self.overlapping)}")
        except Exception:
            pass

        # Disfluencies
        try:
            lines.append(f"{t('label_disfluencies')} {yn(self.disfluencies)}")
        except Exception:
            pass

        # Timestamps
        try:
            lines.append(f"{t('label_timestamps')} {yn(self.timestamps)}")
        except Exception:
            pass

        return "\n".join([ln for ln in lines if ln])
    
class TranscriptionQueue:
    """Manages a queue of transcription jobs"""
    
    def __init__(self):
        self.jobs: List[TranscriptionJob] = []
        self.current_job: Optional[TranscriptionJob] = None  # Track currently running job
    
    def add_job(self, job: TranscriptionJob):
        """Add a job to the queue"""
        self.jobs.append(job)
    
    def get_waiting_jobs(self) -> List[TranscriptionJob]:
        """Get all jobs with WAITING status"""
        return [job for job in self.jobs if job.status == JobStatus.WAITING]
    
    def get_running_jobs(self) -> List[TranscriptionJob]:
        """Get all jobs currently being processed"""
        return [job for job in self.jobs if job.status in [JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION, JobStatus.TRANSCRIPTION, JobStatus.CANCELING]]
    
    def get_finished_jobs(self) -> List[TranscriptionJob]:
        """Get all successfully completed jobs"""
        return [job for job in self.jobs if job.status == JobStatus.FINISHED]
    
    def get_failed_jobs(self) -> List[TranscriptionJob]:
        """Get all jobs that encountered errors"""
        return [job for job in self.jobs if job.status == JobStatus.ERROR]

    def get_canceled_jobs(self) -> List[TranscriptionJob]:
        """Get all jobs that were canceled by the user"""
        return [job for job in self.jobs if job.status == JobStatus.CANCELED]
    
    def has_pending_jobs(self) -> bool:
        """Check if there are jobs waiting to be processed"""
        return len(self.get_waiting_jobs()) > 0
    
    def is_running(self) -> bool:
        """Check if any job are currently beeing processed"""
        return len(self.get_running_jobs()) > 0
    
    def get_next_waiting_job(self) -> Optional[TranscriptionJob]:
        """Get the next job to process"""
        waiting_jobs = self.get_waiting_jobs()
        return waiting_jobs[0] if waiting_jobs else None
    
    def get_queue_summary(self) -> dict:
        """Get summary statistics of the queue"""
        return {
            'total': len(self.jobs),
            'waiting': len(self.get_waiting_jobs()),
            'running': len(self.get_running_jobs()),
            'finished': len(self.get_finished_jobs()),
            'errors': len(self.get_failed_jobs()),
            'canceled': len(self.get_canceled_jobs()),
        }
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.jobs) == 0
    
    def has_output_conflict(self, transcript_file: str, ignore_job: Optional[TranscriptionJob] = None) -> bool:
        """Check if another queue job uses the same output file.
        Ignores jobs in ERROR, CANCELING, CANCELED and optionally a given job."""
        try:
            target = os.path.abspath(transcript_file)
        except Exception:
            return False
        try:
            for j in self.jobs:
                try:
                    if not j or j is ignore_job:
                        continue
                    tf = getattr(j, 'transcript_file', None)
                    if not tf:
                        continue
                    if os.path.abspath(tf) == target and j.status not in [JobStatus.ERROR, JobStatus.CANCELING, JobStatus.CANCELED]:
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def confirm_output_override(self, transcript_file: str, ignore_job: Optional[TranscriptionJob] = None) -> bool:
        """Prompt the user if a conflicting output file is found. Returns True to proceed."""
        try:
            if self.has_output_conflict(transcript_file, ignore_job=ignore_job):
                msg = t('output_override')
                return tk.messagebox.askyesno(title='noScribe', message=msg)
        except Exception:
            pass
        return True
    

# Command Line Interface

def create_transcription_job(audio_file=None, transcript_file=None, start_time=None, stop_time=None,
                           language_name=None, whisper_model_name=None, speaker_detection=None,
                           overlapping=None, timestamps=None, disfluencies=None, pause=None,
                           cli_mode=False) -> TranscriptionJob:
    """Create a TranscriptionJob with all default values
    
    This function handles both CLI and GUI job creation, ensuring all defaults
    are consistent between both modes.
    """
    job = TranscriptionJob()
    
    # File paths
    job.audio_file = audio_file or ''
    job.transcript_file = transcript_file or ''
    if job.transcript_file:
        job.file_ext = os.path.splitext(job.transcript_file)[1][1:]
        if not job.file_ext in ['html', 'txt', 'vtt']:
            raise Exception(t('err_unsupported_output_format', file_type=job.file_ext))
    
    # Time range
    job.start = start_time if start_time is not None else 0
    job.stop = stop_time if stop_time is not None else 0
    
    # Language - handle both language names and codes
    if language_name:
        if language_name in languages.values():
            # Find language name by code
            job.language_name = next(name for name, code in languages.items() if code == language_name)
        elif language_name in languages.keys():
            # Language name provided directly
            job.language_name = language_name
        else:
            raise ValueError(f"Unknown language: {language_name}")
    else:
        job.language_name = 'Auto'
    
    # Model (will be validated later when we have access to the app instance)
    job.whisper_model = whisper_model_name or 'precise'
    
    # Processing options with defaults
    job.speaker_detection = speaker_detection if speaker_detection is not None else 'auto'
    job.overlapping = overlapping if overlapping is not None else True
    job.timestamps = timestamps if timestamps is not None else False
    job.disfluencies = disfluencies if disfluencies is not None else True
    
    # Pause setting
    if pause is not None:
        if isinstance(pause, str):
            pause_options = ['none', '1sec+', '2sec+', '3sec+']
            if pause in pause_options:
                job.pause = pause_options.index(pause)
            else:
                job.pause = 1  # default to '1sec+'
        else:
            job.pause = pause
    else:
        job.pause = 1  # default to '1sec+'
    
    # Config-based options (use defaults from config)
    job.whisper_beam_size = get_config('whisper_beam_size', 1)
    job.whisper_temperature = get_config('whisper_temperature', 0.0)
    job.whisper_compute_type = get_config('whisper_compute_type', 'default')
    job.timestamp_interval = get_config('timestamp_interval', 60_000)
    job.timestamp_color = get_config('timestamp_color', '#78909C')
    job.pause_marker = get_config('pause_seconds_marker', '.')
    job.auto_save = False if get_config('auto_save', 'True') == 'False' else True
        
    job.vad_threshold = float(get_config('voice_activity_detection_threshold', '0.5'))
    
    # Platform-specific XPU settings
    """    
    if platform.system() == "Darwin":  # MAC
        xpu = get_config('pyannote_xpu', 'mps' if platform.mac_ver()[0] >= '12.3' else 'cpu')
        job.pyannote_xpu = 'mps' if xpu == 'mps' else 'cpu'
    elif platform.system() in ('Windows', 'Linux'):
        try:
            cuda_available = torch.cuda.is_available() and get_cuda_device_count() > 0
        except:
            cuda_available = False
        xpu = get_config('pyannote_xpu', 'cuda' if cuda_available else 'cpu')
        job.pyannote_xpu = 'cuda' if xpu == 'cuda' else 'cpu'
        whisper_xpu = get_config('whisper_xpu', 'cuda' if cuda_available else 'cpu')
        job.whisper_xpu = 'cuda' if whisper_xpu == 'cuda' else 'cpu'
    else:
        raise Exception('Platform not supported yet.')
    """    
    
    # Check for invalid VTT options
    if job.file_ext == 'vtt' and (job.pause > 0 or job.overlapping or job.timestamps):
        if cli_mode:
            print("Warning: VTT format doesn't support pause markers, overlapping speech, or timestamps. These options will be disabled.")
        job.pause = 0
        job.overlapping = False
        job.timestamps = False
    
    return job

def create_job_from_cli_args(args) -> TranscriptionJob:
    """Create a TranscriptionJob from command line arguments"""
    # Parse time arguments
    start_time = utils.str_to_ms(args.start) if args.start else None
    stop_time = utils.str_to_ms(args.stop) if args.stop else None

    return create_transcription_job(
        audio_file=args.audio_file,
        transcript_file=args.output_file,
        start_time=start_time,
        stop_time=stop_time,
        language_name=args.language,
        whisper_model_name=args.model,
        speaker_detection=args.speaker_detection,
        overlapping=args.overlapping,
        timestamps=args.timestamps,
        disfluencies=args.disfluencies,
        pause=args.pause,
        cli_mode=True
    )

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='noScribe - AI-powered Audio Transcription',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python noScribe.py audio.wav transcript.html
  python noScribe.py audio.mp3 transcript.txt --language en --speaker-detection 2
  python noScribe.py audio.wav transcript.vtt --start 00:01:30 --stop 00:05:00
  python noScribe.py --help-models  # Show available models
        """
    )
    
    # Special argument to show available models
    parser.add_argument('--help-models', action='store_true',
                       help='Show available Whisper models and exit')
    
    # Required arguments (when not using --help-models)
    parser.add_argument('audio_file', nargs='?',
                       help='Input audio file path')
    parser.add_argument('output_file', nargs='?', 
                       help='Output transcript file path (.html, .txt, or .vtt)')
    
    # Optional arguments
    parser.add_argument('--no-gui', action='store_true', default=False,
                       help='Run without showing the GUI (headless mode)')
    parser.add_argument('--start', default=None,
                       help='Start time (format: HH:MM:SS)')
    parser.add_argument('--stop', default=None,
                       help='Stop time (format: HH:MM:SS)')
    parser.add_argument('--language', default=None,
                       help='Language code (e.g., en, de, fr) or "auto" for auto-detection')
    parser.add_argument('--model', default=None,
                       help='Whisper model to use (use --help-models to see available models)')
    parser.add_argument('--speaker-detection', choices=['none', 'auto', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], default=None,
                       help='Speaker detection/diarization setting')
    parser.add_argument('--overlapping', action='store_true', default=None, 
                       help='Enable overlapping speech detection')
    parser.add_argument('--no-overlapping', action='store_false', dest='overlapping', default=None,
                       help='Disable overlapping speech detection')
    parser.add_argument('--timestamps', action='store_true', default=None,
                       help='Include visible timestamps in the transcript')
    parser.add_argument('--no-timestamps', action='store_false', dest='timestamps', default=None,
                       help='Exclude visible timestamps from the transcript')
    parser.add_argument('--disfluencies', action='store_true', default=None,
                       help='Include disfluencies (uh, um, etc.) in transcript')
    parser.add_argument('--no-disfluencies', action='store_false', dest='disfluencies', default=None,
                       help='Exclude disfluencies from transcript')
    parser.add_argument('--pause', choices=['none', '1sec+', '2sec+', '3sec+'], default=None,
                       help='Mark pauses in transcript')
    
    return parser.parse_args()


class TimeEntry(ctk.CTkEntry): # special Entry box to enter time in the format hh:mm:ss
                               # based on https://stackoverflow.com/questions/63622880/how-to-make-python-automatically-put-colon-in-the-format-of-time-hhmmss
    def __init__(self, master, **kwargs):
        ctk.CTkEntry.__init__(self, master, **kwargs)
        vcmd = self.register(self.validate)

        self.bind('<Key>', self.format)
        self.configure(validate="all", validatecommand=(vcmd, '%P'))

        self.valid = re.compile(r'^\d{0,2}(:\d{0,2}(:\d{0,2})?)?$', re.I)

    def validate(self, text):
        if text == '':
            return True
        elif ''.join(text.split(':')).isnumeric():
            return not self.valid.match(text) is None
        else:
            return False

    def format(self, event):
        if event.keysym not in ['BackSpace', 'Shift_L', 'Shift_R', 'Control_L', 'Control_R']:
            i = self.index('insert')
            if i in [2, 5]:
                if event.char != ':':
                    if self.get()[i:i+1] != ':':
                        self.insert(i, ':')

class JobEntryFrame(ctk.CTkFrame, CTkScalingBaseClass):
    """A custom frame that can display a progress bar as its background with text overlays"""
    
    def __init__(self, master, progress=0.0, progress_color=None, **kwargs):
        ctk.CTkFrame.__init__(self, master, **kwargs)
        CTkScalingBaseClass.__init__(self, scaling_type="widget")
        if not progress_color:
            progress_color = ctk.ThemeManager.theme['CTkProgressBar']['progress_color'][1]
        
        self.progress = progress
        self.progress_color = progress_color
        self.base_color = self._fg_color
        self.show_progress = False  # Only show progress during processing
        
        # Store text content
        self.name_text = ""
        self.status_text = ""
        self.status_color = "lightgray"
        
        # Create a canvas to draw the progress background and text
        self.progress_canvas = tk.Canvas(self, highlightthickness=0)
        self.progress_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Forward mouse events from canvas to frame for CTkToolTip functionality
        self.progress_canvas.bind("<Enter>", self._on_canvas_enter)
        self.progress_canvas.bind("<Leave>", self._on_canvas_leave)
        
        # Bind to configure event to redraw when size changes
        self.bind('<Configure>', self._on_configure)
        
        # Update the progress display
        self._update_progress_display()
    
    def destroy(self):
        """Override destroy to properly clean up scaling callbacks"""
        CTkScalingBaseClass.destroy(self)
        ctk.CTkFrame.destroy(self)
    
    def set_progress(self, progress, show_progress=True):
        """Set the progress value (0.0 to 1.0) and whether to show progress bar"""
        self.progress = max(0.0, min(1.0, progress))
        self.show_progress = show_progress
        self._update_progress_display()
    
    def set_name_text(self, text):
        """Set the name text to display"""
        self.name_text = text
        self._update_progress_display()
    
    def set_status_text(self, text, color="lightgray"):
        """Set the status text and color to display"""
        self.status_text = text
        self.status_color = color
        self._update_progress_display()
    
    def bind_click(self, callback):
        """Bind click event to the canvas"""
        self.progress_canvas.bind("<Button-1>", callback)
    
    def unbind_click(self):
        """Unbind click event from the canvas"""
        self.progress_canvas.unbind("<Button-1>")
    
    def configure_cursor(self, cursor):
        """Configure cursor for the canvas"""
        self.progress_canvas.configure(cursor=cursor)
            
    def _on_configure(self, event=None):
        """Handle resize events"""
        self._update_progress_display()
    
    def _get_scaled_font_size(self):
        """Calculate font size based on frame height and use CustomTkinter's scaling"""
        try:
            font = ctk.CTkFont()
            scaled_font = self._apply_font_scaling(font)
            return scaled_font[1]
        except:
            return 13  # Fallback
    
    def _on_canvas_enter(self, event):
        """Forward canvas Enter event to frame for CTkToolTip"""
        # Generate a synthetic Enter event for the frame
        self.event_generate("<Enter>")
    
    def _on_canvas_leave(self, event):
        """Forward canvas Leave event to frame for CTkToolTip"""
        # Generate a synthetic Leave event for the frame
        self.event_generate("<Leave>")
    
    def _update_progress_display(self):
        """Update the progress bar display and text"""
        if not self.progress_canvas.winfo_exists():
            return
            
        # Clear the canvas
        self.progress_canvas.delete("all")
        
        # Get canvas dimensions
        width = self.progress_canvas.winfo_width()
        height = self.progress_canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            # Canvas not ready yet
            self.after(10, self._update_progress_display)
            return
        
        # Calculate button area width to avoid overlap (1 button = 30px + padding)
        # Reserve space for up to 3 buttons (X, ⟲/✔, ✔)
        button_area_width = 3 * self._apply_widget_scaling(30 + 5)
               
        # Draw base background
        base_color = self.base_color[1] if isinstance(self.base_color, tuple) else self.base_color
        self.progress_canvas.configure(bg=base_color)
        
        # Draw progress bar only if show_progress is True and there's progress
        if self.show_progress and self.progress > 0:
            progress_width = int((width - button_area_width) * self.progress)
            self.progress_canvas.create_rectangle(
                0, 0, progress_width, height,
                fill=self.progress_color,
                outline=""
            )
        
        # Calculate font size based on screen scaling
        font_size = self._get_scaled_font_size()
                
        # Draw text overlays
        if self.name_text:
            self.progress_canvas.create_text(
                10, height // 2,
                text=self.name_text,
                anchor="w",
                fill="lightgray",
                font=("", font_size)
            )
        
        if self.status_text:
            # Position status text to avoid button overlap
            status_x = width - button_area_width - self._apply_widget_scaling(5)
            self.progress_canvas.create_text(
                status_x, height // 2,
                text=self.status_text,
                anchor="e",
                fill=self.status_color,
                font=("", font_size)
            )

class SpeakerNamingDialog(ctk.CTkToplevel):
    """Modal dialog shown after diarization to let the user assign names to
    detected speakers and optionally persist their voice signatures."""

    def __init__(self, parent, speakers_data: list):
        """
        Parameters
        ----------
        speakers_data : list of dict
            Each entry has keys:
              'label'        – pyannote label, e.g. 'SPEAKER_01'
              'short_label'  – display label, e.g. 'S01'
              'matched_name' – name from DB, or None
              'similarity'   – cosine similarity (0–1)
              'embedding'    – list of floats, or None
        """
        super().__init__(parent)
        self.title(t('speaker_naming_title'))
        self.resizable(False, False)
        self.result = {}          # {label: name} – populated on OK
        self._save_results = {}   # {label: (name, embedding)} to save
        self._entries = {}        # {label: tk.StringVar}
        self._save_vars = {}      # {label: tk.BooleanVar}
        self._speakers_data = speakers_data

        self._build_ui()
        self.grab_set()
        self.focus_force()
        # Center relative to parent
        self.update_idletasks()
        pw = parent.winfo_x() + parent.winfo_width() // 2
        ph = parent.winfo_y() + parent.winfo_height() // 2
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"+{pw - w // 2}+{ph - h // 2}")

    def _build_ui(self):
        pad = {'padx': 20, 'pady': (5, 2)}

        ctk.CTkLabel(
            self,
            text=t('speaker_naming_title'),
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(pady=(15, 3), padx=20)

        ctk.CTkLabel(
            self,
            text=t('speaker_naming_hint'),
            wraplength=440,
            justify='left'
        ).pack(pady=(0, 8), padx=20)

        # Scrollable frame for speakers
        visible = min(len(self._speakers_data), 6)
        scroll_frame = ctk.CTkScrollableFrame(self, width=460,
                                               height=visible * 56 + 10)
        scroll_frame.pack(padx=15, pady=5, fill='x')

        for spk in self._speakers_data:
            row = ctk.CTkFrame(scroll_frame)
            row.pack(fill='x', padx=5, pady=3)

            ctk.CTkLabel(row, text=spk['short_label'],
                         width=45,
                         font=ctk.CTkFont(weight="bold")).pack(
                             side='left', padx=(8, 4))

            entry_var = tk.StringVar(value=spk.get('matched_name') or '')
            entry = ctk.CTkEntry(row, textvariable=entry_var, width=185,
                                  placeholder_text=t('speaker_name_placeholder'))
            entry.pack(side='left', padx=4)
            self._entries[spk['label']] = entry_var

            # Confidence badge
            matched = spk.get('matched_name')
            sim = spk.get('similarity', 0.0)
            if matched and sim >= speaker_db.SIMILARITY_THRESHOLD:
                badge_text = f"{int(sim * 100)}%"
                badge_color = "#4CAF50"
            elif sim > 0.55:
                badge_text = f"~{int(sim * 100)}%"
                badge_color = "#FF9800"
            else:
                badge_text = t('speaker_new_badge')
                badge_color = "#78909C"
            ctk.CTkLabel(row, text=badge_text, text_color=badge_color,
                         width=46).pack(side='left', padx=2)

            # Save checkbox (only useful when we have an embedding)
            save_var = tk.BooleanVar(
                value=bool(matched) and spk.get('embedding') is not None
            )
            if spk.get('embedding') is not None:
                ctk.CTkCheckBox(row, text=t('speaker_save_checkbox'),
                                variable=save_var, width=65).pack(
                                    side='left', padx=(4, 8))
            self._save_vars[spk['label']] = save_var

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=12)
        ctk.CTkButton(btn_frame, text=t('btn_ok'),
                      command=self._on_ok, width=110).pack(side='left', padx=10)
        ctk.CTkButton(btn_frame, text=t('btn_skip'),
                      command=self._on_skip, width=110).pack(side='left', padx=10)

    def _on_ok(self):
        for spk in self._speakers_data:
            label = spk['label']
            name = self._entries[label].get().strip()
            if name:
                self.result[label] = name
                save_var = self._save_vars.get(label)
                if save_var and save_var.get() and spk.get('embedding'):
                    try:
                        speaker_db.save_speaker(name, spk['embedding'])
                    except Exception:
                        pass
        self.destroy()

    def _on_skip(self):
        self.result = {}
        self.destroy()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        _init_app_state(self)

        # configure window
        self.title('noScribe - ' + t('app_header'))
        if platform.system() in ("Darwin", "Linux"):
            self.geometry(f"{1100}x{765}")
        else:
            self.geometry(f"{1100}x{690}")
        if platform.system() in ("Darwin", "Windows"):
            self.iconbitmap(os.path.join(app_dir, 'noScribeLogo.ico'))
        if platform.system() == "Linux":
            if hasattr(sys, "_MEIPASS"):
                self.iconphoto(True, tk.PhotoImage(file=os.path.join(sys._MEIPASS, "noScribeLogo.png")))
            else:
                self.iconphoto(True, tk.PhotoImage(file='noScribeLogo.png'))

        # header
        self.frame_header = ctk.CTkFrame(self, height=100)
        self.frame_header.pack(padx=0, pady=0, anchor='nw', fill='x')

        self.frame_header_logo = ctk.CTkFrame(self.frame_header, fg_color='transparent')
        self.frame_header_logo.pack(anchor='w', side='left')

        # logo
        self.logo_label = ctk.CTkLabel(self.frame_header_logo, text="noScribe", font=ctk.CTkFont(size=42, weight="bold"))
        self.logo_label.pack(padx=20, pady=[40, 0], anchor='w')

        # sub header
        self.header_label = ctk.CTkLabel(self.frame_header_logo, text=t('app_header'), font=ctk.CTkFont(size=16, weight="bold"))
        self.header_label.pack(padx=20, pady=[0, 20], anchor='w')
        # graphic
        self.header_graphic = ctk.CTkImage(dark_image=Image.open(os.path.join(app_dir, 'graphic_sw.png')), size=(926,119))
        self.header_graphic_label = ctk.CTkLabel(self.frame_header, image=self.header_graphic, text='')
        self.header_graphic_label.pack(anchor='ne', side='right', padx=[30,30])

        # main window
        self.frame_main = ctk.CTkFrame(self)
        self.frame_main.pack(padx=0, pady=0, anchor='nw', expand=True, fill='both')

        # create sidebar frame for options
        self.sidebar_frame = ctk.CTkFrame(self.frame_main, width=300, corner_radius=0, fg_color='transparent')
        self.sidebar_frame.pack(padx=0, pady=0, fill='y', expand=False, side='left')

        # create options scrollable frame
        self.scrollable_options = ctk.CTkScrollableFrame(self.sidebar_frame, width=300, corner_radius=0, fg_color='transparent')
        self.scrollable_options.pack(padx=0, pady=0, anchor='w', fill='both', expand=True)
        self.bind('<Configure>', self.on_resize) # Bind the configure event of options_frame to a check_scrollbar requirement function
        
        # input audio file
        self.label_audio_file = ctk.CTkLabel(self.scrollable_options, text=t('label_audio_file'))
        self.label_audio_file.pack(padx=20, pady=[20,0], anchor='w')

        self.frame_audio_file = ctk.CTkFrame(self.scrollable_options, width=260, height=33, corner_radius=8, border_width=2)
        self.frame_audio_file.pack(padx=20, pady=[0,10], anchor='w')

        self.button_audio_file_name = ctk.CTkButton(self.frame_audio_file, width=200, corner_radius=8, bg_color='transparent', 
                                                    fg_color='transparent', hover_color=self.frame_audio_file._bg_color, 
                                                    border_width=0, anchor='w',  
                                                    text=t('label_audio_file_name'), command=self.button_audio_file_event)
        self.button_audio_file_name.place(x=3, y=3)

        self.button_audio_file = ctk.CTkButton(self.frame_audio_file, width=45, height=29, text='📂', command=self.button_audio_file_event)
        self.button_audio_file.place(x=213, y=2)

        # input transcript file name
        self.label_transcript_file = ctk.CTkLabel(self.scrollable_options, text=t('label_transcript_file'))
        self.label_transcript_file.pack(padx=20, pady=[10,0], anchor='w')

        self.frame_transcript_file = ctk.CTkFrame(self.scrollable_options, width=260, height=33, corner_radius=8, border_width=2)
        self.frame_transcript_file.pack(padx=20, pady=[0,10], anchor='w')

        self.button_transcript_file_name = ctk.CTkButton(self.frame_transcript_file, width=200, corner_radius=8, bg_color='transparent', 
                                                    fg_color='transparent', hover_color=self.frame_transcript_file._bg_color, 
                                                    border_width=0, anchor='w',  
                                                    text=t('label_transcript_file_name'), command=self.button_transcript_file_event)
        self.button_transcript_file_name.place(x=3, y=3)

        self.button_transcript_file = ctk.CTkButton(self.frame_transcript_file, width=45, height=29, text='📂', command=self.button_transcript_file_event)
        self.button_transcript_file.place(x=213, y=2)

        # Options grid
        self.frame_options = ctk.CTkFrame(self.scrollable_options, width=250, fg_color='transparent')
        self.frame_options.pack_propagate(False)
        self.frame_options.pack(padx=20, pady=10, anchor='w', fill='x')

        # self.frame_options.grid_configure .resizable(width=False, height=True)
        self.frame_options.grid_columnconfigure(0, weight=1, minsize=0)
        self.frame_options.grid_columnconfigure(1, weight=0)

        # Start/stop
        self.label_start = ctk.CTkLabel(self.frame_options, text=t('label_start'))
        self.label_start.grid(column=0, row=0, sticky='w', pady=[0,5])

        self.entry_start = TimeEntry(self.frame_options, width=100)
        self.entry_start.grid(column='1', row='0', sticky='e', pady=[0,5])
        self.entry_start.insert(0, '00:00:00')

        self.label_stop = ctk.CTkLabel(self.frame_options, text=t('label_stop'))
        self.label_stop.grid(column=0, row=1, sticky='w', pady=[5,10])

        self.entry_stop = TimeEntry(self.frame_options, width=100)
        self.entry_stop.grid(column='1', row='1', sticky='e', pady=[5,10])

        # language
        self.label_language = ctk.CTkLabel(self.frame_options, text=t('label_language'))
        self.label_language.grid(column=0, row=2, sticky='w', pady=5)

        self.option_menu_language = ctk.CTkOptionMenu(self.frame_options, width=100, values=list(languages.keys()), dynamic_resizing=False)
        self.option_menu_language.grid(column=1, row=2, sticky='e', pady=5)
        last_language = get_config('last_language', 'auto')
        if last_language in languages.keys():
            self.option_menu_language.set(last_language)
        else:
            self.option_menu_language.set('Auto')
        
        # Whisper Model Selection   
        class CustomCTkOptionMenu(ctk.CTkOptionMenu):
            # Custom version that reads available models on drop down
            def __init__(self, noScribe_parent, master, width = 140, height = 28, corner_radius = None, bg_color = "transparent", fg_color = None, button_color = None, button_hover_color = None, text_color = None, text_color_disabled = None, dropdown_fg_color = None, dropdown_hover_color = None, dropdown_text_color = None, font = None, dropdown_font = None, values = None, variable = None, state = tk.NORMAL, hover = True, command = None, dynamic_resizing = True, anchor = "w", **kwargs):
                super().__init__(master, width, height, corner_radius, bg_color, fg_color, button_color, button_hover_color, text_color, text_color_disabled, dropdown_fg_color, dropdown_hover_color, dropdown_text_color, font, dropdown_font, values, variable, state, hover, command, dynamic_resizing, anchor, **kwargs)
                self.noScribe_parent = noScribe_parent
                self.old_value = ''

            def _clicked(self, event=0):
                self.old_value = self.get()
                self._values = self.noScribe_parent.get_whisper_models()
                self._values.append('--------------------')
                self._values.append(t('label_add_custom_models'))
                self._dropdown_menu.configure(values=self._values)
                super()._clicked(event)
                
            def _dropdown_callback(self, value: str):
                if value == self._values[-2]:  # divider
                    return
                if value == self._values[-1]:  # Add custom model
                    # show custom model folder
                    path = self.noScribe_parent.user_models_dir
                    try:
                        os_type = platform.system()
                        if os_type == "Windows":
                            os.startfile(path)
                        elif os_type == "Darwin":
                            run(["open", path])
                        elif os_type == "Linux":
                            run(["xdg-open", path])
                        else:
                            raise OSError(f"Unsupported operating system: {os_type}")
                    except Exception as e:
                        self.noScribe_parent.logn(f"Failed to open folder: {e}")
                else:
                    super()._dropdown_callback(value)
        
        self.label_whisper_model = ctk.CTkLabel(self.frame_options, text=t('label_whisper_model'))
        self.label_whisper_model.grid(column=0, row=3, sticky='w', pady=5)

        models = self.get_whisper_models()
        self.option_menu_whisper_model = CustomCTkOptionMenu(self, 
                                                       self.frame_options, 
                                                       width=100,
                                                       values=models,
                                                       dynamic_resizing=False)
        self.option_menu_whisper_model.grid(column=1, row=3, sticky='e', pady=5)
        last_whisper_model = get_config('last_whisper_model', 'precise')
        if last_whisper_model in models:
            self.option_menu_whisper_model.set(last_whisper_model)
        elif len(models) > 0:
            self.option_menu_whisper_model.set(models[0])

        # Mark pauses
        self.label_pause = ctk.CTkLabel(self.frame_options, text=t('label_pause'))
        self.label_pause.grid(column=0, row=4, sticky='w', pady=5)

        self.option_menu_pause = ctk.CTkOptionMenu(self.frame_options, width=100, values=['none', '1sec+', '2sec+', '3sec+'])
        self.option_menu_pause.grid(column=1, row=4, sticky='e', pady=5)
        self.option_menu_pause.set(get_config('last_pause', '1sec+'))

        # Speaker Detection (Diarization)
        self.label_speaker = ctk.CTkLabel(self.frame_options, text=t('label_speaker'))
        self.label_speaker.grid(column=0, row=5, sticky='w', pady=5)

        self.option_menu_speaker = ctk.CTkOptionMenu(self.frame_options, width=100, values=['none', 'auto', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        self.option_menu_speaker.grid(column=1, row=5, sticky='e', pady=5)
        self.option_menu_speaker.set(get_config('last_speaker', 'auto'))

        # Overlapping Speech (Diarization)
        self.label_overlapping = ctk.CTkLabel(self.frame_options, text=t('label_overlapping'))
        self.label_overlapping.grid(column=0, row=6, sticky='w', pady=5)

        self.check_box_overlapping = ctk.CTkCheckBox(self.frame_options, text = '')
        self.check_box_overlapping.grid(column=1, row=6, sticky='e', pady=5)
        overlapping = config.get('last_overlapping', True)
        if overlapping:
            self.check_box_overlapping.select()
        else:
            self.check_box_overlapping.deselect()
            
        # Disfluencies
        self.label_disfluencies = ctk.CTkLabel(self.frame_options, text=t('label_disfluencies'))
        self.label_disfluencies.grid(column=0, row=7, sticky='w', pady=5)

        self.check_box_disfluencies = ctk.CTkCheckBox(self.frame_options, text = '')
        self.check_box_disfluencies.grid(column=1, row=7, sticky='e', pady=5)
        check_box_disfluencies = config.get('last_disfluencies', True)
        if check_box_disfluencies:
            self.check_box_disfluencies.select()
        else:
            self.check_box_disfluencies.deselect()

        # Timestamps in text
        self.label_timestamps = ctk.CTkLabel(self.frame_options, text=t('label_timestamps'))
        self.label_timestamps.grid(column=0, row=8, sticky='w', pady=5)

        self.check_box_timestamps = ctk.CTkCheckBox(self.frame_options, text = '')
        self.check_box_timestamps.grid(column=1, row=8, sticky='e', pady=5)
        check_box_timestamps = config.get('last_timestamps', False)
        if check_box_timestamps:
            self.check_box_timestamps.select()
        else:
            self.check_box_timestamps.deselect()
        
        # Start control: single CTkOptionMenu styled like a button
        # Create a container so we can show/hide as one control
        self.start_button_container = ctk.CTkFrame(self.sidebar_frame, fg_color='transparent')
        self.start_button_container.pack(padx=[30,30], pady=[20,30], expand=False, fill='x', anchor='sw')

        class StartActionOptionMenu(ctk.CTkOptionMenu):
            """A full-width option menu that looks like a button.
            - Left-click (main area) runs Start immediately.
            - Clicking the arrow opens a dropdown with 'Send to queue'.
            """
            def __init__(self, noScribe_parent, master, **kwargs):
                # Style to match CTkButton
                btn_theme = ctk.ThemeManager.theme.get('CTkButton', {})
                kwargs.setdefault('height', 42)
                kwargs.setdefault('dynamic_resizing', False)
                kwargs.setdefault('anchor', 'center')
                kwargs.setdefault('fg_color', btn_theme.get('fg_color'))
                kwargs.setdefault('button_color', btn_theme.get('hover_color'))
                kwargs.setdefault('button_hover_color', btn_theme.get('hover_color'))

                super().__init__(master, values=['Start'], **kwargs)
                self.noScribe_parent = noScribe_parent
                try:
                    self.set(t('start_button'))
                except Exception:
                    self.set('Start')
                # Bind click on the text label to run Start immediately
                try:
                    self._text_label.bind("<Button-1>", self._on_text_label_click)
                except Exception:
                    pass

            def _clicked(self, event=None):
                # Open dropdown with the single queue action for non-text-label clicks
                try:
                    self._values = [t('send_queue'), t('start_queue')]
                    self._dropdown_menu.configure(values=self._values)
                except Exception:
                    pass
                super()._clicked(event)

            def _dropdown_callback(self, value: str):
                if value == t('send_queue'):
                    try:
                        self.noScribe_parent.create_job(enqueue=True)
                    finally:
                        try:
                            self.set(t('start_button'))
                        except Exception:
                            self.set('Start')
                elif value == t('start_queue'):
                    try:
                        self.noScribe_parent.create_job(enqueue=False)
                    finally:
                        try:
                            self.set(t('start_button'))
                        except Exception:
                            self.set('Start')
                else:
                    super()._dropdown_callback(value)

            def _on_text_label_click(self, event):
                try:
                    self.noScribe_parent.create_job(enqueue=False)
                except Exception:
                    pass
                return "break"

        self.start_action_menu = StartActionOptionMenu(self, self.start_button_container)
        self.start_action_menu.pack(padx=[0,0], fill='x', expand=True)
        
        # create queue view and log textbox
        self.frame_right = ctk.CTkFrame(self.frame_main, corner_radius=0, fg_color='transparent')
        self.frame_right.pack(padx=0, pady=0, fill='both', expand=True, side='top')
        
        self.tabview = ctk.CTkTabview(self.frame_right, anchor="nw", border_width=0, fg_color='transparent', corner_radius=0)
        self.tabview.pack(padx=[10,30], pady=[0,30], fill='both', expand=True, side='top')
        self.tab_log = self.tabview.add(t("tab_log")) 
        self.tab_queue = self.tabview.add(t("tab_queue")) 
        self.tabview.set(t("tab_log"))  # set currently visible tab

        self.log_frame = ctk.CTkFrame(self.tab_log, fg_color='transparent', border_width=1, corner_radius=0)
        self.log_frame.pack(padx=0, pady=0, expand=True, fill='both')
        self.log_textbox = ctk.CTkTextbox(self.log_frame, wrap='word', state="disabled", font=("",16), text_color="lightgray", bg_color='transparent', fg_color='transparent')
        self.log_textbox.tag_config('highlight', foreground='darkorange')
        self.log_textbox.tag_config('error', foreground='yellow')
        self.log_textbox.pack(padx=5, pady=5, expand=True, fill='both')
        self.log_len = 0
        
        self.log_progress_frame = ctk.CTkFrame(self.log_frame, fg_color='transparent')
        self.log_progress_frame.pack(padx=10, pady=10, fill='x', expand=False, anchor='center') 
        self.log_edit_btn = ctk.CTkButton(
            self.log_progress_frame,
            text=t('editor_button'),
            width=100,
            fg_color=self.log_textbox._scrollbar_button_color,            
            command=lambda: self.launch_editor()
        )
        self.log_edit_btn.pack(side='right', padx=(0, 0), pady=0)
        self.log_stop_btn = ctk.CTkButton(
            self.log_progress_frame,
            text=t('stop_button'),
            fg_color='darkred',
            hover_color='darkred',
            width=100,
            state=ctk.DISABLED,
            command=lambda: self.on_queue_stop()
        )
        self.log_stop_btn.pack(side='right', padx=(0, 10), pady=0)

        self.log_progress_bar = ctk.CTkProgressBar(self.log_progress_frame, mode='determinate', fg_color="gray17")
        self.log_progress_bar.set(0)
        
        self.hyperlink = HyperlinkManager(self.log_textbox._textbox)

        # Queue table
        self.queue_frame = ctk.CTkFrame(self.tab_queue, fg_color='transparent', border_width=1, corner_radius=0)
        self.queue_frame.pack(padx=0, pady=0, expand=True, fill='both')        
        self.queue_frame = ctk.CTkFrame(self.queue_frame, fg_color='transparent')
        self.queue_frame.pack(padx=5, pady=5, fill='both', expand=True)
                
        # Scrollable frame for queue entries
        self.queue_scrollable = ctk.CTkScrollableFrame(self.queue_frame, bg_color='transparent', fg_color='transparent')
        self.queue_scrollable.pack(fill='both', expand=True, padx=0, pady=(0, 0))

        # Controls row at the bottom of the queue tab
        self.queue_controls_frame = ctk.CTkFrame(self.queue_frame, fg_color='transparent')
        self.queue_controls_frame.pack(fill='x', side='bottom', padx=0, pady=(0, 0))

        self.queue_edit_btn = ctk.CTkButton(
            self.queue_controls_frame,
            text=t('editor_button'),
            width=100,
            fg_color=self.log_textbox._scrollbar_button_color,            
            command=lambda: self.launch_editor()
        )
        self.queue_edit_btn.pack(side='right', padx=(0, 5), pady=5)

        self.queue_stop_btn = ctk.CTkButton(
            self.queue_controls_frame,
            text=t('stop_button'),
            fg_color='darkred',
            hover_color='darkred',
            width=100,
            command=lambda: self.on_queue_stop()
        )
        self.queue_stop_btn.pack(side='right', padx=(0, 10), pady=5)

        self.queue_run_btn = ctk.CTkButton(
            self.queue_controls_frame,
            text=t('queue_run_button'),
            width=100,
            command=lambda: self.on_queue_run()
        )
        self.queue_run_btn.pack(side='right', padx=(0, 10), pady=5)

        # Mapping for diff-based queue rows (job_key -> widgets)
        self.queue_row_widgets = {}

        self.update_queue_table()

        self.update_scrollbar_visibility()
        
        self.log(translation_error, 'error') # will be empty if no error occurred        

        self.logn(t('welcome_message'), 'highlight')
        self.log(t('welcome_credits', v=app_version, y=app_year))
        self.logn('https://github.com/kaixxx/noScribe', link='https://github.com/kaixxx/noScribe#readme')
        self.logn(t('welcome_instructions'))
        
        # check for new releases
        if get_config('check_for_update', 'True') == 'True':
            try:
                latest_release = json.loads(urllib.request.urlopen(
                    urllib.request.Request('https://api.github.com/repos/kaixxx/noScribe/releases/latest',
                    headers={'Accept': 'application/vnd.github.v3+json'},),
                    timeout=2).read())
                latest_release_version = str(latest_release['tag_name']).lstrip('v')
                if version_higher(latest_release_version, app_version, subversion_level=1) == 1:
                    # Only major release changes like 0.6 ->_0.7 (subversion_level 1) are indicated in the
                    # UI, not smaller subversion like 0.7.3 -> 0.7.4
                    self.logn(t('new_release', v=latest_release_version), 'highlight')
                    self.logn(str(latest_release['body'])) # release info
                    self.log(t('new_release_download'))
                    self.logn(str(latest_release['html_url']), link=str(latest_release['html_url']))
                    self.logn()
            except:
                pass
            
    # Events and Methods

    def get_whisper_models(self):
        self.whisper_model_paths = {}
        
        def collect_models(dir):        
            for entry in os.listdir(dir):
                entry_path = os.path.join(dir, entry)
                if os.path.isdir(entry_path):
                    if entry in self.whisper_model_paths:
                        self.logn(t('err_invalid_model', entry), 'error')
                    else:
                        self.whisper_model_paths[entry]=entry_path 
       
        # collect system models:
        collect_models(os.path.join(app_dir, 'models'))
        
        # collect user defined models:        
        collect_models(self.user_models_dir)

        return list(self.whisper_model_paths.keys())
    
    def on_whisper_model_selected(self, value):
        print(self.option_menu_whisper_model.old_value)
        print(value)
        
    def on_resize(self, event):
        self.update_scrollbar_visibility()

    def update_scrollbar_visibility(self):
        # Get the size of the scroll region and current canvas size
        canvas = self.scrollable_options._parent_canvas  
        scroll_region_height = canvas.bbox("all")[3]
        canvas_height = canvas.winfo_height()        
        
        scrollbar = self.scrollable_options._scrollbar

        if scroll_region_height > canvas_height:
            scrollbar.grid()
        else:
            scrollbar.grid_remove()  # Hide the scrollbar if not needed    
                        
    def update_queue_table(self):
        """Update the queue table by diffing: update existing rows, add new ones, remove missing."""
        if getattr(self, '_headless', False):
            return
        current_keys = []
        for i in range(len(self.queue.jobs)):
            job = self.queue.jobs[i]
            job_key = id(job)
            current_keys.append(job_key)

            # Compute display values
            audio_name = os.path.basename(job.audio_file) if job.audio_file else "No file"
            status_color = "lightgray"
            job_tooltip = ''
            if job.status == JobStatus.WAITING:
                status_color = "gray"
                job_tooltip = t('job_tt_waiting')
            elif job.status in [JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION, JobStatus.TRANSCRIPTION]:
                status_color = "orange"
                audio_name = '\u23F5 ' + audio_name
                job_tooltip = t('job_tt_running')
            elif job.status == JobStatus.CANCELING:
                status_color = "yellow"
                audio_name = '\u23F5 ' + audio_name
                job_tooltip = t('job_tt_canceling')
            elif job.status == JobStatus.CANCELED:
                status_color = "yellow"
                job_tooltip = t('job_tt_canceled')
            elif job.status == JobStatus.FINISHED:
                status_color = "lightgreen"
                job_tooltip = t('job_tt_finished')
            elif job.status == JobStatus.ERROR:
                status_color = "yellow"
                msg = job.error_message if job.error_message else ''
                job_tooltip = t('job_tt_error', error_msg=msg)

            # Append a real, concise summary of the job's options
            try:
                job_tooltip += '\n\n' + job.format_summary()
            except Exception:
                pass

            status_text = t(str(job.status.value))
            
            btn_color = ctk.ThemeManager.theme['CTkScrollbar']['button_color']

            if hasattr(self, 'queue_row_widgets') and job_key in self.queue_row_widgets:
                # Update existing row
                row = self.queue_row_widgets[job_key]
                # Update text directly on the JobEntryFrame canvas
                row['frame'].set_name_text(audio_name)
                row['frame'].set_status_text(status_text, status_color)
                
                # Update progress bar visibility based on job status
                is_processing = job.status in [JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION, JobStatus.TRANSCRIPTION]
                if is_processing:
                    row['frame'].set_progress(job.progress, show_progress=True)
                else:
                    row['frame'].set_progress(0.0, show_progress=False)

                # Add repeat button for status ERROR and CANCELED only
                try:
                    if job.status in [JobStatus.ERROR, JobStatus.CANCELED]:
                        if 'repeat_btn' not in row or row['repeat_btn'] is None:
                            repeat_btn = ctk.CTkButton(
                                row['frame'],
                                text='⟲',
                                width=24,
                                height=20,
                                fg_color=btn_color,
                                hover_color='darkred',
                                command=lambda j=job: self._on_queue_row_repeat(j)
                            )
                            repeat_btn.pack(side='right', padx=(0, 4), pady=5)
                            row['repeat_btn'] = repeat_btn
                            row['repeat_tt'] = CTkToolTip(repeat_btn, text=t('queue_tt_repeat_job')) 
                        else:
                            if not row['repeat_btn'].winfo_ismapped():
                                row['repeat_btn'].pack(side='right', padx=(0, 4), pady=2)
                            row['repeat_btn'].configure(state=ctk.NORMAL, command=lambda j=job: self._on_queue_row_repeat(j))
                    else:
                        # hide the repeat button if it exists for other states
                        if 'repeat_btn' in row and row['repeat_btn'] is not None:
                            if row['repeat_btn'].winfo_ismapped():
                                row['repeat_btn'].pack_forget()
                except Exception:
                    pass

                # Cancel button for running jobs
                if 'cancel_btn' in row and row['cancel_btn'] is not None:
                    try:
                        row['cancel_btn'].configure(command=lambda j=job: self._on_queue_row_action(j))
                        # Color: red if running, gray otherwise
                        if job.status in [JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION, JobStatus.TRANSCRIPTION]:
                            row['cancel_btn'].configure(fg_color='darkred', hover_color='darkred')
                        else:
                            row['cancel_btn'].configure(fg_color=btn_color, hover_color='darkred')
                        # Make sure it is visible
                        if not row['cancel_btn'].winfo_ismapped():
                            row['cancel_btn'].pack(side='right', padx=(0, 6), pady=2)
                        # Update tooltip on the X button to reflect current status
                        if job.status == JobStatus.WAITING:
                            cancel_tt_text = t('queue_tt_remove_waiting')
                        elif job.status in [JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION, JobStatus.TRANSCRIPTION, JobStatus.CANCELING]:
                            cancel_tt_text = t('queue_tt_remove_entry')
                        else:
                            cancel_tt_text = t('queue_tt_remove_entry')
                        if 'cancel_tt' in row and row['cancel_tt'] is not None:
                            try:
                                row['cancel_tt'].set_text(cancel_tt_text)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Add "open partial" button for failed jobs with partial HTML transcript
                try:
                    if job.status in [JobStatus.ERROR, JobStatus.CANCELED] and getattr(job, 'has_partial_transcript', False):
                        if 'partial_btn' not in row or row['partial_btn'] is None:
                            partial_btn = ctk.CTkButton(
                                row['frame'],
                                text='✔',
                                width=24,
                                height=20,
                                fg_color=btn_color,
                                hover_color='darkred',
                                command=lambda j=job: self._on_queue_row_open_partial(j)
                            )
                            partial_btn.pack(side='right', padx=(0, 4), pady=5)
                            row['partial_btn'] = partial_btn
                            row['partial_tt'] = CTkToolTip(partial_btn, text=t('queue_tt_open_partial_job')) 
                        else:
                            if not row['partial_btn'].winfo_ismapped():
                                row['partial_btn'].pack(side='right', padx=(0, 4), pady=2)
                            row['partial_btn'].configure(state=ctk.NORMAL, command=lambda j=job: self._on_queue_row_open_partial(j))
                    else:
                        # hide the partial button otherwise
                        if 'partial_btn' in row and row['partial_btn'] is not None and row['partial_btn'].winfo_ismapped():
                            row['partial_btn'].pack_forget()
                except Exception:
                    pass

                # Add edit button for finished jobs
                try:
                    if job.status == JobStatus.FINISHED:
                        if 'edit_btn' not in row or row['edit_btn'] is None:
                            edit_btn = ctk.CTkButton(
                                row['frame'],
                                text='✔',
                                width=24,
                                height=20,
                                fg_color=btn_color,
                                hover_color='darkred',
                                command=lambda j=job: self._on_queue_row_edit(j)
                            )
                            edit_btn.pack(side='right', padx=(0, 4), pady=5)
                            row['edit_btn'] = edit_btn
                            row['edit_tt'] = CTkToolTip(edit_btn, text=t('queue_tt_edit_job')) 
                        else:
                            if not row['edit_btn'].winfo_ismapped():
                                row['edit_btn'].pack(side='right', padx=(0, 4), pady=2)
                            row['edit_btn'].configure(state=ctk.NORMAL, command=lambda j=job: self._on_queue_row_edit(j))
                    else:
                        # hide the edit button if it exists for other states
                        if 'edit_btn' in row and row['edit_btn'] is not None:
                            if row['edit_btn'].winfo_ismapped():
                                row['edit_btn'].pack_forget()
                except Exception:
                    pass

                row['status'] = job.status
                row['tooltip_text'] = job_tooltip
                # Update tooltip messages if available
                if 'tooltips' in row:
                    for tt in row['tooltips']:
                        tt.set_text(job_tooltip)
            else:
                # Create new row with progress bar background
                fg_color = ctk.ThemeManager.theme['CTkSegmentedButton']['unselected_color'][1]
                entry_frame = JobEntryFrame(self.queue_scrollable, progress=job.progress, progress_color=None, fg_color=fg_color)
                entry_frame.pack(fill='x', padx=(0, 5), pady=2)
                
                # Set the text directly on the JobEntryFrame canvas
                entry_frame.set_name_text(audio_name)
                entry_frame.set_status_text(status_text, status_color)
                
                # Set progress bar visibility based on job status
                is_processing = job.status in [JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION, JobStatus.TRANSCRIPTION]
                if is_processing:
                    entry_frame.set_progress(job.progress, show_progress=True)
                else:
                    entry_frame.set_progress(0.0, show_progress=False)

                # Add small action buttons to job row
                # X Button
                cancel_btn = ctk.CTkButton(
                    entry_frame,
                    text='X',
                    width=24,
                    height=20,
                    fg_color=('darkred' if job.status in [JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION, JobStatus.TRANSCRIPTION] else btn_color),
                    hover_color=('darkred'),
                    command=lambda j=job: self._on_queue_row_action(j)
                )
                cancel_btn.pack(side='right', padx=(0, 6), pady=5)   
                # Tooltip for X button per status
                if job.status == JobStatus.WAITING:
                    cancel_tt_text = t('queue_tt_remove_waiting')
                elif job.status in [JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION, JobStatus.TRANSCRIPTION]:
                    cancel_tt_text = t('queue_tt_cancel_running')
                else:
                    cancel_tt_text = t('queue_tt_remove_entry')
                cancel_tt = CTkToolTip(cancel_btn, text=cancel_tt_text)
                
                # Repeat button (job status canceled or error only)                               
                repeat_btn = None
                repeat_tt = None
                if job.status in [JobStatus.ERROR, JobStatus.CANCELED]:
                    repeat_btn = ctk.CTkButton(
                        entry_frame,
                        text='⟲',
                        width=24,
                        height=20,
                        fg_color=btn_color,
                        hover_color=('darkred'),
                        command=lambda j=job: self._on_queue_row_repeat(j)
                    )
                    repeat_btn.pack(side='right', padx=(0, 4), pady=5)
                    repeat_tt = CTkToolTip(repeat_btn, text=t('queue_tt_repeat_job'))

                # Open partial button (failed job with partial transcript, HTML only)
                partial_btn = None
                partial_tt = None
                if job.status in [JobStatus.ERROR, JobStatus.CANCELED] and getattr(job, 'has_partial_transcript', False):
                    partial_btn = ctk.CTkButton(
                        entry_frame,
                        text='✔',
                        width=24,
                        height=20,
                        fg_color=btn_color,
                        hover_color='darkred',
                        command=lambda j=job: self._on_queue_row_open_partial(j)
                    )
                    partial_btn.pack(side='right', padx=(0, 4), pady=5)
                    partial_tt = CTkToolTip(partial_btn, text=t('queue_tt_open_partial_job'))

                # Edit button (finished jobs only)
                edit_btn = None
                edit_tt = None
                if job.status == JobStatus.FINISHED:
                    edit_btn = ctk.CTkButton(
                        entry_frame,
                        text='✔',
                        width=24,
                        height=20,
                        fg_color=btn_color,
                        hover_color='darkred',
                        command=lambda j=job: self._on_queue_row_edit(j)
                    )
                    edit_btn.pack(side='right', padx=(0, 4), pady=5)
                    edit_tt = CTkToolTip(edit_btn, text=t('queue_tt_edit_job'))                 

                # Row tooltip (create once per row)
                tt_frame = CTkToolTip(entry_frame, text=job_tooltip) #, bg_color='gray')

                if not hasattr(self, 'queue_row_widgets'):
                    self.queue_row_widgets = {}
                self.queue_row_widgets[job_key] = {
                    'frame': entry_frame,
                    'status': job.status,
                    'tooltip_text': job_tooltip,
                    'tooltips': [tt_frame],
                    'cancel_btn': cancel_btn,
                    'cancel_tt': cancel_tt,
                    'repeat_btn': repeat_btn,
                    'repeat_tt': repeat_tt,
                    'partial_btn': partial_btn,
                    'partial_tt': partial_tt,
                    'edit_btn': edit_btn,
                    'edit_tt': edit_tt
                }

        # Remove rows no longer present
        if hasattr(self, 'queue_row_widgets'):
            to_remove = [key for key in list(self.queue_row_widgets.keys()) if key not in current_keys]
            for key in to_remove:
                row = self.queue_row_widgets.pop(key)
                if row['frame'].winfo_exists():
                    row['frame'].destroy()
                    
        # Update queue tab title
        new_name = f'{t("tab_queue")} ({len(self.queue.jobs) - len(self.queue.get_waiting_jobs()) - len(self.queue.get_running_jobs())}/{len(self.queue.jobs)})'
        old_name = self.tabview._name_list[1]
        if new_name != old_name:
            self.tabview.rename(old_name, new_name)
            if self.tabview.get() == old_name:
                self.tabview.set(new_name)
        # Update controls state
        try:
            self.update_queue_controls()
        except Exception:
            pass

    def update_queue_controls(self):
        """Enable/disable and label the queue control buttons based on state."""
        if getattr(self, '_headless', False):
            return
        try:
            has_running = len(self.queue.get_running_jobs()) > 0
            has_pending = self.queue.has_pending_jobs()

            # Run button: enabled only if there are pending jobs and nothing is running
            self.queue_run_btn.configure(text=t('queue_run_button'))
            if (not has_running) and has_pending:
                self.queue_run_btn.configure(state=ctk.NORMAL)
            else:
                self.queue_run_btn.configure(state=ctk.DISABLED)

            # Stop button: enabled if something is running or pending
            if has_running or has_pending:
                self.queue_stop_btn.configure(state=ctk.NORMAL)
            else:
                self.queue_stop_btn.configure(state=ctk.DISABLED)
        except Exception:
            pass

    def on_queue_run(self):
        """Start processing pending jobs if idle."""
        try:
            has_running = len(self.queue.get_running_jobs()) > 0
            has_pending = self.queue.has_pending_jobs()
            if (not has_running) and has_pending:
                wkr = Thread(target=self.transcription_worker, args=(), daemon=True)
                self._worker_threads.append(wkr)
                wkr.start()
            self.update_queue_controls()
        except Exception:
            pass

    def on_queue_stop(self, ask_before_canceling=True) -> bool:
        """Ask for confirmation, then cancel running job and mark all pending jobs as canceled.
        Returns False if user does not confirm cancelation."""
        try:
            if (ask_before_canceling and
                   (self.queue.is_running() or self.queue.has_pending_jobs()) and 
                   not tk.messagebox.askyesno(title='noScribe', message=t('queue_cancel_all_confirm'))):
                return False
            # Mark waiting jobs as canceled immediately
            for job in self.queue.get_waiting_jobs():
                try:
                    job.set_canceled(t('err_user_cancelation'))
                except Exception:
                    job.set_canceled('Canceled by user')
            # If something is running, reflect canceling state and signal cancel
            for job in self.queue.get_running_jobs():
                if job.status != JobStatus.CANCELING:
                    job.status = JobStatus.CANCELING
            self.cancel = True
            self._cancel_job_only = False
            self.update_queue_table()
        except Exception:
            pass
        return True

    def _on_queue_row_action(self, job: TranscriptionJob):
        """Handle click on the small X button for a job row."""
        try:
            if job.status == JobStatus.WAITING:
                # Confirm deletion of waiting job
                if tk.messagebox.askyesno(title='noScribe', message=t('queue_remove_waiting')):
                    try:
                        self.queue.jobs.remove(job)
                    except ValueError:
                        pass
                    self.update_queue_table()
            elif job.status in [JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION, JobStatus.TRANSCRIPTION]:
                # Confirm cancel of running job
                if tk.messagebox.askyesno(title='noScribe', message=t('transcription_canceled')):
                    self.logn()
                    self.logn(t('start_canceling'))
                    self.update()
                    # reflect canceling state in queue immediately
                    try:
                        job.status = JobStatus.CANCELING
                        self.update_queue_table()
                    except Exception:
                        pass
                    # Only cancel the current job, not the entire queue
                    self._cancel_job_only = True
                    self.cancel = True
                    # Try to terminate active whisper subprocess if present
                    try:
                        if getattr(self, "_mp_proc", None) is not None and self._mp_proc.is_alive():
                            try:
                                self._mp_proc.terminate()
                            except Exception:
                                pass
                            try:
                                self._mp_proc.join(timeout=0.3)
                            except Exception:
                                pass
                            try:
                                self._mp_proc.close()
                            except Exception:
                                pass
                    finally:
                        self._mp_proc = None
                        self._mp_queue = None
            else:
                # Finished, canceling or error -> remove from list after confirmation
                if tk.messagebox.askyesno(title='noScribe', message=t('queue_remove_entry')):
                    try:
                        self.queue.jobs.remove(job)
                    except ValueError:
                        pass
                    self.update_queue_table()
        except Exception as e:
            # Log any UI handling error silently
            self.logn(f'Queue action error: {e}', 'error')

    def _on_queue_row_repeat(self, job: TranscriptionJob):
        """Repeat a job: set to WAITING if others are running, else start immediately."""
        try:
            if job.status not in [JobStatus.ERROR, JobStatus.CANCELED]:
                return
            # Confirm override if output file conflicts with other jobs (ignore this job itself)
            if not self.queue.confirm_output_override(job.transcript_file, ignore_job=job):
                return
            # reset job timing and messages
            job.error_message = None
            job.error_tb = None
            job.started_at = None
            job.finished_at = None
            job.status = JobStatus.WAITING
            self.update_queue_table()

            has_running = len(self.queue.get_running_jobs()) > 0
            if has_running:
                return

            # no running jobs: start this one immediately
            try:
                start_idx = self.queue.jobs.index(job)
            except ValueError:
                start_idx = None
            if start_idx is not None:
                wkr = Thread(target=self.transcription_worker, kwargs={"start_job_index": start_idx}, daemon=True)
                self._worker_threads.append(wkr)
                wkr.start()
        except Exception as e:
            self.logn(f'Queue repeat error: {e}', 'error')
    
    def _on_queue_row_edit(self, job: TranscriptionJob):
        self.openLink(f'file://{job.transcript_file}')

    def _on_queue_row_open_partial(self, job: TranscriptionJob):
        """Open the partial transcript file (HTML in editor, TXT/VTT via default app)."""
        try:
            path = getattr(job, 'transcript_file', '') or ''
            if not path or not os.path.exists(path):
                try:
                    self.logn(t('err_partial_not_found'), 'error')
                except Exception:
                    pass
                try:
                    tk.messagebox.showerror(title='noScribe', message=t('err_partial_not_found'))
                except Exception:
                    pass
                return
            try:
                self.logn(t('log_open_partial', file=path))
            except Exception:
                pass
            self.openLink(f'file://{path}')
        except Exception:
            pass

    def launch_editor(self, file=''):
        # Launch the editor in a separate process so that in can stay running even if noScribe quits.
        # Source: https://stackoverflow.com/questions/13243807/popen-waiting-for-child-process-even-when-the-immediate-child-has-terminated/13256908#13256908 
        # set system/version dependent "start_new_session" analogs
  
        if file == '':
            # get last finished job (if any)
            jobs = self.queue.get_finished_jobs()
            if len(jobs) > 0:
                file = jobs[-1].transcript_file
            
        if file == '':
            # no file or finished job to open
            if not tk.messagebox.askyesno(title='noScribe', message=t('err_editor_no_file')):
                return

        ext = os.path.splitext(file)[1][1:]
        if file != '' and ext != 'html':
            # wrong format
            file = ''
            if not tk.messagebox.askyesno(title='noScribe', message=t('err_editor_invalid_format')):
                return

        program: str = None
        if platform.system() == 'Windows':
            program = os.path.join(app_dir, 'noScribeEdit', 'noScribeEdit.exe')
        elif platform.system() == "Darwin": # = MAC
            # use local copy in development, installed one if used as an app:
            program = os.path.join(app_dir, 'noScribeEdit', 'noScribeEdit')
            if not os.path.exists(program):
                program = os.path.join(os.sep, 'Applications', 'noScribeEdit.app', 'Contents', 'MacOS', 'noScribeEdit')
        elif platform.system() == "Linux":
            if hasattr(sys, "_MEIPASS"):
                program = os.path.join(sys._MEIPASS, 'noScribeEdit', "noScribeEdit")
            else:
                program = os.path.join(app_dir, 'noScribeEdit', "noScribeEdit.py")
        kwargs = {}
        if platform.system() == 'Windows':
            # from msdn [1]
            CREATE_NEW_PROCESS_GROUP = 0x00000200  # note: could get it from subprocess
            DETACHED_PROCESS = 0x00000008          # 0x8 | 0x200 == 0x208
            kwargs.update(creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)  
        else:  # should work on all POSIX systems, Linux and macOS 
            kwargs.update(start_new_session=True)

        if program is not None and os.path.exists(program):
            popenargs = [program]
            if platform.system() == "Linux" and not hasattr(sys, "_MEIPASS"): # only do this, if you run as python script; Linux python vs. executable needs refinement
                popenargs = [sys.executable, program]
            if file != '':
                popenargs.append(file)
            Popen(popenargs, **kwargs)
        else:
            self.logn(t('err_noScribeEdit_not_found'), 'error')

    def openLink(self, link: str) -> None:
        if link.startswith('file://') and link.endswith('.html'):
            self.launch_editor(link[7:])
        else: 
            webbrowser.open(link)
    
    def log(self, txt: str = '', tags: list = [], where: str = 'both', link: str = '', tb: str = '') -> None:
        """ Log to main window (where can be 'screen', 'file', or 'both') 
        tb = formatted traceback of the error, only logged to file
        """
        
        # Handle screen logging if requested and textbox exists
        if where != 'file': 
            if txt[:-1] != t('welcome_instructions'):
                print(txt, end='')            
            if not getattr(self, '_headless', False) and hasattr(self, 'log_textbox') and self.log_textbox.winfo_exists():
                try:
                    self.log_textbox.configure(state=tk.NORMAL)
                    # To prevent slowing down the UI, limit the content of log_textbox to max 5000 characters
                    if self.log_len > 5000:
                       self.log_textbox.delete("1.0", f"1.0 + {self.log_len - 3000} chars") # keep the last 3000
                       self.log_len = 3000 
                       
                    if link:
                        tags = tags + self.hyperlink.add(partial(self.openLink, link))
                                      
                    self.log_textbox.insert(tk.END, txt, tags)
                    self.log_textbox.yview_moveto(1)  # Scroll to last line
                    self.log_len += len(txt)
                    
                    # Schedule disabling the textbox in the main thread
                    self.log_textbox.after(0, lambda: self.log_textbox.configure(state=tk.DISABLED))
                except Exception as e:
                    # Log screen errors only to file to prevent recursion
                    if where == 'both':
                        self.log(f"Error updating log_textbox: {str(e)}\nOriginal error: {txt}", tags='error', where='file', tb=tb)

        # Handle file logging if requested
        if where != 'screen' and self.log_file and not self.log_file.closed:
            try:
                if tags == 'error':
                    txt = f'ERROR: {txt}'
                if tb != '':
                    txt = f'{txt}\nTraceback:\n{tb}' 
                self.log_file.write(txt)
                self.log_file.flush()
            except Exception as e:
                # If we get here, both screen and file logging failed
                # As a last resort, print to stderr to not lose the error
                import sys
                print(f"Critical error - both screen and file logging failed: {str(e)}\nOriginal error: {txt}\nOriginal traceback:\n{tb}", file=sys.stderr)

    def logn(self, txt: str = '', tags: list = [], where: str = 'both', link:str = '', tb: str = '') -> None:
        """ Log with a newline appended """
        self.log(f'{txt}\n', tags, where, link, tb)

    def logr(self, txt: str = '', tags: list = [], where: str = 'both', link:str = '', tb: str = '') -> None:
        """ Replace the last line of the log """
        if where != 'file' and not getattr(self, '_headless', False) and hasattr(self, 'log_textbox') and self.log_textbox.winfo_exists():
            self.log_textbox.configure(state=ctk.NORMAL)
            tmp_txt = self.log_textbox.get("end-1c linestart", "end-1c")
            self.log_textbox.delete("end-1c linestart", "end-1c")
            self.log_len -= len(tmp_txt)
        self.log(txt, tags, where, link, tb)

    def create_default_transcript_names(self, dir=None):
        self.transcript_files_list = []
        if 'default_filetype' not in config:
            config['default_filetype'] = 'html'

        # Collect audio file names.
        for f in self.audio_files_list:
            f = Path(f)
            if dir:
                self.transcript_files_list.append(Path(dir) / f'{f.stem}.{config["default_filetype"]}')
            else:
                self.transcript_files_list.append(f'{f.with_name(f.stem)}.{config["default_filetype"]}')

        # Ensure to not override anything and that we have unique file names.
        # Make sure here that every file is a `Path`.
        self.transcript_files_list = utils.create_unique_filenames([Path(x) for x in self.transcript_files_list])

        if len(self.transcript_files_list) > 1:
            self.button_transcript_file_name.configure(text=t('multiple_audio_files'))
        elif len(self.transcript_files_list) == 1:
            self.button_transcript_file_name.configure(text=self.transcript_files_list[0].name)
        else:
            self.button_transcript_file_name.configure(text='')

        self.logn()
        log_msg = t('log_transcript_filename')
        for fn in self.transcript_files_list:
            log_msg += f'\n{fn}'
        self.logn(log_msg)

    def button_audio_file_event(self):
        fn = tk.filedialog.askopenfilename(initialdir=os.path.dirname(self.audio_files_list[0] if len(self.audio_files_list) > 0 else ''), 
                                           initialfile=" ".join(f'"{os.path.basename(path)}"' for path in self.audio_files_list),  
                                           multiple=True)
        if fn and len(fn) > 0:
            self.audio_files_list = fn
            msg = t('log_audio_file_selected')
            for f in fn:
                msg += f'\n{f}'
            self.logn()
            self.logn(msg)
            if len(fn) == 1:
                self.button_audio_file_name.configure(text=os.path.basename(self.audio_files_list[0]))
            else:
                self.button_audio_file_name.configure(text=t('multiple_audio_files'))
            self.create_default_transcript_names()

    def button_transcript_file_event(self):
        if len(self.audio_files_list) == 0:
            # select audio first
            tk.messagebox.showerror(title='noScribe', message=t('err_no_audio_file'))
            return                    
        if len(self.transcript_files_list) > 0:
            _initialdir = os.path.dirname(self.transcript_files_list[0])
            _initialfile = os.path.basename(self.transcript_files_list[0])
        else:
            _initialdir = ''
            _initialfile = ''            
        if not ('last_filetype' in config):
            config['last_filetype'] = 'html'
        filetypes = [
            ('noScribe Transcript','*.html'), 
            ('Text only','*.txt'),
            ('WebVTT Subtitles (also for EXMARaLDA)', '*.vtt')
        ]
        for i, ft in enumerate(filetypes):
            if ft[1] == f'*.{config["last_filetype"]}':
                filetypes.insert(0, filetypes.pop(i))
                break
        
        if len(self.audio_files_list) > 1:
            # multiple audio files, select an output directory
            tk.messagebox.showinfo(title='noScribe', message=t('output_dir_selection'))
            dir = tk.filedialog.askdirectory(title="noScribe", initialdir=_initialdir)
            if dir:
                self.create_default_transcript_names(dir)
            else:
                return
        else:
            # single audio file, select an output file name
            fn = tk.filedialog.asksaveasfilename(initialdir=_initialdir, initialfile=_initialfile, 
                                                filetypes=filetypes, 
                                                defaultextension=config['last_filetype'])
            if fn:
                file_ext = os.path.splitext(fn)[1][1:].lower()
                if not file_ext in ['html', 'txt', 'vtt']:
                    tk.messagebox.showerror(title='noScribe', message=t('err_unsupported_output_format', file_type=file_ext))
                    return                    
                self.transcript_files_list = [fn]
                self.button_transcript_file_name.configure(text=os.path.basename(fn))
                config['last_filetype'] = file_ext
            else:
                return
        
        self.logn()
        log_msg = t('log_transcript_filename')
        for fn in self.transcript_files_list:
            log_msg += f'\n{fn}'
        self.logn(log_msg)
        
    def set_progress(self, step, value, speaker_detection='none'):
        """ Update state of the progress bar """
        if getattr(self, '_headless', False):
            return
        progr = -1
        if step == 1:
            progr = value * 0.05 / 100
        elif step == 2:
            progr = 0.05 # (step 1)
            progr = progr + (value * 0.45 / 100)
        elif step == 3:
            if speaker_detection != 'none':
                progr = 0.05 + 0.45 # (step 1 + step 2)
                progr_factor = 0.5
            else:
                progr = 0.05 # (step 1)
                progr_factor = 0.95
            progr = progr + (value * progr_factor / 100)
        if progr >= 1:
            progr = 0.99 # whisper sometimes still needs some time to finish even if the progress is already at 100%. This can be confusing, so we never go above 99%...

        if abs(progr - self.current_progress) < 0.01:
            # stop updating progress bars if the change is less than 1% (0.01)
            return
        self.current_progress = progr
        
        # Update log_progress_bar
        if self.current_progress > 0:
            self.log_progress_bar.set(self.current_progress)
            if not self.log_progress_bar.winfo_ismapped():
                self.log_progress_bar.pack(padx=(0,10), pady=0, expand=True, fill='x', anchor='sw', side='left')
                self.log_stop_btn.configure(state=ctk.NORMAL)
        else:
            self.log_progress_bar.set(0)
            if self.log_progress_bar.winfo_ismapped():
                self.log_progress_bar.pack_forget()
                self.log_stop_btn.configure(state=ctk.DISABLED)
        
        # Update progress of currently running job in queue table
        if progr >= 0:
            running_jobs = self.queue.get_running_jobs()
            if running_jobs:
                current_job = running_jobs[0]  # Get the first running job
                current_job.progress = progr
                
                # Update the progress bar background for this job
                job_key = id(current_job)
                if hasattr(self, 'queue_row_widgets') and job_key in self.queue_row_widgets:
                    row = self.queue_row_widgets[job_key]
                    if hasattr(row['frame'], 'set_progress'):
                        row['frame'].set_progress(progr)

    def collect_transcription_options(self) -> TranscriptionQueue:
        """Collect all transcription options from UI and config and creates a 
        TranscriptionQueue for each audio file"""
        # Validate required inputs
        if len(self.audio_files_list) == 0:
            raise ValueError(t('err_no_audio_file'))
        
        if len(self.transcript_files_list) == 0:
            raise ValueError(t('err_no_transcript_file'))
        
        # Parse time range from UI
        start_time = None
        val = self.entry_start.get()
        if val != '':
            start_time = utils.str_to_ms(val)
        
        stop_time = None
        val = self.entry_stop.get()
        if val != '':
            stop_time = utils.str_to_ms(val)
        
        # Get whisper model path
        sel_whisper_model = self.option_menu_whisper_model.get()
        if sel_whisper_model not in self.whisper_model_paths.keys():
            raise FileNotFoundError(f"The whisper model '{sel_whisper_model}' does not exist.")
        whisper_model_path = self.whisper_model_paths[sel_whisper_model]
        
        queue = TranscriptionQueue()
        if len(self.audio_files_list) != len(self.transcript_files_list):
            self.create_default_transcript_names()
        
        for i in range(len(self.audio_files_list)):
            job = create_transcription_job(
                audio_file=self.audio_files_list[i],
                transcript_file=self.transcript_files_list[i],
                start_time=start_time,
                stop_time=stop_time,
                language_name=self.option_menu_language.get(),
                whisper_model_name=whisper_model_path,  # Pass the full path
                speaker_detection=self.option_menu_speaker.get(),
                overlapping=self.check_box_overlapping.get(),
                timestamps=self.check_box_timestamps.get(),
                disfluencies=self.check_box_disfluencies.get(),
                pause=self.option_menu_pause.get(),  # Pass string value
                cli_mode=False
            )
            # Handle VTT format warnings in GUI mode
            if job.file_ext == 'vtt' and (job.pause > 0 or job.overlapping or job.timestamps):
                self.logn()
                self.logn(t('err_vtt_invalid_options'), 'error')
            
            queue.add_job(job)
        
        return queue

    def transcription_worker(self, start_job_index=None):
        """Process transcription jobs from the queue"""
        queue_start_time = datetime.datetime.now()
        queue_jobs_processed = 0
        job = None
        self.cancel = False

        try:
            # Log queue summary
            summary = self.queue.get_queue_summary()
            self.logn()
            self.logn(t('queue_start'), 'highlight')
            pending = len(self.queue.get_waiting_jobs())
            if pending > 0:
                self.logn(t('queue_start_jobs', total=pending))
            else:
                self.logn(t('queue_none_waiting'))                
            # Process each job in the queue
            while self.queue.has_pending_jobs():
                # If global cancel was requested (via Stop button), cancel all waiting jobs
                if self.cancel and not self._cancel_job_only:
                    for job in self.queue.get_waiting_jobs():
                        job.set_canceled(t('err_user_cancelation'))
                        self.update_queue_table()
                    break
                
                # Get next job
                job = None
                if start_job_index and start_job_index < len(self.queue.jobs):
                    job = self.queue.jobs[start_job_index]
                    if job.status != JobStatus.WAITING:
                        job = None
                if job is None:
                    job = self.queue.get_next_waiting_job()
                if not job:
                    break
                
                # Process the job
                try:
                    self.logn()
                    self.logn(t('start_job', audio_file=os.path.basename(job.audio_file)), 'highlight')
  
                    # Process single job
                    self._process_single_job(job)
                    
                    queue_jobs_processed += 1
                    job.set_finished()
                    self.update_queue_table()
                    
                except Exception as e:
                    # Distinguish cancellation from real errors
                    error_msg = job.error_message or str(e)
                    if str(e) == t('err_user_cancelation') or self.cancel:
                        job.set_canceled(t('err_user_cancelation'))
                    else:
                        job.set_error(error_msg)
                    self.update_queue_table()
                    self.logn(error_msg, 'error')
                    traceback_str = job.error_tb or traceback.format_exc()
                    self.logn(f"Job error details: {traceback_str}", where='file')
                    print(f"Job error details: {traceback_str}")
                finally:
                    # If we were canceling only the current job, reset flags after it stops
                    if self._cancel_job_only:
                        self.cancel = False
                        self._cancel_job_only = False
            
            # Log final summary
            final_summary = self.queue.get_queue_summary()
            self.logn()
            self.logn(t('queue_complete'), 'highlight')
            self.logn(t('total_jobs', total=final_summary['total']))
            self.logn(t('completed', finished=final_summary['finished']))
            self.logn(t('failed', errors=final_summary['errors']))
            self.logn(t('canceled_summary', canceled=final_summary['canceled']))
            
            # Log total processing time
            total_time = datetime.datetime.now() - queue_start_time
            total_seconds = "{:02d}".format(int(total_time.total_seconds() % 60))
            total_time_str = f'{int(total_time.total_seconds() // 60)}:{total_seconds}'
            self.logn(t('processing_time', total_time_str=total_time_str))
            
            # open editor if only a single file was processed
            if not getattr(self, '_headless', False) \
                    and queue_jobs_processed == 1 \
                    and job \
                    and job.file_ext == 'html' \
                    and job.status == JobStatus.FINISHED \
                    and get_config('auto_edit_transcript', 'True') == 'True':
                self.launch_editor(job.transcript_file)
            elif queue_jobs_processed > 1 and not getattr(self, '_headless', False):
                # if more than one job has been processed, switch to queue tab for an overview 
                self.tabview.set(self.tabview._name_list[1])
            
        except Exception as e:
            self.logn(f"Queue processing error: {str(e)}", 'error')
            traceback_str = traceback.format_exc()
            self.logn(f"Queue error details: {traceback_str}", where='file')
        
        finally:
            # Hide progress
            self.set_progress(0, 0)
            try:
                self.update_queue_controls()
            except Exception:
                pass

    def _process_single_job(self, job: TranscriptionJob):
        """Process a single transcription job"""
        proc_start_time = datetime.datetime.now()
        job.set_running()
        self.update_queue_table()
        
        tmpdir = TemporaryDirectory('noScribe')
        tmp_audio_file = os.path.join(tmpdir.name, 'tmp_audio.wav')
        orig_transcript_file = job.transcript_file

        try:
            # Create option info string for logging
            option_info = ''
            if job.start > 0:
                option_info += f'{t("label_start")} {utils.ms_to_str(job.start)} | '.replace(':', '꞉') # replace the normal colon here in the header with a special character so that MAXQDA does not misinterpret it as a time marker in the transcript.
            if job.stop > 0:
                option_info += f'{t("label_stop")} {utils.ms_to_str(job.stop)} | '.replace(':', '꞉')
            option_info += f'{t("label_language")} {job.language_name} ({languages[job.language_name]}) | '
            option_info += f'{t("label_speaker")} {job.speaker_detection} | '
            option_info += f'{t("label_overlapping")} {job.overlapping} | '
            option_info += f'{t("label_timestamps")} {job.timestamps} | '
            option_info += f'{t("label_disfluencies")} {job.disfluencies} | '
            option_info += f'{t("label_pause")} {job.pause}'

            # Create log file
            if not os.path.exists(f'{config_dir}/log'):
                os.makedirs(f'{config_dir}/log')
            self.log_file = open(f'{config_dir}/log/{Path(job.transcript_file).stem}.log', 'w', encoding="utf-8")

            # Log job configuration
            self.logn(f'whisper beam size: {job.whisper_beam_size}', where='file')
            self.logn(f'whisper temperature: {job.whisper_temperature}', where='file')
            self.logn(f'whisper compute type: {job.whisper_compute_type}', where='file')
            self.logn(f'timestamp_interval: {job.timestamp_interval}', where='file')
            self.logn(f'timestamp_color: {job.timestamp_color}', where='file')

            # Log CPU capabilities
            self.logn("=== CPU FEATURES ===", where="file")
            if platform.system() == 'Windows':
                self.logn("System: Windows", where="file")
                for key, value in cpufeature.CPUFeature.items():
                    self.logn('    {:24}: {}'.format(key, value), where="file")
            elif platform.system() == "Darwin": # = MAC
                self.logn(f"System: MAC {platform.machine()}", where="file")
                """
                if platform.mac_ver()[0] >= '12.3': # MPS needs macOS 12.3+
                    if job.pyannote_xpu == 'mps':
                        self.logn("macOS version >= 12.3:\nUsing MPS (with PYTORCH_ENABLE_MPS_FALLBACK enabled)", where="file")
                    elif job.pyannote_xpu == 'cpu':
                        self.logn("macOS version >= 12.3:\nUser selected to use CPU (results will be better, but you might wanna make yourself a coffee)", where="file")
                    else:
                        self.logn("macOS version >= 12.3:\nInvalid option for 'pyannote_xpu' in config.yml (should be 'mps' or 'cpu')\nYou might wanna change this\nUsing MPS anyway (with PYTORCH_ENABLE_MPS_FALLBACK enabled)", where="file")
                else:
                    self.logn("macOS version < 12.3:\nMPS not available: Using CPU\nPerformance might be poor\nConsider updating macOS, if possible", where="file")
                """
            try:

                #-------------------------------------------------------
                # 1) Convert Audio

                try:
                    self.logn()
                    self.logn(t('start_audio_conversion'), 'highlight')
                
                    if int(job.stop) > 0: # transcribe only part of the audio
                        end_pos_cmd = f'-to {job.stop}ms'
                    else: # tranbscribe until the end
                        end_pos_cmd = ''

                    arguments = f' -loglevel warning -hwaccel auto -y -ss {job.start}ms {end_pos_cmd} -i \"{job.audio_file}\" -ar 16000 -ac 1 -c:a pcm_s16le "{tmp_audio_file}"'
                    if platform.system() == 'Windows':
                        ffmpeg_path = os.path.join(app_dir, 'ffmpeg.exe')
                        ffmpeg_cmd = ffmpeg_path + arguments
                    elif platform.system() == "Darwin":  # = MAC
                        ffmpeg_path = os.path.join(app_dir, 'ffmpeg')
                        ffmpeg_cmd = shlex.split(ffmpeg_path + arguments)
                    elif platform.system() == "Linux":
                        # TODO: Use system ffmpeg if available
                        ffmpeg_path = os.path.join(app_dir, 'ffmpeg-linux-x86_64')
                        ffmpeg_cmd = shlex.split(ffmpeg_path + arguments)
                    else:
                        raise Exception('Platform not supported yet.')

                    self.logn(ffmpeg_cmd, where='file')

                    if platform.system() == 'Windows':
                        # (suppresses the terminal, see: https://stackoverflow.com/questions/1813872/running-a-process-in-pythonw-with-popen-without-a-console)
                        startupinfo = STARTUPINFO()
                        startupinfo.dwFlags |= STARTF_USESHOWWINDOW
                        ffmpeg_proc = Popen(
                            ffmpeg_cmd,
                            stdout=DEVNULL,
                            stderr=STDOUT,
                            universal_newlines=True,
                            encoding='utf-8',
                            startupinfo=startupinfo
                        )
                    elif platform.system() in ("Darwin", "Linux"):
                        ffmpeg_proc = Popen(
                            ffmpeg_cmd,
                            stdout=DEVNULL,
                            stderr=STDOUT,
                            universal_newlines=True,
                            encoding='utf-8'
                        )

                    # Track process for external cancel/close handling
                    self._ffmpeg_proc = ffmpeg_proc

                    try:
                        # Poll loop to allow responsive cancel during conversion
                        while True:
                            rc = ffmpeg_proc.poll()
                            if rc is not None:
                                break
                            if self.cancel:
                                try:
                                    ffmpeg_proc.terminate()
                                except Exception:
                                    pass
                                # Ensure process does not linger
                                try:
                                    ffmpeg_proc.wait(timeout=1.0)
                                except Exception:
                                    try:
                                        ffmpeg_proc.kill()
                                    except Exception:
                                        pass
                                raise Exception(t('err_user_cancelation'))
                            time.sleep(0.1)

                        if ffmpeg_proc.returncode and ffmpeg_proc.returncode > 0:
                            raise Exception(t('err_ffmpeg'))
                    finally:
                        self._ffmpeg_proc = None
                    self.logn(t('audio_conversion_finished'))
                    self.set_progress(1, 100, job.speaker_detection)
                except Exception as e:
                    traceback_str = traceback.format_exc()
                    # Distinguish cancel vs. real error during audio conversion
                    if str(e) == t('err_user_cancelation') or self.cancel:
                        job.set_canceled(t('err_user_cancelation'))
                        self.update_queue_table()
                        raise Exception(t('err_user_cancelation'))
                    else:
                        job.set_error(f"{t('err_converting_audio')}: {e}", traceback_str)
                        self.update_queue_table()
                        raise Exception(job.error_message)

                #-------------------------------------------------------
                # 2) Speaker identification (diarization) with pyannote

                # Helper Functions:

                def overlap_len(ss_start, ss_end, ts_start, ts_end):
                    # ss...: speaker segment start and end in milliseconds (from pyannote)
                    # ts...: transcript segment start and end (from whisper.cpp)
                    # returns overlap percentage, i.e., "0.8" = 80% of the transcript segment overlaps with the speaker segment from pyannote  
                    if ts_end < ss_start: # no overlap, ts is before ss
                        return None

                    if ts_start > ss_end: # no overlap, ts is after ss
                        return 0.0

                    ts_len = ts_end - ts_start
                    if ts_len <= 0:
                        return None

                    # ss & ts have overlap
                    overlap_start = max(ss_start, ts_start) # Whichever starts later
                    overlap_end = min(ss_end, ts_end) # Whichever ends sooner

                    ol_len = overlap_end - overlap_start + 1
                    return ol_len / ts_len

                # Mapping from pyannote label (e.g. "SPEAKER_01") to a human
                # name confirmed via the speaker naming dialog.  Populated
                # after diarization if voice signatures are available.
                speaker_name_map = {}

                def find_speaker(diarization, transcript_start, transcript_end) -> str:
                    # Looks for the shortest segment in diarization that has at least 80% overlap
                    # with transcript_start - trancript_end.
                    # Returns the speaker name if found.
                    # If only an overlap < 80% is found, this speaker name ist returned.
                    # If no overlap is found, an empty string is returned.
                    spkr = ''
                    overlap_found = 0
                    overlap_threshold = 0.8
                    segment_len = 0
                    is_overlapping = False

                    for segment in diarization:
                        t = overlap_len(segment["start"], segment["end"], transcript_start, transcript_end)
                        if t is None: # we are already after transcript_end
                            break

                        current_segment_len = segment["end"] - segment["start"] # Length of the current segment
                        lbl = segment["label"]
                        # Use confirmed name from dialog if available, else fall
                        # back to short label like "S01"
                        current_segment_spkr = speaker_name_map.get(
                            lbl, f'S{lbl[8:]}')  # "SPEAKER_01" -> "S01"

                        if overlap_found >= overlap_threshold: # we already found a fitting segment, compare length now
                            if (t >= overlap_threshold) and (current_segment_len < segment_len): # found a shorter (= better fitting) segment that also overlaps well
                                is_overlapping = True
                                overlap_found = t
                                segment_len = current_segment_len
                                spkr = current_segment_spkr
                        elif t > overlap_found: # no segment with good overlap yet, take this if the overlap is better then previously found
                            overlap_found = t
                            segment_len = current_segment_len
                            spkr = current_segment_spkr

                    if job.overlapping and is_overlapping:
                        return f"//{spkr}"
                    else:
                        return spkr

                # Start Diarization:

                if job.speaker_detection != 'none':
                    try:
                        job.status = JobStatus.SPEAKER_IDENTIFICATION
                        self.update_queue_table()

                        self.logn()
                        self.logn(t('start_identifying_speakers'), 'highlight')
                        self.logn(t('loading_pyannote'))
                        # self.set_progress(1, 100, job.speaker_detection)

                        while True:
                            try:
                                diarization, _embeddings = self._run_diarize_subprocess(tmp_audio_file, job)
                                break
                            except Exception as err:
                                if self._handle_cuda_fallback('pyannote', err):
                                    self.logn(t('pyannote_cuda_retry'), 'highlight')
                                    continue
                                raise

                        # write segments to log file
                        for segment in diarization:
                            line = f'{utils.ms_to_str(job.start + segment["start"], include_ms=True)} - {utils.ms_to_str(job.start + segment["end"], include_ms=True)} {segment["label"]}'
                            self.logn(line, where='file')

                        self.logn()

                        # --------------------------------------------------
                        # Speaker naming dialog: match embeddings against the
                        # stored database and ask the user to confirm / assign
                        # names for this session.
                        # --------------------------------------------------
                        if _embeddings:
                            # Build data list for the dialog
                            seen_labels = set()
                            speakers_data = []
                            for seg in diarization:
                                lbl = seg["label"]
                                if lbl in seen_labels:
                                    continue
                                seen_labels.add(lbl)
                                short = f'S{lbl[8:]}'  # "SPEAKER_01" -> "S01"
                                emb = _embeddings.get(lbl)
                                matched_name, sim = (None, 0.0)
                                if emb:
                                    try:
                                        matched_name, sim = speaker_db.find_match(emb)
                                    except Exception:
                                        pass
                                speakers_data.append({
                                    'label': lbl,
                                    'short_label': short,
                                    'matched_name': matched_name,
                                    'similarity': sim,
                                    'embedding': emb,
                                })

                            # Show the dialog in the main GUI thread and wait
                            import threading as _threading
                            _result_holder = [{}]
                            _dialog_done = _threading.Event()

                            def _open_naming_dialog():
                                try:
                                    _result_holder[0] = self._run_speaker_naming_dialog(
                                        speakers_data)
                                except Exception:
                                    _result_holder[0] = {}
                                finally:
                                    _dialog_done.set()

                            self.after(0, _open_naming_dialog)
                            _dialog_done.wait()
                            speaker_name_map.update(_result_holder[0])

                    except Exception as e:
                        traceback_str = traceback.format_exc()
                        if str(e) == t('err_user_cancelation') or self.cancel:
                            job.set_canceled(t('err_user_cancelation'))
                            self.update_queue_table()
                            raise Exception(t('err_user_cancelation'))
                        else:
                            job.set_error(f"{t('err_identifying_speakers')}: {e}", traceback_str)
                            self.update_queue_table()
                            raise Exception(job.error_message)

                #-------------------------------------------------------
                # 3) Transcribe with faster-whisper

                job.status = JobStatus.TRANSCRIPTION
                self.update_queue_table()

                self.logn()
                self.logn(t('start_transcription'), 'highlight')
                self.logn(t('loading_whisper'))

                info = None
                transcription_success = False
                while True:
                    retry_cuda = False

                    # prepare transcript html
                    d = AdvancedHTMLParser.AdvancedHTMLParser()
                    d.parseStr(default_html)                

                    # add audio file path:
                    tag = d.createElement("meta")
                    tag.name = "audio_source"
                    tag.content = job.audio_file
                    d.head.appendChild(tag)

                    # add app version:
                    """ # removed because not really necessary
                    tag = d.createElement("meta")
                    tag.name = "noScribe_version"
                    tag.content = app_version
                    d.head.appendChild(tag)
                    """

                    #add WordSection1 (for line numbers in MS Word) as main_body
                    main_body = d.createElement('div')
                    main_body.addClass('WordSection1')
                    d.body.appendChild(main_body)

                    # header               
                    p = d.createElement('p')
                    p.setStyle('font-weight', '600')
                    p.appendText(Path(job.audio_file).stem) # use the name of the audio file (without extension) as the title
                    main_body.appendChild(p)

                    # subheader
                    p = d.createElement('p')
                    s = d.createElement('span')
                    s.setStyle('color', '#909090')
                    s.setStyle('font-size', '0.8em')
                    s.appendText(t('doc_header', version=app_version))
                    br = d.createElement('br')
                    s.appendChild(br)

                    s.appendText(t('doc_header_audio', file=job.audio_file))
                    br = d.createElement('br')
                    s.appendChild(br)

                    s.appendText(f'({html.escape(option_info, quote=False)})')

                    p.appendChild(s)
                    main_body.appendChild(p)

                    p = d.createElement('p')
                    main_body.appendChild(p)

                    speaker = ''
                    prev_speaker = ''
                    last_auto_save = datetime.datetime.now()

                    def save_doc():
                        nonlocal last_auto_save
                        txt = ''
                        if job.file_ext == 'html':
                            txt = d.asHTML()
                        elif job.file_ext == 'txt':
                            txt = html_to_text(d)
                        elif job.file_ext == 'vtt':
                            txt = html_to_webvtt(d, job.audio_file)
                        else:
                            raise TypeError(f'Invalid file type "{job.file_ext}".')
                        try:
                            if txt != '':
                                with open(job.transcript_file, 'w', encoding="utf-8") as f:
                                    f.write(txt)
                                    f.flush()
                                last_auto_save = datetime.datetime.now()
                        except Exception:
                            # other error while saving, maybe the file is already open in Word and cannot be overwritten
                            # try saving to a different filename
                            try:
                                job.transcript_file = utils.create_unique_filenames([Path(job.transcript_file)])[0]
                            except RuntimeError as e:
                                # File name already exists and a new one could not
                                # be found.
                                raise RuntimeError(t('rescue_saving_failed')) from e

                            # `job.transcript_file` is for sure a `Path` here as we
                            # called `create_unique_filenames`.
                            job.transcript_file.write_text(txt, encoding="utf-8")

                            self.logn()
                            self.logn(t('rescue_saving', file=job.transcript_file), 'error', link=f'file://{job.transcript_file}')
                            last_auto_save = datetime.datetime.now()

                    # Prepare VAD data locally for pause adjustment (audio is 16kHz mono after ffmpeg conversion)
                    try:
                        job.vad_threshold = float(config['voice_activity_detection_threshold'])
                    except Exception:
                        config['voice_activity_detection_threshold'] = '0.5'
                        job.vad_threshold = 0.5
                    sampling_rate = 16000
                    audio = decode_audio(tmp_audio_file, sampling_rate=sampling_rate)
                    duration = audio.shape[0] / sampling_rate
                    try:
                        vad_parameters = VadOptions(min_silence_duration_ms=500,
                                                    threshold=job.vad_threshold,
                                                    speech_pad_ms=0)
                    except TypeError:
                        vad_parameters = VadOptions(min_silence_duration_ms=500,
                                                    onset=job.vad_threshold,
                                                    speech_pad_ms=0)
                    speech_chunks = get_speech_timestamps(audio, vad_parameters)

                    def adjust_for_pause(segment):
                        """Adjusts start and end of segment if it falls into a pause
                        identified by the VAD"""
                        pause_extend = 0.2  # extend the pauses by 200ms to make the detection more robust

                        original_start = segment.start
                        original_end = segment.end

                        # iterate through the pauses and adjust segment boundaries accordingly
                        for i in range(0, len(speech_chunks)):
                            pause_start = (speech_chunks[i]['end'] / sampling_rate) - pause_extend
                            if i == (len(speech_chunks) - 1):
                                pause_end = duration + pause_extend # last segment, pause till the end
                            else:
                                pause_end = (speech_chunks[i+1]['start']  / sampling_rate) + pause_extend

                            if pause_start > segment.end:
                                break  # we moved beyond the segment, stop going further
                            if segment.start > pause_start and segment.start < pause_end:
                                segment.start = pause_end - pause_extend
                            if segment.end > pause_start and segment.end < pause_end:
                                segment.end = pause_start + pause_extend

                        # Safety check: if pause adjustments caused start >= end
                        # (e.g. short segment fully inside a pause), revert to
                        # original boundaries to avoid negative durations (#253)
                        if segment.start >= segment.end:
                            segment.start = original_start
                            segment.end = original_end

                        return segment
                                    
                    # Run Faster-Whisper in a spawned subprocess and stream segments
                    last_segment_end = 0
                    last_timestamp_ms = 0
                    first_segment = True

                    def on_segment(seg):
                        nonlocal first_segment, last_segment_end, last_timestamp_ms, p, speaker, prev_speaker
                        # Map dict to simple object-like for existing code
                        class _Seg:
                            __slots__ = ("start", "end", "text", "words")
                            def __init__(self, d):
                                self.start = d.get('start')
                                self.end = d.get('end')
                                self.text = d.get('text')
                                self.words = d.get('words')
                        segment = _Seg(seg)

                        segment = adjust_for_pause(segment)

                        # get time of the segment in milliseconds
                        start = round(segment.start * 1000.0)
                        end = round(segment.end * 1000.0)
                        # if we skipped a part at the beginning of the audio we have to add this here again, otherwise the timestamps will not match the original audio:
                        orig_audio_start = job.start + start
                        orig_audio_end = job.start + end

                        if job.timestamps:
                            ts = utils.ms_to_str(orig_audio_start)
                            ts = f'[{ts}]'

                        # check for pauses and mark them in the transcript
                        if (job.pause > 0) and (start - last_segment_end >= job.pause * 1000): # (more than x seconds with no speech)
                            pause_len = round((start - last_segment_end)/1000)
                            if pause_len >= 60: # longer than 60 seconds
                                pause_str = ' ' + t('pause_minutes', minutes=round(pause_len/60))
                            elif pause_len >= 10: # longer than 10 seconds
                                pause_str = ' ' + t('pause_seconds', seconds=pause_len)
                            else: # less than 10 seconds
                                pause_str = ' (' + (job.pause_marker * pause_len) + ')'

                            if first_segment:
                                pause_str = pause_str.lstrip() + ' '

                            orig_audio_start_pause = job.start + last_segment_end
                            orig_audio_end_pause = job.start + start
                            a = d.createElement('a')
                            a.name = f'ts_{orig_audio_start_pause}_{orig_audio_end_pause}_{speaker}'
                            a.appendText(pause_str)
                            p.appendChild(a)
                            self.log(pause_str)
                            if first_segment:
                                self.logn()
                                self.logn()
                        last_segment_end = end

                        # write text to the doc
                        # diarization (speaker detection)?
                        seg_text = segment.text
                        seg_html = html.escape(seg_text, quote=False)

                        if job.speaker_detection != 'none':
                            new_speaker = find_speaker(diarization, start, end)
                            if (speaker != new_speaker) and (new_speaker != ''): # speaker change
                                if new_speaker[:2] == '//': # is overlapping speech, create no new paragraph
                                    prev_speaker = speaker
                                    speaker = new_speaker
                                    seg_text = f' {speaker}:{seg_text}'
                                    seg_html = html.escape(seg_text, quote=False)                                
                                elif (speaker[:2] == '//') and (new_speaker == prev_speaker): # was overlapping speech and we are returning to the previous speaker 
                                    speaker = new_speaker
                                    seg_text = f'//{seg_text}'
                                    seg_html = html.escape(seg_text, quote=False)
                                else: # new speaker, not overlapping
                                    if speaker[:2] == '//': # was overlapping speech, mark the end
                                        last_elem = p.lastElementChild
                                        if last_elem:
                                            last_elem.appendText('//')
                                        else:
                                            p.appendText('//')
                                        self.log('//')
                                    p = d.createElement('p')
                                    main_body.appendChild(p)
                                    if not first_segment:
                                        self.logn()
                                        self.logn()
                                    speaker = new_speaker
                                    # add timestamp
                                    if job.timestamps:
                                        seg_html = f'{speaker}: <span style="color: {job.timestamp_color}" >{ts}</span>{html.escape(seg_text, quote=False)}'
                                        seg_text = f'{speaker}: {ts}{seg_text}'
                                        last_timestamp_ms = start
                                    else:
                                        if job.file_ext != 'vtt': # in vtt files, speaker names are added as special voice tags so skip this here
                                            seg_text = f'{speaker}:{seg_text}'
                                            seg_html = html.escape(seg_text, quote=False)
                                        else:
                                            seg_html = html.escape(seg_text, quote=False).lstrip()
                                            seg_text = f'{speaker}:{seg_text}'
                                        
                            else: # same speaker
                                if job.timestamps:
                                    if (start - last_timestamp_ms) > job.timestamp_interval:
                                        seg_html = f' <span style=\"color: {job.timestamp_color}\" >{ts}</span>{html.escape(seg_text, quote=False)}'
                                        seg_text = f' {ts}{seg_text}'
                                        last_timestamp_ms = start
                                    else:
                                        seg_html = html.escape(seg_text, quote=False)

                        else: # no speaker detection
                            if job.timestamps and (first_segment or (start - last_timestamp_ms) > job.timestamp_interval):
                                seg_html = f' <span style=\"color: {job.timestamp_color}\" >{ts}</span>{html.escape(seg_text, quote=False)}'
                                seg_text = f' {ts}{seg_text}'
                                last_timestamp_ms = start
                            else:
                                seg_html = html.escape(seg_text, quote=False)
                            # avoid leading whitespace in first paragraph
                            if first_segment:
                                seg_text = seg_text.lstrip()
                                seg_html = seg_html.lstrip()

                        # Create bookmark with audio timestamps start to end and add the current segment.
                        a_html = f'<a name=\"ts_{orig_audio_start}_{orig_audio_end}_{speaker}\" >{seg_html}</a>'
                        a = d.createElementFromHTML(a_html)
                        p.appendChild(a)

                        self.log(seg_text)
                        
                        first_segment = False

                        # auto save periodically
                        nonlocal last_auto_save
                        if job.auto_save:
                            if (datetime.datetime.now() - last_auto_save).total_seconds() > 5:
                                save_doc()
                                job.has_partial_transcript = True

                        # per-segment progress based on total duration
                        try:
                            progr = round((segment.end/duration) * 100)
                            self.set_progress(3, progr, job.speaker_detection)
                        except Exception:
                            pass
                    
                    try:
                        info = self._run_whisper_subprocess_stream(tmp_audio_file, job, on_segment)
                        transcription_success = True
                        # if self.cancel:
                        #    raise Exception(t('err_user_cancelation')) 
                        
                        job.has_partial_transcript = False # transcript is finished
                        self.logn()
                        self.logn()
                        self.logn(t('transcription_finished'), 'highlight')
                    except Exception as err:
                        if self._handle_cuda_fallback('whisper', err):
                            retry_cuda = True
                        else:
                            raise
                    finally:
                        if not first_segment:
                            save_doc()
                            job.has_partial_transcript = job.status != JobStatus.FINISHED
                        else:
                            job.has_partial_transcript = False
                        if transcription_success:
                            if job.transcript_file != orig_transcript_file: # used alternative filename because saving under the initial name failed
                                self.logn(
                                    t('rescue_saving', file=job.transcript_file),
                                    link=f'file://{job.transcript_file}'
                                )
                            else:
                                self.logn(
                                    t('transcription_saved', file=job.transcript_file),
                                    link=f'file://{job.transcript_file}'
                                )
                    if retry_cuda:
                        self.logn(t('whisper_cuda_retry'), 'highlight')
                        continue
                    else:
                        break

                # log duration of the whole process
                proc_time = datetime.datetime.now() - proc_start_time
                proc_seconds = "{:02d}".format(int(proc_time.total_seconds() % 60))
                proc_time_str = f'{int(proc_time.total_seconds() // 60)}:{proc_seconds}' 
                self.logn(t('trancription_time', duration=proc_time_str)) 
            finally:
                self.log_file.close()
                self.log_file = None

        finally:
            # hide progress
            self.set_progress(0, 0)
            
    def create_job(self, enqueue=False):
        try:
            show_queue_tab = enqueue
            # Collect transcription options from UI
            new_queue = self.collect_transcription_options()
            
            # Confirm override if output file conflicts with jobs in queue
            for job in new_queue.jobs:
                if self.queue.has_output_conflict(job.transcript_file):
                    if not self.queue.confirm_output_override(job.transcript_file):
                        return
                    else:
                        break

            # Add the jobs to the queue
            for job in new_queue.jobs:
                self.queue.add_job(job)            
                if not enqueue and not self.queue.is_running(): # Start transcription worker with the queue
                    wkr = Thread(target=self.transcription_worker, kwargs={"start_job_index": len(self.queue.jobs) - 1}, daemon=True)
                    self._worker_threads.append(wkr)
                    wkr.start()
                    enqueue = True
                else: # just add it to the queue
                    show_queue_tab = True
                    self.logn()
                    self.logn(t('queue_added_job', audio_file=os.path.basename(job.audio_file)), 'highlight')
            
            self.update_queue_table()
            if show_queue_tab:
                try:
                    self.tabview.set(self.tabview._name_list[1]) # Switch to queue tab for visual feedback
                except Exception:
                    pass
                            
        except (ValueError, FileNotFoundError) as e:
            # Handle validation errors from collect_transcription_options
            self.logn(str(e), 'error')
            tk.messagebox.showerror(title='noScribe', message=str(e))
        except Exception as e:
            # Handle unexpected errors
            self.logn(f'Error starting transcription: {str(e)}', 'error')
            tk.messagebox.showerror(title='noScribe', message=f'Error starting transcription: {str(e)}')

    def _handle_cuda_fallback(self, component: str, error: Exception) -> bool:
        global force_pyannote_cpu
        global force_whisper_cpu
        message = str(error).strip()
        if not _is_cuda_error_message(message):
            return False

        if component == 'pyannote':
            if force_pyannote_cpu:
                return False
            prompt = t('pyannote_cuda_error_prompt', error=message)
            if tk.messagebox.askyesno(title='noScribe', message=prompt):
                force_pyannote_cpu = True
                config['force_pyannote_cpu'] = 'true'
                save_config()
                return True
            return False

        if component == 'whisper':
            if force_whisper_cpu:
                return False
            prompt = t('whisper_cuda_error_prompt', error=message)
            if tk.messagebox.askyesno(title='noScribe', message=prompt):
                force_whisper_cpu = True
                config['force_whisper_cpu'] = 'true'
                save_config()
                return True
            return False

        return False

    def _run_whisper_subprocess_stream(self, tmp_audio_file: str, job, on_segment):
        """Spawn a subprocess to run Faster-Whisper and stream segments.
        Calls on_segment(dict) for each segment streamed by the child.
        Returns a simple info object (duration at least).
        """
        global force_whisper_cpu
        # Language code for non-auto/multilingual
        language_code = None
        if job.language_name not in ('Auto', 'Multilingual'):
            try:
                language_code = languages[job.language_name]
            except Exception:
                language_code = None

        # VAD threshold from config
        try:
            vad_threshold = float(config.get('voice_activity_detection_threshold', '0.5'))
        except Exception:
            vad_threshold = 0.5

        args = {
            "model_name_or_path": job.whisper_model,
            "device": 'cpu' if force_whisper_cpu else 'auto',
            "compute_type": job.whisper_compute_type,
            "cpu_threads": number_threads,
            "local_files_only": True,
            "audio_path": tmp_audio_file,
            "language_name": job.language_name,
            "language_code": language_code,
            "disfluencies": job.disfluencies,
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
            "vad_threshold": vad_threshold,
            "locale": app_locale,
        }

        # Spawn child process using spawn start method
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        from whisper_mp_worker import whisper_proc_entrypoint
        proc = ctx.Process(target=whisper_proc_entrypoint, args=(args, q))
        proc.start()
        # Expose to allow cancel to terminate the child
        self._mp_proc = proc
        self._mp_queue = q

        info = None
        try:
            while True:
                try:
                    msg = q.get(timeout=0.1)
                except pyqueue.Empty:
                    if self.cancel:
                        # User requested cancel; terminate child
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        raise Exception(t('err_user_cancelation'))
                    if not proc.is_alive():
                        # Process died without sending result
                        exitcode = proc.exitcode
                        self.logn(f"Transcription worker exited unexpectedly (code {exitcode}).", 'error')
                        raise Exception('Subprocess terminated unexpectedly')
                    continue

                mtype = msg.get("type") if isinstance(msg, dict) else None
                if mtype == "log":
                    level = msg.get("level", "info")
                    txt = msg.get("msg", "")
                    if level == 'error':
                        self.logn(txt, 'error')
                    else:
                        self.logn(txt)
                elif mtype == "progress":
                    pct = msg.get("pct")
                    detail = msg.get("detail")
                    try:
                        if pct is not None:
                            self.set_progress(3, float(pct), job.speaker_detection)
                    except Exception:
                        pass
                elif mtype == "segment":
                    seg = msg.get("segment") or {}
                    try:
                        on_segment(seg)
                    except Exception as e:
                        # If on_segment fails, stop child and raise
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        raise
                elif mtype == "result":
                    if msg.get("ok"):
                        info = msg.get("info", {})
                    else:
                        err = msg.get('error', 'Transcription failed')
                        trc = msg.get('trace')
                        self.logn(f"Transcription failed: {err}", 'error')
                        if trc:
                            self.logn(trc, where='file')
                        raise Exception(err)
                    break
                # keep looping until we get a result
        finally:
            try:
                proc.join(timeout=0.2)
            except Exception:
                pass
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.close()
            except Exception:
                pass
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
            # Clear exposed handles
            self._mp_proc = None
            self._mp_queue = None

        class _Info:
            __slots__ = ("duration",)
            def __init__(self, d):
                self.duration = d.get('duration')
        info_obj = _Info(info or {})
        return info_obj

    def _run_diarize_subprocess(self, tmp_audio_file: str, job):
        """Spawn a subprocess to run diarization and return (segments, embeddings).

        *segments* is a list of dicts {start, end, label}.
        *embeddings* is a dict {label: [float, ...]} with per-speaker voice
        embeddings, or an empty dict when extraction was not possible.
        Streams child logs/progress back to GUI and honors cancel.
        """
        global force_pyannote_cpu
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        from pyannote_mp_worker import pyannote_proc_entrypoint
        args = {
            "device": 'cpu' if force_pyannote_cpu else '',
            "audio_path": tmp_audio_file,
            "num_speakers": (int(job.speaker_detection) if str(job.speaker_detection).isdigit() else None),
        }
        proc = ctx.Process(target=pyannote_proc_entrypoint, args=(args, q))
        proc.start()
        # Keep handles for cancel
        self._mp_proc = proc
        self._mp_queue = q

        diarization = None
        embeddings = {}
        try:
            while True:
                try:
                    msg = q.get(timeout=0.1)
                except pyqueue.Empty:
                    if self.cancel:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        raise Exception(t('err_user_cancelation'))
                    if not proc.is_alive():
                        exitcode = proc.exitcode
                        self.logn(f"Diarization worker exited unexpectedly (code {exitcode}). UI remains responsive.", 'error')
                        raise Exception('Subprocess terminated unexpectedly')
                    continue

                mtype = msg.get("type") if isinstance(msg, dict) else None
                if mtype == "log":
                    txt = msg.get("msg", "")
                    self.logn('PyAnnote ' + txt, where='file')
                elif mtype == "progress":
                    step_name = str(msg.get("step", ""))
                    progress_percent = int(msg.get("pct", 0))
                    self.logr(f'{step_name}: {progress_percent}%')
                    if step_name == 'segmentation':
                        self.set_progress(2, progress_percent * 0.3, job.speaker_detection)
                    elif step_name == 'embeddings':
                        self.set_progress(2, 30 + (progress_percent * 0.7), job.speaker_detection)
                elif mtype == "result":
                    if msg.get("ok"):
                        diarization = msg.get("segments", [])
                        embeddings = msg.get("embeddings", {})
                    else:
                        err = msg.get('error', 'Diarization failed')
                        trc = msg.get('trace')
                        self.logn(f"PyAnnote error: {err}", 'error')
                        if trc:
                            self.logn(trc, where='file')
                        raise Exception(err)
                    break

        finally:
            try:
                proc.join(timeout=0.2)
            except Exception:
                pass
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.close()
            except Exception:
                pass
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
            self._mp_proc = None
            self._mp_queue = None

        return diarization or [], embeddings

    def _run_speaker_naming_dialog(self, speakers_data: list) -> dict:
        """Show SpeakerNamingDialog in the main (GUI) thread and return the
        resulting {label: name} mapping.  Must be called from the main thread.
        """
        dialog = SpeakerNamingDialog(self, speakers_data)
        self.wait_window(dialog)
        return dialog.result
    
    def on_closing(self):
        # (see: https://stackoverflow.com/questions/111155/how-do-i-handle-the-window-close-event-in-tkinter)
        global force_pyannote_cpu
        global force_whisper_cpu
        #if messagebox.askokcancel("Quit", "Do you want to quit?"):

        # Stop all running jobs:
        try:
            if not self.on_queue_stop(ask_before_canceling=True):
                return # user has aborted cancelation of waiting jobs
        except:
            pass

        # Signal shutdown and try to stop background activity gracefully
        self._shutting_down = True
        self.cancel = True
        try:
            # Terminate active multiprocessing child (diarization/whisper) if present
            try:
                if getattr(self, "_mp_proc", None) is not None and self._mp_proc.is_alive():
                    try:
                        self._mp_proc.terminate()
                    except Exception:
                        pass
                    try:
                        self._mp_proc.join(timeout=1.0)
                    except Exception:
                        pass
            finally:
                try:
                    if getattr(self, "_mp_queue", None) is not None:
                        try:
                            self._mp_queue.close()
                        except Exception:
                            pass
                        try:
                            self._mp_queue.join_thread()
                        except Exception:
                            pass
                finally:
                    self._mp_proc = None
                    self._mp_queue = None

            # Terminate ffmpeg if currently converting
            try:
                if getattr(self, "_ffmpeg_proc", None) is not None and self._ffmpeg_proc.poll() is None:
                    try:
                        self._ffmpeg_proc.terminate()
                    except Exception:
                        pass
                    try:
                        self._ffmpeg_proc.wait(timeout=1.0)
                    except Exception:
                        try:
                            self._ffmpeg_proc.kill()
                        except Exception:
                            pass
            finally:
                self._ffmpeg_proc = None

            # Join worker threads briefly to give them a chance to exit
            try:
                for th in list(getattr(self, "_worker_threads", [])):
                    try:
                        th.join(timeout=2.0)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

        # remember some settings for the next run
        try:
            config['last_language'] = self.option_menu_language.get()
            config['last_speaker'] = self.option_menu_speaker.get()
            config['last_whisper_model'] = self.option_menu_whisper_model.get()
            config['last_pause'] = self.option_menu_pause.get()
            config['last_overlapping'] = self.check_box_overlapping.get()
            config['last_timestamps'] = self.check_box_timestamps.get()
            config['last_disfluencies'] = self.check_box_disfluencies.get()
            config['force_pyannote_cpu'] = str(force_pyannote_cpu)
            config['force_whisper_cpu'] = str(force_whisper_cpu)

            save_config()
        finally:
            try:
                self.quit()
            except Exception:
                pass
            self.destroy()


class HeadlessApp(App):
    def __init__(self):
        # Do not initialize Tk/CTk to avoid DISPLAY requirements
        _init_app_state(self)
        self._headless = True

    def __getattr__(self, name):
        # Avoid Tk attribute delegation recursion when CTk isn't initialized
        raise AttributeError(name)


def _cleanup_app(app):
    """Cleanup app instance for proper process exit in CLI mode."""
    try:
        # Signal shutdown to stop background activity
        app._shutting_down = True
        app.cancel = True

        # Terminate active multiprocessing child (diarization/whisper) if present
        if getattr(app, "_mp_proc", None) is not None:
            try:
                if app._mp_proc.is_alive():
                    app._mp_proc.terminate()
                    app._mp_proc.join(timeout=1.0)
            except Exception:
                pass
            finally:
                app._mp_proc = None

        # Close multiprocessing queue
        if getattr(app, "_mp_queue", None) is not None:
            try:
                app._mp_queue.close()
                app._mp_queue.join_thread()
            except Exception:
                pass
            finally:
                app._mp_queue = None

        # Terminate ffmpeg if currently converting
        if getattr(app, "_ffmpeg_proc", None) is not None:
            try:
                if app._ffmpeg_proc.poll() is None:
                    app._ffmpeg_proc.terminate()
                    app._ffmpeg_proc.wait(timeout=1.0)
            except Exception:
                try:
                    app._ffmpeg_proc.kill()
                except Exception:
                    pass
            finally:
                app._ffmpeg_proc = None

        # Join worker threads briefly
        for th in list(getattr(app, "_worker_threads", [])):
            try:
                th.join(timeout=0.5)
            except Exception:
                pass
    except Exception:
        pass

    if not getattr(app, '_headless', False):
        try:
            app.quit()
        except Exception:
            pass
        try:
            app.destroy()
        except Exception:
            pass

def run_cli_mode(args):
    """Run noScribe in CLI mode"""
    app = None
    try:
        # Create a headless app instance (no GUI initialization)
        app = HeadlessApp()
        
        # Validate and set the whisper model
        available_models = app.get_whisper_models()
        if args.model:
            if args.model not in available_models:
                print(f"Error: Model '{args.model}' not found.")
                print(f"Available models: {', '.join(available_models)}")
                return 1
        else:
            # Use default model
            if 'precise' in available_models:
                args.model = 'precise'
            elif available_models:
                args.model = available_models[0]
            else:
                print("Error: No Whisper models found.")
                return 1
        
        # Create job from CLI arguments
        job = create_job_from_cli_args(args)
        
        # Set the whisper model path
        job.whisper_model = app.whisper_model_paths[args.model]
        
        # Validate files
        if not os.path.exists(job.audio_file):
            print(f"Error: Audio file '{job.audio_file}' not found.")
            return 1
        
        # Check output directory exists
        output_dir = os.path.dirname(os.path.abspath(job.transcript_file))
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                print(f"Error: Cannot create output directory '{output_dir}': {e}")
                return 1
        
        # Add the job to the queue
        app.queue.add_job(job)
        
        print(f"Starting transcription of '{job.audio_file}'...")
        print(f"Output will be saved to '{job.transcript_file}'")
        print(f"Language: {job.language_name}")
        print(f"Model: {args.model}")
        print(f"Speaker detection: {job.speaker_detection}")
        print()
        
        # Start transcription worker with the queue
        app.transcription_worker()
        
        # Check results
        final_summary = app.queue.get_queue_summary()
        if final_summary['finished'] > 0:
            print(f"\nTranscription completed successfully!")
            print(f"Output saved to: {job.transcript_file}")
            return 0
        else:
            if final_summary.get('canceled', 0) > 0:
                print(f"\nTranscription canceled by user.")
                return 1
            print(f"\nTranscription failed!")
            failed_jobs = app.queue.get_failed_jobs()
            if failed_jobs:
                print(f"Error: {failed_jobs[0].error_message}")
            return 1
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        if app is not None:
            _cleanup_app(app)

def show_available_models():
    """Show available Whisper models"""
    app = None
    try:
        # Create headless app instance to get models
        app = HeadlessApp()
        models = app.get_whisper_models()
        
        print("Available Whisper models:")
        for model in models:
            print(f"  - {model}")
        
        if not models:
            print("  No models found. Please check your installation.")
    except Exception as e:
        print(f"Error getting models: {str(e)}")
    finally:
        if app is not None:
            _cleanup_app(app)

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_cli_args()

    # Handle special case: show available models
    if args.help_models:
        show_available_models()
        sys.exit(0)

    # If explicit headless requested, run pure CLI mode
    if getattr(args, 'no_gui', False):
        if args.audio_file and args.output_file:
            exit_code = run_cli_mode(args)
            sys.exit(exit_code)
        else:
            print("Error: --no-gui requires both audio_file and output_file.")
            print("Usage: python noScribe.py <audio_file> <output_file> [options] --no-gui")
            sys.exit(1)

    # Default: show GUI, even with CLI args
    app = App()

    # If arguments were provided, prefill and optionally auto-start
    try:
        # Prefill selected model if provided
        desired_model_name = None
        available_models = app.get_whisper_models()
        if getattr(args, 'model', None):
            if args.model in available_models:
                desired_model_name = args.model
            else:
                app.logn(f"Warning: Model '{args.model}' not found. Using default GUI selection.")

        if desired_model_name:
            try:
                app.option_menu_whisper_model.set(desired_model_name)
            except Exception:
                pass

        # Prefill files if provided
        if getattr(args, 'audio_file', None):
            app.audio_files_list = [args.audio_file]
            try:
                app.button_audio_file_name.configure(text=os.path.basename(args.audio_file))
            except Exception:
                pass
            app.logn()
            app.logn(t('log_audio_file_selected') + f'\n{args.audio_file}')

        if getattr(args, 'output_file', None):
            app.transcript_files_list = [args.output_file]
            try:
                app.button_transcript_file_name.configure(text=os.path.basename(args.output_file))
            except Exception:
                pass
            app.logn()
            app.logn(t('log_transcript_filename') + f'\n{args.output_file}')

        # Prefill other options if provided
        if getattr(args, 'start', None):
            app.entry_start.delete(0, 'end')
            app.entry_start.insert(0, args.start)
        if getattr(args, 'stop', None):
            app.entry_stop.delete(0, 'end')
            app.entry_stop.insert(0, args.stop)
        if getattr(args, 'language', None):
            app.option_menu_language.set(args.language)
        if getattr(args, 'pause', None):
            app.option_menu_pause.set(args.pause)
        if getattr(args, 'speaker_detection', None):
            app.option_menu_speaker.set(args.speaker_detection)
        if getattr(args, 'overlapping', None) is not None:
            if args.overlapping:
                app.check_box_overlapping.select()
            else:
                app.check_box_overlapping.deselect()
        if getattr(args, 'disfluencies', None) is not None:
            if args.disfluencies:
                app.check_box_disfluencies.select()
            else:
                app.check_box_disfluencies.deselect()
        if getattr(args, 'timestamps', None) is not None:
            if args.timestamps:
                app.check_box_timestamps.select()
            else:
                app.check_box_timestamps.deselect()

        # If both files provided, create a job and auto-start in GUI
        if len(app.audio_files_list) > 0 and len(app.transcript_files_list) > 0:
            # Start the job
            app.create_job()

    except Exception as e:
        # Non-fatal: continue to show GUI
        print(f"Warning: Failed to prefill GUI from CLI args: {e}")

    # Enter GUI main loop
    app.mainloop()
