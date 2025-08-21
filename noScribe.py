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
# In the compiled version (no command line), stdout is None which might lead to errors
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import tkinter as tk
import customtkinter as ctk
# from CTkToolTip import CTkToolTip
from CTkToolTips import CTkToolTip
from tkHyperlinkManager import HyperlinkManager
import webbrowser
from functools import partial
from PIL import Image
import os
import platform
import yaml
import locale
import appdirs
from subprocess import run, call, Popen, PIPE, STDOUT
if platform.system() == 'Windows':
    # import torch.cuda # to check with torch.cuda.is_available()
    from subprocess import STARTUPINFO, STARTF_USESHOWWINDOW
if platform.system() in ("Windows", "Linux"):
    from ctranslate2 import get_cuda_device_count
    import torch
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
import time
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
import multiprocessing
import gc
import traceback
from enum import Enum
from typing import Optional, List

 # Pyinstaller fix, used to open multiple instances on Mac
multiprocessing.freeze_support()

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

app_version = '0.6.2'
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

try:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        if not config:
            raise # config file is empty (None)        
except: # seems we run it for the first time and there is no config file
    config = {}
    
def get_config(key: str, default):
    """ Get a config value, set it if it doesn't exist """
    if key not in config:
        config[key] = default
    return config[key]

    
def version_higher(version1, version2) -> int:
    """Will return 
    1 if version1 is higher
    2 if version2 is higher
    0  if both are equal """
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
    # must be completly equal
    return 0
    
# In versions < 0.4.5 (Windows/Linux only), 'pyannote_xpu' was always set to 'cpu'.
# Delete this so we can determine the optimal value  
if platform.system() in ('Windows', 'Linux'):
    try:
        if version_higher('0.4.5', config['app_version']) == 1:
            del config['pyannote_xpu'] 
    except:
        pass

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
i18n.set('locale', app_locale)
config['locale'] = app_locale

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

def millisec(timeStr: str) -> int:
    """ Convert 'hh:mm:ss' string into milliseconds """
    try:
        h, m, s = timeStr.split(':')
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 # https://stackoverflow.com/a/6402859
    except:
        raise Exception(t('err_invalid_time_string', time = timeStr))

def ms_to_str(milliseconds: float, include_ms=False):
    """ Convert milliseconds into formatted timestamp 'hh:mm:ss' """
    seconds, milliseconds = divmod(milliseconds,1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    formatted = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    if include_ms:
        formatted += f'.{milliseconds:03d}'
    return formatted 

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
    txt = html.escape(txt)
    while txt.find('\n\n') > -1:
        txt = txt.replace('\n\n', '\n')
    return txt    

def ms_to_webvtt(milliseconds) -> str:
    """converts milliseconds to the time stamp of WebVTT (HH:MM:SS.mmm)
    """
    # 1 hour = 3600000 milliseconds
    # 1 minute = 60000 milliseconds
    # 1 second = 1000 milliseconds
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, milliseconds)

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
                start = ms_to_webvtt(int(name_elems[1]))
                end = ms_to_webvtt(int(name_elems[2]))
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
        
        # File paths
        self.audio_file: str = ''
        self.transcript_file: str = ''
        
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
        self.auto_edit_transcript: bool = True
        self.pyannote_xpu: str = 'cpu'
        self.whisper_xpu: str = 'cpu'  # Windows/Linux only
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
    
    def get_duration(self) -> Optional[datetime.timedelta]:
        """Get processing duration if job is completed"""
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None
    
    def is_completed(self) -> bool:
        """Check if job is completed (finished or error)"""
        return self.status in [JobStatus.FINISHED, JobStatus.ERROR]

class TranscriptionQueue:
    """Manages a queue of transcription jobs"""
    
    def __init__(self):
        self.jobs: List[TranscriptionJob] = []
    
    def add_job(self, job: TranscriptionJob):
        """Add a job to the queue"""
        self.jobs.append(job)
    
    def get_waiting_jobs(self) -> List[TranscriptionJob]:
        """Get all jobs with WAITING status"""
        return [job for job in self.jobs if job.status == JobStatus.WAITING]
    
    def get_running_jobs(self) -> List[TranscriptionJob]:
        """Get all jobs currently being processed"""
        return [job for job in self.jobs if job.status in [JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION, JobStatus.TRANSCRIPTION]]
    
    def get_finished_jobs(self) -> List[TranscriptionJob]:
        """Get all successfully completed jobs"""
        return [job for job in self.jobs if job.status == JobStatus.FINISHED]
    
    def get_failed_jobs(self) -> List[TranscriptionJob]:
        """Get all jobs that encountered errors"""
        return [job for job in self.jobs if job.status == JobStatus.ERROR]
    
    def has_pending_jobs(self) -> bool:
        """Check if there are jobs waiting to be processed"""
        return len(self.get_waiting_jobs()) > 0
    
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
            'errors': len(self.get_failed_jobs())
        }
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.jobs) == 0

# Command Line Interface

def create_transcription_job(audio_file=None, transcript_file=None, start_time=None, stop_time=None,
                           language_name=None, whisper_model_name=None, speaker_detection=None,
                           overlapping=None, timestamps=None, disfluencies=None, pause=None,
                           auto_edit_transcript=None, cli_mode=False) -> TranscriptionJob:
    """Create a TranscriptionJob with all default values set in one place.
    
    This function handles both CLI and GUI job creation, ensuring all defaults
    are consistent between both modes.
    """
    job = TranscriptionJob()
    
    # File paths
    job.audio_file = audio_file or ''
    job.transcript_file = transcript_file or ''
    if job.transcript_file:
        job.file_ext = os.path.splitext(job.transcript_file)[1][1:]
    
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
    
    # Auto-edit transcript setting
    if auto_edit_transcript is not None:
        job.auto_edit_transcript = auto_edit_transcript
    elif cli_mode:
        job.auto_edit_transcript = 'False'  # Don't auto-open editor in CLI mode
    else:
        job.auto_edit_transcript = get_config('auto_edit_transcript', 'True')
    
    job.vad_threshold = float(get_config('voice_activity_detection_threshold', '0.5'))
    
    # Platform-specific XPU settings
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
    start_time = millisec(args.start) if args.start else None
    stop_time = millisec(args.stop) if args.stop else None
    
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
    parser.add_argument('--no-gui', action='store_true',
                       help='Run without showing the GUI (headless mode)')
    parser.add_argument('--start', 
                       help='Start time (format: HH:MM:SS)')
    parser.add_argument('--stop',
                       help='Stop time (format: HH:MM:SS)')
    parser.add_argument('--language', 
                       help='Language code (e.g., en, de, fr) or "auto" for auto-detection')
    parser.add_argument('--model',
                       help='Whisper model to use (use --help-models to see available models)')
    parser.add_argument('--speaker-detection', choices=['none', 'auto', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                       help='Speaker detection/diarization setting')
    parser.add_argument('--overlapping', action='store_true',
                       help='Enable overlapping speech detection')
    parser.add_argument('--timestamps', action='store_true',
                       help='Include timestamps in transcript')
    parser.add_argument('--disfluencies', action='store_true', default=True,
                       help='Include disfluencies (uh, um, etc.) in transcript')
    parser.add_argument('--no-disfluencies', action='store_false', dest='disfluencies',
                       help='Exclude disfluencies from transcript')
    parser.add_argument('--pause', choices=['none', '1sec+', '2sec+', '3sec+'],
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

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.user_models_dir = os.path.join(config_dir, 'whisper_models')
        os.makedirs(self.user_models_dir, exist_ok=True)
        whisper_models_readme = os.path.join(self.user_models_dir, 'readme.txt')
        if not os.path.exists(whisper_models_readme):
            with open(whisper_models_readme, 'w') as file:
                file.write('You can download custom Whisper-models for the transcription into this folder. \n' 
                           'See here for more information: https://github.com/kaixxx/noScribe/wiki/Add-custom-Whisper-models-for-transcription')            
        
        self.queue = TranscriptionQueue()
        self.audio_file = ''
        self.transcript_file = ''
        self.log_file = None
        self.cancel = False # if set to True, transcription will be canceled
        # If True, cancel only the currently running job (triggered from queue row "X")
        self._cancel_job_only = False

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
        
        # Start Button
        self.start_button = ctk.CTkButton(self.sidebar_frame, height=42, text=t('start_button'), command=self.button_start_event)
        self.start_button.pack(padx=[20, 0], pady=[20,30], expand=False, fill='x', anchor='sw')

        # Stop Button
        self.stop_button = ctk.CTkButton(self.sidebar_frame, height=42, fg_color='darkred', hover_color='darkred', text=t('stop_button'), command=self.button_stop_event)
        
        # create queue view and log textbox
        self.frame_right = ctk.CTkFrame(self.frame_main, corner_radius=0, fg_color='transparent')
        self.frame_right.pack(padx=0, pady=0, fill='both', expand=True, side='top')
        
        self.tabview = ctk.CTkTabview(self.frame_right, anchor="nw", border_width=2)
        self.tabview.pack(padx=20, pady=[20,20], fill='both', expand=True, side='top')
        self.tab_log = self.tabview.add(t("tab_log")) 
        self.tab_queue = self.tabview.add(t("tab_queue")) 
        self.tabview.set(t("tab_log"))  # set currently visible tab

        self.log_textbox = ctk.CTkTextbox(self.tab_log, wrap='word', state="disabled", font=("",16), text_color="lightgray", bg_color='transparent', fg_color='transparent')
        self.log_textbox.tag_config('highlight', foreground='darkorange')
        self.log_textbox.tag_config('error', foreground='yellow')
        self.log_textbox.pack(padx=0, pady=[0,0], expand=True, fill='both')

        self.hyperlink = HyperlinkManager(self.log_textbox._textbox)

        # Queue table
        self.queue_frame = ctk.CTkFrame(self.tab_queue)
        self.queue_frame.pack(padx=0, pady=0, fill='both', expand=True)
        
        # Queue table header
        #self.queue_header_frame = ctk.CTkFrame(self.queue_frame)
        #self.queue_header_frame.pack(fill='x', padx=0, pady=0)
        
        #self.queue_header_name = ctk.CTkLabel(self.queue_header_frame, text="Name", font=ctk.CTkFont(weight="bold"))
        #self.queue_header_name.pack(side='left', padx=(20, 0))
        
        #self.queue_header_status = ctk.CTkLabel(self.queue_header_frame, text="Status", font=ctk.CTkFont(weight="bold"))
        #self.queue_header_status.pack(side='right', padx=(0, 20))
        
        # Scrollable frame for queue entries
        self.queue_scrollable = ctk.CTkScrollableFrame(self.queue_frame, bg_color='transparent', fg_color='transparent')
        self.queue_scrollable.pack(fill='both', expand=True, padx=0, pady=(0, 0))

        # Mapping for diff-based queue rows (job_key -> widgets)
        self.queue_row_widgets = {}
        
        self.update_queue_table()

        # Frame progress bar / edit button
        self.frame_edit = ctk.CTkFrame(self.frame_main, height=20, corner_radius=0, fg_color=self.log_textbox._fg_color)
        self.frame_edit.pack(padx=20, pady=[0,30], anchor='sw', fill='x', side='bottom')

        # Edit Button
        self.edit_button = ctk.CTkButton(self.frame_edit, fg_color=self.log_textbox._scrollbar_button_color, 
                                         text=t('editor_button'), command=self.launch_editor, width=140)
        self.edit_button.pack(padx=[20,10], pady=[10,10], expand=False, anchor='se', side='right')

        # Progress bar
        self.progress_textbox = ctk.CTkTextbox(self.frame_edit, wrap='none', height=15, state="disabled", font=("",16), text_color="lightgray")
        self.progress_textbox.pack(padx=[10,10], pady=[5,0], expand=True, fill='x', anchor='sw', side='left')

        self.update_scrollbar_visibility()        
        #self.progress_bar = ctk.CTkProgressBar(self.frame_edit, mode='determinate', progress_color='darkred', fg_color=self.log_textbox._fg_color)
        #self.progress_bar.set(0)
        # self.progress_bar.pack(padx=[10,10], pady=[10,10], expand=True, fill='x', anchor='sw', side='left')

        # status bar bottom
        #self.frame_status = ctk.CTkFrame(self, height=20, corner_radius=0)
        #self.frame_status.pack(padx=0, pady=[0,0], anchor='sw', fill='x', side='bottom')

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
                if version_higher(latest_release_version, app_version) == 1:
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
            elif job.status == JobStatus.FINISHED:
                status_color = "lightgreen"
                job_tooltip = t('job_tt_finished')
            elif job.status == JobStatus.ERROR:
                status_color = "yellow"
                msg = job.error_message if job.error_message else ''
                job_tooltip = t('job_tt_error', error_msg=msg)

            status_text = t(str(job.status.value))

            if hasattr(self, 'queue_row_widgets') and job_key in self.queue_row_widgets:
                # Update existing row
                row = self.queue_row_widgets[job_key]
                row['name_label'].configure(text=audio_name)
                row['status_label'].configure(text=status_text, text_color=status_color)

                # Show or hide cancel/delete button depending on status
                is_unfinished = job.status not in [JobStatus.FINISHED, JobStatus.ERROR]
                if 'cancel_btn' in row and row['cancel_btn'] is not None:
                    try:
                        if is_unfinished:
                            row['cancel_btn'].configure(command=lambda j=job: self._on_queue_row_action(j))
                            # Ensure it is visible
                            if not row['cancel_btn'].winfo_ismapped():
                                row['cancel_btn'].pack(side='right', padx=(0, 6), pady=2)
                        else:
                            # Hide the button for finished/error jobs
                            if row['cancel_btn'].winfo_ismapped():
                                row['cancel_btn'].pack_forget()
                    except Exception:
                        pass

                # Update click bindings only on transition to/from FINISHED
                was_finished = row.get('status') == JobStatus.FINISHED
                is_finished = job.status == JobStatus.FINISHED
                if is_finished and not was_finished:
                    def on_click(event, transcript_file=job.transcript_file):
                        if transcript_file and os.path.exists(transcript_file):
                            self.after(100, lambda: self.launch_editor(transcript_file))
                    row['frame'].configure(cursor="hand2")
                    row['name_label'].configure(cursor="hand2")
                    row['status_label'].configure(cursor="hand2")
                    row['frame'].bind("<Button-1>", on_click)
                    row['name_label'].bind("<Button-1>", on_click)
                    row['status_label'].bind("<Button-1>", on_click)
                elif was_finished and not is_finished:
                    row['frame'].configure(cursor="")
                    row['name_label'].configure(cursor="")
                    row['status_label'].configure(cursor="")
                    row['frame'].unbind("<Button-1>")
                    row['name_label'].unbind("<Button-1>")
                    row['status_label'].unbind("<Button-1>")

                row['status'] = job.status
                row['tooltip_text'] = job_tooltip
                # Update tooltip messages if available
                if 'tooltips' in row:
                    for tt in row['tooltips']:
                        tt.set_text(job_tooltip)
            else:
                # Create new row
                entry_frame = ctk.CTkFrame(self.queue_scrollable, fg_color='#4A4A4A')  # #1D1E1E
                entry_frame.pack(fill='x', padx=0, pady=2)

                name_label = ctk.CTkLabel(entry_frame, text=audio_name, anchor='w', text_color="lightgray")
                name_label.pack(side='left', padx=(10, 0), pady=2, fill='x', expand=True)

                # Add small cancel/delete button for unfinished jobs
                cancel_btn = None
                if job.status not in [JobStatus.FINISHED, JobStatus.ERROR]:
                    cancel_btn = ctk.CTkButton(
                        entry_frame,
                        text='X',
                        width=24,
                        height=20,
                        fg_color='#6b6b6b',
                        hover_color='#8a2a2a',
                        command=lambda j=job: self._on_queue_row_action(j)
                    )
                    cancel_btn.pack(side='right', padx=(0, 6), pady=2)

                status_label = ctk.CTkLabel(entry_frame, text=status_text, text_color=status_color, anchor='e')
                status_label.pack(side='right', padx=(0, 10), pady=2)

                # Tooltips (create once per row)
                tt_frame = CTkToolTip(entry_frame, text=job_tooltip) #, bg_color='gray')
                tt_name = CTkToolTip(name_label, text=job_tooltip) #, bg_color='gray')
                tt_status = CTkToolTip(status_label, text=job_tooltip) #, bg_color='gray')
                if cancel_btn is not None:
                    CTkToolTip(cancel_btn, text=t('transcription_canceled'))

                if job.status == JobStatus.FINISHED:
                    entry_frame.configure(cursor="hand2")
                    name_label.configure(cursor="hand2")
                    status_label.configure(cursor="hand2")
                    def on_click(event, transcript_file=job.transcript_file):
                        if transcript_file and os.path.exists(transcript_file):
                            self.after(100, lambda: self.launch_editor(transcript_file))
                    entry_frame.bind("<Button-1>", on_click)
                    name_label.bind("<Button-1>", on_click)
                    status_label.bind("<Button-1>", on_click)

                if not hasattr(self, 'queue_row_widgets'):
                    self.queue_row_widgets = {}
                self.queue_row_widgets[job_key] = {
                    'frame': entry_frame,
                    'name_label': name_label,
                    'status_label': status_label,
                    'status': job.status,
                    'tooltip_text': job_tooltip,
                    'tooltips': [tt_frame, tt_name, tt_status],
                    'cancel_btn': cancel_btn,
                }

        # Remove rows no longer present
        if hasattr(self, 'queue_row_widgets'):
            to_remove = [key for key in list(self.queue_row_widgets.keys()) if key not in current_keys]
            for key in to_remove:
                row = self.queue_row_widgets.pop(key)
                if row['frame'].winfo_exists():
                    row['frame'].destroy()
                    
        # Udate queue tab title
        new_name = f'{t("tab_queue")} ({len(self.queue.get_finished_jobs())}/{len(self.queue.jobs)})'
        old_name = self.tabview._name_list[1]
        if new_name != old_name:
            self.tabview.rename(old_name, new_name)

    def _on_queue_row_action(self, job: TranscriptionJob):
        """Handle click on the small X button for a job row."""
        try:
            if job.status == JobStatus.WAITING:
                # Confirm deletion of waiting job
                if tk.messagebox.askyesno(title='noScribe', message='Remove this job from the queue?'):
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
                    # Only cancel the current job, not the entire queue
                    self._cancel_job_only = True
                    self.cancel = True
            else:
                # Do nothing for finished/error
                pass
        except Exception as e:
            # Log any UI handling error silently
            self.logn(f'Queue action error: {e}', 'error')

    def launch_editor(self, file=''):
        # Launch the editor in a seperate process so that in can stay running even if noScribe quits.
        # Source: https://stackoverflow.com/questions/13243807/popen-waiting-for-child-process-even-when-the-immediate-child-has-terminated/13256908#13256908 
        # set system/version dependent "start_new_session" analogs
        if file == '':
            file = self.transcript_file
        ext = os.path.splitext(self.transcript_file)[1][1:]
        if file != '' and ext != 'html':
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
            if hasattr(self, 'log_textbox') and self.log_textbox.winfo_exists():
                try:
                    self.log_textbox.configure(state=tk.NORMAL)
                    
                    if link:
                        tags = tags + self.hyperlink.add(partial(self.openLink, link))
                    
                    self.log_textbox.insert(tk.END, txt, tags)
                    self.log_textbox.yview_moveto(1)  # Scroll to last line
                    
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
        if where != 'file':
            self.log_textbox.configure(state=ctk.NORMAL)
            self.log_textbox.delete("end-1c linestart", "end-1c")
        self.log(txt, tags, where, link, tb)

    def button_audio_file_event(self):
        fn = tk.filedialog.askopenfilename(initialdir=os.path.dirname(self.audio_file), initialfile=os.path.basename(self.audio_file))
        if fn:
            self.audio_file = fn
            self.logn(t('log_audio_file_selected') + self.audio_file)
            self.button_audio_file_name.configure(text=os.path.basename(self.audio_file))

    def button_transcript_file_event(self):
        if self.transcript_file != '':
            _initialdir = os.path.dirname(self.transcript_file)
            _initialfile = os.path.basename(self.transcript_file)
        else:
            _initialdir = os.path.dirname(self.audio_file)
            _initialfile = Path(os.path.basename(self.audio_file)).stem
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
        fn = tk.filedialog.asksaveasfilename(initialdir=_initialdir, initialfile=_initialfile, 
                                             filetypes=filetypes, 
                                             defaultextension=config['last_filetype'])
        if fn:
            self.transcript_file = fn
            self.logn(t('log_transcript_filename') + self.transcript_file)
            self.button_transcript_file_name.configure(text=os.path.basename(self.transcript_file))
            config['last_filetype'] = os.path.splitext(self.transcript_file)[1][1:]
            
    def set_progress(self, step, value, speaker_detection='none'):
        """ Update state of the progress bar """
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

        # Update progress_textbox
        if progr < 0:
            progr_str = ''
        else:
            progr_str = f'({t("overall_progress")}{round(progr * 100)}%)'
        self.progress_textbox.configure(state=ctk.NORMAL)        
        self.progress_textbox.delete('1.0', tk.END)
        self.progress_textbox.insert(tk.END, progr_str)
        self.progress_textbox.configure(state=ctk.DISABLED)

    def collect_transcription_options(self) -> TranscriptionJob:
        """Collect all transcription options from UI and config into a TranscriptionJob object"""
        # Validate required inputs
        if self.audio_file == '':
            raise ValueError(t('err_no_audio_file'))
        
        if self.transcript_file == '':
            raise ValueError(t('err_no_transcript_file'))
        
        # Parse time range from UI
        start_time = None
        val = self.entry_start.get()
        if val != '':
            start_time = millisec(val)
        
        stop_time = None
        val = self.entry_stop.get()
        if val != '':
            stop_time = millisec(val)
        
        # Get whisper model path
        sel_whisper_model = self.option_menu_whisper_model.get()
        if sel_whisper_model not in self.whisper_model_paths.keys():
            raise FileNotFoundError(f"The whisper model '{sel_whisper_model}' does not exist.")
        whisper_model_path = self.whisper_model_paths[sel_whisper_model]
        
        # Create job using unified function
        job = create_transcription_job(
            audio_file=self.audio_file,
            transcript_file=self.transcript_file,
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
        
        return job


    ################################################################################################
    # Main function

    def transcription_worker(self):
        """Process transcription jobs from the queue"""
        queue_start_time = datetime.datetime.now()
        self.cancel = False

        # Show the stop button
        self.start_button.pack_forget() # hide
        self.stop_button.pack(padx=[20, 0], pady=[20,30], expand=False, fill='x', anchor='sw')

        try:
            # Log queue summary
            summary = self.queue.get_queue_summary()
            self.logn()
            self.logn(t('queue_start'), 'highlight')
            self.logn(t('queue_start_jobs', total=summary['total']))
            # Process each job in the queue
            while self.queue.has_pending_jobs():
                # If global cancel was requested (via Stop button), cancel all waiting jobs
                if self.cancel and not self._cancel_job_only:
                    for job in self.queue.get_waiting_jobs():
                        job.set_error(t('err_user_cancelation'))
                        self.update_queue_table()
                    break
                
                # Get next job
                job = self.queue.get_next_waiting_job()
                if not job:
                    break
                
                # Process the job
                try:
                    self.logn()
                    self.logn(t('start_job', audio_file=os.path.basename(job.audio_file)), 'highlight')
                    
                    # Process single job
                    self._process_single_job(job)
                    
                    job.set_finished()
                    self.update_queue_table()
                    
                except Exception as e:
                    error_msg = job.error_message or str(e)
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
            
            # Log total processing time
            total_time = datetime.datetime.now() - queue_start_time
            total_seconds = "{:02d}".format(int(total_time.total_seconds() % 60))
            total_time_str = f'{int(total_time.total_seconds() // 60)}:{total_seconds}'
            self.logn(f"Total processing time: {total_time_str}")
            
        except Exception as e:
            self.logn(f"Queue processing error: {str(e)}", 'error')
            traceback_str = traceback.format_exc()
            self.logn(f"Queue error details: {traceback_str}", where='file')
        
        finally:
            # Hide the stop button
            self.stop_button.pack_forget() # hide
            self.start_button.pack(padx=[20, 0], pady=[20,30], expand=False, fill='x', anchor='sw')
            # Hide progress
            self.set_progress(0, 0)

    def _process_single_job(self, job: TranscriptionJob):
        """Process a single transcription job"""
        proc_start_time = datetime.datetime.now()
        job.set_running()
        self.update_queue_table()
        
        tmpdir = TemporaryDirectory('noScribe')
        tmp_audio_file = os.path.join(tmpdir.name, 'tmp_audio.wav')
        my_transcript_file = job.transcript_file

        try:
            # Create option info string for logging
            option_info = ''
            if job.start > 0:
                option_info += f'{t("label_start")} {ms_to_str(job.start)} | '
            if job.stop > 0:
                option_info += f'{t("label_stop")} {ms_to_str(job.stop)} | '
            option_info += f'{t("label_language")} {job.language_name} ({languages[job.language_name]}) | '
            option_info += f'{t("label_speaker")} {job.speaker_detection} | '
            option_info += f'{t("label_overlapping")} {job.overlapping} | '
            option_info += f'{t("label_timestamps")} {job.timestamps} | '
            option_info += f'{t("label_disfluencies")} {job.disfluencies} | '
            option_info += f'{t("label_pause")} {job.pause}'

            # Create log file
            if not os.path.exists(f'{config_dir}/log'):
                os.makedirs(f'{config_dir}/log')
            self.log_file = open(f'{config_dir}/log/{Path(my_transcript_file).stem}.log', 'w', encoding="utf-8")

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
                if platform.mac_ver()[0] >= '12.3': # MPS needs macOS 12.3+
                    if job.pyannote_xpu == 'mps':
                        self.logn("macOS version >= 12.3:\nUsing MPS (with PYTORCH_ENABLE_MPS_FALLBACK enabled)", where="file")
                    elif job.pyannote_xpu == 'cpu':
                        self.logn("macOS version >= 12.3:\nUser selected to use CPU (results will be better, but you might wanna make yourself a coffee)", where="file")
                    else:
                        self.logn("macOS version >= 12.3:\nInvalid option for 'pyannote_xpu' in config.yml (should be 'mps' or 'cpu')\nYou might wanna change this\nUsing MPS anyway (with PYTORCH_ENABLE_MPS_FALLBACK enabled)", where="file")
                else:
                    self.logn("macOS version < 12.3:\nMPS not available: Using CPU\nPerformance might be poor\nConsider updating macOS, if possible", where="file")

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
                        # (supresses the terminal, see: https://stackoverflow.com/questions/1813872/running-a-process-in-pythonw-with-popen-without-a-console)
                        startupinfo = STARTUPINFO()
                        startupinfo.dwFlags |= STARTF_USESHOWWINDOW
                        with Popen(ffmpeg_cmd, stdout=PIPE, stderr=STDOUT, bufsize=1,universal_newlines=True,encoding='utf-8', startupinfo=startupinfo) as ffmpeg_proc:
                            for line in ffmpeg_proc.stdout:
                                self.logn('ffmpeg: ' + line)
                    elif platform.system() in ("Darwin", "Linux"):
                        with Popen(ffmpeg_cmd, stdout=PIPE, stderr=STDOUT, bufsize=1,universal_newlines=True,encoding='utf-8') as ffmpeg_proc:
                            for line in ffmpeg_proc.stdout:
                                self.logn('ffmpeg: ' + line)
                    if ffmpeg_proc.returncode > 0:
                        raise Exception(t('err_ffmpeg'))
                    self.logn(t('audio_conversion_finished'))
                    self.set_progress(1, 50, job.speaker_detection)
                except Exception as e:
                    traceback_str = traceback.format_exc()
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
                        current_segment_spkr = f'S{segment["label"][8:]}' # shorten the label: "SPEAKER_01" > "S01"

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
                        self.logn(t('start_identifiying_speakers'), 'highlight')
                        self.logn(t('loading_pyannote'))
                        self.set_progress(1, 100, job.speaker_detection)

                        diarize_output = os.path.join(tmpdir.name, 'diarize_out.yaml')
                        diarize_abspath = 'python ' + os.path.join(app_dir, 'diarize.py')
                        diarize_abspath_win = os.path.join(app_dir, '..', 'diarize.exe')
                        diarize_abspath_mac = os.path.join(app_dir, '..', 'MacOS', 'diarize')
                        diarize_abspath_lin = os.path.join(app_dir, '..', 'diarize')
                        if platform.system() == 'Windows' and os.path.exists(diarize_abspath_win):
                            diarize_abspath = diarize_abspath_win
                        elif platform.system() == 'Darwin' and os.path.exists(diarize_abspath_mac): # = MAC
                            diarize_abspath = diarize_abspath_mac
                        elif platform.system() == 'Linux' and os.path.exists(diarize_abspath_lin):
                            diarize_abspath = diarize_abspath_lin
                        diarize_cmd = f'{diarize_abspath} {job.pyannote_xpu} "{tmp_audio_file}" "{diarize_output}" {job.speaker_detection}'
                        diarize_env = None
                        if job.pyannote_xpu == 'mps':
                            diarize_env = os.environ.copy()
                            diarize_env["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1) # Necessary since some operators are not implemented for MPS yet.
                        self.logn(diarize_cmd, where='file')

                        if platform.system() == 'Windows':
                            # (supresses the terminal, see: https://stackoverflow.com/questions/1813872/running-a-process-in-pythonw-with-popen-without-a-console)
                            startupinfo = STARTUPINFO()
                            startupinfo.dwFlags |= STARTF_USESHOWWINDOW
                        elif platform.system() in ('Darwin', "Linux"): # = MAC
                            diarize_cmd = shlex.split(diarize_cmd)
                            startupinfo = None
                        else:
                            raise Exception('Platform not supported yet.')

                        with Popen(diarize_cmd,
                                   stdout=PIPE,
                                   stderr=STDOUT,
                                   encoding='UTF-8',
                                   startupinfo=startupinfo,
                                   env=diarize_env,
                                   close_fds=True) as pyannote_proc:
                            for line in pyannote_proc.stdout:
                                if self.cancel:
                                    pyannote_proc.kill()
                                    raise Exception(t('err_user_cancelation')) 
                                print(line)
                                if line.startswith('progress '):
                                    progress = line.split()
                                    step_name = progress[1]
                                    progress_percent = int(progress[2])
                                    self.logr(f'{step_name}: {progress_percent}%')                       
                                    if step_name == 'segmentation':
                                        self.set_progress(2, progress_percent * 0.3, job.speaker_detection)
                                    elif step_name == 'embeddings':
                                        self.set_progress(2, 30 + (progress_percent * 0.7), job.speaker_detection)
                                elif line.startswith('error '):
                                    self.logn('PyAnnote error: ' + line[5:], 'error')
                                elif line.startswith('log: '):
                                    self.logn('PyAnnote ' + line, where='file')
                                    if line.strip() == "log: 'pyannote_xpu: cpu' was set.": # The string needs to be the same as in diarize.py `print("log: 'pyannote_xpu: cpu' was set.")`.
                                        job.pyannote_xpu = 'cpu'
                                        config['pyannote_xpu'] = 'cpu'

                        if pyannote_proc.returncode > 0:
                            raise Exception('')

                        # load diarization results
                        with open(diarize_output, 'r') as file:
                            diarization = yaml.safe_load(file)

                        # write segments to log file 
                        for segment in diarization:
                            line = f'{ms_to_str(job.start + segment["start"], include_ms=True)} - {ms_to_str(job.start + segment["end"], include_ms=True)} {segment["label"]}'
                            self.logn(line, where='file')

                        self.logn()

                    except Exception as e:
                        traceback_str = traceback.format_exc()
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

                s.appendText(f'({html.escape(option_info)})')

                p.appendChild(s)
                main_body.appendChild(p)

                p = d.createElement('p')
                main_body.appendChild(p)

                speaker = ''
                prev_speaker = ''
                last_auto_save = datetime.datetime.now()

                def save_doc():
                    nonlocal my_transcript_file  # Tell Python we’re modifying the outer variable
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
                            with open(my_transcript_file, 'w', encoding="utf-8") as f:
                                f.write(txt)
                                f.flush()
                            last_auto_save = datetime.datetime.now()
                    except Exception as e:
                        # other error while saving, maybe the file is already open in Word and cannot be overwritten
                        # try saving to a different filename
                        transcript_path = Path(my_transcript_file)
                        my_transcript_file = f'{transcript_path.parent}/{transcript_path.stem}_1{job.file_ext}'
                        if os.path.exists(my_transcript_file):
                            # the alternative filename also exists already, don't want to overwrite, giving up
                            raise Exception(t('rescue_saving_failed'))
                        else:
                            # htmlStr = d.asHTML()
                            with open(my_transcript_file, 'w', encoding="utf-8") as f:
                                f.write(txt)
                                f.flush()
                            self.logn()
                            self.logn(t('rescue_saving', file=my_transcript_file), 'error', link=f'file://{my_transcript_file}')
                            last_auto_save = datetime.datetime.now()

                from faster_whisper import WhisperModel
                if platform.system() == "Darwin": # = MAC
                    whisper_device = 'auto'
                elif platform.system() in ('Windows', 'Linux'):
                    whisper_device = 'cpu'
                    whisper_device = job.whisper_xpu
                else:
                    raise Exception('Platform not supported yet.')
                model = WhisperModel(job.whisper_model,
                                        device=whisper_device,  
                                        cpu_threads=number_threads, 
                                        compute_type=job.whisper_compute_type, 
                                        local_files_only=True)
                self.logn('model loaded', where='file')

                if self.cancel:
                    raise Exception(t('err_user_cancelation')) 

                multilingual = False
                if job.language_name == 'Multilingual':
                    multilingual = True
                    whisper_lang = None
                elif job.language_name == 'Auto':
                    whisper_lang = None
                else:
                    whisper_lang = languages[job.language_name]
                
                # VAD 
                    
                try:
                    job.vad_threshold = float(config['voice_activity_detection_threshold'])
                except:
                    config['voice_activity_detection_threshold'] = '0.5'
                    job.vad_threshold = 0.5                     

                sampling_rate = model.feature_extractor.sampling_rate
                audio = decode_audio(tmp_audio_file, sampling_rate=sampling_rate)
                duration = audio.shape[0] / sampling_rate
                
                self.logn('Voice Activity Detection')
                try:
                    vad_parameters = VadOptions(min_silence_duration_ms=1000, 
                                            threshold=job.vad_threshold,
                                            speech_pad_ms=0)
                except TypeError:
                    # parameter threshold was temporarily renamed to 'onset' in pyannote 3.1:  
                    vad_parameters = VadOptions(min_silence_duration_ms=1000, 
                                            onset=job.vad_threshold,
                                            speech_pad_ms=0)
                speech_chunks = get_speech_timestamps(audio, vad_parameters)
                
                def adjust_for_pause(segment):
                    """Adjusts start and end of segment if it falls into a pause 
                    identified by the VAD"""
                    pause_extend = 0.2  # extend the pauses by 200ms to make the detection more robust
                    
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
                    
                    return segment
                
                # transcribe
                
                if self.cancel:
                    raise Exception(t('err_user_cancelation')) 

                vad_parameters.speech_pad_ms = 400

                # detect language                    
                if job.language_name == 'auto':
                    language, language_probability, all_language_probs = model.detect_language(
                        audio,
                        vad_filter=True,
                        vad_parameters=vad_parameters
                    )
                    self.logn("Detected language '%s' with probability %f" % (language, language_probability))
                    whisper_lang = language

                if job.disfluencies:                    
                    try:
                        with open(os.path.join(app_dir, 'prompt.yml'), 'r', encoding='utf-8') as file:
                            prompts = yaml.safe_load(file)
                    except:
                        prompts = {}
                    prompt = prompts.get(whisper_lang, '') # Fetch language prompt, default to empty string
                else:
                    prompt = ''
                
                del audio
                gc.collect()
                
                segments, info = model.transcribe(
                    tmp_audio_file, # audio, 
                    language=whisper_lang,
                    multilingual=multilingual, 
                    beam_size=5, 
                    #temperature=job.whisper_temperature, 
                    word_timestamps=True, 
                    #initial_prompt=prompt,
                    hotwords=prompt, 
                    vad_filter=True,
                    vad_parameters=vad_parameters,
                    # length_penalty=0.5
                )

                if self.cancel:
                    raise Exception(t('err_user_cancelation')) 

                self.logn(t('start_transcription'))
                self.logn()

                last_segment_end = 0
                last_timestamp_ms = 0
                first_segment = True

                for segment in segments:
                    # check for user cancelation
                    if self.cancel:
                        if job.auto_save:
                            save_doc()
                            self.logn()
                            self.log(t('transcription_saved'))
                            self.logn(my_transcript_file, link=f'file://{my_transcript_file}')

                        raise Exception(t('err_user_cancelation')) 

                    segment = adjust_for_pause(segment)

                    # get time of the segment in milliseconds
                    start = round(segment.start * 1000.0)
                    end = round(segment.end * 1000.0)
                    # if we skipped a part at the beginning of the audio we have to add this here again, otherwise the timestaps will not match the original audio:
                    orig_audio_start = job.start + start
                    orig_audio_end = job.start + end

                    if job.timestamps:
                        ts = ms_to_str(orig_audio_start)
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
                    seg_html = html.escape(seg_text)

                    if job.speaker_detection != 'none':
                        new_speaker = find_speaker(diarization, start, end)
                        if (speaker != new_speaker) and (new_speaker != ''): # speaker change
                            if new_speaker[:2] == '//': # is overlapping speech, create no new paragraph
                                prev_speaker = speaker
                                speaker = new_speaker
                                seg_text = f' {speaker}:{seg_text}'
                                seg_html = html.escape(seg_text)                                
                            elif (speaker[:2] == '//') and (new_speaker == prev_speaker): # was overlapping speech and we are returning to the previous speaker 
                                speaker = new_speaker
                                seg_text = f'//{seg_text}'
                                seg_html = html.escape(seg_text)
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
                                    seg_html = f'{speaker}: <span style="color: {job.timestamp_color}" >{ts}</span>{html.escape(seg_text)}'
                                    seg_text = f'{speaker}: {ts}{seg_text}'
                                    last_timestamp_ms = start
                                else:
                                    if job.file_ext != 'vtt': # in vtt files, speaker names are added as special voice tags so skip this here
                                        seg_text = f'{speaker}:{seg_text}'
                                        seg_html = html.escape(seg_text)
                                    else:
                                        seg_html = html.escape(seg_text).lstrip()
                                        seg_text = f'{speaker}:{seg_text}'
                                    
                        else: # same speaker
                            if job.timestamps:
                                if (start - last_timestamp_ms) > job.timestamp_interval:
                                    seg_html = f' <span style="color: {job.timestamp_color}" >{ts}</span>{html.escape(seg_text)}'
                                    seg_text = f' {ts}{seg_text}'
                                    last_timestamp_ms = start
                                else:
                                    seg_html = html.escape(seg_text)

                    else: # no speaker detection
                        if job.timestamps and (first_segment or (start - last_timestamp_ms) > job.timestamp_interval):
                            seg_html = f' <span style="color: {job.timestamp_color}" >{ts}</span>{html.escape(seg_text)}'
                            seg_text = f' {ts}{seg_text}'
                            last_timestamp_ms = start
                        else:
                            seg_html = html.escape(seg_text)
                        # avoid leading whitespace in first paragraph
                        if first_segment:
                            seg_text = seg_text.lstrip()
                            seg_html = seg_html.lstrip()

                    # Mark confidence level (not implemented yet in html)
                    # cl_level = round((segment.avg_logprob + 1) * 10)
                    # TODO: better cl_level for words based on https://github.com/Softcatala/whisper-ctranslate2/blob/main/src/whisper_ctranslate2/transcribe.py
                    # if cl_level > 0:
                    #     r.style = d.styles[f'noScribe_cl{cl_level}']

                    # Create bookmark with audio timestamps start to end and add the current segment.
                    # This way, we can jump to the according audio position and play it later in the editor.
                    a_html = f'<a name="ts_{orig_audio_start}_{orig_audio_end}_{speaker}" >{seg_html}</a>'
                    a = d.createElementFromHTML(a_html)
                    p.appendChild(a)

                    self.log(seg_text)

                    first_segment = False

                    # auto save
                    if job.auto_save:
                        if (datetime.datetime.now() - last_auto_save).total_seconds() > 20:
                            save_doc()

                    progr = round((segment.end/info.duration) * 100)
                    self.set_progress(3, progr, job.speaker_detection)

                save_doc()
                self.logn()
                self.logn()
                self.logn(t('transcription_finished'), 'highlight')
                if job.transcript_file != my_transcript_file: # used alternative filename because saving under the initial name failed
                    self.log(t('rescue_saving'))
                    self.logn(my_transcript_file, link=f'file://{my_transcript_file}')
                else:
                    self.log(t('transcription_saved'))
                    self.logn(my_transcript_file, link=f'file://{my_transcript_file}')
                # log duration of the whole process
                proc_time = datetime.datetime.now() - proc_start_time
                proc_seconds = "{:02d}".format(int(proc_time.total_seconds() % 60))
                proc_time_str = f'{int(proc_time.total_seconds() // 60)}:{proc_seconds}' 
                self.logn(t('trancription_time', duration=proc_time_str)) 

                # auto open transcript in editor
                if (job.auto_edit_transcript == 'True') and (job.file_ext == 'html'):
                    self.launch_editor(my_transcript_file)
            
            finally:
                self.log_file.close()
                self.log_file = None

        finally:
            # hide the stop button
            self.stop_button.pack_forget() # hide
            self.start_button.pack(padx=[20, 0], pady=[20,30], expand=False, fill='x', anchor='sw')

            # hide progress
            self.set_progress(0, 0)
            
    def button_start_event(self):
        try:
            # Collect transcription options from UI
            job = self.collect_transcription_options()
            
            # Add the job to the queue
            self.queue.add_job(job)
            self.update_queue_table()
            
            # Start transcription worker with the queue
            wkr = Thread(target=self.transcription_worker, args=())
            wkr.start()
            
        except (ValueError, FileNotFoundError) as e:
            # Handle validation errors from collect_transcription_options
            self.logn(str(e), 'error')
            tk.messagebox.showerror(title='noScribe', message=str(e))
        except Exception as e:
            # Handle unexpected errors
            self.logn(f'Error starting transcription: {str(e)}', 'error')
            tk.messagebox.showerror(title='noScribe', message=f'Error starting transcription: {str(e)}')
    
    # End main function Button Start        
    ################################################################################################

    def button_stop_event(self):
        if tk.messagebox.askyesno(title='noScribe', message=t('transcription_canceled')) == True:
            self.logn()
            self.logn(t('start_canceling'))
            self.update()
            self.cancel = True

    def on_closing(self):
        # (see: https://stackoverflow.com/questions/111155/how-do-i-handle-the-window-close-event-in-tkinter)
        #if messagebox.askokcancel("Quit", "Do you want to quit?"):
        try:
            # remember some settings for the next run
            config['last_language'] = self.option_menu_language.get()
            config['last_speaker'] = self.option_menu_speaker.get()
            config['last_whisper_model'] = self.option_menu_whisper_model.get()
            config['last_pause'] = self.option_menu_pause.get()
            config['last_overlapping'] = self.check_box_overlapping.get()
            config['last_timestamps'] = self.check_box_timestamps.get()
            config['last_disfluencies'] = self.check_box_disfluencies.get()

            save_config()
        finally:
            self.destroy()

def run_cli_mode(args):
    """Run noScribe in CLI mode"""
    try:
        # Create a minimal app instance to access model paths and logging
        app = App()
        # Hide GUI window for headless execution
        try:
            app.withdraw()
        except Exception:
            pass
        
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
            print(f"\nTranscription failed!")
            failed_jobs = app.queue.get_failed_jobs()
            if failed_jobs:
                print(f"Error: {failed_jobs[0].error_message}")
            return 1
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

def show_available_models():
    """Show available Whisper models"""
    try:
        # Create minimal app instance to get models
        app = App()
        models = app.get_whisper_models()
        
        print("Available Whisper models:")
        for model in models:
            print(f"  - {model}")
        
        if not models:
            print("  No models found. Please check your installation.")
            
    except Exception as e:
        print(f"Error getting models: {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_cli_args()

    # Handle special case: show available models
    if args.help_models:
        show_available_models()
        sys.exit(0)

    # If explicit headless requested, keep old CLI behavior
    if getattr(args, 'no_gui', False):
        if args.audio_file and args.output_file:
            exit_code = run_cli_mode(args)
            sys.exit(exit_code)
        else:
            print("Error: --no-gui requires both audio_file and output_file.")
            print("Usage: python noScribe.py <audio_file> <output_file> [options] --no-gui")
            sys.exit(1)

    # Default: show GUI and keep it usable, even with CLI args
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
                print(f"Warning: Model '{args.model}' not found. Using default GUI selection.")

        if desired_model_name:
            try:
                app.option_menu_whisper_model.set(desired_model_name)
            except Exception:
                pass

        # Prefill files if provided
        if getattr(args, 'audio_file', None):
            app.audio_file = args.audio_file
            try:
                app.button_audio_file_name.configure(text=os.path.basename(app.audio_file))
            except Exception:
                pass
            app.logn(t('log_audio_file_selected') + app.audio_file)

        if getattr(args, 'output_file', None):
            app.transcript_file = args.output_file
            try:
                app.button_transcript_file_name.configure(text=os.path.basename(app.transcript_file))
            except Exception:
                pass
            app.logn(t('log_transcript_filename') + app.transcript_file)

        # If both files provided, create a job and auto-start in GUI
        if getattr(args, 'audio_file', None) and getattr(args, 'output_file', None):
            # Build job from args but use GUI defaults (not headless)
            start_time = millisec(args.start) if getattr(args, 'start', None) else None
            stop_time = millisec(args.stop) if getattr(args, 'stop', None) else None

            # Resolve model path: preferred from CLI if available, otherwise from current GUI selection
            model_name = desired_model_name or app.option_menu_whisper_model.get()
            if model_name in getattr(app, 'whisper_model_paths', {}):
                model_path = app.whisper_model_paths[model_name]
            else:
                # Ensure model paths are populated
                app.get_whisper_models()
                model_path = app.whisper_model_paths.get(model_name, None)

            job = create_transcription_job(
                audio_file=args.audio_file,
                transcript_file=args.output_file,
                start_time=start_time,
                stop_time=stop_time,
                language_name=getattr(args, 'language', None),
                whisper_model_name=model_path if model_path else None,
                speaker_detection=getattr(args, 'speaker_detection', None),
                overlapping=getattr(args, 'overlapping', None),
                timestamps=getattr(args, 'timestamps', None),
                disfluencies=getattr(args, 'disfluencies', None),
                pause=getattr(args, 'pause', None),
                cli_mode=False,
            )

            # Add job to queue and start worker thread
            app.queue.add_job(job)
            try:
                app.update_queue_table()
            except Exception:
                pass
            wkr = Thread(target=app.transcription_worker, args=())
            wkr.start()
    except Exception as e:
        # Non-fatal: continue to show GUI
        print(f"Warning: Failed to prefill GUI from CLI args: {e}")

    # Enter GUI main loop
    app.mainloop()
