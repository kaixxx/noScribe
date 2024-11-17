# noScribe - AI-powered Audio Transcription
# Copyright (C) 2023 Kai Dröge
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
# In the compiled version (no command line), stdout is None which might lead to errors
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import tkinter as tk
import customtkinter as ctk
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
import re
if platform.system() == "Darwin": # = MAC
    from subprocess import check_output
    if platform.machine() == "x86_64":
        os.environ['KMP_DUPLICATE_LIB_OK']='True' # prevent OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.
    # import torch.backends.mps # loading torch modules leads to segmentation fault later
import AdvancedHTMLParser
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

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

app_version = '0.5'
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
    number_threads = cpufeature.CPUFeature["num_physical_cores"]
elif platform.system() == "Linux":
    number_threads = os.cpu_count()
    number_threads = 4 if number_threads is None else number_threads
elif platform.system() == "Darwin": # = MAC
    if platform.machine() == "arm64":
        cpu_count = int(check_output(["sysctl", "-n", "hw.perflevel0.logicalcpu_max"]))
    elif platform.machine() == "x86_64":
        cpu_count = int(check_output(["sysctl", "-n", "hw.logicalcpu_max"]))
    else:
        raise Exception("Unsupported mac")
    number_threads = int(cpu_count * 0.75)
else:
    raise Exception('Platform not supported yet.')

# timestamp regex
timestamp_re = re.compile('\[\d\d:\d\d:\d\d.\d\d\d --> \d\d:\d\d:\d\d.\d\d\d\]')

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
        return node
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
    txt = txt.replace('&', '&amp;')
    txt = txt.replace('<', '&lt;')
    txt = txt.replace('>', '&gt;')
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
    
class TimeEntry(ctk.CTkEntry): # special Entry box to enter time in the format hh:mm:ss
                               # based on https://stackoverflow.com/questions/63622880/how-to-make-python-automatically-put-colon-in-the-format-of-time-hhmmss
    def __init__(self, master, **kwargs):
        ctk.CTkEntry.__init__(self, master, **kwargs)
        vcmd = self.register(self.validate)

        self.bind('<Key>', self.format)
        self.configure(validate="all", validatecommand=(vcmd, '%P'))

        self.valid = re.compile('^\d{0,2}(:\d{0,2}(:\d{0,2})?)?$', re.I)

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

        self.audio_file = ''
        self.transcript_file = ''
        self.log_file = None
        self.cancel = False # if set to True, transcription will be canceled

        # configure window
        self.title('noScribe - ' + t('app_header'))
        if platform.system() in ("Darwin", "Linux"):
            self.geometry(f"{1100}x{725}")
        else:
            self.geometry(f"{1100}x{650}")
        if platform.system() in ("Darwin", "Windows"):
            self.iconbitmap('noScribeLogo.ico')
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
        self.sidebar_frame = ctk.CTkFrame(self.frame_main, width=270, corner_radius=0, fg_color='transparent')
        self.sidebar_frame.pack(padx=0, pady=0, fill='y', expand=False, side='left')

        # input audio file
        self.label_audio_file = ctk.CTkLabel(self.sidebar_frame, text=t('label_audio_file'))
        self.label_audio_file.pack(padx=20, pady=[20,0], anchor='w')

        self.frame_audio_file = ctk.CTkFrame(self.sidebar_frame, width=250, height=33, corner_radius=8, border_width=2)
        self.frame_audio_file.pack(padx=20, pady=[0,10], anchor='w')

        self.button_audio_file_name = ctk.CTkButton(self.frame_audio_file, width=200, corner_radius=8, bg_color='transparent', 
                                                    fg_color='transparent', hover_color=self.frame_audio_file._bg_color, 
                                                    border_width=0, anchor='w',  
                                                    text=t('label_audio_file_name'), command=self.button_audio_file_event)
        self.button_audio_file_name.place(x=3, y=3)

        self.button_audio_file = ctk.CTkButton(self.frame_audio_file, width=45, height=29, text='📂', command=self.button_audio_file_event)
        self.button_audio_file.place(x=203, y=2)

        # input transcript file name
        self.label_transcript_file = ctk.CTkLabel(self.sidebar_frame, text=t('label_transcript_file'))
        self.label_transcript_file.pack(padx=20, pady=[10,0], anchor='w')

        self.frame_transcript_file = ctk.CTkFrame(self.sidebar_frame, width=250, height=33, corner_radius=8, border_width=2)
        self.frame_transcript_file.pack(padx=20, pady=[0,10], anchor='w')

        self.button_transcript_file_name = ctk.CTkButton(self.frame_transcript_file, width=200, corner_radius=8, bg_color='transparent', 
                                                    fg_color='transparent', hover_color=self.frame_transcript_file._bg_color, 
                                                    border_width=0, anchor='w',  
                                                    text=t('label_transcript_file_name'), command=self.button_transcript_file_event)
        self.button_transcript_file_name.place(x=3, y=3)

        self.button_transcript_file = ctk.CTkButton(self.frame_transcript_file, width=45, height=29, text='📂', command=self.button_transcript_file_event)
        self.button_transcript_file.place(x=203, y=2)

        # Options grid
        self.frame_options = ctk.CTkFrame(self.sidebar_frame, width=250, fg_color='transparent')
        self.frame_options.pack(padx=20, pady=10, anchor='w', fill='x')

        self.frame_options.grid_columnconfigure(0, weight=1)
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

        self.langs = ('auto', 'en (english)', 'zh (chinese)', 'de (german)', 'es (spanish)', 'ru (russian)', 'ko (korean)', 'fr (french)', 'ja (japanese)', 'pt (portuguese)', 'tr (turkish)', 'pl (polish)', 'ca (catalan)', 'nl (dutch)', 'ar (arabic)', 'sv (swedish)', 'it (italian)', 'id (indonesian)', 'hi (hindi)', 'fi (finnish)', 'vi (vietnamese)', 'he (hebrew)', 'uk (ukrainian)', 'el (greek)', 'ms (malay)', 'cs (czech)', 'ro (romanian)', 'da (danish)', 'hu (hungarian)', 'ta (tamil)', 'no (norwegian)', 'th (thai)', 'ur (urdu)', 'hr (croatian)', 'bg (bulgarian)', 'lt (lithuanian)', 'la (latin)', 'mi (maori)', 'ml (malayalam)', 'cy (welsh)', 'sk (slovak)', 'te (telugu)', 'fa (persian)', 'lv (latvian)', 'bn (bengali)', 'sr (serbian)', 'az (azerbaijani)', 'sl (slovenian)', 'kn (kannada)', 'et (estonian)', 'mk (macedonian)', 'br (breton)', 'eu (basque)', 'is (icelandic)', 'hy (armenian)', 'ne (nepali)', 'mn (mongolian)', 'bs (bosnian)', 'kk (kazakh)', 'sq (albanian)', 'sw (swahili)', 'gl (galician)', 'mr (marathi)', 'pa (punjabi)', 'si (sinhala)', 'km (khmer)', 'sn (shona)', 'yo (yoruba)', 'so (somali)', 'af (afrikaans)', 'oc (occitan)', 'ka (georgian)', 'be (belarusian)', 'tg (tajik)', 'sd (sindhi)', 'gu (gujarati)', 'am (amharic)', 'yi (yiddish)', 'lo (lao)', 'uz (uzbek)', 'fo (faroese)', 'ht (haitian   creole)', 'ps (pashto)', 'tk (turkmen)', 'nn (nynorsk)', 'mt (maltese)', 'sa (sanskrit)', 'lb (luxembourgish)', 'my (myanmar)', 'bo (tibetan)', 'tl (tagalog)', 'mg (malagasy)', 'as (assamese)', 'tt (tatar)', 'haw (hawaiian)', 'ln (lingala)', 'ha (hausa)', 'ba (bashkir)', 'jw (javanese)', 'su (sundanese)')

        self.option_menu_language = ctk.CTkOptionMenu(self.frame_options, width=100, values=self.langs, dynamic_resizing=False)
        self.option_menu_language.grid(column=1, row=2, sticky='e', pady=5)
        self.option_menu_language.set(get_config('last_language', 'auto'))
        
        # Quality (Model Selection)
        self.label_quality = ctk.CTkLabel(self.frame_options, text=t('label_quality'))
        self.label_quality.grid(column=0, row=3, sticky='w', pady=5)

        self.option_menu_quality = ctk.CTkOptionMenu(self.frame_options, width=100, values=['precise', 'fast'])
        self.option_menu_quality.grid(column=1, row=3, sticky='e', pady=5)
        self.option_menu_quality.set(get_config('last_quality', 'precise'))

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

        # Timestamps in text
        self.label_timestamps = ctk.CTkLabel(self.frame_options, text=t('label_timestamps'))
        self.label_timestamps.grid(column=0, row=7, sticky='w', pady=5)

        self.check_box_timestamps = ctk.CTkCheckBox(self.frame_options, text = '')
        self.check_box_timestamps.grid(column=1, row=7, sticky='e', pady=5)
        check_box_timestamps = config.get('last_timestamps', False)
        if check_box_timestamps:
            self.check_box_timestamps.select()
        else:
            self.check_box_timestamps.deselect()

        # Start Button
        self.start_button = ctk.CTkButton(self.sidebar_frame, height=42, text=t('start_button'), command=self.button_start_event)
        self.start_button.pack(padx=20, pady=[0,30], expand=True, fill='x', anchor='sw')

        # Stop Button
        self.stop_button = ctk.CTkButton(self.sidebar_frame, height=42, fg_color='darkred', hover_color='darkred', text=t('stop_button'), command=self.button_stop_event)
        
        # create log textbox
        self.log_frame = ctk.CTkFrame(self.frame_main, corner_radius=0, fg_color='transparent')
        self.log_frame.pack(padx=0, pady=0, fill='both', expand=True, side='top')

        self.log_textbox = ctk.CTkTextbox(self.log_frame, wrap='word', state="disabled", font=("",16), text_color="lightgray")
        self.log_textbox.tag_config('highlight', foreground='darkorange')
        self.log_textbox.tag_config('error', foreground='yellow')
        self.log_textbox.pack(padx=20, pady=[20,0], expand=True, fill='both')

        self.hyperlink = HyperlinkManager(self.log_textbox._textbox)

        # Frame progress bar / edit button
        self.frame_edit = ctk.CTkFrame(self.frame_main, height=20, corner_radius=0, fg_color=self.log_textbox._fg_color)
        self.frame_edit.pack(padx=20, pady=[0,30], anchor='sw', fill='x', side='bottom')

        # Edit Button
        self.edit_button = ctk.CTkButton(self.frame_edit, fg_color=self.log_textbox._scrollbar_button_color, 
                                         text=t('editor_button'), command=self.launch_editor, width=140)
        self.edit_button.pack(padx=[20,10], pady=[10,10], expand=False, anchor='se', side='right')

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.frame_edit, mode='determinate', progress_color='darkred', fg_color=self.log_textbox._fg_color)
        self.progress_bar.set(0)
        # self.progress_bar.pack(padx=[0,10], pady=[10,10], expand=True, fill='x', anchor='sw', side='left')

        # status bar bottom
        #self.frame_status = ctk.CTkFrame(self, height=20, corner_radius=0)
        #self.frame_status.pack(padx=0, pady=[0,0], anchor='sw', fill='x', side='bottom')

        self.logn(t('welcome_message'), 'highlight')
        self.log(t('welcome_credits', v=app_version))
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
            program = os.path.join(os.sep, 'Applications', 'noScribeEdit.app', 'Contents', 'MacOS', 'noScribeEdit')
        elif platform.system() == "Linux":
            if hasattr(sys, "_MEIPASS"):
                program = os.path.join(sys._MEIPASS, 'noScribeEdit', "noScribeEdit")
            else:
                program = os.path.join(app_dir, 'noScribeEdit', "noScribeEdit")
        kwargs = {}
        if platform.system() == 'Windows':
            # from msdn [1]
            CREATE_NEW_PROCESS_GROUP = 0x00000200  # note: could get it from subprocess
            DETACHED_PROCESS = 0x00000008          # 0x8 | 0x200 == 0x208
            kwargs.update(creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)  
        else:  # should work on all POSIX systems, Linux and macOS 
            kwargs.update(start_new_session=True)

        if program is not None and os.path.exists(program):
            if file != '':
                Popen([program, file], **kwargs)
            else:
                Popen([program], **kwargs)
        else:
            self.logn(t('err_noScribeEdit_not_found'), 'error')

    def openLink(self, link: str) -> None:
        if link.startswith('file://') and link.endswith('.html'):
            self.launch_editor(link[7:])
        else: 
            webbrowser.open(link)

    def log(self, txt: str = '', tags: list = [], where: str = 'both', link: str = '') -> None:
        """ Log to main window (where can be 'screen', 'file', or 'both') """
        if where != 'file':
            self.log_textbox.configure(state=ctk.NORMAL)
            if link != '':
                tags = tags + self.hyperlink.add(partial(self.openLink, link))
            self.log_textbox.insert(ctk.END, txt, tags)
            self.log_textbox.yview_moveto(1) # scroll to last line
            self.log_textbox.configure(state=ctk.DISABLED)
        if (where != 'screen') and (self.log_file != None) and (self.log_file.closed == False):
            if tags == 'error':
                txt = f'ERROR: {txt}'
            self.log_file.write(txt)
            self.log_file.flush()

    def logn(self, txt: str = '', tags: list = [], where: str = 'both', link:str = '') -> None:
        """ Log with a newline appended """
        self.log(f'{txt}\n', tags, where, link)

    def logr(self, txt: str = '', tags: list = [], where: str = 'both', link:str = '') -> None:
        """ Replace the last line of the log """
        if where != 'file':
            self.log_textbox.configure(state=ctk.NORMAL)
            self.log_textbox.delete("end-1c linestart", "end-1c")
        self.log(txt, tags, where, link)

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
            
    def set_progress(self, step, value):
        """ Update state of the progress bar """
        if step == 1:
            self.progress_bar.set(value * 0.05 / 100)
        elif step == 2:
            progr = 0.05 # (step 1)
            progr = progr + (value * 0.45 / 100)
            self.progress_bar.set(progr)
        elif step == 3:
            if self.speaker_detection == 'auto':
                progr = 0.05 + 0.45 # (step 1 + step 2)
                progr_factor = 0.5
            else:
                progr = 0.05 # (step 1)
                progr_factor = 0.95
            progr = progr + (value * progr_factor / 100)
            self.progress_bar.set(progr)
        else:
            self.progress_bar.set(0)


    ################################################################################################
    # Main function

    def transcription_worker(self):
        # This is the main function where all the magic happens
        # We put this in a seperate thread so that it does not block the main ui

        proc_start_time = datetime.datetime.now()
        self.cancel = False

        # Show the stop button
        self.start_button.pack_forget() # hide
        self.stop_button.pack(padx=20, pady=[0,30], expand=True, fill='x', anchor='sw')

        # Show the progress bar
        self.progress_bar.set(0)
        self.progress_bar.pack(padx=[10,10], pady=[10,10], expand=True, fill='x', anchor='sw', side='left')
        # self.progress_bar.pack(padx=[0,10], pady=[10,25], expand=True, fill='x', anchor='sw', side='left')
        # self.progress_bar.pack(padx=20, pady=[10,20], expand=True, fill='both')

        tmpdir = TemporaryDirectory('noScribe')
        self.tmp_audio_file = os.path.join(tmpdir.name, 'tmp_audio.wav')

        try:
            # collect all the options
            option_info = ''

            if self.audio_file == '':
                self.logn(t('err_no_audio_file'), 'error')
                tk.messagebox.showerror(title='noScribe', message=t('err_no_audio_file'))
                return

            if self.transcript_file == '':
                self.logn(t('err_no_transcript_file'), 'error')
                tk.messagebox.showerror(title='noScribe', message=t('err_no_transcript_file'))
                return

            self.my_transcript_file = self.transcript_file
            self.file_ext = os.path.splitext(self.my_transcript_file)[1][1:]

            # create log file
            if not os.path.exists(f'{config_dir}/log'):
                os.makedirs(f'{config_dir}/log')
            self.log_file = open(f'{config_dir}/log/{Path(self.my_transcript_file).stem}.log', 'w', encoding="utf-8")

            # options for faster-whisper
            self.whisper_precise_beam_size = get_config('whisper_precise_beam_size', 1)
            self.logn(f'whisper precise beam size: {self.whisper_precise_beam_size}', where='file')

            self.whisper_fast_beam_size = get_config('whisper_fast_beam_size', 1)
            self.logn(f'whisper fast beam size: {self.whisper_fast_beam_size}', where='file')

            self.whisper_precise_temperature = get_config('whisper_precise_temperature', 0.0)
            self.logn(f'whisper precise temperature: {self.whisper_precise_temperature}', where='file')

            self.whisper_fast_temperature = get_config('whisper_fast_temperature', 0.0)
            self.logn(f'whisper fast temperature: {self.whisper_fast_temperature}', where='file')

            self.whisper_precise_compute_type = get_config('whisper_precise_compute_type', 'default')
            self.logn(f'whisper precise compute type: {self.whisper_precise_compute_type}', where='file')

            self.whisper_fast_compute_type = get_config('whisper_fast_compute_type', 'default')
            self.logn(f'whisper fast compute type: {self.whisper_fast_compute_type}', where='file')

            self.timestamp_interval = get_config('timestamp_interval', 60_000) # default: add a timestamp every minute
            self.logn(f'timestamp_interval: {self.timestamp_interval}', where='file')

            self.timestamp_color = get_config('timestamp_color', '#78909C') # default: light gray/blue
            self.logn(f'timestamp_color: {self.timestamp_color}', where='file')

            # get UI settings
            val = self.entry_start.get()
            if val == '':
                self.start = 0
            else:
                self.start = millisec(val)
                option_info += f'{t("label_start")} {val} | ' 

            val = self.entry_stop.get()
            if val == '':
                self.stop = '0'
            else:
                self.stop = millisec(val)
                option_info += f'{t("label_stop")} {val} | '

            if self.option_menu_quality.get() == 'fast':
                self.whisper_model = os.path.join(app_dir, 'models', 'faster-whisper-small')
                self.whisper_beam_size = self.whisper_fast_beam_size
                self.whisper_temperature = self.whisper_fast_temperature
                self.whisper_compute_type = self.whisper_fast_compute_type
            else:
                self.whisper_model = os.path.join(app_dir, 'models', 'faster-whisper-large-v2')
                self.whisper_beam_size = self.whisper_precise_beam_size
                self.whisper_temperature = self.whisper_precise_temperature
                self.whisper_compute_type = self.whisper_precise_compute_type
            option_info += f'{t("label_quality")} {self.option_menu_quality.get()} | '

            try:
                with open(os.path.join(app_dir, 'prompt.yml'), 'r', encoding='utf-8') as file:
                    prompts = yaml.safe_load(file)
            except:
                prompts = {}

            self.language = self.option_menu_language.get()
            if self.language != 'auto':
                self.language = self.language[0:3].strip()

            self.prompt = prompts.get(self.language, '') # Fetch language prompt, default to empty string

            option_info += f'{t("label_language")} {self.language} | '

            self.speaker_detection = self.option_menu_speaker.get()
            option_info += f'{t("label_speaker")} {self.speaker_detection} | '

            self.overlapping = self.check_box_overlapping.get()
            option_info += f'{t("label_overlapping")} {self.overlapping} | '

            self.timestamps = self.check_box_timestamps.get()
            option_info += f'{t("label_timestamps")} {self.timestamps} | '

            self.pause = self.option_menu_pause._values.index(self.option_menu_pause.get())
            option_info += f'{t("label_pause")} {self.pause}'

            self.pause_marker = get_config('pause_seconds_marker', '.') # Default to . if marker not in config

            # Default to True if auto save not in config or invalid value
            self.auto_save = False if get_config('auto_save', 'True') == 'False' else True 
            
            # Open the finished transript in the editor automatically?
            self.auto_edit_transcript = get_config('auto_edit_transcript', 'True')
            
            # Check for invalid vtt options
            if self.file_ext == 'vtt' and (self.pause > 0 or self.overlapping or self.timestamps):
                self.logn()
                self.logn(t('err_vtt_invalid_options'), 'error')
                self.pause = 0
                self.overlapping = False
                self.timestamps = False           

            if platform.system() == "Darwin": # = MAC
                # if (platform.mac_ver()[0] >= '12.3' and
                #     # torch.backends.mps.is_built() and # not necessary since depends on packaged PyTorch
                #     torch.backends.mps.is_available()):
                # Default to mps on 12.3 and newer, else cpu
                xpu = get_config('pyannote_xpu', 'mps' if platform.mac_ver()[0] >= '12.3' else 'cpu')
                self.pyannote_xpu = 'mps' if xpu == 'mps' else 'cpu'
            elif platform.system() in ('Windows', 'Linux'):
                # Use cuda if available and not set otherwise in config.yml, fallback to cpu: 
                xpu = get_config('pyannote_xpu', 'cuda' if get_cuda_device_count() > 0 else 'cpu')
                self.pyannote_xpu = 'cuda' if xpu == 'cuda' else 'cpu'
                whisper_xpu = get_config('whisper_xpu', 'cuda' if get_cuda_device_count() > 0 else 'cpu')
                self.whisper_xpu = 'cuda' if whisper_xpu == 'cuda' else 'cpu'
            else:
                raise Exception('Platform not supported yet.')

            # log CPU capabilities
            self.logn("=== CPU FEATURES ===", where="file")
            if platform.system() == 'Windows':
                self.logn("System: Windows", where="file")
                for key, value in cpufeature.CPUFeature.items():
                    self.logn('    {:24}: {}'.format(key, value), where="file")
            elif platform.system() == "Darwin": # = MAC
                self.logn(f"System: MAC {platform.machine()}", where="file")
                if platform.mac_ver()[0] >= '12.3': # MPS needs macOS 12.3+
                    if config['pyannote_xpu'] == 'mps':
                        self.logn("macOS version >= 12.3:\nUsing MPS (with PYTORCH_ENABLE_MPS_FALLBACK enabled)", where="file")
                    elif config['pyannote_xpu'] == 'cpu':
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
                
                    if int(self.stop) > 0: # transcribe only part of the audio
                        end_pos_cmd = f'-to {self.stop}ms'
                    else: # tranbscribe until the end
                        end_pos_cmd = ''

                    arguments = f' -loglevel warning -y -ss {self.start}ms {end_pos_cmd} -i \"{self.audio_file}\" -ar 16000 -ac 1 -c:a pcm_s16le "{self.tmp_audio_file}"'
                    if platform.system() == 'Windows':
                        ffmpeg_path = 'ffmpeg.exe'
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
                    self.set_progress(1, 50)
                except Exception as e:
                    self.logn(t('err_converting_audio'), 'error')
                    self.logn(e, 'error')
                    return

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
                        
                    if self.overlapping and is_overlapping:
                        return f"//{spkr}"
                    else:
                        return spkr

                # Start Diarization:

                if self.speaker_detection != 'none':
                    try:
                        self.logn()
                        self.logn(t('start_identifiying_speakers'), 'highlight')
                        self.logn(t('loading_pyannote'))
                        self.set_progress(1, 100)

                        diarize_output = os.path.join(tmpdir.name, 'diarize_out.yaml')
                        diarize_abspath = 'python ' + os.path.join(app_dir, 'diarize.py')
                        diarize_abspath_win = os.path.join(app_dir, 'diarize.exe')
                        diarize_abspath_mac = os.path.join(app_dir, '..', 'MacOS', 'diarize')
                        diarize_abspath_lin = os.path.join(app_dir, 'diarize')
                        if platform.system() == 'Windows' and os.path.exists(diarize_abspath_win):
                            diarize_abspath = diarize_abspath_win
                        elif platform.system() == 'Darwin' and os.path.exists(diarize_abspath_mac): # = MAC
                            diarize_abspath = diarize_abspath_mac
                        elif platform.system() == 'Linux' and os.path.exists(diarize_abspath_lin):
                            diarize_abspath = diarize_abspath_lin
                        diarize_cmd = f'{diarize_abspath} {self.pyannote_xpu} "{self.tmp_audio_file}" "{diarize_output}" {self.speaker_detection}'
                        diarize_env = None
                        if self.pyannote_xpu == 'mps':
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
                                        self.set_progress(2, progress_percent * 0.3)
                                    elif step_name == 'embeddings':
                                        self.set_progress(2, 30 + (progress_percent * 0.7))
                                elif line.startswith('error '):
                                    self.logn('PyAnnote error: ' + line[5:], 'error')
                                elif line.startswith('log: '):
                                    self.logn('PyAnnote ' + line, where='file')
                                    if line.strip() == "log: 'pyannote_xpu: cpu' was set.": # The string needs to be the same as in diarize.py `print("log: 'pyannote_xpu: cpu' was set.")`.
                                        self.pyannote_xpu = 'cpu'
                                        config['pyannote_xpu'] = 'cpu'

                        if pyannote_proc.returncode > 0:
                            raise Exception('')

                        # load diarization results
                        with open(diarize_output, 'r') as file:
                            diarization = yaml.safe_load(file)

                        # write segments to log file 
                        for segment in diarization:
                            line = f'{ms_to_str(self.start + segment["start"], include_ms=True)} - {ms_to_str(self.start + segment["end"], include_ms=True)} {segment["label"]}'
                            self.logn(line, where='file')

                        self.logn()

                    except Exception as e:
                        self.logn(t('err_identifying_speakers'), 'error')
                        self.logn(e, 'error')
                        return

                #-------------------------------------------------------
                # 3) Transcribe with faster-whisper

                self.logn()
                self.logn(t('start_transcription'), 'highlight')
                self.logn(t('loading_whisper'))

                # prepare transcript html
                d = AdvancedHTMLParser.AdvancedHTMLParser()
                d.parseStr(default_html)                

                # add audio file path:
                tag = d.createElement("meta")
                tag.name = "audio_source"
                tag.content = self.audio_file
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
                p.appendText(Path(self.audio_file).stem) # use the name of the audio file (without extension) as the title
                main_body.appendChild(p)

                # subheader
                p = d.createElement('p')
                s = d.createElement('span')
                s.setStyle('color', '#909090')
                s.setStyle('font-size', '0.8em')
                s.appendText(t('doc_header', version=app_version))
                br = d.createElement('br')
                s.appendChild(br)

                s.appendText(t('doc_header_audio', file=self.audio_file))
                br = d.createElement('br')
                s.appendChild(br)

                s.appendText(f'({option_info})')

                p.appendChild(s)
                main_body.appendChild(p)

                p = d.createElement('p')
                main_body.appendChild(p)

                speaker = ''
                prev_speaker = ''
                self.last_auto_save = datetime.datetime.now()

                def save_doc():
                    txt = ''
                    if self.file_ext == 'html':
                        txt = d.asHTML()
                    elif self.file_ext == 'txt':
                        txt = html_to_text(d)
                    elif self.file_ext == 'vtt':
                        txt = html_to_webvtt(d, self.audio_file)
                    else:
                        raise TypeError(f'Invalid file type "{self.file_ext}".')
                    try:
                        if txt != '':
                            with open(self.my_transcript_file, 'w', encoding="utf-8") as f:
                                f.write(txt)
                                f.flush()
                            self.last_auto_save = datetime.datetime.now()
                    except Exception as e:
                        # other error while saving, maybe the file is already open in Word and cannot be overwritten
                        # try saving to a different filename
                        transcript_path = Path(self.my_transcript_file)
                        self.my_transcript_file = f'{transcript_path.parent}/{transcript_path.stem}_1{self.file_ext}'
                        if os.path.exists(self.my_transcript_file):
                            # the alternative filename also exists already, don't want to overwrite, giving up
                            raise Exception(t('rescue_saving_failed'))
                        else:
                            # htmlStr = d.asHTML()
                            with open(self.my_transcript_file, 'w', encoding="utf-8") as f:
                                f.write(txt)
                                f.flush()
                            self.logn()
                            self.logn(t('rescue_saving', file=self.my_transcript_file), 'error', link=f'file://{self.my_transcript_file}')
                            self.last_auto_save = datetime.datetime.now()

                try:
                    from faster_whisper import WhisperModel
                    if platform.system() == "Darwin": # = MAC
                        whisper_device = 'auto'
                    elif platform.system() in ('Windows', 'Linux'):
                        whisper_device = 'cpu'
                        whisper_device = self.whisper_xpu
                    else:
                        raise Exception('Platform not supported yet.')
                    model = WhisperModel(self.whisper_model,
                                         device=whisper_device,  
                                         cpu_threads=number_threads, 
                                         compute_type=self.whisper_compute_type, 
                                         local_files_only=True)
                    self.logn('model loaded', where='file')

                    if self.cancel:
                        raise Exception(t('err_user_cancelation')) 

                    whisper_lang = self.language if self.language != 'auto' else None
   
                    try:
                        self.vad_threshold = float(config['voice_activity_detection_threshold'])
                    except:
                        config['voice_activity_detection_threshold'] = '0.5'
                        self.vad_threshold = 0.5
                    
                    segments, info = model.transcribe(
                        self.tmp_audio_file, language=whisper_lang, 
                        beam_size=1, temperature=0, word_timestamps=True, 
                        initial_prompt=self.prompt, vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=200, 
                                            threshold=self.vad_threshold))

                    if self.language == "auto":
                        self.logn("Detected language '%s' with probability %f" % (info.language, info.language_probability))

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
                            if self.auto_save:
                                save_doc()
                                self.logn()
                                self.log(t('transcription_saved'))
                                self.logn(self.my_transcript_file, link=f'file://{self.my_transcript_file}')
  
                            raise Exception(t('err_user_cancelation')) 

                        # get time of the segment in milliseconds
                        start = round(segment.start * 1000.0)
                        end = round(segment.end * 1000.0)
                        # if we skipped a part at the beginning of the audio we have to add this here again, otherwise the timestaps will not match the original audio:
                        orig_audio_start = self.start + start
                        orig_audio_end = self.start + end

                        if self.timestamps:
                            ts = ms_to_str(orig_audio_start)
                            ts = f'[{ts}]'

                        # check for pauses and mark them in the transcript
                        if (self.pause > 0) and (start - last_segment_end >= self.pause * 1000): # (more than x seconds with no speech)
                            pause_len = round((start - last_segment_end)/1000)
                            if pause_len >= 60: # longer than 60 seconds
                                pause_str = ' ' + t('pause_minutes', minutes=round(pause_len/60))
                            elif pause_len >= 10: # longer than 10 seconds
                                pause_str = ' ' + t('pause_seconds', seconds=pause_len)
                            else: # less than 10 seconds
                                pause_str = ' (' + (self.pause_marker * pause_len) + ')'

                            if first_segment:
                                pause_str = pause_str.lstrip() + ' '

                            orig_audio_start_pause = self.start + last_segment_end
                            orig_audio_end_pause = self.start + start
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
                        seg_html = seg_text

                        if self.speaker_detection != 'none':
                            new_speaker = find_speaker(diarization, start, end)
                            if (speaker != new_speaker) and (new_speaker != ''): # speaker change
                                if new_speaker[:2] == '//': # is overlapping speech, create no new paragraph
                                    prev_speaker = speaker
                                    speaker = new_speaker
                                    seg_text = f' {speaker}:{seg_text}'
                                    seg_html = seg_text                                
                                elif (speaker[:2] == '//') and (new_speaker == prev_speaker): # was overlapping speech and we are returning to the previous speaker 
                                    speaker = new_speaker
                                    seg_text = f'//{seg_text}'
                                    seg_html = seg_text
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
                                    if self.timestamps:
                                        seg_html = f'{speaker} <span style="color: {self.timestamp_color}" >{ts}</span>:{seg_text}'
                                        seg_text = f'{speaker} {ts}:{seg_text}'
                                        last_timestamp_ms = start
                                    else:
                                        if self.file_ext != 'vtt': # in vtt files, speaker names are added as special voice tags so skip this here
                                            seg_text = f'{speaker}:{seg_text}'
                                            seg_html = seg_text
                                        else:
                                            seg_html = seg_text.lstrip()
                                            seg_text = f'{speaker}:{seg_text}'
                                        
                            else: # same speaker
                                if self.timestamps:
                                    if (start - last_timestamp_ms) > self.timestamp_interval:
                                        seg_html = f' <span style="color: {self.timestamp_color}" >{ts}</span>{seg_text}'
                                        seg_text = f' {ts}{seg_text}'
                                        last_timestamp_ms = start
                                    else:
                                        seg_html = seg_text

                        else: # no speaker detection
                            if self.timestamps and (first_segment or (start - last_timestamp_ms) > self.timestamp_interval):
                                seg_html = f' <span style="color: {self.timestamp_color}" >{ts}</span>{seg_text}'
                                seg_text = f' {ts}{seg_text}'
                                last_timestamp_ms = start
                            else:
                                seg_html = seg_text
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
                        if self.auto_save:
                            if (datetime.datetime.now() - self.last_auto_save).total_seconds() > 20:
                                save_doc()

                        progr = round((segment.end/info.duration) * 100)
                        self.set_progress(3, progr)

                    save_doc()
                    self.logn()
                    self.logn()
                    self.logn(t('transcription_finished'), 'highlight')
                    if self.transcript_file != self.my_transcript_file: # used alternative filename because saving under the initial name failed
                        self.log(t('rescue_saving'))
                        self.logn(self.my_transcript_file, link=f'file://{self.my_transcript_file}')
                    else:
                        self.log(t('transcription_saved'))
                        self.logn(self.my_transcript_file, link=f'file://{self.my_transcript_file}')
                    # log duration of the whole process in minutes
                    proc_time = datetime.datetime.now() - proc_start_time
                    self.logn(t('trancription_time', duration=int(proc_time.total_seconds() / 60))) 
                    
                    # auto open transcript in editor
                    if (self.auto_edit_transcript == 'True') and (self.file_ext == 'html'):
                        self.launch_editor(self.my_transcript_file)
                
                except Exception as e:
                    self.logn()
                    self.logn(t('err_transcription'), 'error')
                    self.logn(e, 'error')
                    return

            finally:
                self.log_file.close()
                self.log_file = None

        except Exception as e:
            self.logn(t('err_options'), 'error')
            self.logn(e, 'error')
            return

        finally:
            # hide the stop button
            self.stop_button.pack_forget() # hide
            self.start_button.pack(padx=20, pady=[0,30], expand=True, fill='x', anchor='sw')

            # hide progress bar
            self.progress_bar.pack_forget()

    def button_start_event(self):
        wkr = Thread(target=self.transcription_worker)
        wkr.start()
        while wkr.is_alive():
            self.update()
            time.sleep(0.1)
    
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
            config['last_quality'] = self.option_menu_quality.get()
            config['last_pause'] = self.option_menu_pause.get()
            config['last_overlapping'] = self.check_box_overlapping.get()
            config['last_timestamps'] = self.check_box_timestamps.get()

            save_config()
        finally:
            self.destroy()

if __name__ == "__main__":

    app = App()

    app.mainloop()
