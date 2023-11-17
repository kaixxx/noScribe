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
if platform.system() == 'Windows':
    import cpufeature
if platform.system() == "Darwin": # = MAC
    import shlex
    import Foundation
import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

app_version = '0.4.3'
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
try:
    with open(f'{config_dir}/config.yml', 'r') as file:
        config = yaml.safe_load(file)
        if not config:
            raise # config file is empty (None)        
except: # seems we run it for the first time and there is no config file
    config = {}
    
config['app_version'] = app_version

def save_config():
    with open(f'{config_dir}/config.yml', 'w') as file:
        yaml.safe_dump(config, file)

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
elif platform.system() == "Darwin": # = MAC
    if platform.machine() == "arm64":
            number_threads = int(int(check_output(["sysctl",
                                                   "-n",
                                                   "hw.perflevel0.logicalcpu_max"])) * 0.75)
    elif platform.machine() == "x86_64":
            number_threads = int(int(check_output(["sysctl",
                                                   "-n",
                                                   "hw.logicalcpu_max"])) * 0.75)
else:
    raise Exception('Platform not supported yet.')

# timestamp regex
timestamp_re = re.compile('\[\d\d:\d\d:\d\d.\d\d\d --> \d\d:\d\d:\d\d.\d\d\d\]')

# Helper functions

def millisec(timeStr): # convert 'hh:mm:ss' string to milliseconds
    try:
        spl = timeStr.split(':')
        s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
        return s
    except:
        raise Exception(t('err_invalid_time_string', time = timeStr))
    
def ms_to_str(t, include_ms=False): # convert milliseconds to timestamp 'hh:mm:ss'
    hh = t//(60*60*1000) # hours
    t = t-hh*(60*60*1000)
    mm = t//(60*1000) # minutes
    t = t-mm*(60*1000)
    ss = t//1000 # seconds
    if include_ms:
        sss = t-ss*1000 # milliseconds
        return(f'{hh:02d}:{mm:02d}:{ss:02d}.{sss}')
    else:
        return(f'{hh:02d}:{mm:02d}:{ss:02d}')

def iter_except(function, exception):
        # Works like builtin 2-argument `iter()`, but stops on `exception`.
        try:
            while True:
                yield function()
        except exception:
            return

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
        if platform.system() == "Darwin":
            self.geometry(f"{1100}x{695}")
        else:
            self.geometry(f"{1100}x{650}")
        self.iconbitmap('noScribeLogo.ico')

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

        self.langs = ('auto', 'en (english)', 'zh (chinese)', 'de (german)', 'es (spanish)', 'ru (russian)', 'ko (korean)', 'fr (french)', 'ja (japanese)', 'pt (portuguese)', 'tr (turkish)', 'pl (polish)', 'ca (catalan)', 'nl (dutch)', 'ar (arabic)', 'sv (swedish)', 'it (italian)', 'id (indonesian)', 'hi (hindi)', 'fi (finnish)', 'vi (vietnamese)', 'iw (hebrew)', 'uk (ukrainian)', 'el (greek)', 'ms (malay)', 'cs (czech)', 'ro (romanian)', 'da (danish)', 'hu (hungarian)', 'ta (tamil)', 'no (norwegian)', 'th (thai)', 'ur (urdu)', 'hr (croatian)', 'bg (bulgarian)', 'lt (lithuanian)', 'la (latin)', 'mi (maori)', 'ml (malayalam)', 'cy (welsh)', 'sk (slovak)', 'te (telugu)', 'fa (persian)', 'lv (latvian)', 'bn (bengali)', 'sr (serbian)', 'az (azerbaijani)', 'sl (slovenian)', 'kn (kannada)', 'et (estonian)', 'mk (macedonian)', 'br (breton)', 'eu (basque)', 'is (icelandic)', 'hy (armenian)', 'ne (nepali)', 'mn (mongolian)', 'bs (bosnian)', 'kk (kazakh)', 'sq (albanian)', 'sw (swahili)', 'gl (galician)', 'mr (marathi)', 'pa (punjabi)', 'si (sinhala)', 'km (khmer)', 'sn (shona)', 'yo (yoruba)', 'so (somali)', 'af (afrikaans)', 'oc (occitan)', 'ka (georgian)', 'be (belarusian)', 'tg (tajik)', 'sd (sindhi)', 'gu (gujarati)', 'am (amharic)', 'yi (yiddish)', 'lo (lao)', 'uz (uzbek)', 'fo (faroese)', 'ht (haitian   creole)', 'ps (pashto)', 'tk (turkmen)', 'nn (nynorsk)', 'mt (maltese)', 'sa (sanskrit)', 'lb (luxembourgish)', 'my (myanmar)', 'bo (tibetan)', 'tl (tagalog)', 'mg (malagasy)', 'as (assamese)', 'tt (tatar)', 'haw (hawaiian)', 'ln (lingala)', 'ha (hausa)', 'ba (bashkir)', 'jw (javanese)', 'su (sundanese)')

        self.option_menu_language = ctk.CTkOptionMenu(self.frame_options, width=100, values=self.langs, dynamic_resizing=False)
        self.option_menu_language.grid(column=1, row=2, sticky='e', pady=5)
        try:
            self.option_menu_language.set(config['last_language'])
        except:
            pass

        # Quality (Model Selection)
        self.label_quality = ctk.CTkLabel(self.frame_options, text=t('label_quality'))
        self.label_quality.grid(column=0, row=3, sticky='w', pady=5)
        
        self.option_menu_quality = ctk.CTkOptionMenu(self.frame_options, width=100, values=['precise', 'fast'])
        self.option_menu_quality.grid(column=1, row=3, sticky='e', pady=5)
        try:
            self.option_menu_quality.set(config['last_quality'])
        except:
            pass
        
        # Mark pauses
        self.label_pause = ctk.CTkLabel(self.frame_options, text=t('label_pause'))
        self.label_pause.grid(column=0, row=4, sticky='w', pady=5)

        self.option_menu_pause = ctk.CTkOptionMenu(self.frame_options, width=100, values=['none', '1sec+', '2sec+', '3sec+'])
        self.option_menu_pause.grid(column=1, row=4, sticky='e', pady=5)
        try:
            val = config['last_pause']
            if val in self.option_menu_pause._values:
                self.option_menu_pause.set(val)
            else:
                raise
        except:
            self.option_menu_pause.set(self.option_menu_pause._values[1])    

        # Speaker Detection (Diarization)
        self.label_speaker = ctk.CTkLabel(self.frame_options, text=t('label_speaker'))
        self.label_speaker.grid(column=0, row=5, sticky='w', pady=5)

        self.option_menu_speaker = ctk.CTkOptionMenu(self.frame_options, width=100, values=['auto', 'none'])
        self.option_menu_speaker.grid(column=1, row=5, sticky='e', pady=5)
        try:
            self.option_menu_speaker.set(config['last_speaker'])
        except:
            pass
        
        # Overlapping Speech (Diarization)
        self.label_overlapping = ctk.CTkLabel(self.frame_options, text=t('label_overlapping'))
        self.label_overlapping.grid(column=0, row=6, sticky='w', pady=5)

        self.check_box_overlapping = ctk.CTkCheckBox(self.frame_options, text = '')
        self.check_box_overlapping.grid(column=1, row=6, sticky='e', pady=5)
        try:
            if config['last_overlapping']:
                self.check_box_overlapping.select()
            else:
                self.check_box_overlapping.deselect()
        except:
            self.check_box_overlapping.select() # defaults to on

        # Timestamps in text
        self.label_timestamps = ctk.CTkLabel(self.frame_options, text=t('label_timestamps'))
        self.label_timestamps.grid(column=0, row=7, sticky='w', pady=5)

        self.check_box_timestamps = ctk.CTkCheckBox(self.frame_options, text = '')
        self.check_box_timestamps.grid(column=1, row=7, sticky='e', pady=5)
        try:
            if config['last_timestamps']:
                self.check_box_timestamps.select()
            else:
                self.check_box_timestamps.deselect()
        except:
            self.check_box_timestamps.deselect() # defaults to off

        # Start Button
        self.start_button = ctk.CTkButton(self.sidebar_frame, height=42, text=t('start_button'), command=self.button_start_event)
        self.start_button.pack(padx=20, pady=[0,10], expand=True, fill='x', anchor='sw')

        # Stop Button
        self.stop_button = ctk.CTkButton(self.sidebar_frame, height=42, fg_color='darkred', hover_color='darkred', text=t('stop_button'), command=self.button_stop_event)
    
        # create log textbox
        self.log_frame = ctk.CTkFrame(self.frame_main, corner_radius=0, fg_color='transparent')
        self.log_frame.pack(padx=0, pady=0, fill='both', expand=True, side='right')

        self.log_textbox = ctk.CTkTextbox(self.log_frame, wrap='word', state="disabled", font=("",16), text_color="lightgray")
        self.log_textbox.tag_config('highlight', foreground='darkorange')
        self.log_textbox.tag_config('error', foreground='yellow')
        self.log_textbox.pack(padx=20, pady=[20,10], expand=True, fill='both')

        self.hyperlink = HyperlinkManager(self.log_textbox._textbox)
        
        # status bar bottom
        self.frame_status = ctk.CTkFrame(self, height=20, corner_radius=0)
        self.frame_status.pack(padx=0, pady=[0,0], anchor='sw', fill='x', side='bottom')

        self.progress_bar = ctk.CTkProgressBar(self.frame_status, height=5, mode='determinate')
        self.progress_bar.set(0)
        
        self.logn(t('welcome_message'), 'highlight')
        self.log(t('welcome_credits', v=app_version))
        self.logn('https://github.com/kaixxx/noScribe', link='https://github.com/kaixxx/noScribe#readme')
        self.logn(t('welcome_instructions'))       
        
    # Events and Methods
    
    def launch_editor(self, file):
        # Launch the editor in a seperate process so that in can stay running even if noScribe quits.
        # Source: https://stackoverflow.com/questions/13243807/popen-waiting-for-child-process-even-when-the-immediate-child-has-terminated/13256908#13256908 
        # set system/version dependent "start_new_session" analogs
        if platform.system() == 'Windows':
            program = os.path.join(app_dir, 'noScribeEdit', 'noScribeEdit.exe')
        elif platform.system() == "Darwin": # = MAC
            program = os.path.join(os.sep, 'Applications', 'noScribeEdit.app', 'Contents', 'MacOS', 'noScribeEdit')
        kwargs = {}
        if platform.system() == 'Windows':
            # from msdn [1]
            CREATE_NEW_PROCESS_GROUP = 0x00000200  # note: could get it from subprocess
            DETACHED_PROCESS = 0x00000008          # 0x8 | 0x200 == 0x208
            kwargs.update(creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)  
        else:  # should work on all POSIX systems, Linux and macOS 
            kwargs.update(start_new_session=True)

        # p = Popen(["C"], stdin=PIPE, stdout=PIPE, stderr=PIPE, **kwargs)
        p = Popen([program, file], **kwargs)
        # assert not p.poll()
    
    def openLink(self, link):
        if link.startswith('file://') and link.endswith('.html'):
            self.launch_editor(link[7:])
        else: 
            webbrowser.open(link)
    
    def log(self, txt='', tags=[], where='both', link=''): # log to main window (log can be 'screen', 'file' or 'both')
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
    
    def logn(self, txt='', tags=[], where='both', link=''): # log with newline
        self.log(f'{txt}\n', tags, where, link)

    def logr(self, txt='', tags=[], where='both', link=''): # replace the last line of the log
        if where != 'file':
            self.log_textbox.configure(state=ctk.NORMAL)
            self.log_textbox.delete("end-1c linestart", "end-1c")
        self.log(txt, tags, where, link)
        
    def reader_thread(self, q):
        try:
            with self.process.stdout as pipe:
                for line in iter(pipe.readline, b''):
                    q.put(line)
        finally:
            q.put(None)

    def button_audio_file_event(self):
        fn = tk.filedialog.askopenfilename(initialdir=os.path.dirname(self.audio_file), initialfile=os.path.basename(self.audio_file))
        if fn != '':
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
        fn = tk.filedialog.asksaveasfilename(initialdir=_initialdir, initialfile=_initialfile, 
                                             filetypes=[('noScribe Transcript','*.html')], defaultextension='html')
        if fn != '':
            self.transcript_file = fn
            self.logn(t('log_transcript_filename') + self.transcript_file)
            self.button_transcript_file_name.configure(text=os.path.basename(self.transcript_file))
    
    def set_progress(self, step, value):
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
    # Button Start

    def transcription_worker(self):
        # This is the main function where all the magic happens
        # We put this in a seperate thread so that it does not block the main ui
        
        proc_start_time = datetime.datetime.now()
        self.cancel = False

        # Show the stop button
        self.start_button.pack_forget() # hide
        self.stop_button.pack(padx=20, pady=[0,10], expand=True, fill='x', anchor='sw')
        
        # Show the progress bar
        self.progress_bar.set(0)
        self.progress_bar.pack(padx=20, pady=[10,20], expand=True, fill='both')

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
            
            # create log file
            if not os.path.exists(f'{config_dir}/log'):
                os.makedirs(f'{config_dir}/log')
            self.log_file = open(f'{config_dir}/log/{Path(self.my_transcript_file).stem}.log', 'w', encoding="utf-8")

            # options for faster-whisper
            try:
                self.whisper_precise_beam_size = config['whisper_precise_beam_size']
            except:
                config['whisper_precise_beam_size'] = 1
                self.whisper_precise_beam_size = 1
            self.logn(f'whisper precise beam size: {self.whisper_precise_beam_size}', where='file')
            try:
                self.whisper_fast_beam_size = config['whisper_fast_beam_size']
            except:
                config['whisper_fast_beam_size'] = 1
                self.whisper_fast_beam_size = 1
            self.logn(f'whisper fast beam size: {self.whisper_fast_beam_size}', where='file')

            try:
                self.whisper_precise_temperature = config['whisper_precise_temperature']
            except:
                config['whisper_precise_temperature'] = 0.0
                self.whisper_precise_temperature = 0.0
            self.logn(f'whisper precise temperature: {self.whisper_precise_temperature}', where='file')
            try:
                self.whisper_fast_temperature = config['whisper_fast_temperature']
            except:
                config['whisper_fast_temperature'] = 0.0
                self.whisper_fast_temperature = 0.0
            self.logn(f'whisper fast temperature: {self.whisper_fast_temperature}', where='file')

            try:
                self.whisper_precise_compute_type = config['whisper_precise_compute_type']
            except:
                config['whisper_precise_compute_type'] = 'default'
                self.whisper_precise_compute_type = 'default'
            self.logn(f'whisper precise compute type: {self.whisper_precise_compute_type}', where='file')
            try:
                self.whisper_fast_compute_type = config['whisper_fast_compute_type']
            except:
                config['whisper_fast_compute_type'] = 'default'
                self.whisper_fast_compute_type = 'default'
            self.logn(f'whisper fast compute type: {self.whisper_fast_compute_type}', where='file')
            
            try:
                self.timestamp_interval = config['timestamp_interval']
            except:
                config['timestamp_interval'] = 60000 # default: add a timestamp every minute
                self.timestamp_interval = 60000
            self.logn(f'timestamp_interval: {self.timestamp_interval}', where='file')
            
            try:
                self.timestamp_color = config['timestamp_color']
            except:
                config['timestamp_color'] = '#78909C' # default: light gray/blue
                self.timestamp_color = '#78909C'
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

            self.prompt = ''
            try:
                with open(os.path.join(app_dir, 'prompt.yml'), 'r', encoding='utf-8') as file:
                    prompts = yaml.safe_load(file)
            except:
                prompts = {}

            self.language = self.option_menu_language.get()
            if self.language != 'auto':
                self.language = self.language[0:3].strip()
                try:
                    self.prompt = prompts[self.language]
                except:
                    self.prompt = ''
            option_info += f'{t("label_language")} {self.language} | '
            
            self.speaker_detection = self.option_menu_speaker.get()
            option_info += f'{t("label_speaker")} {self.speaker_detection} | '
            
            self.overlapping = self.check_box_overlapping.get()
            option_info += f'{t("label_overlapping")} {self.overlapping} | '
            
            self.timestamps = self.check_box_timestamps.get()
            option_info += f'{t("label_timestamps")} {self.timestamps} | '
            
            self.pause = self.option_menu_pause._values.index(self.option_menu_pause.get())
            option_info += f'{t("label_pause")} {self.pause}'
            try:
                self.pause_marker = config['pause_seconds_marker']
            except:
                config['pause_seconds_marker'] = '.'
                self.pause_marker = '.' 

            try:
                if config['auto_save'] == 'True': # auto save during transcription (every 20 sec)?
                    self.auto_save = True
                else:
                    self.auto_save = False
            except:
                config['auto_save'] = 'True'
                self.auto_save = True 

            if platform.system() == "Darwin": # = MAC
                # if (platform.mac_ver()[0] >= '12.3' and
                #     # torch.backends.mps.is_built() and # not necessary since depends on packaged PyTorch
                #     torch.backends.mps.is_available()):
                if platform.mac_ver()[0] >= '12.3': # loading torch modules leads to segmentation fault later
                    try:
                        if config['pyannote_xpu'] == 'cpu':
                            self.pyannote_xpu = 'cpu'
                        else:
                            self.pyannote_xpu = 'mps'
                    except:
                        config['pyannote_xpu'] = 'mps'
                        self.pyannote_xpu = 'mps'
                else:
                    try:
                        if config['pyannote_xpu'] == 'cpu':
                            self.pyannote_xpu = 'cpu'
                        else:
                            self.pyannote_xpu = 'cpu'
                    except:
                        config['pyannote_xpu'] = 'cpu'
                        self.pyannote_xpu = 'cpu'
            else:
                # on other platforms, cuda can be used for pyannote if set in config.yml (experimental)   
                try:
                    if config['pyannote_xpu'] == 'cuda':
                        self.pyannote_xpu = 'cuda'
                    else:
                        self.pyannote_xpu = 'cpu'
                except:
                    config['pyannote_xpu'] = 'cpu'
                    self.pyannote_xpu = 'cpu'

            # log CPU capabilities
            self.logn("=== CPU FEATURES ===", where="file")
            if platform.system() == 'Windows':
                self.logn("System: Windows", where="file")
                for key, value in cpufeature.CPUFeature.items():
                    self.logn('    {:24}: {}'.format(key, value), where="file")
            elif platform.system() == "Darwin": # = MAC
                if platform.machine() == "arm64":
                    self.logn("System: MAC arm64", where="file")
                elif platform.machine() == "x86_64":
                    self.logn("System: MAC x86_64", where="file")
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

                    if platform.system() == 'Windows':
                        ffmpeg_cmd = f'ffmpeg.exe -loglevel warning -y -ss {self.start}ms {end_pos_cmd} -i \"{self.audio_file}\" -ar 16000 -ac 1 -c:a pcm_s16le {self.tmp_audio_file}'
                    elif platform.system() == "Darwin":  # = MAC
                        ffmpeg_abspath = os.path.join(app_dir, 'ffmpeg')
                        ffmpeg_cmd = f'{ffmpeg_abspath} -nostdin -loglevel warning -y -ss {self.start}ms {end_pos_cmd} -i \"{self.audio_file}\" -ar 16000 -ac 1 -c:a pcm_s16le {self.tmp_audio_file}'
                        ffmpeg_cmd = shlex.split(ffmpeg_cmd)
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
                    elif platform.system() == "Darwin":  # = MAC
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
                        return -1   
                    elif ts_start > ss_end: # no overlap, ts is after ss
                        return 0
                    else: # ss & ts have overlap
                        if ts_start > ss_start: # ts starts after ss
                            overlap_start = ts_start
                        else:
                            overlap_start = ss_start
                        if ts_end > ss_end: # ts ends after ss
                            overlap_end = ss_end
                        else:
                            overlap_end = ts_end
                        ol_len = overlap_end - overlap_start + 1
                        ts_len = ts_end - ts_start
                        if ts_len == 0:
                            return -1
                        else:
                            return ol_len / ts_len

                def find_speaker(diarization, transcript_start, transcript_end):
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
                        if t == -1: # we are already after transcript_end
                            break
                        else:
                            if overlap_found >= overlap_threshold: # we already found a fitting segment, compare length now
                                if (t >= overlap_threshold) and ((segment["end"] - segment["start"]) < segment_len): # found a shorter (= better fitting) segment that also overlaps well
                                    is_overlapping = True
                                    overlap_found = t
                                    segment_len = segment["end"] - segment["start"]
                                    spkr = f'S{segment["label"][8:]}' # shorten the label: "SPEAKER_01" > "S01"
                            elif t > overlap_found: # no segment with good overlap jet, take this if the overlap is better then previously found 
                                overlap_found = t
                                segment_len = segment["end"] - segment["start"]
                                spkr = f'S{segment["label"][8:]}' # shorten the label: "SPEAKER_01" > "S01"

                    if self.overlapping and is_overlapping:
                        return f"//{spkr}"
                    else:
                        return spkr

                # Start Diarization:

                if self.speaker_detection == 'auto':
                    try:
                        self.logn()
                        self.logn(t('start_identifiying_speakers'), 'highlight')
                        self.logn(t('loading_pyannote'))
                        self.set_progress(1, 100)

                        diarize_output = os.path.join(tmpdir.name, 'diarize_out.yaml')
                        if platform.system() == 'Windows':
                            diarize_abspath = os.path.join(app_dir, 'diarize.exe')
                        elif platform.system() == 'Darwin': # = MAC
                            # No check for arm64 or x86_64 necessary, since the correct version will be compiled and bundled
                            diarize_abspath = os.path.join(app_dir, '..', 'MacOS', 'diarize')
                        if not os.path.exists(diarize_abspath): # Run the compiled version of diarize if it exists, otherwise the python script:
                            diarize_abspath = 'python ' + os.path.join(app_dir, 'diarize.py')
                        diarize_cmd = f'{diarize_abspath} {self.pyannote_xpu} "{self.tmp_audio_file}" "{diarize_output}"'
                        diarize_env = None
                        if self.pyannote_xpu == 'mps':
                            diarize_env = os.environ.copy()
                            diarize_env["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1) # Necessary since some operators are not implemented for MPS yet.
                        self.logn(diarize_cmd, where='file')
                        
                        if platform.system() == 'Windows':
                            # (supresses the terminal, see: https://stackoverflow.com/questions/1813872/running-a-process-in-pythonw-with-popen-without-a-console)
                            startupinfo = STARTUPINFO()
                            startupinfo.dwFlags |= STARTF_USESHOWWINDOW
                        elif platform.system() == 'Darwin': # = MAC
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
                    try:
                        htmlStr = d.asHTML()
                        with open(self.my_transcript_file, 'w', encoding="utf-8") as f:
                            f.write(htmlStr)
                            f.flush()
                        self.last_auto_save = datetime.datetime.now()
                    except:
                        # saving failed, maybe the file is already open in Word and cannot be overwritten
                        # try saving to a different filename
                        transcript_path = Path(self.my_transcript_file)
                        self.my_transcript_file = f'{transcript_path.parent}/{transcript_path.stem}_1.html'
                        if os.path.exists(self.my_transcript_file):
                            # the alternative filename also exists already, don't want to overwrite, giving up
                            raise Exception(t('rescue_saving_failed'))
                        else:
                            htmlStr = d.asHTML()
                            with open(self.my_transcript_file, 'w', encoding="utf-8") as f:
                                f.write(htmlStr)
                                f.flush()
                            self.logn()
                            self.logn(t('rescue_saving', file=self.my_transcript_file), 'error', link=f'file://{self.my_transcript_file}')
                            self.last_auto_save = datetime.datetime.now()
            
                try:
                    from faster_whisper import WhisperModel
                    if platform.system() == 'Windows':
                        whisper_device = 'cpu'
                    elif platform.system() == "Darwin": # = MAC
                        whisper_device = 'auto'
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

                    if self.language != "auto":
                        whisper_lang = self.language
                    else:
                        whisper_lang = None
                     
                    """
                    # > VAD made the pause-detection actually worse, so I removed it. Strange...   
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
                    """
                    segments, info = model.transcribe(
                        self.tmp_audio_file, language=whisper_lang, 
                        beam_size=self.whisper_beam_size, 
                        temperature=self.whisper_temperature, 
                        word_timestamps=True, 
                        initial_prompt=self.prompt, vad_filter=False)

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
                            else:    
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
                            if pause_len >= 10:
                                if pause_len >= 60: # > 1 minute
                                    pause_str = ' ' + t('pause_minutes', minutes=round(pause_len/60))
                                else: # > 10 sec < 1 min
                                    pause_str = ' ' + t('pause_seconds', seconds=pause_len)
                            else: # < 10 sec, note one 'pause_marker' per second (default to a dot)
                                pause_str = ' (' + (self.pause_marker * pause_len) + ')'
                                
                            if first_segment:
                                pause_str = pause_str.lstrip() + ' '
                            
                            orig_audio_start_pause = self.start + last_segment_end
                            orig_audio_end_pause = self.start + start
                            a = d.createElement('a')
                            a.name = f'ts_{orig_audio_start_pause}_{orig_audio_end_pause}'
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
                        
                        if self.speaker_detection == 'auto':
                            new_speaker = find_speaker(diarization, start, end)
                            if (speaker != new_speaker) & (new_speaker != ''): # speaker change
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
                                        seg_text = f'{speaker}:{seg_text}'
                                        seg_html = seg_text
                            else: # same speaker
                                if self.timestamps:
                                    if (start - last_timestamp_ms) > self.timestamp_interval:
                                        seg_html = f' <span style="color: {self.timestamp_color}" >{ts}</span>{seg_text}'
                                        seg_text = f' {ts}{seg_text}'
                                        last_timestamp_ms = start
                                    else:
                                        seg_html = seg_text

                        else: # no speaker detection
                            if self.timestamps:
                                if first_segment or (start - last_timestamp_ms) > self.timestamp_interval:
                                    seg_html = f' <span style="color: {self.timestamp_color}" >{ts}</span>{seg_text}'
                                    seg_text = f' {ts}{seg_text}'
                                    last_timestamp_ms = start
                                else:
                                    seg_html = seg_text
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
                        a_html = f'<a name="ts_{orig_audio_start}_{orig_audio_end}" >{seg_html}</a>'
                        a = d.createElementFromHTML(a_html)
                        p.appendChild(a)
                        
                        self.log(seg_text)
                        
                        first_segment = False
                                    
                        # auto save
                        if self.auto_save == True:
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
            self.start_button.pack(padx=20, pady=[0,10], expand=True, fill='x', anchor='sw')

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