The "fast" setting uses Whisper v3 turbo also, but in an int8 quantization. In my testing on CPU, this lead to about 30% faster speed.  

Download all the files from here into this folder:
https://huggingface.co/mukowaty/faster-whisper-int8/tree/main/faster-whisper-large-v3-turbo-int8

If the files are named like so: "faster-whisper-large-v3-turbo-int8_config.json", please remove the prefix "faster-whisper-large-v3-turbo-int8_". The resulting file in this example must be named "config.json". Do this to all files in the folder.

Model license
-------------
The model weights in this folder are a converted version of OpenAI's Whisper
large-v3-turbo model. The conversion used by noScribe is linked above.
Original model: https://huggingface.co/openai/whisper-large-v3-turbo

OpenAI releases the Whisper model weights under the MIT License. The full
original license is reproduced below.
Source: https://github.com/openai/whisper/blob/main/LICENSE

MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
