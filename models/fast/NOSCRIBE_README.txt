The "fast" setting uses Whisper v3 turbo also, but in an int8 quantization. In my testing on CPU, this lead to about 30% faster speed.  

Download all the files from here into this folder:
https://huggingface.co/mukowaty/faster-whisper-int8/tree/main/faster-whisper-large-v3-turbo-int8

If the files are named like so: "faster-whisper-large-v3-turbo-int8_config.json", please remove the prefix "faster-whisper-large-v3-turbo-int8_". The resulting file in this example must be named "config.json". Do this to all files in the folder. 