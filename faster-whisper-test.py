from faster_whisper import WhisperModel

def format_timestamp(seconds: float, always_include_hours: bool = True, decimal_marker: str = "."):
# from: https://github.com/Softcatala/whisper-ctranslate2/blob/main/src/whisper_ctranslate2/writers.py

    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )
    
model_size = "large-v2"

# Run on GPU with FP16
model = WhisperModel("./models/faster-whisper-large-v2", device="auto", compute_type="auto", local_files_only=True)

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("../Intw/Ein Gespräch mit Heikedine Körting l Die EUROPA Hörspiel-Königing.mp3", beam_size=5, word_timestamps=True)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))