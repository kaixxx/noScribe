Swiss German Whisper Model
==========================

This model is a fine-tuned version of Whisper Large V3 Turbo optimized for
Swiss German (Schweizerdeutsch) automatic speech recognition.

Source: https://huggingface.co/Flurin17/whisper-large-v3-turbo-swiss-german

Model Details:
- 809M parameters
- Trained on 350+ hours of Swiss German speech
- Supports all major Swiss German dialects
- Outputs Standard German text
- License: Apache 2.0

Installation
------------

The model must be converted from HuggingFace Transformers format to
CTranslate2 format for use with faster-whisper.

1. Install conversion tools (in your noScribe virtual environment):

   pip install ctranslate2 transformers[torch]

2. Convert the model (downloads ~1.6GB, takes ~5 minutes):

   ct2-transformers-converter --model Flurin17/whisper-large-v3-turbo-swiss-german \
       --output_dir /path/to/MeetingMemory/models/swiss-german \
       --quantization float16 --force

   Replace /path/to/MeetingMemory with your actual installation path.

3. Copy tokenizer files from the precise model (same base architecture):

   cp models/precise/tokenizer.json models/swiss-german/
   cp models/precise/preprocessor_config.json models/swiss-german/

4. After conversion, this directory should contain:
   - model.bin
   - config.json
   - tokenizer.json
   - preprocessor_config.json
   - vocabulary.json

Requirements:
- ~8GB RAM during conversion
- ~1.6GB disk space for final model
