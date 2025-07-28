# noScribe Enhanced Version

This is an enhanced version of noScribe with additional features for improved workflow efficiency.

## New Features

### Auto-filename Generation
- Checkbox to automatically use input file name as output
- Generates appropriate file extensions (.html, .txt, .vtt, .srt)
- Persists settings between sessions

### Directory Processing
- Process entire directories of audio/video files
- Recursive file discovery in subdirectories
- Support for all ffmpeg-compatible formats
- Batch processing with progress tracking
- Warning dialog with file count before processing

### Supported Media Formats
**Audio**: .mp3, .wav, .m4a, .flac, .aac, .ogg, .wma, .aiff, .opus, .amr, .ac3, .dts
**Video**: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v, .3gp, .ogv, .ts, .mts, .m2ts
**Other**: .m4b, .m4p, .3g2, .asf, .divx, .f4v, .f4p, .f4a, .f4b, .mxf, .bwf, .aif, .caf, .pcm, .raw, .au, .snd

## Download

### macOS (Apple Silicon)
- **Download**: [noScribe-macOS-arm64.tar.gz](https://github.com/falcon-onpurpose/noScribe_queue/releases/latest/download/noScribe-macOS-arm64.tar.gz)
- **Extract**: `tar -xzf noScribe-macOS-arm64.tar.gz`
- **Run**: `./noScribe/noScribe`

## Quick Start

1. **Download** the appropriate version for your platform
2. **Extract** the archive
3. **Run** the application
4. **Select** audio/video file or directory
5. **Choose** output format and settings
6. **Start** transcription

## Usage

### Single File Processing
- Click the document button to select a single audio/video file
- Use the checkbox "Use input file name as output" for auto-filename generation
- Choose output format and settings
- Click Start to begin transcription

### Directory Processing
- Click the folder button to select a directory
- Review the warning dialog showing file count
- Confirm to begin batch processing
- Monitor progress in the main window

## Requirements

- macOS 10.15+ (for Apple Silicon version)
- 8GB+ RAM recommended for large files
- Internet connection for first-time model download

## Backward Compatibility

All original noScribe features are preserved. The enhanced version is fully backward compatible with existing workflows.

## License

Same as original noScribe project (GPL-3.0)

## Contributing

This enhanced version is based on the original noScribe project by kaixxx.
Pull requests for additional features are welcome!

## Links

- **Original Project**: https://github.com/kaixxx/noScribe
- **Enhanced Fork**: https://github.com/falcon-onpurpose/noScribe_queue
- **Issues**: https://github.com/falcon-onpurpose/noScribe_queue/issues 