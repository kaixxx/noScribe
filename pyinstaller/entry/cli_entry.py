import sys
from noScribe import parse_cli_args, run_cli_mode, show_available_models, App

def run_cli(argv):
    sys.argv = [sys.argv[0]] + list(argv)
    args = parse_cli_args()
    if args.help_models:
        show_available_models()
        return 0
    if args.audio_file and args.output_file:
        return run_cli_mode(args)
    elif args.audio_file or args.output_file:
        print("Error: Both audio_file and output_file are required for CLI mode.")
        print("Use --help for usage information.")
        return 1
    else:
        app = App()
        app.mainloop()
        return 0

if __name__ == "__main__":
    code = run_cli(sys.argv[1:])
    raise SystemExit(code if isinstance(code, int) else 0)
