class CommandLine:
    @staticmethod
    def show_available_models(model_names: set):
        """
        Show available Whisper models.
        """

        if not model_names:
            print("No models found. Please check your installation.")
            return
            
        print("Available Whisper models:")
        for name in model_names:
            print(f" - {name}")

    @staticmethod
    def download_model(model_name: str):
        print(f"Downloading model '{model_name}'.")