class ModelAlreadyExists(Exception):
    """
    A model already exists in the model directory and cannot be downloaded
    again.
    """

    pass

    
class ModelDoesNotExist(Exception):
    """
    A desired model does not exist or is not specified.
    """

    pass
