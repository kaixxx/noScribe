import multiprocessing as mp

import noScribe

if __name__ == "__main__":
    # Must come before anything touches noScribe.main. Under PyInstaller a
    # multiprocessing "spawn" child is the frozen binary started again, so it
    # runs this script from the top; freeze_support() is what recognises such a
    # child, runs its target and exits. main.py calls it too, but only after
    # importing tkinter, customtkinter and PyAV -- which is the whole GUI stack
    # this package's lazy __getattr__ exists to keep out of worker processes.
    mp.freeze_support()
    # Exceptions are deliberately not caught here. Swallowing them printed a
    # bare message and still exited 0 -- and in a windowed build main.py's
    # stdout fix has not run yet at import time, so the message went nowhere
    # and a failed start looked like a successful one.
    noScribe.main.noScribeMain()
