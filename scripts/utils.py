import os, sys
import warnings

def tqdm(iterable):
    if not os.isatty(sys.stdout.fileno()):
        warnings.warn("TQDM disabled because output is not a TTY.")
        return iterable
    else:
        from tqdm.auto import tqdm as _tqdm
        return _tqdm(iterable)
