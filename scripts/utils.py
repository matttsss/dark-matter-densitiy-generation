import os, sys
import warnings

def tqdm(iterable, display_training_bar=True):
    if not (display_training_bar and sys.stdout.isatty()):
        return iterable
    else:
        from tqdm.auto import tqdm as _tqdm
        return _tqdm(iterable, disable=not display_training_bar)