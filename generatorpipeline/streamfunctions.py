# Copyright (C) 2020 Stephan Kuschel

'''
This module offers functions to help with stream manipulation.
'''

from .helper import isgenerator


def filterNone(gen):
    '''
    Removes elements from the stream, which are `None`.
    '''
    if not isgenerator(gen):
        raise ValueError(f'{gen} must be a generator.')
    for el in gen:
        if el is not None:
            yield el
