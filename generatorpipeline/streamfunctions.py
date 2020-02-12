# Copyright (C) 2020 Stephan Kuschel

'''
This module offers functions to help with stream manipulation.
'''

from collections import deque
from .helper import isgenerator


def simplecache(gen, length=8):
    '''
    simple cache. elements on the beginning will be discarded in order to
    always return a list of length `length`.
    element `[-1]` will be the latest element,
    element `[-2]` the element before that, till
    element `[-length]` is the oldest element to be accessed. Equal to element `[0]`.
    '''
    if not isgenerator(gen):
        raise ValueError(f'{gen} must be a generator.')
    cache = deque(maxlen=length)
    for el in gen:
        cache.append(el)
        # fill cache
        if len(cache) < length:
            continue
        # return data
        yield list(cache)
        # no need to remove the oldest element, because `maxlen`
        # was specified when the deque was initialized.
