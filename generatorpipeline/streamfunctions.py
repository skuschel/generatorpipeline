# Copyright (C) 2020 Stephan Kuschel
#
# This file is part of generatorpipeline.
#
# generatorpipeline is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# generatorpipeline is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with generatorpipeline. If not, see <http://www.gnu.org/licenses/>.
#

'''
This module offers functions which are gernerally helpful when working with generators.
'''

import time
from collections import deque
from .helper import isiterator


def simplecache(gen, length=8):
    '''
    simple cache. elements on the beginning will be discarded in order to
    always return a list of length `length`.
    element `[-1]` will be the latest element,
    element `[-2]` the element before that, till
    element `[-length]` is the oldest element to be accessed. Equal to element `[0]`.
    '''
    if not isiterator(gen):
        raise ValueError(f'{gen} must be an iterator.')
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


def observe(gen, *funcs, interval=1):
    '''
    calls all functions in `funcs` for every element in the generator. The return value of those
    functions will be discarded.
    The functions MUST NOT change the elements in the generator.

    kwargs
    ------
      interval=1: int
        using `interval=n` will call all functions on every `n`th value from the
        generator.
    '''
    interval = int(interval)
    for i, el in enumerate(gen):
        if not i % interval:
            for f in funcs:
                f(el)
        yield el


def observe_time(gen, *funcs, interval=0.5):
    '''
    calls all functions in `funcs`. The return value of those
    functions will be discarded.
    The functions MUST NOT change the elements in the generator.

    kwargs
    ------
      interval=0.5: float, time in seconds
        using `interval=n` will attempt to call all functions once every n
        seconds.
    '''
    interval = int(float(interval)*1e9)
    tlast = 0
    for el in gen:
        t = time.time_ns()
        if t - tlast > interval:
            tlast = t
            for f in funcs:
                f(el)
        yield el
