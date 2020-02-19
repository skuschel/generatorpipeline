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
