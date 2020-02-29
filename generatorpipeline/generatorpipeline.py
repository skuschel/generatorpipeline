# Copyright (C) 2019 Stephan Kuschel
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

import functools
from multiprocessing import Pool
import os
from collections import deque
from .helper import isiterator


__all__ = ['pipeline']


class pipeline():

    def __init__(self, nworkers=0, *, skipNone=True, extracache=0, verbose=False):
        '''
        Create a pipeline decorator.

        kwargs
        -------
        nworkers = 0
          number of workers to be used for this execution.
          0 (default) will be single threaded execution in the current process.
          Will create about 0.6 micro-second overhead
          per call.
          Caution: If workers > 0, then the inter-process communication will
          create overhead of about 125 micro-seconds per function call!

        skipNone = True,
          when False, also `None` objects will be returned. Default is to skip `None`.
        extracache = 0,
          changes the number of cached elements. By default, the cache will
          hold `nworkers` many elements. `extracache` will specify additional
          elements to be cached, without having additional workers.
        verbose = False,
          activate verbose print statements. For debugging only.
        '''
        self.nworkers = nworkers
        self.cachelen = nworkers + extracache
        self.verbose = verbose
        self.skipNone = skipNone

    def __call__(self, func):
        '''
        Decorate the function `func` to be used as a pipeline.

        kwargs to the decorated function will be forwarded to every call of f.
        '''
        if not callable(func):
            raise TypeError("must be a callable")
        return self._build_pipeline(func)

    def _build_pipeline(self, f):

        def return_generator_serial(arg, **kwargs):
            if self.verbose:
                print(f'serial execution of "{f.__name__}"')
            for el in arg:
                ret = wrapper(el, **kwargs)  # f(el)
                if ret is not None or not self.skipNone:
                    yield ret

        def return_generator_parallel(arg, **kwargs):
            if self.verbose:
                print(f'parallel execution of "{f.__name__}" with {self.nworkers} workers.')
            pool = Pool(self.nworkers)
            cache = deque()
            for el in arg:
                cache.append(pool.apply_async(wrapper, (el,), kwargs))
                if len(cache) < self.cachelen:
                    # fill cache
                    continue
                ret = cache.popleft().get()
                if ret is not None or not self.skipNone:
                    yield ret
            # flush cache
            while len(cache) > 0:
                ret = cache.popleft().get()
                if ret is not None or not self.skipNone:
                    yield ret
            pool.close()
            return

        @functools.wraps(f)
        def wrapper(arg, **kwargs):
            if isiterator(arg):
                if self.nworkers == 0:
                    return return_generator_serial(arg, **kwargs)
                else:
                    return return_generator_parallel(arg, **kwargs)
            else:
                if self.verbose:
                    print(f'executing wrapped function "{f.__name__}" (PID: {os.getpid()}).')
                return f(arg, **kwargs)
        wrapper.nworkers = self.nworkers
        wrapper.cachelen = self.cachelen
        wrapper.skipNone = self.skipNone
        return wrapper
