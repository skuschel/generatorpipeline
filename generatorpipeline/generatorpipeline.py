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


class Pipeline():

    def __init__(self, func, nworkers=0, *,
                 skipNone=True,
                 extracache=0,
                 verbose=False,
                 maxtasksperchild=None):
        '''
        Create a pipeline decorator.

        kwargs
        -------
        nworkers = 0
          given to `multiprocessing.Pool`.
          Number of workers to be used for this execution.
          0 (default) will be single threaded execution in the CURRENT process.
          The decorator will create about 0.6 micro-second overhead
          per function call.
          If workers > 0, then the inter-process communication will
          create overhead of about 125 micro-seconds per function call.

        skipNone = True,
          when False, also `None` objects will be returned.
          The default is to skip `None` and instead continue with
          the next non-None element in the generator.
        extracache = 0,
          changes the number of cached elements. By default, the cache will
          hold `nworkers` many elements. `extracache` will specify additional
          elements to be cached, without having additional workers.
        verbose = False,
          activate verbose print statements. For debugging only.
        maxtasksperchild = None,
          given to `multiprocessing.Pool`. Resets a worker after `maxtasksperchild` have
          been executed. By using this, memory leakage, too many open file handles or otherwise
          limited and blocked resources will be freed again.
          However, this only counteracts the symptoms: If you need this, your
          decorated function (or a package it is using) does have a memory leak or is
          getting blocked by another resource. This is also one of the first things to try
          when execution on a large dataset fails, that has worked on a small dataset.
          Try setting this to the number of elements in the small dataset.
        '''
        if not callable(func):
            raise TypeError("{} must be a callable".format(func))
        self.func = func
        self.nworkers = nworkers
        self.cachelen = nworkers + extracache
        self.verbose = verbose
        self.skipNone = skipNone
        self.maxtasksperchild = maxtasksperchild
        functools.update_wrapper(self, func)
        # collect statistics
        self.el_processed = 0
        self.el_yielded = 0

    def __call__(self, arg, **kwargs):
        # No Docstring! It has been set in `__init__` by `functools.update_wrapper`
        if isiterator(arg):
            if self.nworkers == 0:
                return self._call_serial(arg, **kwargs)
            else:
                return self._call_parallel(arg, **kwargs)
        else:
            if self.verbose:
                print(f'executing wrapped function "{self.func.__name__}" (PID: {os.getpid()}).')
            return self.func(arg, **kwargs)

    def _call_serial(self, arg, **kwargs):
        if self.verbose:
            print(f'serial execution of "{self.func.__name__}"')
        for el in arg:
            ret = self(el, **kwargs)  # f(el)
            self.el_processed += 1
            if not isiterator(ret):
                ret = (ret,)
            for r in ret:
                if r is not None or not self.skipNone:
                    self.el_yielded += 1
                    yield r

    def _call_parallel(self, arg, **kwargs):
        if self.verbose:
            print(f'parallel execution of "{self.func.__name__}" with {self.nworkers} workers.')
        with Pool(self.nworkers, maxtasksperchild=self.maxtasksperchild) as pool:
            cache = deque()
            for el in arg:
                cache.append(pool.apply_async(self.func, (el,), kwargs))
                if len(cache) < self.cachelen:
                    # fill cache
                    continue
                ret = cache.popleft().get()
                self.el_processed += 1
                if ret is not None or not self.skipNone:
                    self.el_yielded += 1
                    yield ret
            # flush cache
            while len(cache) > 0:
                ret = cache.popleft().get()
                self.el_processed += 1
                if ret is not None or not self.skipNone:
                    self.el_yielded += 1
                    yield ret

    def pipe_info(self):
        return Pipe_info(self.el_processed, self.el_yielded)


def pipeline(*args, **kwargs):
    def ret(func):
        return Pipeline(func, *args, **kwargs)
    return ret


pipeline.__doc__ = Pipeline.__doc__


class Pipe_info():

    def __init__(self, processed=0, yielded=0):
        self.processed = processed
        self.yielded = yielded

    def __str__(self):
        if self.processed > 0:
            s = 'Pipe_info(processed={p}, yielded={y})[{r:.2%}]'
            return s.format(p=self.processed, y=self.yielded, r=self.yielded/self.processed)
        s = 'Pipe_info(processed={p}, yielded={y})'
        return s.format(p=self.processed, y=self.yielded)

    __repr__ = __str__
