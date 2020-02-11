# Copyright (C) 2019 Stephan Kuschel


import functools
from multiprocessing import Pool
import os
from collections import deque
from .helper import isgenerator


__all__ = ['pipeline']


class pipeline():

    def __init__(self, workers=0, *, verbose=False):
        '''
        Create a pipeline decorator.

        kwargs
        -------
        workers = 0
          number of workers to be used for this execution.
          0 (default) will be single threaded execution in the current process.
          Will create about 0.6 micro-second overhead
          per call.
          Caution: If workers > 0, then the inter-process communication will
          create overhead of about 125 micro-seconds per function call!

        verbose = False,
          activate verbose print statements. For debugging only.
        '''
        self.workers = workers
        self.verbose = verbose

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
                yield wrapper(el, **kwargs)  # f(el)

        def return_generator_parallel(arg, **kwargs):
            if self.verbose:
                print(f'parallel execution of "{f.__name__}" with {self.workers} workers.')
            pool = Pool(self.workers)
            cache = deque()
            for el in arg:
                cache.append(pool.apply_async(wrapper, (el,), kwargs))
                if len(cache) < self.workers:
                    # fill cache
                    continue
                yield cache.popleft().get()
            # flush cache
            while len(cache) > 0:
                yield cache.popleft().get()
            pool.close()
            return

        @functools.wraps(f)
        def wrapper(arg, **kwargs):
            if isgenerator(arg):
                if self.workers == 0:
                    return return_generator_serial(arg, **kwargs)
                else:
                    return return_generator_parallel(arg, **kwargs)
            else:
                if self.verbose:
                    print(f'executing wrapped function "{f.__name__}" (PID: {os.getpid()}).')
                return f(arg, **kwargs)
        wrapper.workers = self.workers
        return wrapper
