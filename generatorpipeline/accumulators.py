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
Accumulators, which can be used as potential endpoints of the pipeline.
Examples include the calculation of a mean or a running mean over various
parts of the data.
'''

import abc
import numpy as np


class Accumulator(abc.ABC):
    '''
    The Accumulator base class. All Accumulators must extend this class.
    '''

    @abc.abstractmethod
    def accumulate_obj(self, other):
        pass

    def accumulate_other(self, other):
        s = '`accumulate_other(self, other)` must be defined to accumulate two accumulators.'
        raise NotImplementedError(s)

    @property
    @abc.abstractmethod
    def value(self):
        pass

    @property
    @abc.abstractmethod
    def n(self):
        pass

    def __repr__(self):
        s = '<{cls} of {n} objects>'
        return s.format(n=self.n, cls=self.__class__.__name__)

    __str__ = __repr__

    def __array__(self, dtype=None):
        return np.asanyarray(self.value, dtype=dtype)

    def accumulate(self, other):
        if isinstance(other, self.__class__):
            self.accumulate_other(other)
        else:
            self.accumulate_obj(other)
        return self

    __iadd__ = accumulate


# Some basic accumulators.

class Mean(Accumulator):
    '''
    Calculate the Mean over all data.
    '''

    def __init__(self):
        self.acc = 0
        self._n = 0

    def accumulate_obj(self, obj):
        self.acc += obj
        self._n += 1

    def accumulate_other(self, other):
        self.acc += other.acc
        self._n += other._n

    @property
    def value(self):
        return self.acc / self.n

    @property
    def n(self):
        return self._n


class RunningMean(Accumulator):
    '''
    Calculate the exponential running mean.

    Note: `accumulate_other` is not implemented as the order of
    elements matters.
    '''

    def __init__(self, lifetime=10):
        self.acc = 0
        self._n = 0
        self.alpha = 1/lifetime

    def accumulate_obj(self, obj):
        self.acc = self.acc * (1 - self.alpha) + obj * self.alpha
        self._n += 1

    @property
    def value(self):
        return self.acc

    @property
    def n(self):
        return self._n


class Variance(Accumulator):
    '''
    Calculate the Variance over all data.
    '''

    def __init__(self):
        self.mean = Mean()
        self.meansq = Mean()

    def accumulate_obj(self, obj):
        self.mean += obj
        self.meansq += obj**2

    def accumulate_other(self, other):
        self.mean += other.mean
        self.meansq += other.meansq

    @property
    def n(self):
        return self.mean.n

    @property
    def value(self):
        return self.meansq.value - self.mean.value**2

    @property
    def std(self):
        return np.sqrt(self.value)


class RunningVariance(Variance):
    '''
    Calculate the exponential running Variance.

    Note: `accumulate_other` will raise a NotImplementedError
    inside RunningMean.
    '''

    def __init__(self, lifetime=10):
        self.mean = RunningMean(lifetime=lifetime)
        self.meansq = RunningMean(lifetime=lifetime)
