# Copyright (C) 2020-2023 Stephan Kuschel
#               2022 Robert Radloff
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
from collections import deque
import random
import time
import heapq


class Accumulator(abc.ABC):
    '''
    The Accumulator base class. All Accumulators must extend this class.
    '''

    @abc.abstractmethod
    def _accumulate_obj(self, obj):
        pass

    def _accumulate_other(self, other):
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
            self._accumulate_other(other)
        else:
            self._accumulate_obj(other)
        return self

    __iadd__ = accumulate


class Counter(Accumulator):
    '''
    Count the number of accumulated objects.
    Objects can be None and will still be counted.
    '''

    def __init__(self, n=0):
        if not n >= 0:
            raise ValueError('n >=0 required, but n={} found.', format(n))
        self._n = n

    def _accumulate_obj(self, obj):
        self._n += 1

    def _accumulate_other(self, other):
        self._n += other._n

    @property
    def value(self):
        return self._n

    @property
    def n(self):
        return self._n


class _BinaryOpAccumulatorNumpy(Accumulator):
    '''
    Baseclass for Accumulation with a binary operation.
    '''
    _operator = None

    def __init__(self):
        self.acc = None
        self._n = 0

    def _accumulate_obj(self, obj):
        self._n += 1
        if self.acc is None:
            self.acc = np.asarray(obj)
            return
        self.__class__._operator(self.acc, obj, out=self.acc)

    def _accumulate_other(self, other):
        self.__class__._operator(self.acc, other.acc, out=self.acc)
        self._n += other._n

    @property
    def value(self):
        return self.acc

    @property
    def n(self):
        return self._n


# Some basic accumulators.

class Minimum(_BinaryOpAccumulatorNumpy):
    '''
    Calculate the minimum over all data.
    '''
    _operator = np.minimum


class Maximum(_BinaryOpAccumulatorNumpy):
    '''
    Calculate the maximum over all data.
    '''
    _operator = np.maximum


class Mean(Accumulator):
    '''
    Calculate the Mean over all data.
    '''

    def __init__(self, value=0, n=0):
        if not n >= 0:
            raise ValueError('n >=0 required, but n={} found.', format(n))
        self._val = value
        self._n = n

    def _accumulate_obj(self, obj):
        self._n += 1
        self._val += obj / self._n - self._val / self._n

    def _accumulate_other(self, other):
        ntot = self.n + other.n
        self._val = self._val * (self.n / ntot) + other._val * (other.n / ntot)
        self._n += other._n

    @property
    def value(self):
        return self._val

    @property
    def sum(self):
        return self._val * self.n

    @property
    def n(self):
        return self._n


class RunningMean(Accumulator):
    '''
    Calculate the exponential running mean.

    Note: `_accumulate_other` is not implemented as the order of
    elements matters.
    '''

    def __init__(self, lifetime=10):
        self.acc = 0
        self._n = 0
        self.lifetime = lifetime

    def _accumulate_obj(self, obj):
        self._n += 1
        alpha = max(self.alpha, 1 / self._n)
        self.acc = self.acc * (1 - alpha) + obj * alpha

    @property
    def value(self):
        return self.acc

    @property
    def n(self):
        return self._n

    @property
    def lifetime(self):
        return 1 / self.alpha

    @lifetime.setter
    def lifetime(self, x):
        self.alpha = 1 / x


class Variance(Accumulator):
    '''
    Calculate the Variance over all data.

    Internally Welfords Algorithm is used:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    '''

    def __init__(self):
        self.mean = Mean()
        self.var = Mean()

    def _accumulate_obj(self, obj):
        delta1 = obj - self.mean.value
        self.mean += obj
        # (obj - M_n-1) * (obj - M_n) -- last and current iteration mean
        self.var += delta1 * (obj - self.mean.value)

    def _accumulate_other(self, other):
        # for explanation of the formulas, see
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        dmean = self.mean.value - other.mean.value
        newn = self.n + other.n
        newvar = self.var.sum + other.var.sum + dmean ** 2 * self.n * other.n / newn
        self.mean += other.mean
        self.var = Mean(value=newvar / newn, n=newn)

    @property
    def n(self):
        return self.mean.n

    @property
    def value(self):
        return self.var.value * (self.n / (self.n - 1))

    @property
    def rms(self):
        return self.var.value

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
        self.var = RunningMean(lifetime=lifetime)

    @property
    def lifetime(self):
        return self.mean.lifetime

    @lifetime.setter
    def lifetime(self, x):
        self.mean.lifetime = x
        self.var.lifetime = x


class Covariance(Accumulator):
    '''
    Calculate the Covariance (matrix).

    Returns the same as `numpy.cov`.
    '''

    def __init__(self):
        self.mean = Mean()
        self._cov = Mean()

    def _accumulate_obj(self, obj):
        delta1 = obj - self.mean.value
        self.mean += obj
        delta2 = (obj - self.mean.value)
        D = np.outer(delta1, delta2)
        self._cov += D

    def _accumulate_other(self, other):
        # for explanation of the formulas, see
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        dmean = self.mean.value - other.mean.value
        newn = self.n + other.n
        newvar = self._cov.sum + other._cov.sum + np.outer(dmean, dmean) * self.n * other.n / newn
        self.mean += other.mean
        self._cov = Mean(value=newvar / newn, n=newn)

    @property
    def n(self):
        return self.mean.n

    @property
    def value(self):
        return self._cov.value * (self.n / (self.n - 1))

    @property
    def rms(self):
        return self._cov.value


class RunningCovariance(Covariance):
    '''
    Calculate the exponential running Covariance(matrix).

    Note: `accumulate_other` will raise a NotImplementedError
    inside RunningMean.
    '''

    def __init__(self, lifetime=10):
        self.mean = RunningMean(lifetime=lifetime)
        self._cov = RunningMean(lifetime=lifetime)

    @property
    def lifetime(self):
        return self.mean.lifetime

    @lifetime.setter
    def lifetime(self, x):
        self.mean.lifetime = x
        self._cov.lifetime = x


class CacheAccumulator(Accumulator):
    '''
    A cache that implements the Accumulator interface. The value it returns will be the
    last `length` elements, which have been added.
    No calculation is performed.

    length=1 the number of previous elements to return.

    acc.value[-1] is the last element
    acc.value[-2] is the one before the last element
    and so on.
    '''

    def __init__(self, length=1):
        self._n = 0
        self._cache = deque(maxlen=length)
        self._timecache = deque(maxlen=length)

    def _accumulate_obj(self, obj):
        self._n += 1
        self._cache.append(obj)
        # no need to remove any element, because `maxlen` is defined.
        self._timecache.append(time.time_ns())

    def _accumulate_other(self, other):
        self._n += other.n
        cache = deque(maxlen=self._cache.maxlen)
        ctimes = deque(maxlen=self._cache.maxlen)
        it1 = zip(self._timecache, self._cache)
        it2 = zip(other._timecache, other._cache)
        m = heapq.merge(it1, it2, key=lambda x: x[0])
        for t, el in m:
            cache.append(el)
            ctimes.append(t)
        self._cache = cache
        self._timecache = ctimes

    @property
    def value(self):
        '''
        value[-1] is the last element.
        '''
        return list(self._cache)

    @property
    def n(self):
        return self._n


class CacheMaximum(Accumulator):
    """
    length: Number of retained elements
    key: method that cast element to number. Elements with highest number are retained
    time_key: way in which a time is assigned to the elements.
              default: None -> use system time
    timeout: maximum time between the retained elements: If the number of elements would be smaller
             than length then the newest elements are used
             default: None -> no timeout is used
    """

    def __init__(self, length=10, key=lambda x: x, time_key=None, timeout=None):
        self._n = 0
        self.length = length
        self._cache = []
        self.sortkeyf = key
        self._time_keyf = time_key
        self.timeout = timeout

    def time_keyf(self, obj):
        if self._time_keyf is None:
            return time.time_ns()
        else:
            return self._time_keyf(obj)

    def _accumulate_obj(self, obj):
        self._n += 1
        # Filling of Cache when lenght is not reached
        if self._n <= self.length:
            heapq.heappush(self._cache, (self.sortkeyf(obj), self.time_keyf(obj), obj))
        else:
            if self.timeout is None:
                # Push the newest element onto the heap and remove the smallest one
                # Note: If the heap consists of tupels as it is the case here, the first elements
                # of the tupel will be compared first
                heapq.heappushpop(self._cache, (self.sortkeyf(obj), self.time_keyf(obj), obj))
            else:
                # Push the newest(shot passed to the function) shot onto the heap. Necessary to
                # check wether the oldest shot is older than timeout
                heapq.heappush(self._cache, (self.sortkeyf(obj), self.time_keyf(obj), obj))
                # Put times onto array for easy comparison
                time_arr = [el[1] for el in self._cache]
                # check if the oldest shot is older than timeout
                if np.max(time_arr)-np.min(time_arr) >= self.timeout:
                    # Take the n newest shots(Later times are newer)
                    self._cache = heapq.nlargest(self.length, self._cache, key=lambda x: x[1])
                    heapq.heapify(self._cache)
                else:
                    heapq.heappop(self._cache)

    # Uses timeout of gathering Accumulator and implicitly assumes that key and time_key are
    # the same for both accumulators
    def _accumulate_other(self, other):
        self._n += other.n
        # Inputs are not necessarily sorted, so the output isn't either
        merged = heapq.merge((self._cache, other._cache))
        # remove the shots which are too old first
        if self.timeout is not None:
            time_arr = [el[1] for el in self._cache]
            j = 0
            while np.max(time_arr) - np.min(time_arr) >= self.timeout and other.length > j:
                # remove the oldest shot
                j += 1
                self._cache = heapq.nlargest(self.length + other.length-j,
                                             self._cache, key=lambda x: x[1])
                time_arr = [el[1] for el in self._cache]
        self._cache = heapq.nlargest(self.length, self._cache, key=lambda x: x[0])
        heapq.heapify(self._cache)

    @property
    def value(self):
        '''
        value[0] is the brightest element.
        '''
        # Sort the output according to 'brightness'
        lst = sorted(self._cache, key=lambda x: x[0])
        return [el[2] for el in lst]

    @property
    def n(self):
        return self._n


class ReservoirSampling(Accumulator):
    '''
    Samples a random collection from a series of elements.

    This class chooses a simple random sample, without replacement,
    of k items from a population of unknown size n in a single pass over the items.
    The size of the population n is not known to the algorithm and is typically too large
    for all n items to fit into main memory.
    The Accumulator is non-deterministic.
    Uses the Algorithm R (https://en.wikipedia.org/wiki/Reservoir_sampling)

    length: size of the reservoir
    '''

    def __init__(self, length=10):
        self._n = 0
        self.length = length
        self._reservoir = []

    def _accumulate_obj(self, obj):
        self._n += 1
        # Filling of Reservoir until length is reached
        if self._n <= self.length:
            self._reservoir.append(obj)
        else:
            j = random.randint(1, self._n)
            if j <= self.length:
                self._reservoir[j-1] = obj

    @property
    def value(self):
        return list(self._reservoir)

    @property
    def n(self):
        return self._n


class CDFEstimator(Accumulator):
    '''
    Estimates the Cumulative Distribution Function (CDF).
    Arguments:
    ----------
      * points - number of positions for CDF sampling
        OR
        list of sampling positions

    This implementation follows the P^2 algorithm
    proposed by Jain and Chlamtac in the paper
    https://doi.org/10.1145/4372.4378.

    The algorithm to calculate the p-quantile
    (p = 0.5 for the median) roughly works the following:

    Accumulate the first five observations and sort them.
    This yields a list with 5 marker heights.
    Each marker is assigned a marker position (1, 2, ..., 5)

    For each subsequent observation two operations are performed:
    First:
    Check between which markers the observation fits and increment
    the marker heights of every marker that is higher.
    Only replace the marker height by the value of the observation
    if a new minimum or maximum is found.

    Second:
    Adjust the height of the non-extreme markers (markers 2 to 4)
    if they differ from their desired position 1 or more
    and if incrementing the position of considered marker
    does not lead to a collision with another marker
    (the difference of position of the next higher or lower
    marker and the considered marker must be > 1 (, -1)).

    The adjustment of the marker heights and positions is
    performed via a parabolic interpolation (linear interpolation
    in some situations where the parabolic methode cannot be used).

    Marker positions are always only adjusted in steps of 1, -1!

    The estimation of the p-quantile is then given by the marker height
    of the third marker (`self.m_height[2]`).

    Obtain the approximate value via `self.value`.

    Robert Radloff 2022
    '''

    def __init__(self, points):
        if np.asanyarray(points).shape == ():
            # linear spacing (equiprobable cells)
            self.q_desired = np.asarray(np.linspace(0, 1, points))
        else:
            # just use the cells given
            self.q_desired = np.array(points, dtype=float)
            self.q_desired.sort()
        self.q_desired.setflags(write=False)
        if self.q_desired[0] != 0 or self.q_desired[-1] != 1:
            raise ValueError('points must be 0 in first and 1 in the last element.')
        self._n = 0
        # Important note:
        # in this implementation n
        # counts the number of observations
        # but is incremented only at the end
        # of _accumulate_obj.
        # Thus, during the call of
        # _accumulate_obj it acts as N-1 instead.

        # if shape is not given the marker positions and heights
        # will be set in a shape fitting the first observation.
        # The values of the marker heights and positions are in the last
        # dimension of the according array.
        self.m_pos = None
        self.m_height = None

    def _init_m_pos(self, shape):
        if self.m_pos is not None:
            raise ValueError(f'`{self}.m_pos` was already initialized.')
        self.m_pos = np.ones((len(self.q_desired), *shape), dtype=float)  # marker positions
        a = np.arange(len(self.q_desired), dtype=float)
        a.shape = (len(self.q_desired), *[1 for _ in range(len(shape))])
        self.m_pos *= a

    def _init_m_height(self, shape):
        if self.m_height is not None:
            raise ValueError(f'`{self}.m_height` was already initialized.')
        # marker heights
        self.m_height = np.ones((len(self.q_desired), *shape), dtype=float) * np.nan

    def _accumulate_obj(self, obj):
        obj = np.asarray(obj)
        if self.m_pos is None or self.m_height is None:
            self._init_m_pos(obj.shape)
            self._init_m_height(obj.shape)
        # TODO write test to assure obj shape doesn't change
        #  once `m_pos` and `m_height` is initialized.
        if self._n < len(self.q_desired) - 1:
            self.m_height[self._n] = obj
            self._n += 1
            return
        elif self._n == len(self.q_desired) - 1:
            self.m_height[self._n] = obj
            self.m_height = np.sort(self.m_height, axis=0)
        else:
            # Check for new Min
            cond_min = obj < self.m_height[0]
            self.m_height[0] = np.where(cond_min, obj, self.m_height[0])
            # Check for new Max
            cond_max = obj > self.m_height[-1]
            self.m_height[-1] = np.where(cond_max, obj, self.m_height[-1])
            # Increment Marker positions
            self.m_pos[1:] += (obj <= self.m_height[1:])

            # assert self.m_pos[0] == 0 and self.m_pos[-1] == self.n
        self._adjust_heights()
        # assert np.all(self.m_height[:-1] <= self.m_height[1:]),
        # f'Problem: Heights unsorted at {self.n}'
        self._n += 1

    def _m_posdiff(self, i):
        '''
        Calculate the difference between the marker positions
        `m_pos` and the desired marker positions `_m_desired`.
        '''
        return self._m_desired[i] - self.m_pos[i]

    @property
    def _m_desired(self):
        return self.q_desired * self.n

    def _adjust_heights(self):
        '''
        This function implements step B3 from box 1 in the Jain and Chlamtac paper.
        '''

        def adjust_possible(pdiff, positions):
            l_step = np.logical_and(pdiff <= -1, positions[0] - positions[1] < -1)
            r_step = np.logical_and(pdiff >= 1, positions[2] - positions[1] > 1)
            return np.logical_or(l_step, r_step)

        def parabolic_possible(h_new, heights):  # p for previous, n for next
            # could potentially be replaced
            # by a check if heights are still strictly increasing
            return np.logical_and(heights[0] < h_new, h_new < heights[2])

        # assert np.all(self._m_posdiff[..., 0]) == 0 and np.all(self._m_posdiff[..., -1] == 0)
        for i in range(1, len(self.q_desired) - 1):
            posdiff = self._m_posdiff(i)
            direction = np.sign(posdiff)
            heights = (self.m_height[i-1],
                       self.m_height[i],
                       self.m_height[i+1])
            positions = (self.m_pos[i-1],
                         self.m_pos[i],
                         self.m_pos[i+1])
            # calc parabolic and linear interp for all observations
            par = self._parabolic(heights, positions, direction)
            # depending on the step direction _linear requires different arguments.
            lin = self._linear((heights[1], np.where(direction < 0, heights[0], heights[2])),
                               (positions[1], np.where(direction < 0, positions[0], positions[2])),
                               direction)
            pos = self.m_pos[i] + direction
            # Apply height changes where needed.
            adj = adjust_possible(posdiff, positions)
            self.m_height[i] = np.where(adj,
                                        np.where(parabolic_possible(par, heights), par, lin),
                                        self.m_height[i])
            # Don't forget to adjust marker positions where necessary
            self.m_pos[i] = np.where(adj,
                                     pos,
                                     self.m_pos[i])

    @staticmethod
    def _linear(q, n, d):
        '''
        Calculate the new marker height by using linear interpolation.
        '''
        # if len(q) != 2:
        #     raise ValueError('q does not contain 2 elements!')
        # if len(n) != 2:
        #     raise ValueError('n does not contain 2 elements!')
        q_i, q_d = q
        n_i, n_d = n
        q_new = q_i + d*((q_d - q_i) / (n_d - n_i))
        return q_new

    @staticmethod
    def _parabolic(heights, positions, d):
        '''
        Calculate marker height at the new position
        using the piecewise parabolic formula described in
        https://doi.org/10.1145/4372.4378.
        '''
        # TODO: im not quite happy with this implementation.
        #  Also add error handling again
        q1, q2, q3 = heights
        n1, n2, n3 = positions
        q_new = q2 + d / (n3 - n1) * ((n2 - n1 + d)
                                      * (q3 - q2) / (n3 - n2)
                                      + (n3 - n2 - d)
                                      * (q2 - q1) / (n2 - n1))
        return q_new

    @property
    def n(self):
        return self._n

    @property
    def q_actual(self):
        '''
        The actual quantile marker positions.

        similar to `self.q_desired` but accurately calculated.
        '''
        return np.asarray(self.m_pos, dtype=float) / (self.n - 1)

    @property
    def cdf(self):
        '''
        Cumulative Distribution Function
        '''
        return self.m_height, self.q_actual

    def cdf_interp(self, v):
        '''
        Interpolation of the CDF at value(s) `v`.
        Returns the quantile(s) `q` at which you would find `v` in the distribution.

        Uses `np.interp` for the interpolation.
        '''
        return np.interp(v, *self.cdf)

    @property
    def quantile(self):
        '''
        Quantiles and corresponding values.

        This is equivalent to `cdf` with swapped axis.
        '''
        return self.q_actual,  self.m_height

    def quantile_interp(self, q):
        '''
        Interpolation of the value for the quantile(s) `q`.

        This is equivalent to the interpolation of the inverse CDF at value(s) `q`.
        Returns the value(s) `v` which is at the `q`-quantile of the distribution.

        `quantile_interp(0.5)` returns an interpolated median.

        Uses `np.interp` for the interpolation.
        '''
        return np.interp(q, *self.quantile)

    @property
    def pdf(self):
        '''
        Probability Density Function

        This is simply the derivative of the CDF.
        '''
        x, y = self.cdf
        pdf = x, np.gradient(y) / np.gradient(x)
        return pdf

    @property
    def min(self):
        return self.m_height[0]

    @property
    def max(self):
        return self.m_height[-1]

    @property
    def value(self):
        return self.cdf

    @property
    def _debug_info(self):
        return self._n, self.m_pos, self.m_height


class QuantileEstimator(CDFEstimator):

    def __init__(self, p):
        self.p = p
        # desired quantile markers
        super().__init__(np.asarray([0, 0.5 * p, p, 0.5 * (p + 1), 1], dtype=float))

    @property
    def value(self):
        return self.m_height[2]


class MedianEstimator(QuantileEstimator):
    '''
    Calculate the approximate median.
    Uses QuantileEstimator with p=0.5.
    '''
    def __init__(self):
        super().__init__(0.5)


class BinSorter(Accumulator):

    def __init__(self, bin_edges, binaccumulatorcls=Counter, /, kwargs={},
                 key=lambda x: x, datakey=lambda x: x):
        '''
        Sorts objects into accumulators for each bin. The simplest use case is to
        create a histogram. However, the accumulator class to be used in each bin
        can be set by the `binaccumulatorcls`. It will accumulate the data according
        to the the `datakey` function. The data will be sorted into the right bins according to
        the `key` function.

        `bin_edges` have to be given explicitly.

        the number of bins `nbins` is `len(bin_edges) - 1`.
        '''
        self._bin_edges = bin_edges
        # + 2 for min and max bin
        self._binaccs = [binaccumulatorcls(**kwargs) for _ in range(self.nbins + 2)]
        self.sortkey = key
        self.datakey = datakey
        self._n = 0

    @property
    def nbins(self):
        return len(self.bin_edges) - 1

    @property
    def bin_edges(self):
        return self._bin_edges

    def _accumulate_obj(self, obj):
        self._n += 1
        s = self.sortkey(obj)  # sort element
        d = self.datakey(obj)  # data element
        idx = np.digitize(s, self.bin_edges)
        self._binaccs[idx].accumulate(d)

    @property
    def value(self):
        return self.bin_edges, self._binaccs[1:-1]

    @property
    def histogram(self):
        '''
        returns `bin_edges` and `histogram`, identical to `np.histogram`.
        `bin_edges` always has one element more than `histogram`.
        '''
        return self.bin_edges, [x.n for x in self._binaccs[1:-1]]

    @property
    def histogram_density(self):
        '''
        The Density in the histogram.
        This is what you are looking for when the bins are not equally spaced.

        The Integral of this function is the total number of observations.
        '''
        e, h = self.histogram
        return e, h / np.diff(e)

    @property
    def n(self):
        return self._n


class DynamicBinSorter(BinSorter):

    def __init__(self, nbins, binaccumulatorcls=Counter, /, kwargs={},
                 key=lambda x: x, datakey=lambda x: x):
        '''
        Same as the `BinSorter`, but the `bin_edges` are dynamically adjusted using the
        `CDFEstimator`.
        '''
        self._nbins = nbins
        self.cdfestimator = CDFEstimator(nbins + 1)
        self._binaccs = [binaccumulatorcls(**kwargs) for _ in range(nbins)]
        self.sortkey = key
        self.datakey = datakey
        self._n = 0

    @property
    def nbins(self):
        return self._nbins

    @property
    def bin_edges(self):
        return self.cdfestimator.m_height

    def _accumulate_obj(self, obj):
        self._n += 1
        s = self.sortkey(obj)  # sort element
        self.cdfestimator.accumulate(s)
        if self._n <= self.nbins:
            # bin_edges must be available and the cdfestimator requires more than nbins
            # many elements. Therefor only start after that many elements.
            return
        d = self.datakey(obj)  # data element
        idx = np.digitize(s, self.bin_edges) - 1
        # -1 because a new minimum will never be found.
        # The cdfestimater has already adjusted to that.
        if idx == self.nbins:
            # element is new max
            # print('new max found with element {}'.format(s))
            idx -= 1
        self._binaccs[idx].accumulate(d)

    @property
    def value(self):
        return self.bin_edges, self._binaccs

    @property
    def histogram(self):
        '''
        returns `bin_edges` and `histogram`, identical to `np.histogram`.
        `bin_edges` always has one element more than `histogram`.
        '''
        return self.bin_edges, [x.n for x in self._binaccs]
