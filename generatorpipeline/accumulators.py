# Copyright (C) 2020-2022 Stephan Kuschel
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

    def __init__(self, points,  shape=None):
        if np.asanyarray(points).shape == ():
            # linear spacing (equiprobable cells)
            self.q_desired = np.linspace(0, 1, points)
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
        if shape is not None:
            self._init_m_pos(shape)
            self._init_m_height(shape)
        else:
            self.m_pos = None
            self.m_height = None

    def _init_m_pos(self, shape):
        if self.m_pos is not None:
            raise ValueError(f'`{self}.m_pos` was already initialized.')
        self.m_pos = np.ones((*shape, len(self.q_desired)), dtype=float)  # marker positions
        self.m_pos *= np.arange(len(self.q_desired), dtype=float)

    def _init_m_height(self, shape):
        if self.m_height is not None:
            raise ValueError(f'`{self}.m_height` was already initialized.')
        self.m_height = np.ones((*shape, len(self.q_desired)), dtype=float) * np.nan  # marker heights

    def _accumulate_obj(self, obj):
        obj = np.asarray(obj)
        if self.m_pos is None or self.m_height is None:
            self._init_m_pos(obj.shape)
            self._init_m_height(obj.shape)
        # TODO write test to assure obj shape doesn't change
        #  once `m_pos` and `m_height` is initialized.
        if self._n < len(self.q_desired) - 1:
            self.m_height[..., self._n] = obj
            self._n += 1
            return
        elif self._n == len(self.q_desired) - 1:
            self.m_height[..., self._n] = obj
            self.m_height = np.sort(self.m_height, axis=-1)
        else:
            # Check for new Min
            cond_min = obj < self.m_height[..., 0]
            self.m_height[..., 0] = np.where(cond_min, obj, self.m_height[..., 0])
            # Check for new Max
            cond_max = obj > self.m_height[..., -1]
            self.m_height[..., -1] = np.where(cond_max, obj, self.m_height[..., -1])
            # Increment Marker positions
            self.m_pos[..., 1:] += (obj[..., None] <= self.m_height[..., 1:])

            assert self.m_pos[0] == 0 and self.m_pos[-1] == self.n
        self._adjust_heights()
        assert np.all(self.m_height[..., :-1] <= self.m_height[..., 1:]), f'Problem: Heights unsorted at {self.n}'
        self._n += 1

    @property
    def _m_posdiff(self):
        '''
        Calculate the difference between the marker positions
        `m_pos` and the desired marker positions `_m_desired`.
        '''
        return self._m_desired - self.m_pos

    @property
    def _m_desired(self):
        return self.q_desired * self.n

    def _adjust_heights(self):
        '''
        This function implements step B3 from box 1 in the Jain and Chlamtac paper.
        '''
        assert np.all(self._m_posdiff[..., 0]) == 0 and np.all(self._m_posdiff[..., -1] == 0)
        for i in range(1, len(self.q_desired) - 1):
            posdiff = self._m_posdiff
            if ((posdiff[i] >= 1) and (self.m_pos[i+1] - self.m_pos[i] > 1))\
                    or ((posdiff[i] <= -1) and (self.m_pos[i-1] - self.m_pos[i] < -1)):
                d = np.sign(posdiff[..., i])
                # h2 is the height that will be adjusted (self.m_height[...,i])
                heights = np.split(self.m_height[..., i - 1:i + 2], 3, axis=-1)
                positions = np.split(self.m_pos[..., i-1:i+2], 3, axis=-1)
                h_new = self._parabolic(heights, positions, d)
                if self.m_height[i-1] < h_new and h_new < self.m_height[i+1]:
                    # assures that heights are in ascending order after the adjustment
                    # TODO: this most likely needs a different solution, as does the other if condition
                    # as not all values of the array wil satisfy the condition
                    # thinking about it it would maybe be best, to calculate bot the parabolic and the
                    # linear interpolation and then choose the according value, either by nested np.where
                    # expressions or (probably better) np.select
                    # so basically: compute posdiff, compute step direction (what happens if its 0?)
                    # compute parabolic interpolation, compute linear interpolation
                    # select original value if interpol was not necessary, select linear if parabolic is not
                    # possible, else select parabolic.
                    self.m_height[..., i] = h_new  # set marker height to new value
                else:
                    heights = (self.m_height[..., i], np.where(d < 0, self.m_height[..., i-1], self.m_height[..., i+1]))
                    positions = (self.m_pos[..., i], np.where(d < 0, self.m_pos[..., i-1], self.m_pos[..., i+1]))
                    self.m_height[..., i] = self._linear(heights, positions, d)
                self.m_pos[..., i] += d

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
        q_new = q_i + d[..., None]*((q_d - q_i) / (n_d - n_i))
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
        d = np.asarray(d)
        q1, q2, q3 = heights
        n1, n2, n3 = positions
        q_new = q2 + d[..., None] / (n3 - n1) * ((n2 - n1 + d[..., None])
                                                 * (q3 - q2) / (n3 - n2)
                                                 + (n3 - n2 - d[..., None])
                                                 * (q2 - q1) / (n2 - n1))
        return q_new.squeeze()

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
