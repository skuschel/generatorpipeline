#!/usr/bin/env python3

# Stephan Kuschel 2021

import unittest
import generatorpipeline as gp
import numpy as np


class _TestAccumulator():

    def test_1d(self):
        np.random.seed(42)
        data = np.random.random(100)
        acc = self.testacc()
        for d in data:
            acc += d
        np.testing.assert_array_almost_equal(acc.value, self.ref(data))
        self.assertEqual(acc.n, len(data))

    def test_1d_accumulate_other(self):
        np.random.seed(42)
        data = np.random.random(100)
        acc = self.testacc()
        acc2 = self.testacc()
        for d in data[:len(data)//2]:
            acc += d
        for d in data[len(data)//2:]:
            acc2 += d
        acc += acc2
        np.testing.assert_array_almost_equal(acc.value, self.ref(data))
        self.assertEqual(acc.n, len(data))

    def test_2d(self):
        np.random.seed(42)
        data = np.random.random((100, 5))
        acc = self.testacc()
        for d in data:
            acc += d
        np.testing.assert_array_almost_equal(acc.value, self.ref(data))
        self.assertEqual(acc.n, len(data))

    def test_2d(self):
        np.random.seed(42)
        data = np.random.random((100, 5))
        acc = self.testacc()
        acc2 = self.testacc()
        for d in data[:len(data)//2]:
            acc += d
        for d in data[len(data)//2:]:
            acc2 += d
        acc += acc2
        np.testing.assert_array_almost_equal(acc.value, self.ref(data))
        self.assertEqual(acc.n, len(data))


class TestMean(_TestAccumulator, unittest.TestCase):

    def testacc(self):
        return gp.accumulators.Mean()

    def ref(self, x):
        # reference function
        return np.mean(x, axis=0)


class TestVariance(_TestAccumulator, unittest.TestCase):

    def testacc(self):
        return gp.accumulators.Variance()

    def ref(self, x):
        # reference function
        return np.var(x, axis=0, ddof=1)


class TestCovariance(_TestAccumulator, unittest.TestCase):

    def testacc(self):
        return gp.accumulators.Covariance()

    def ref(self, x):
        # reference function
        return np.cov(x.T)


if __name__ == '__main__':
    unittest.main()
