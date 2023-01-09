#!/usr/bin/env python3

# Robert Radloff 2022

import unittest
import numpy as np
import generatorpipeline as gp

sample_values = [0.02, 0.15, 0.74, 3.39, 0.83, 22.37, 10.15, 15.43, 38.62, 15.92, 34.6, 10.28, 1.47, 0.4, 0.05, 11.39,
                 0.27, 0.42, 0.09, 11.37, -100]  # Observables given in the paper are wrong, 2nd value must be 0.15 not 0.5

# sample position taken from the paper. The values are reduced by 1
# to account for python indexing starting at 0, whereas the paper
# starts indexing with 1.

sample_m_pos = [[0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 5],
                [0, 1, 2, 4, 6],
                [0, 1, 3, 5, 7],
                [0, 2, 4, 6, 8],
                [0, 2, 4, 6, 9],
                [0, 2, 5, 7, 10],
                [0, 2, 5, 8, 11],
                [0, 3, 6, 9, 12],
                [0, 4, 7, 10, 13],
                [0, 4, 7, 11, 14],
                [0, 4, 7, 12, 15],
                [0, 4, 8, 12, 16],
                [0, 5, 9, 13, 17],
                [0, 5, 9, 14, 18],
                [0, 5, 9, 15, 19],
                [0, 5, 10, 15, 20]]

sample_m_height = [[0.02, np.nan, np.nan, np.nan, np.nan],
                    [0.02, 0.15, np.nan, np.nan, np.nan],
                    [0.02, 0.15, 0.74, np.nan, np.nan],
                    [0.02, 0.15, 0.74, 3.39, np.nan],
                    [0.02, 0.15, 0.74, 0.83, 3.39],
                    [0.02, 0.15, 0.74, 0.83, 22.37],
                    [0.02, 0.15, 0.74, 4.465, 22.37],
                    [0.02, 0.15, 2.178333333333333, 8.592500000000001, 22.37],
                    [0.02, 0.8694444444444442, 4.752685185185185, 15.516990740740741, 38.62],
                    [0.02, 0.8694444444444442, 4.752685185185185, 15.516990740740741, 38.62],
                    [0.02, 0.8694444444444442, 9.274704861111111, 21.572663194444445, 38.62],
                    [0.02, 0.8694444444444442, 9.274704861111111, 21.572663194444445, 38.62],
                    [0.02, 2.1324631076388885, 9.274704861111111, 21.572663194444445, 38.62],
                    [0.02, 2.1324631076388885, 9.274704861111111, 21.572663194444445, 38.62],
                    [0.02, 0.7308431712962957, 6.297302000661376, 21.572663194444445, 38.62],
                    [0.02, 0.7308431712962957, 6.297302000661376, 21.572663194444445, 38.62],
                    [0.02, 0.5886745370370365, 6.297302000661376, 17.203904274140214, 38.62],
                    [0.02, 0.5886745370370365, 6.297302000661376, 17.203904274140214, 38.62],
                    [0.02, 0.4938954475308638, 4.440634353260338, 17.203904274140214, 38.62],
                    [0.02, 0.4938954475308638, 4.440634353260338, 17.203904274140214, 38.62],
                    [-100, -8.37393820297956, 4.440634353260338, 13.463286481667751, 38.62]]  # Last value is custom


class TestQuantile(unittest.TestCase):
    def test_1d_paperdata(self):
        quant = gp.accumulators.QuantileEstimator(p=0.5)
        for v, p, h in zip(sample_values, sample_m_pos, sample_m_height):
            quant.accumulate(v)
            n, pos, height = quant._debug_info
            pos, height = list(pos), list(height)
            self.assertEqual(p, pos, f'Position test failed after value {n}: {v}.')
            self.assertTrue(np.allclose(h, height, equal_nan=True), f'Height test failed after value {n}: {v}.')

    def test_1d_median(self):
        median = gp.accumulators.MedianEstimator()
        for v, p, h in zip(sample_values, sample_m_pos, sample_m_height):
            median.accumulate(v)
            n, pos, height = median._debug_info
            self.assertEqual(p, pos, f'Position test failed after value {n}: {v}.')
            for a, b in zip(h, height):
                self.assertAlmostEqual(a, b, 10, f'Height test failed after value {n}: {v}.')


if __name__ == '__main__':
    unittest.main()
