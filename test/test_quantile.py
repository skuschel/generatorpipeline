#!/usr/bin/env python3

# Robert Radloff 2022
# Stephan Kuschel 2022-2023

import unittest
import generatorpipeline as gp


# These sample values are taken from the Jain and Chlamtac (https://doi.org/10.1145/4372.4378).
# Detailed test, including also testing internal positions and heights in every step
# have been removed due to convenience for testing entire numpy arrays.
# Last commit that includes also checking positions and heights is
# 00e53fb038b4610326f55a91e450cfb156070fc2

paper_sample_values = [0.02, 0.15, 0.74, 3.39, 0.83, 22.37, 10.15, 15.43, 38.62, 15.92, 34.6, 10.28, 1.47, 0.4, 0.05, 11.39,
                 0.27, 0.42, 0.19, 11.37, -100]  # There is a typo in the paper: 2nd value must be 0.15 not 0.5

paper_sample_medianestimates = [None, None, 0.74, 0.74, 0.74, 0.74, 0.74, 2.178333333333333, 4.752685185185185,
                          4.752685185185185, 9.274704861111111, 9.274704861111111, 9.274704861111111,
                          9.274704861111111, 6.297302000661376, 6.297302000661376, 6.297302000661376,
                          6.297302000661376, 4.440634353260338, 4.440634353260338, 4.440634353260338]

class TestMedian(unittest.TestCase):

    def test_1d_median(self):
        median = gp.accumulators.MedianEstimator()
        for v, m in zip(paper_sample_values, paper_sample_medianestimates):
            median.accumulate(v)
            self.assertAlmostEqual(median.value, m, msg='median {} and expected median {} differ after sample value {}'.format(median.value, m, v))



if __name__ == '__main__':
    unittest.main()
