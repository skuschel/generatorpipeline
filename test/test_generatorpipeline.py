#!/usr/bin/env python3

import unittest
import generatorpipeline as gp


@gp.pipeline()
def square_serial(el):
    return el**2


@gp.pipeline(2)
def square_parallel(el):
    return el**2


class _TestPipeline():

    def test_el(self):
        # function should behave as undecorated with normal arguments
        r = self.squaref(7)
        self.assertEqual(r, 49)

    def test_gen(self):
        # do the job for each element in the generator
        gen = (i for i in range(20))
        gen = self.squaref(gen)
        sq = [i**2 for i in range(20)]
        self.assertListEqual(list(gen), sq)


class TestPipeline_serial(_TestPipeline, unittest.TestCase):

    def setUp(self):
        self.squaref = square_serial


class TestPipeline_parallel(_TestPipeline, unittest.TestCase):

    def setUp(self):
        self.squaref = square_parallel


if __name__ == '__main__':
    unittest.main()
