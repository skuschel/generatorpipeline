#!/usr/bin/env python3

import generatorpipeline as gp
import numpy as np
import time


def bench_covariance(n):
    np.random.seed(42)
    cov = gp.accumulators.Covariance()

    t0 = time.time()
    for i in range(n):
        observations = np.random.random(500)
        cov += observations
    t1 =  time.time()
    passtime = (t1 - t0) * 1e3 / n
    print('covmatrix (500x500): {:.3f} ms/observation'.format(passtime))

    t0 = time.time()
    observations = np.random.random((n,500))
    covnp = np.cov(observations)
    t1 =  time.time()
    passtime = (t1 - t0) * 1e3 / n
    print('covmatrix numpy (500x500): {:.3f} ms/observation'.format(passtime))


def bench_variance(n):
    np.random.seed(42)
    var = gp.accumulators.Variance()

    t0 = time.time()
    for i in range(n):
        observations = np.random.random(10000)
        var += observations
    t1 = time.time()
    passtime = (t1 - t0) * 1e3 / n
    print('varaince ({:n}): {:.3f} ms/observation'.format(n, passtime))

    t0 = time.time()
    observations = np.random.random((n, 10000))
    covnp = np.var(observations)
    t1 = time.time()
    passtime = (t1 - t0) * 1e3 / n
    print('varaince numpy ({:n}): {:.3f} ms/observation'.format(n, passtime))


def bench_mean(n):
    np.random.seed(42)
    acc = gp.accumulators.Mean()

    t0 = time.time()
    for i in range(n):
        observations = np.random.random(500)
        acc += observations
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('Mean ({:n}): {:.3f} us/observation'.format(n, passtime))

    t0 = time.time()
    observations = np.random.random((n,500))
    _ = np.mean(observations, axis=0)
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('Mean numpy ({:n}): {:.3f} us/observation'.format(n, passtime))

def bench_median(n):
    np.random.seed(42)
    acc = gp.accumulators.MedianEstimator()

    t0 = time.time()
    for _ in range(n):
        obs = np.random.random()
        acc += obs
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('MedianEstimator ({:n}): {:.3f} us/observation'.format(n, passtime))

def bench_cdf(n):
    np.random.seed(42)
    acc = gp.accumulators.CDFEstimator(50)

    t0 = time.time()
    for _ in range(n):
        obs = np.random.random()
        acc += obs
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('MedianEstimator ({:n}): {:.3f} us/observation'.format(n, passtime))

def main(n=int(500)):
    bench_covariance(n)
    bench_mean(n*10)
    bench_variance(n*2)
    bench_median(500)
    bench_cdf(400)



if __name__ == '__main__':
    main()
