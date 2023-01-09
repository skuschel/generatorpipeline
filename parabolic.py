import numpy as np
import generatorpipeline as gp

parabolic = gp.accumulators.CDFEstimator._parabolic


def parabolic_ndim(q, n, d):
    '''
    Calculate marker height at the new position
    using the piecewise parabolic formula described in
    https://doi.org/10.1145/4372.4378.
    '''
    if len(q) != 3:
        raise ValueError('q does not contain 3 elements!')
    if len(n) != 3:
        raise ValueError('n does not contain 3 elements!')

    #if not all([n[i] <= n[i+1] for i in range(len(n)-1)]):
    #    raise ValueError('n must be sorted!')
    q1, q2, q3 = q
    n1, n2, n3 = n
    q_new = q2 + d / (n3 - n1) * ((n2 - n1 + d)
                                  * (q3 - q2) / (n3 - n2)
                                  + (n3 - n2 - d)
                                  * (q2 - q1) / (n2 - n1))
    return q_new


def main():
    np.random.seed(42)
    test = np.sort(np.random.random(size=(4, 10, 5)), axis=-1)
    print(test)


if __name__ == '__main__':
    main()
