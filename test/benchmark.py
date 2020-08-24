#!/usr/bin/env python3

import generatorpipeline as gp
import time

def pass_serial_nopipe(el):
    return el

@gp.pipeline()
def pass_serial(el):
    return el

nprocs = 4
@gp.pipeline(nprocs)
def pass_parallel(el):
    return el


@gp.pipeline(nprocs)
def work(el):
    for _ in range(100):
        el = pass_serial(el)
    return el


def main(n=1e3):
    n = int(n)

    gen = iter(range(n))
    t0 = time.time()
    for el in gen:
        pass
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('empty for loop: {:.3f} us/iter'.format(passtime))

    gen = iter(range(n))
    t0 = time.time()
    for el in gen:
        pass_serial_nopipe(el)
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('fcall in for loop: {:.3f} us/iter'.format(passtime))

    gen = iter(range(n))
    t0 = time.time()
    for el in gen:
        pass_serial(el)
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('pipe fcall in for loop: {:.3f} us/iter'.format(passtime))

    gen = iter(range(n))
    gen = pass_serial(gen)
    t0 = time.time()
    for el in gen:
        pass
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('pass 1 pipe: {:.3f} us/iter (serial)'.format(passtime))

    gen = iter(range(n))
    gen = pass_parallel(gen)
    t0 = time.time()
    for el in gen:
        pass
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('pass 1 pipe: {:.3f} us/iter (parallel)'.format(passtime))

    gen = iter(range(n))
    for _ in range(10):
        gen = pass_serial(gen)
    t0 = time.time()
    for el in gen:
        pass
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('pass 10 pipe: {:.3f} us/iter (serial)'.format(passtime))

    gen = iter(range(n//10))
    for _ in range(10):
        # you should actually never to that, as each pass_parallel call
        # spwans its own set of `nprocs` many processes and the data will be sent
        # between them. This setup results in a total of 10*nprocs many processes.
        # Always use parallel pipelines at the highest  possible level.
        gen = pass_parallel(gen)
    t0 = time.time()
    for el in gen:
        pass
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('pass 10 pipe: {:.3f} us/iter (parallel each -- dont do that!)'.format(passtime))

    gen = iter(range(n))
    for _ in range(100):
        gen = pass_serial(gen)
    t0 = time.time()
    for el in gen:
        pass
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('pass 100 pipe: {:.3f} us/iter (serial)'.format(passtime))


    gen = iter(range(n))
    gen = work(gen)
    t0 = time.time()
    for el in gen:
        pass
    t1 =  time.time()
    passtime = (t1 - t0) * 1e6 / n
    print('pass 100 pipe: {:.3f} us/iter (parallel good)'.format(passtime))


if __name__ == '__main__':
    main()
