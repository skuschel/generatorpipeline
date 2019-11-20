# Copyright (C) 2019 Stephan Kuschel


import types


__all__ = ['isgenerator']


def isgenerator(x):
    return isinstance(x, types.GeneratorType)
