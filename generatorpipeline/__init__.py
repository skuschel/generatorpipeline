# Stephan Kuschel, 2019


'''
The generatorpipeline package provides a clean and easy interface to build
a data processing pipeline using python generators and the multiprocessing library.
'''

from .generatorpipeline import pipeline
from .helper import isgenerator


__all__ = ['pipeline']
__all__ += ['isgenerator']
