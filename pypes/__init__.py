# Stephan Kuschel, 2019


'''
The pypes package provides a clean and easy interface to build a data processing pipeline
using python generators and the multiprocessing library.
'''

from .generatorpipeline import *

__all__ = []
__all__ += generatorpipeline.__all__
