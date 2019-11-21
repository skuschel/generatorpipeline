# Stephan Kuschel, 2019


'''
The generatorpipeline package provides a clean and easy interface to build a data processing pipeline
using python generators and the multiprocessing library.
'''

from .generatorpipeline import *
from .helper import *


__all__ = []
__all__ += generatorpipeline.__all__
__all__ += helper.__all__
