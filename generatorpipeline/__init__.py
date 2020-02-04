# Stephan Kuschel, 2019


'''
The generatorpipeline package provides a clean and easy interface to build
a data processing pipeline using python generators and the multiprocessing library.
'''

from .generatorpipeline import pipeline
from .helper import isgenerator
from .streamfunctions import filterNone


__all__ = ['pipeline']
__all__ += ['isgenerator']
__all__ += ['filterNone']

from ._version import get_versions  # noqa
__version__ = get_versions()['version']
__git_version__ = get_versions()['full-revisionid']
# work around if zip is downloaded from github and current version does not have a tag.
if __version__ == '0+unknown':
    __version__ = __git_version__
del get_versions
