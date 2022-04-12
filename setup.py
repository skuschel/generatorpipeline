#!/usr/bin/env python

# Copyright (C) 2019-2020 Stephan Kuschel
#
# This file is part of generatorpipeline.
#
# generatorpipeline is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# generatorpipeline is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with generatorpipeline. If not, see <http://www.gnu.org/licenses/>.
#

from setuptools import setup, find_packages
import versioneer


setup(name='generatorpipeline',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      packages=find_packages(include=['generatorpipeline*']),
      python_requires='>=3.6',
      license='GPLv3+',
      author='Stephan Kuschel',
      author_email='stephan.kuschel@gmail.com',
      url='https://github.com/skuschel/generatorpipeline',
      install_requires=['numpy'],
      description='Parallelize your data-processing pipelines with just a decorator.')
