#!/usr/bin/env python

# Copyright (C) 2019 Stephan Kuschel

from setuptools import setup
import versioneer


setup(name='generatorpipeline',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author='Stephan Kuschel',
      author_email='stephan.kuschel@gmail.com',
      description='Build data-processing pipelines with generators.')
