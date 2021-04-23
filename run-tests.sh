#!/usr/bin/env bash
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
# Copyright Stephan Kuschel, 2019-2020

# run all tests and pep8 verification of this project.
# It is HIGHLY RECOMMENDED to link it as a git pre-commit hook!
# Please see pre-commit for instructions.

# THIS FILE MUST RUN WITHOUT ERROR ON EVERY COMMIT!

set -o xtrace

flake8 --max-line-length=99 --show-source --statistics generatorpipeline
nosetests . --exe

./test/benchmark.py
./test/benchmark_accumulator.py
