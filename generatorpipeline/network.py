# Copyright (C) 2022 Stephan Kuschel
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

'''
This module offers functions for using generators over the network with zmq.
This module requires pyzmq.
'''

import zmq


def zmqgeneratorsend(gen, address):
    '''
    provides data of a generator over a zmq connection element by element.

    Example:
        `zmqgeneratorsend(gen, "tcp://*:22313")`
        Another Machine (or a different process on the same) can receive the data using
        `gen = zmqgeneratorrecv("tcp://127.0.0.1:22313")`
    '''
    context = zmq.Context()
    with context.socket(zmq.REP) as socket:
        socket.bind(address)
        for el in gen:
            _ = socket.recv()  # wait for request
            socket.send_pyobj((0, el))
        _ = socket.recv()  # wait last request
        socket.send_pyobj((None, None))


def zmqgeneratorrecv(address):
    '''
    receives data of a generator over a zmq connection element by element.

    Example:
        `zmqgeneratorsend(gen, "tcp://*:22313")`
        Another Machine (or a different process on the same) can receive the data using
        `gen = zmqgeneratorrecv("tcp://127.0.0.1:22313")`
    '''
    context = zmq.Context()
    with context.socket(zmq.REQ) as socket:
        socket.connect(address)
        while True:
            socket.send(b'next')  # request new data
            status, ret = socket.recv_pyobj()
            if status is None:
                # generator is exhausted
                return
            yield ret
