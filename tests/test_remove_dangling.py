#
# Copyright 2025 Hannes Holey
#
# This file is part of beam_networks. beam_networks is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. beam_networks is distributed in
# the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# beam_networks. If not, see <https://www.gnu.org/licenses/>.
#
import numpy as np
from beam_networks.utils import _remove_isolated_nodes_edges


def test_linear_chain():

    _x = np.linspace(0., 1., 5)
    nodes = np.vstack([_x, np.zeros_like(_x)]).T
    _xid = np.arange(4)
    _yid = np.arange(1, 5)
    edges = np.vstack([_xid, _yid]).T

    new_nodes, new_edges = _remove_isolated_nodes_edges(nodes, edges, max_depth=1)

    np.testing.assert_array_equal(new_nodes, nodes[1:-1])
    np.testing.assert_array_equal(new_edges, edges[:2])

    new_nodes, new_edges = _remove_isolated_nodes_edges(nodes, edges, max_depth=2)

    np.testing.assert_array_equal(new_nodes, np.empty(shape=(0, 2)))
    np.testing.assert_array_equal(new_edges, np.empty(shape=(0, 2)))


def test_cube():
    """

    +
    |
    +     +
    |    /
    +---+           +---+
    |   |   ==>     |   |
    +---+           +---+


    """

    nodes = np.array([[0., 0.],  # 0
                      [1., 0.],  # 1
                      [1., 1.],  # 2
                      [0., 1.],  # 3
                      [1., 1.],  # 4
                      [2., 2.],  # 5
                      [0., 2.],  # 6
                      [0., 3.],  # 7
                      ])

    edges = np.array([[0, 1],
                      [0, 3],
                      [1, 2],
                      [2, 3],
                      [2, 5],
                      [3, 6],
                      [3, 7],
                      ])

    new_nodes, new_edges = _remove_isolated_nodes_edges(nodes, edges)

    np.testing.assert_array_equal(new_nodes, nodes[:4])
    np.testing.assert_array_equal(new_edges, edges[:4])
