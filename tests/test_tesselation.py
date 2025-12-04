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
import os
from json import load

import numpy as np

from beam_networks.tesselate import tesselate


def structures_2d_1():
    """
    simple square

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (4,2).
    adj : np.ndarray
        adjacency list of shape (4,2).
    cell : np.ndarray
        cell lengths of shape (2).

    """

    #
    coords = np.array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 1]])
    #
    adj = np.array([[0, 1],
                    [0, 2],
                    [1, 3],
                    [2, 3]])
    cell = np.array([1, 1])
    return coords, adj, cell, "ss-3x3x.json"


def structures_2d_2():
    """
    area centered square (2d equivalent to bcc)

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (5,2).
    adj : np.ndarray
        adjacency list of shape (8,2).
    cell : np.ndarray
        cell lengths of shape (2).

    """

    #
    coords = np.array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 1],
                       [0.5, 0.5]])
    #
    adj = np.array([[0, 1],
                    [0, 2],
                    [1, 3],
                    [2, 3],
                    [0, 4],
                    [1, 4],
                    [2, 4],
                    [3, 4]])
    cell = np.array([1, 1])
    return coords, adj, cell, "acs-3x3x.json"


def structures_2d_3():
    """
    line centered square (2d equivalent to fcc)

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (8,2).
    adj : np.ndarray
        adjacency list of shape (8,2).
    cell : np.ndarray
        cell lengths of shape (2).

    """
    #
    coords = np.array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 1],
                       [.5, 0],
                       [0, 0.5],
                       [0.5, 1],
                       [1, 0.5]])
    #
    adj = np.array([[0, 4],
                    [0, 5],
                    [1, 4],
                    [1, 7],
                    [2, 5],
                    [2, 6],
                    [3, 6],
                    [3, 7]])
    cell = np.array([1, 1])
    return coords, adj, cell, "lcs-3x3x.json"


def structures_2d_4():
    """
    test for inward lying points

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (6,2).
    adj : np.ndarray
        adjacency list of shape (7,2).
    cell : np.ndarray
        cell lengths of shape (2).

    """
    #
    coords = np.array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 1],
                       [0.5, 0.2],
                       [0.5, 0.8]])
    #
    adj = np.array([[0, 1],
                    [0, 4],
                    [1, 5],
                    [2, 4],
                    [2, 3],
                    [3, 5],
                    [4, 5]])
    cell = np.array([1, 1])
    return coords, adj, cell, "2D4-3x3x.json"


def structures_2d_5():
    """
    test for asymmetric cell

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (5,2).
    adj : np.ndarray
        adjacency list of shape (5,2).
    cell : np.ndarray
        cell lengths of shape (2).

    """
    #
    coords = np.array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 1],
                       [.5, 1.]])
    #
    adj = np.array([[0, 1],
                    [0, 2],
                    [1, 3],
                    [2, 4],
                    [3, 4]])
    cell = np.array([1, 1])
    return coords, adj, cell, "2D5-3x3x.json"


def structures_2d_6():
    """
    2nd test for inward lying points

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (6,2).
    adj : np.ndarray
        adjacency list of shape (7,2).
    cell : np.ndarray
        cell lengths of shape (2).

    """
    #
    coords = np.array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 1],
                       [0.5, 0.2],
                       [0.5, 0.8]])
    #
    adj = np.array([[0, 2],
                    [0, 4],
                    [1, 3],
                    [1, 4],
                    [2, 5],
                    [3, 5]])
    cell = np.array([1, 1])
    return coords, adj, cell, "2D6-3x3x.json"


def structure_3d_1():
    """
    sc cell

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (8,3).
    adj : np.ndarray
        adjacency list of shape (12,2).
    cell : np.ndarray
        cell lengths of shape (3).

    """
    #
    coords = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0],
                       [1, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1]])
    adj = np.array([[0, 1],
                    [0, 2],
                    [0, 3],
                    [1, 4],
                    [1, 5],
                    [2, 4],
                    [2, 6],
                    [3, 5],
                    [3, 6],
                    [4, 7],
                    [5, 7],
                    [6, 7]])
    cell = np.array([1, 1, 1])

    return coords, adj, cell, "sc-3x3x3x.json"


def structure_3d_2():
    """
    bcc cell

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (9,3).
    adj : np.ndarray
        adjacency list of shape (20,2).
    cell : np.ndarray
        cell lengths of shape (3).

    """
    #
    coords = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0],
                       [1, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1],
                       [.5, .5, .5]])
    adj = np.array([[0, 1],
                    [0, 2],
                    [0, 3],
                    [1, 4],
                    [1, 5],
                    [2, 4],
                    [2, 6],
                    [3, 5],
                    [3, 6],
                    [4, 7],
                    [5, 7],
                    [6, 7],
                    [0, 8],
                    [1, 8],
                    [2, 8],
                    [3, 8],
                    [4, 8],
                    [5, 8],
                    [6, 8],
                    [7, 8]])
    cell = np.array([1, 1, 1])

    return coords, adj, cell, "bcc-3x3x3x.json"


def structure_3d_3():
    """
    fcc cell

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (14,3).
    adj : np.ndarray
        adjacency list of shape (36,2).
    cell : np.ndarray
        cell lengths of shape (3).

    """
    #
    coords = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0],
                       [1, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1],
                       [.5, .5, 0],
                       [0, .5, .5],
                       [.5, 0, .5],
                       [1, .5, .5],
                       [.5, 1, .5],
                       [.5, .5, 1]])
    adj = np.array([[0, 1],
                    [0, 2],
                    [0, 3],
                    [1, 4],
                    [1, 5],
                    [2, 4],
                    [2, 6],
                    [3, 5],
                    [3, 6],
                    [4, 7],
                    [5, 7],
                    [6, 7],
                    [0, 8],
                    [0, 9],
                    [0, 10],
                    [1, 8],
                    [1, 10],
                    [1, 11],
                    [2, 8],
                    [2, 9],
                    [2, 12],
                    [3, 9],
                    [3, 10],
                    [3, 13],
                    [4, 8],
                    [4, 11],
                    [4, 12],
                    [5, 10],
                    [5, 11],
                    [5, 13],
                    [6, 9],
                    [6, 12],
                    [6, 13],
                    [7, 11],
                    [7, 12],
                    [7, 13]])
    cell = np.array([1, 1, 1])

    return coords, adj, cell, "fcc-3x3x3x.json"


def test_tesselation():
    pbc = [False, False, False]
    for struct in [structures_2d_1,
                   structures_2d_2,
                   structures_2d_3,
                   structures_2d_4,
                   structures_2d_5,
                   structures_2d_6,
                   structure_3d_1,
                   structure_3d_2,
                   structure_3d_3]:

        #
        coords, adj, cell, file = struct()
        # tesselate
        coords_t, adj_t, cell_t = tesselate(coords,
                                            adj,
                                            rep=np.array([3, 3, 3]),
                                            cell=cell,
                                            pbc=pbc)
        # load file with previously checked correct structures
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_files", file), "r") as f:
            solution = load(f)
        # compare coords
        np.testing.assert_almost_equal(coords_t, np.array(solution["coords"]))
        # compare adjacency list
        np.testing.assert_almost_equal(adj_t, np.array(solution["adj"]))
        # compare cells
        np.testing.assert_almost_equal(cell_t, np.array(solution["cell"]))
