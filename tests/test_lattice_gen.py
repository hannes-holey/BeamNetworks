#
# Copyright 2025-2026 Hannes Holey
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
from beam_networks.network import Network


def test_sc():

    bbox = (1., 1., 1.)
    lattice = Network.generate_cubic_lattice(a=1., bbox=bbox, lattice_type='sc')

    assert lattice.num_nodes == 8
    assert lattice.num_edges == 12
    assert np.all(np.amax(lattice.nodes, axis=0) <= bbox)


def test_bcc():

    bbox = (1., 1., 1.)
    lattice = Network.generate_cubic_lattice(a=1., bbox=bbox, lattice_type='bcc')

    assert lattice.num_nodes == 9
    assert lattice.num_edges == 8
    assert np.all(np.amax(lattice.nodes, axis=0) <= bbox)


def test_fcc():

    bbox = (1., 1., 1.)
    lattice = Network.generate_cubic_lattice(a=1., bbox=bbox, lattice_type='fcc')

    assert lattice.num_nodes == 14
    assert lattice.num_edges == 36
    assert np.all(np.amax(lattice.nodes, axis=0) <= bbox)


def test_sc_2d():
    bbox = (1., 1.)
    lattice = Network.generate_square_lattice(a=1., bbox=bbox, lattice_type='sc')

    assert lattice.num_nodes == 4
    assert lattice.num_edges == 4
    assert np.all(np.amax(lattice.nodes, axis=0) <= bbox)


def test_fcc_2d():
    bbox = (1., 1.)
    lattice = Network.generate_square_lattice(a=1., bbox=bbox, lattice_type='fcc')

    assert lattice.num_nodes == 5
    assert lattice.num_edges == 4


def test_dia():
    bbox = (1., 1., 1.)
    lattice = Network.generate_cubic_lattice(a=1., bbox=bbox, lattice_type='dia')

    # 4 corner atoms have all neighbours outside the bbox and are pruned.
    assert lattice.num_nodes == 14
    assert lattice.num_edges == 16
    assert np.all(np.amax(lattice.nodes, axis=0) <= bbox)


def test_dia_coordination():
    """Interior nodes of the diamond lattice must have coordination 4."""
    bbox = (2., 2., 2.)
    lattice = Network.generate_cubic_lattice(a=1., bbox=bbox, lattice_type='dia')
    deg = np.bincount(lattice.edges.ravel(), minlength=lattice.num_nodes)
    assert deg.max() == 4


def test_sc_bcc():
    bbox = (1., 1., 1.)
    lattice = Network.generate_cubic_lattice(a=1., bbox=bbox, lattice_type='sc-bcc')

    assert lattice.num_nodes == 9
    assert lattice.num_edges == 20
    assert np.all(np.amax(lattice.nodes, axis=0) <= bbox)


def test_sc_bcc_coordination():
    """Fully connected interior nodes of sc-bcc must have coordination 14."""
    bbox = (3., 3., 3.)
    lattice = Network.generate_cubic_lattice(a=1., bbox=bbox, lattice_type='sc-bcc')
    deg = np.bincount(lattice.edges.ravel(), minlength=lattice.num_nodes)
    assert deg.max() == 14


def test_triangular():
    bbox = (2., 2.)
    lattice = Network.generate_square_lattice(a=1., bbox=bbox, lattice_type='triangular')

    assert lattice.num_nodes == 8
    assert lattice.num_edges == 13
    assert np.all(np.amax(lattice.nodes, axis=0) <= bbox)


def test_triangular_coordination():
    """Interior nodes of the triangular lattice must have coordination 6."""
    bbox = (4., 4.)
    lattice = Network.generate_square_lattice(a=1., bbox=bbox, lattice_type='triangular')
    deg = np.bincount(lattice.edges.ravel(), minlength=lattice.num_nodes)
    assert deg.max() == 6


def test_hex_alias():
    """'hex' must produce the same lattice as 'triangular'."""
    bbox = (3., 3.)
    lat_tri = Network.generate_square_lattice(a=1., bbox=bbox, lattice_type='triangular')
    lat_hex = Network.generate_square_lattice(a=1., bbox=bbox, lattice_type='hex')

    assert lat_tri.num_nodes == lat_hex.num_nodes
    assert lat_tri.num_edges == lat_hex.num_edges


def test_kagome():
    bbox = (4., 4.)
    lattice = Network.generate_square_lattice(a=1., bbox=bbox, lattice_type='kagome')

    assert lattice.num_nodes == 15
    assert lattice.num_edges == 22
    assert np.all(np.amax(lattice.nodes, axis=0) <= bbox)


def test_kagome_coordination():
    """Interior nodes of the kagome lattice must have coordination 4."""
    bbox = (6., 6.)
    lattice = Network.generate_square_lattice(a=1., bbox=bbox, lattice_type='kagome')
    deg = np.bincount(lattice.edges.ravel(), minlength=lattice.num_nodes)
    assert deg.max() == 4
    assert np.all(np.amax(lattice.nodes, axis=0) <= bbox)
