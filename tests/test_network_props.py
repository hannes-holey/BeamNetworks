#
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
import pytest
from beam_networks.network import Network


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _square_2d():
    """Unit square: 4 nodes, 4 edges."""
    nodes = np.array([[0., 0.],
                      [1., 0.],
                      [0., 1.],
                      [1., 1.]])
    edges = np.array([[0, 1],
                      [0, 2],
                      [1, 3],
                      [2, 3]])
    return Network(nodes, edges, valid=True)


def _tetra_3d():
    """Unit tetrahedron: 4 nodes, 3 edges (star from origin)."""
    nodes = np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])
    edges = np.array([[0, 1],
                      [0, 2],
                      [0, 3]])
    return Network(nodes, edges, valid=True)


# ---------------------------------------------------------------------------
# 2D network properties
# ---------------------------------------------------------------------------

def test_2d_bounds():
    net = _square_2d()
    xlo, xhi, ylo, yhi = net.bounds
    assert np.isclose(xlo, 0.)
    assert np.isclose(xhi, 1.)
    assert np.isclose(ylo, 0.)
    assert np.isclose(yhi, 1.)


def test_2d_boxsize():
    net = _square_2d()
    Lx, Ly = net.boxsize
    assert np.isclose(Lx, 1.)
    assert np.isclose(Ly, 1.)


def test_2d_lx_ly():
    net = _square_2d()
    assert np.isclose(net.Lx, 1.)
    assert np.isclose(net.Ly, 1.)


def test_2d_lz_raises():
    """Accessing Lz on a 2D network should raise IndexError."""
    net = _square_2d()
    with pytest.raises(IndexError):
        _ = net.Lz


def test_2d_bondlengths():
    net = _square_2d()
    # All 4 edges of the unit square have length 1
    assert net.bondlengths.shape == (4,)
    np.testing.assert_allclose(net.bondlengths, 1.)


def test_2d_coordination():
    """Unit square: each node has 2 neighbours → mean coordination = 2."""
    net = _square_2d()
    assert np.isclose(net.coordination, 2.)


def test_2d_pbc_edges_shape():
    net = _square_2d()
    assert net.pbc_edges.shape == (net.num_edges,)
    assert net.pbc_edges.dtype == bool


def test_2d_edge_vectors_shape():
    net = _square_2d()
    assert net.edge_vectors.shape == (net.num_edges, 2)


def test_2d_volume_equals_area():
    """ConvexHull.volume for 2D is the enclosed area (unit square = 1.0)."""
    net = _square_2d()
    assert np.isclose(net.volume, 1.0)


def test_2d_is_connected():
    net = _square_2d()
    assert net.is_connected


# ---------------------------------------------------------------------------
# 3D network properties
# ---------------------------------------------------------------------------

def test_3d_bounds():
    net = _tetra_3d()
    xlo, xhi, ylo, yhi, zlo, zhi = net.bounds
    assert np.isclose(xlo, 0.) and np.isclose(xhi, 1.)
    assert np.isclose(ylo, 0.) and np.isclose(yhi, 1.)
    assert np.isclose(zlo, 0.) and np.isclose(zhi, 1.)


def test_3d_lx_ly_lz():
    net = _tetra_3d()
    assert np.isclose(net.Lx, 1.)
    assert np.isclose(net.Ly, 1.)
    assert np.isclose(net.Lz, 1.)


def test_3d_boxsize():
    net = _tetra_3d()
    Lx, Ly, Lz = net.boxsize
    assert np.isclose(Lx, 1.)
    assert np.isclose(Ly, 1.)
    assert np.isclose(Lz, 1.)


def test_3d_bondlengths():
    net = _tetra_3d()
    assert net.bondlengths.shape == (3,)
    np.testing.assert_allclose(net.bondlengths, 1.)


def test_3d_coordination():
    """Star topology: 3 leaf nodes (degree 1), 1 hub (degree 3) → mean = 1.5."""
    net = _tetra_3d()
    # Each leaf appears once, hub appears 3 times → counts [1,1,1,3], mean=1.5
    assert np.isclose(net.coordination, 1.5)


def test_3d_is_connected():
    net = _tetra_3d()
    assert net.is_connected


# ---------------------------------------------------------------------------
# Lattice factory methods
# ---------------------------------------------------------------------------

def test_generate_cubic_lattice_sc():
    net = Network.generate_cubic_lattice(a=1., bbox=[2., 2., 2.], lattice_type='sc')
    assert net.dim == 3
    assert net.num_nodes > 0
    assert net.num_edges > 0


def test_generate_square_lattice():
    net = Network.generate_square_lattice(a=1., bbox=[3., 3.])
    assert net.dim == 2
    assert net.num_nodes > 0
    assert net.num_edges > 0
