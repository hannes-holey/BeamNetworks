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
import pytest
import numpy as np
from scipy.sparse import issparse

from beam_networks.problem import BeamNetwork
from beam_networks.utils import get_edges_from_disks


@pytest.mark.parametrize('a0,a1,v', [('lil', 'dense', True),
                                     ('lil', 'bsr', True),
                                     ('lil', 'dense', False),
                                     ('lil', 'bsr', False), ])
def test_K_assembly(a0, a1, v):

    # Example usage
    E = 1.
    nu = 0.3
    R = 0.05
    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

    test_path = os.path.dirname(os.path.abspath(__file__))

    nodes = np.loadtxt(os.path.join(test_path, 'triangular.nodes'))
    edges = np.loadtxt(os.path.join(test_path, 'triangular.edges')).astype(int)

    p1 = BeamNetwork(nodes,
                     edges,
                     beam_prop=props,
                     valid=False,
                     outdir='data',
                     options={'matrix': a0, 'vectorize': v, 'verbose': True})

    p2 = BeamNetwork(nodes,
                     edges,
                     beam_prop=props,
                     valid=False,
                     outdir='data',
                     options={'matrix': a1, 'vectorize': v, 'verbose': True})

    if issparse(p1._K):
        K1 = p1._K.todense()
    else:
        K1 = p1._K

    if issparse(p2._K):
        K2 = p2._K.todense()
    else:
        K2 = p2._K

    np.testing.assert_almost_equal(K1, K2)


@pytest.fixture(scope="session")
def system():
    test_path = os.path.dirname(os.path.abspath(__file__))
    nodes, edges, Lx, Ly = get_edges_from_disks(os.path.join(test_path, 'hard_disks.txt'))
    yield nodes, edges, Lx, Ly


@pytest.mark.parametrize('a0,a1,v', [('dense', 'lil', True),
                                     ('dense', 'bsr', True),
                                     ('dense', 'lil', False),
                                     ('dense', 'bsr', False), ])
def test_K_assembly_pbc(system, a0, a1, v):

    # Example usage
    E = 1.
    nu = 0.3
    R = 0.05
    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

    nodes, edges, Lx, Ly = system

    p1 = BeamNetwork(nodes,
                     edges,
                     beam_prop=props,
                     valid=False,
                     outdir='data',
                     periodic=[True, True],
                     boxsize=(Lx, Ly),
                     options={'matrix': a0, 'vectorize': v, 'verbose': True})

    p2 = BeamNetwork(nodes,
                     edges,
                     beam_prop=props,
                     valid=False,
                     outdir='data',
                     periodic=[True, True],
                     boxsize=(Lx, Ly),
                     options={'matrix': a1, 'vectorize': v, 'verbose': True})

    if issparse(p1._K):
        K1 = p1._K.todense()
    else:
        K1 = p1._K

    if issparse(p2._K):
        K2 = p2._K.todense()
    else:
        K2 = p2._K

    np.testing.assert_almost_equal(K1, K2)
