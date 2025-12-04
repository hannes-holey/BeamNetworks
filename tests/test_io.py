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


@pytest.mark.parametrize('a0', ['bsr', 'dense'])
def test_K_io(tmp_path, a0):

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
                     options={'matrix': a0, 'vectorize': True, 'verbose': True})

    tarfile = os.path.join(tmp_path, 'p1.tar.gz')

    p1.save(tarfile)
    p2 = BeamNetwork.load(tarfile, recompute=False)

    # check that stiffness matrix is the same
    if issparse(p1._K):
        np.testing.assert_almost_equal(p1._K.todense(), p2._K.todense())
    else:
        np.testing.assert_almost_equal(p1._K, p2._K)

    # TODO: check more... also with BCs and solution
