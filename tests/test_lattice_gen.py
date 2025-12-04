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
import pytest
import numpy as np
from beam_networks.network import Network


@pytest.mark.skip(reason="")
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
