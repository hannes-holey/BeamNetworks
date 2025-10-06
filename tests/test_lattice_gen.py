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
