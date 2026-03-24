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
import os
import pytest
import numpy as np
from scipy.sparse import issparse

from beam_networks.problem import ElasticNetwork
from beam_networks.io.formats import _to_tar


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TEST_PATH = os.path.dirname(os.path.abspath(__file__))


_PROPS = {'name': 'circle', 'radius': 0.05, 'E': 1.0, 'nu': 0.3}


def _make_network(matrix):
    nodes = np.loadtxt(os.path.join(TEST_PATH, 'triangular.nodes'))
    edges = np.loadtxt(os.path.join(TEST_PATH, 'triangular.edges')).astype(int)
    return ElasticNetwork(
        nodes, edges,
        beam_prop=_PROPS, valid=False, outdir='data',
        options={'matrix': matrix, 'vectorize': True, 'verbose': False})


def _make_solved_network(matrix):
    """Small 4-node square network with BCs and a solution."""
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
    p = ElasticNetwork(
        nodes, edges,
        beam_prop=_PROPS, valid=True, outdir='.',
        options={'matrix': matrix, 'vectorize': True, 'verbose': False})
    p.add_BC('bot', 'D', 'node', [0, 1], [0., 0., 0.])
    p.add_BC('top', 'D', 'node', [2, 3], [None, 0.01, None])
    p.solve()
    return p


def _assert_networks_equal(p1, p2):
    """Check that two ElasticNetwork instances carry identical state."""
    np.testing.assert_array_equal(p1._nodes, p2._nodes)
    np.testing.assert_array_equal(p1._edges, p2._edges)
    np.testing.assert_array_equal(p1._active_edges, p2._active_edges)
    np.testing.assert_equal(p1._beam_prop, p2._beam_prop)
    np.testing.assert_equal(p1._bc, p2._bc)

    if issparse(p1._K):
        np.testing.assert_array_almost_equal(p1._K.todense(), p2._K.todense())
    else:
        np.testing.assert_array_almost_equal(p1._K, p2._K)

    assert p1.has_solution == p2.has_solution
    if p1.has_solution:
        np.testing.assert_array_almost_equal(p1.sol, p2.sol)
        np.testing.assert_array_almost_equal(p1._sVM, p2._sVM)


# ---------------------------------------------------------------------------
# New npz format
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('matrix', ['bsr', 'dense'])
def test_npz_roundtrip_no_solution(tmp_path, matrix):
    """Networks without a solution survive a .npz roundtrip."""
    p1 = _make_network(matrix)
    archive = tmp_path / 'checkpoint.npz'

    p1.save(str(archive))
    assert archive.exists()

    p2 = ElasticNetwork.load(str(archive), recompute=False)
    _assert_networks_equal(p1, p2)


def test_npz_roundtrip_with_solution(tmp_path):
    """Networks that have been solved survive a .npz roundtrip."""
    p1 = _make_solved_network('bsr')

    archive = tmp_path / 'checkpoint.npz'
    p1.save(str(archive))

    p2 = ElasticNetwork.load(str(archive), recompute=False)
    _assert_networks_equal(p1, p2)


def test_npz_suffix_appended_automatically(tmp_path):
    """save() without .npz suffix: numpy appends it; load() still finds it."""
    p1 = _make_network('bsr')
    stem = str(tmp_path / 'checkpoint')  # no extension

    p1.save(stem)
    assert (tmp_path / 'checkpoint.npz').exists()

    p2 = ElasticNetwork.load(stem, recompute=False)
    _assert_networks_equal(p1, p2)


# ---------------------------------------------------------------------------
# Legacy tar.gz format (backward compatibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('matrix', ['bsr', 'dense'])
def test_tar_roundtrip_backward_compat(tmp_path, matrix):
    """Legacy .tar.gz files written by _to_tar are still loadable."""
    p1 = _make_network(matrix)
    archive = str(tmp_path / 'legacy.tar.gz')

    _to_tar(archive, p1)
    assert os.path.exists(archive)

    p2 = ElasticNetwork.load(archive, recompute=False)
    _assert_networks_equal(p1, p2)


@pytest.mark.parametrize('matrix', ['bsr', 'dense'])
def test_tar_roundtrip_with_solution(tmp_path, matrix):
    """Legacy .tar.gz with a stored solution is restored correctly."""
    p1 = _make_solved_network(matrix)

    archive = str(tmp_path / 'legacy_solved.tar.gz')
    _to_tar(archive, p1)

    p2 = ElasticNetwork.load(archive, recompute=False)
    _assert_networks_equal(p1, p2)
