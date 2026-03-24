#
# Copyright 2026 Hannes Holey
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
from beam_networks.fracture import FractureProblem


# ---------------------------------------------------------------------------
# Helper: small 2D square network amenable to fracture
#
#   2 ---- 3        BC '1': Dirichlet uy at top nodes (2, 3)
#   |      |        BC '0': Dirichlet all DOF at bottom nodes (0, 1)
#   0 ---- 1
# ---------------------------------------------------------------------------

_BEAM_PROP = {'name': 'circle', 'radius': 0.05, 'E': 1.0, 'nu': 0.3, 'strength': 0.01}
_OPTS = {'matrix': 'bsr', 'vectorize': True, 'verbose': False}


def _make_square_fracture_problem():
    nodes = np.array([[0., 0.],
                      [1., 0.],
                      [0., 1.],
                      [1., 1.]])
    edges = np.array([[0, 1],   # bottom horizontal
                      [0, 2],   # left vertical
                      [1, 3],   # right vertical
                      [2, 3]])  # top horizontal

    p = FractureProblem(nodes, edges, beam_prop=_BEAM_PROP, valid=True, options=_OPTS)
    # BC '0': clamp bottom nodes fully
    p.add_BC('0', 'D', 'node', [0, 1], [0., 0., 0.])
    # BC '1': prescribe uy at top nodes (run() will modify this)
    p.add_BC('1', 'D', 'node', [2, 3], [None, 0., None])
    return p


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_fracture_init():
    p = _make_square_fracture_problem()
    assert p.has_stiffness          # assemble_on_init=True by default
    assert p.has_bc
    assert p.num_edges == 4
    assert p.num_nodes == 4
    assert hasattr(p, '_output')


def test_fracture_removed_edges_initially_empty():
    p = _make_square_fracture_problem()
    assert len(p.removed_edges) == 0


# ---------------------------------------------------------------------------
# failure_criterion modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('crit', ['vM', 'pS', 'pE'])
def test_failure_criterion_modes(crit):
    """All three criteria should return an array of length num_edges."""
    p = _make_square_fracture_problem()
    p.solve()
    f = p.failure_criterion(name=crit)
    assert f.shape == (p.num_edges,)
    assert np.all(np.isfinite(f))


def test_failure_criterion_bad_name():
    p = _make_square_fracture_problem()
    p.solve()
    with pytest.raises(AssertionError):
        p.failure_criterion(name='bad')


# ---------------------------------------------------------------------------
# _crack_edge
# ---------------------------------------------------------------------------

def test_crack_edge_removes_edge():
    p = _make_square_fracture_problem()
    p.solve()
    n_before = p.num_edges
    p._crack_edge(0)
    assert p.num_edges == n_before - 1


def test_crack_edge_updates_stiffness():
    """Cracking an edge modifies _K (it should no longer equal the original)."""
    p = _make_square_fracture_problem()
    p.solve()
    K_before = p._K.copy()
    p._crack_edge(1)
    # The stiffness matrix should have changed
    diff = (p._K - K_before)
    assert diff.nnz > 0 or np.any(diff.toarray() != 0)


# ---------------------------------------------------------------------------
# run() — full fracture simulation
# ---------------------------------------------------------------------------

def test_fracture_run_completes(tmp_path, monkeypatch):
    """After run(), the original network is restored; during run it disconnected."""
    pytest.importorskip('meshio')
    monkeypatch.chdir(tmp_path)

    p = _make_square_fracture_problem()
    n_edges_before = p.num_edges
    p.run(mode='cascade', sign=-1)

    # Output buffers should have been appended
    assert len(p._output['stress_strain']) == 1
    assert len(p._output['removed_edges']) == 1
    assert len(p._output['cracked_edges']) == 1

    # run() restores the network
    assert p.num_edges == n_edges_before


def test_fracture_run_vtk_written(tmp_path, monkeypatch):
    """VTK files should be created in the working directory during run()."""
    pytest.importorskip('meshio')
    monkeypatch.chdir(tmp_path)

    p = _make_square_fracture_problem()
    p.run(mode='cascade', sign=-1)

    vtk_files = list(tmp_path.glob('fracture-*.vtk'))
    assert len(vtk_files) >= 1


# ---------------------------------------------------------------------------
# write() / _write_hdf5
# ---------------------------------------------------------------------------

def test_write_hdf5(tmp_path, monkeypatch):
    """write() produces a valid HDF5 file with one group per run."""
    pytest.importorskip('meshio')
    h5py = pytest.importorskip('h5py')
    import numpy as np

    monkeypatch.chdir(tmp_path)

    p = _make_square_fracture_problem()
    p._outdir = str(tmp_path)
    p.run(mode='cascade', sign=-1)
    p.run(mode='cascade', sign=-1)
    p.write()

    hdf_file = tmp_path / 'data.h5'
    assert hdf_file.exists()

    expected_keys = {'fracture_energy', 'stress_strain', 'avalanche_size',
                     'removed_edges', 'cracked_edges'}

    with h5py.File(hdf_file, 'r') as f:
        assert len(f) == 2                          # two runs
        for grp_name in f:
            assert set(f[grp_name].keys()) == expected_keys
        # stress_strain must be 2-column
        ss = f['0000']['stress_strain'][()]
        assert ss.ndim == 2 and ss.shape[1] == 2
