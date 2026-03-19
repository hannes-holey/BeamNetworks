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

import pytest
import numpy as np
from beam_networks.problem import ElasticNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROPS_CIRCLE = {'name': 'circle', 'radius': 0.05, 'E': 1.0, 'nu': 0.3}
_OPTS = {'matrix': 'bsr', 'vectorize': True, 'verbose': False}


def _make_2d_cantilever(n=11, length=10.0):
    """Horizontal 2D cantilever, tip loaded in y."""
    x = np.linspace(0., length, n)
    nodes = np.column_stack([x, np.zeros(n)])
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    p = ElasticNetwork(nodes, edges, beam_prop=_PROPS_CIRCLE, valid=True, options=_OPTS)
    p.add_BC('0', 'D', 'node', [0], [0., 0., 0.])
    p.add_BC('1', 'N', 'node', [n - 1], [0., 1e-3, 0.])
    return p


def _make_3d_cantilever(n=11, length=10.0):
    """Horizontal 3D cantilever, tip loaded in y."""
    x = np.linspace(0., length, n)
    nodes = np.column_stack([x, np.zeros(n), np.zeros(n)])
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    p = ElasticNetwork(nodes, edges, beam_prop=_PROPS_CIRCLE, valid=True, options=_OPTS)
    p.add_BC('0', 'D', 'node', [0], [0., 0., 0., 0., 0., 0.])
    p.add_BC('1', 'N', 'node', [n - 1], [0., 1e-3, 0., 0., 0., 0.])
    return p


# ---------------------------------------------------------------------------
# von Mises stress: mode='max' vs mode='mean'
# ---------------------------------------------------------------------------

def test_stress_mode_max_ge_mean():
    """mode='max' >= mode='mean' element-wise."""
    p = _make_2d_cantilever()
    p.solve(stress_mode='max')
    s_max = p._sVM.copy()

    p.compute_equivalent_stress(mode='mean')
    s_mean = p._sVM.copy()

    assert np.all(s_max >= s_mean - 1e-12)


def test_stress_mode_mean_shape():
    p = _make_2d_cantilever()
    p.solve(stress_mode='mean')
    assert p._sVM.shape == (p.num_edges,)


def test_stress_mode_max_shape():
    p = _make_2d_cantilever()
    p.solve(stress_mode='max')
    assert p._sVM.shape == (p.num_edges,)


def test_stress_3d_mode_max_ge_mean():
    """Same mode comparison for 3D (exercises _vmises_stress_3d)."""
    p = _make_3d_cantilever()
    p.solve(stress_mode='max')
    s_max = p._sVM.copy()

    p.compute_equivalent_stress(mode='mean')
    s_mean = p._sVM.copy()

    assert np.all(s_max >= s_mean - 1e-12)


# ---------------------------------------------------------------------------
# Bending-to-total stress ratio
# ---------------------------------------------------------------------------

def test_compute_ratio_2d():
    """compute_ratio returns an array in [0, 1] for 2D."""
    p = _make_2d_cantilever()
    p.solve()
    ratio = p.compute_ratio()

    assert ratio.shape == (p.num_edges,)
    assert np.all(ratio >= 0.)
    assert np.all(ratio <= 1. + 1e-12)


def test_compute_ratio_3d_raises():
    """compute_ratio raises RuntimeError for 3D (not implemented)."""
    p = _make_3d_cantilever()
    p.solve()
    with pytest.raises(RuntimeError, match="Not implemented for 3d"):
        p.compute_ratio()


# ---------------------------------------------------------------------------
# Principal stresses
# ---------------------------------------------------------------------------

def test_compute_principal_stresses_2d():
    """2D principal stresses: shape (num_edges, 3) and sorted descending."""
    p = _make_2d_cantilever()
    p.solve()
    pS = p.compute_principal_stresses()

    assert pS.shape == (p.num_edges, 3)
    # Columns should be sorted in descending order
    assert np.all(pS[:, 0] >= pS[:, 1] - 1e-12)
    assert np.all(pS[:, 1] >= pS[:, 2] - 1e-12)


def test_compute_principal_stresses_3d():
    """3D principal stresses: shape (num_edges, 3)."""
    p = _make_3d_cantilever()
    p.solve()
    pS = p.compute_principal_stresses()

    assert pS.shape == (p.num_edges, 3)
    assert np.all(pS[:, 0] >= pS[:, 1] - 1e-12)
    assert np.all(pS[:, 1] >= pS[:, 2] - 1e-12)


# ---------------------------------------------------------------------------
# Co-rotational vs linear von Mises stress convergence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ndim', [2, 3])
def test_corot_stress_converges_to_linear(ndim):
    """Co-rotational von Mises stress converges to the analytic limit as load -> 0.

    For a cantilever with a transverse tip load P the exact (small-rotation)
    bending stress at a cross-section located at arc-length coordinate x from
    the fixed end is::

        sigma(x) = P * (L - x) * y_max / Iz

    As the tip load increases the nonlinear solver accounts for geometric
    stiffening, so the actual bending moments — and thus stresses — are lower
    than the linear prediction.  The test verifies that the relative error < 0.01%
    at a very small load.

    """
    from beam_networks.geometry.geo import get_geometric_props

    ne = 10
    length = 10.
    props = {'name': 'circle', 'radius': 0.05, 'E': 2e11, 'nu': 0.}
    opts = {'matrix': 'bsr', 'vectorize': True, 'verbose': False}

    _, Iz, _, _, _, _ = get_geometric_props(props)
    ymax = props['radius']
    P_ref = 2. * np.pi * props['E'] * Iz / length**2   # tip-force scale

    x = np.linspace(0., length, ne + 1)
    edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])
    # Analytic bending stress per element: maximum at the left (stiffer) end
    sigma_scale = (length - x[:-1]) * ymax / Iz

    load_fraction = 1e-4

    P = load_fraction * P_ref
    sigma_analytic = P * sigma_scale

    if ndim == 2:
        nodes = np.column_stack([x, np.zeros(ne + 1)])
        net_nl = ElasticNetwork(nodes, edges, beam_prop=props, valid=True, options=opts)
        net_nl.add_BC('fix', 'D', 'node', [0], [0., 0., 0.])
        net_nl.add_BC('load', 'N', 'node', [ne], [0., P, 0.])
        net_nl.solve_nonlinear(n_steps=5, tol=1e-12, verbose=False)
    else:
        nodes = np.column_stack([x, np.zeros(ne + 1), np.zeros(ne + 1)])
        ref_vecs = np.tile([0., 0., 1.], (ne, 1))
        net_nl = ElasticNetwork(nodes, edges, beam_prop=props, valid=True, options=opts)
        net_nl.add_BC('fix', 'D', 'node', [0], [0., 0., 0., 0., 0., 0.])
        net_nl.add_BC('load', 'N', 'node', [ne], [0., P, 0., 0., 0., 0.])
        net_nl.solve_nonlinear(n_steps=5, tol=1e-12, verbose=False,
                               ref_vectors=ref_vecs)

    error = np.max(np.abs(net_nl._sVM - sigma_analytic) / sigma_analytic)

    assert error < 1e-4, f"Small-load error too large: {error:.2e}"
