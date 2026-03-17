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
"""3D geometrically nonlinear solver tests.

Rolling-cantilever benchmark (Crisfield 1990) extended to 3D.  A cantilever
along x is loaded with a tip moment or prescribed tip rotation; the full
2π load bends it into a complete circle whose tip returns to the origin.
Both in-plane (x-y, moment Mz) and out-of-plane (x-z, moment My) cases are
tested, together with a 3D-vs-2D consistency check.
"""
import numpy as np
import pytest

from beam_networks.problem import ElasticNetwork
from beam_networks.geometry.geo import get_geometric_props


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _cantilever_3d(ne: int, matrix: str, ref_vec=(0., 1., 0.)):
    """Return a fresh 3D cantilever ElasticNetwork with a clamped root.

    The beam runs along the x-axis from (0,0,0) to (L,0,0).

    Parameters
    ----------
    ref_vec : tuple
        Reference vector used to define the local e2 axis.  Must not be
        parallel to the element chord at any point during the deformation.
        Use ``(0, 0, 1)`` for bending in the x-y plane (chord stays in x-y,
        so z is always perpendicular) and ``(0, 1, 0)`` for bending in the
        x-z plane (chord stays in x-z, so y is always perpendicular).
    """
    Lx = 10.
    beam_prop = {'b': 0.1, 'h': 0.1, 'E': 2.e11, 'nu': 0., 'name': 'rectangle'}
    x = np.linspace(0., Lx, ne + 1)
    nodes = np.column_stack([x, np.zeros(ne + 1), np.zeros(ne + 1)])
    edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])
    net = ElasticNetwork(nodes, edges, beam_prop=beam_prop,
                         options={'vectorize': False, 'matrix': matrix,
                                  'verbose': False},
                         assemble_on_init=False)
    net.add_BC('clamp', 'D', 'node', [0], [0., 0., 0., 0., 0., 0.])
    ref = np.tile(ref_vec, (ne, 1))
    return net, Lx, beam_prop, ref


# ---------------------------------------------------------------------------
# In-plane (x-y) tests — Neumann (tip moment Mz)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ne, matrix', [
    (20, 'dense'),
    (20, 'bsr'),
    (50, 'bsr'),
])
def test_tip_moment_circle_xy(ne, matrix):
    """Tip moment Mz = 2π EI/L bends beam into a full circle in x-y plane.

    The free tip must return to the clamped-end origin after the full 2π
    load is applied.
    """
    # ref=[0,0,1]: e2=z, always ⊥ to any chord in the x-y plane → no frame singularity
    net, Lx, beam_prop, ref = _cantilever_3d(ne, matrix, ref_vec=(0., 0., 1.))
    Iz = get_geometric_props(beam_prop)[1]
    Mref = 2. * np.pi * beam_prop['E'] * Iz / Lx

    # Neumann BC: moment about z (DOF index 5 per node = θz)
    net.add_BC('load', 'N', 'node', [ne], [None, None, None, None, None, Mref])
    net.solve_nonlinear(n_steps=100, tol=1e-9, verbose=False,
                        matrix=matrix, ref_vectors=ref)

    tip = net.displaced_nodes[-1]
    np.testing.assert_allclose(tip[:2], [0., 0.], atol=1e-4)
    # No out-of-plane displacement
    np.testing.assert_allclose(tip[2], 0., atol=1e-8)


# ---------------------------------------------------------------------------
# In-plane (x-y) tests — Dirichlet (prescribed tip rotation θz)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ne, matrix', [
    (20, 'dense'),
    (20, 'bsr'),
    (50, 'bsr'),
])
def test_tip_rotation_circle_xy(ne, matrix):
    """Prescribed tip rotation θz = 2π produces a full circle in x-y plane.

    Prescribing the rotational DOF θz = 2π at the tip with translational
    DOFs free gives a uniform-curvature arc (energy-minimising shape), so
    the tip must return to the origin — identical to the tip-moment case.
    """
    net, Lx, _, ref = _cantilever_3d(ne, matrix, ref_vec=(0., 0., 1.))
    net.add_BC('tip_rot', 'D', 'node', [ne],
               [None, None, None, None, None, 2. * np.pi])
    net.solve_nonlinear(n_steps=100, tol=1e-9, verbose=False,
                        matrix=matrix, ref_vectors=ref)

    tip = net.displaced_nodes[-1]
    np.testing.assert_allclose(tip[:2], [0., 0.], atol=1e-4)
    np.testing.assert_allclose(tip[2], 0., atol=1e-8)


@pytest.mark.parametrize('ne', [20, 50])
def test_tip_rotation_semicircle_xy(ne):
    """Prescribed θz = π produces a semicircle: tip at (0, 2L/π, 0).

    For a uniform arc of total rotation θ::

        x_tip = (L/θ) sin θ,   y_tip = (L/θ)(1 − cos θ)

    For θ = π:  x_tip = 0,  y_tip = 2L/π.
    """
    net, Lx, _, ref = _cantilever_3d(ne, 'bsr', ref_vec=(0., 0., 1.))
    net.add_BC('tip_rot', 'D', 'node', [ne],
               [None, None, None, None, None, np.pi])
    net.solve_nonlinear(n_steps=100, tol=1e-9, verbose=False,
                        matrix='bsr', ref_vectors=ref)

    tip = net.displaced_nodes[-1]
    np.testing.assert_allclose(tip[0], 0., atol=1e-4)
    # y discretisation error ~ O((L/ne)²); 1 % of y_exp covers ne = 20
    np.testing.assert_allclose(tip[1], 2. * Lx / np.pi, rtol=1e-2)
    np.testing.assert_allclose(tip[2], 0., atol=1e-8)


# ---------------------------------------------------------------------------
# Out-of-plane (x-z) tests — Neumann (tip moment My)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ne, matrix', [
    (20, 'dense'),
    (20, 'bsr'),
    (50, 'bsr'),
])
def test_tip_moment_circle_xz(ne, matrix):
    """Tip moment My = 2π EI/L bends beam into a full circle in x-z plane.

    Mirrors the x-y test using the second transverse bending plane.
    My is applied at DOF θy (index 4 per node).  For a square cross-section
    Iy = Iz, so the same reference moment drives the same geometry.
    """
    net, Lx, beam_prop, ref = _cantilever_3d(ne, matrix)
    Iy = get_geometric_props(beam_prop)[0]
    Mref = 2. * np.pi * beam_prop['E'] * Iy / Lx

    # Moment about y (DOF index 4 per node = θy)
    net.add_BC('load', 'N', 'node', [ne], [None, None, None, None, Mref, None])
    net.solve_nonlinear(n_steps=100, tol=1e-9, verbose=False,
                        matrix=matrix, ref_vectors=ref)

    tip = net.displaced_nodes[-1]
    np.testing.assert_allclose(tip[0], 0., atol=1e-4)
    np.testing.assert_allclose(tip[2], 0., atol=1e-4)
    # No in-plane displacement
    np.testing.assert_allclose(tip[1], 0., atol=1e-8)


# ---------------------------------------------------------------------------
# Out-of-plane (x-z) — Dirichlet (prescribed tip rotation θy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ne', [20, 50])
def test_tip_rotation_circle_xz(ne):
    """Prescribed tip rotation θy = 2π produces a full circle in x-z plane."""
    net, Lx, _, ref = _cantilever_3d(ne, 'bsr')
    net.add_BC('tip_rot', 'D', 'node', [ne],
               [None, None, None, None, 2. * np.pi, None])
    net.solve_nonlinear(n_steps=100, tol=1e-9, verbose=False,
                        matrix='bsr', ref_vectors=ref)

    tip = net.displaced_nodes[-1]
    np.testing.assert_allclose(tip[0], 0., atol=1e-4)
    np.testing.assert_allclose(tip[2], 0., atol=1e-4)
    np.testing.assert_allclose(tip[1], 0., atol=1e-8)


# ---------------------------------------------------------------------------
# 3D vs 2D consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ne', [20])
def test_3d_matches_2d_inplane(ne):
    """3D in-plane solution must match the 2D solution exactly.

    The 3D cantilever with tip moment Mz should give the same tip x, y
    coordinates as the 2D cantilever with the same tip moment.
    """
    from beam_networks.problem import ElasticNetwork

    Lx = 10.
    beam_prop = {'b': 0.1, 'h': 0.1, 'E': 2.e11, 'nu': 0., 'name': 'rectangle'}
    Iz = get_geometric_props(beam_prop)[1]
    Mref = 2. * np.pi * beam_prop['E'] * Iz / Lx

    x = np.linspace(0., Lx, ne + 1)

    # 2D network
    net2d = ElasticNetwork(
        np.column_stack([x, np.zeros(ne + 1)]),
        np.column_stack([np.arange(ne), np.arange(ne) + 1]),
        beam_prop=beam_prop,
        options={'vectorize': False, 'matrix': 'bsr', 'verbose': False},
        assemble_on_init=False,
    )
    net2d.add_BC('clamp', 'D', 'node', [0], [0., 0., 0.])
    net2d.add_BC('load', 'N', 'node', [ne], [None, None, Mref])
    net2d.solve_nonlinear(n_steps=100, tol=1e-9, verbose=False, matrix='bsr')

    # 3D network — ref=[0,0,1] to avoid frame singularity in x-y bending
    net3d, _, _, ref = _cantilever_3d(ne, 'bsr', ref_vec=(0., 0., 1.))
    net3d.add_BC('load', 'N', 'node', [ne], [None, None, None, None, None, Mref])
    net3d.solve_nonlinear(n_steps=100, tol=1e-9, verbose=False,
                          matrix='bsr', ref_vectors=ref)

    tip2d = net2d.displaced_nodes[-1]    # (x, y)
    tip3d = net3d.displaced_nodes[-1]    # (x, y, z)

    np.testing.assert_allclose(tip3d[:2], tip2d, atol=1e-6)
    np.testing.assert_allclose(tip3d[2], 0., atol=1e-8)
