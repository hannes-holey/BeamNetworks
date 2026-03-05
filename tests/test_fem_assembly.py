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
import pytest

from beam_networks.problem import BeamNetwork
from beam_networks.geo import get_geometric_props
from beam_networks.stiffness import (_fem_element_stiffness_2d,
                                     _fem_element_stiffness_3d,
                                     get_fem_element_stiffness_global)
from beam_networks.reference_solutions.cantilever import cantilever_analytic


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

PROPS = {'name': 'circle', 'radius': 0.05, 'E': 1., 'nu': 0.3}


def _cantilever_2d(num_nodes, length, options):
    x = np.linspace(0., length, num_nodes)
    nodes = np.column_stack([x, np.zeros(num_nodes)])
    edges = np.column_stack([np.arange(num_nodes - 1), np.arange(1, num_nodes)])
    return BeamNetwork(nodes, edges, beam_prop=PROPS, valid=True, options=options)


def _tip_disp_fem(length, density, poly_order=1, n_gauss=None):
    """Two-node cantilever with transverse tip load; return uy at tip."""
    _, Iz, _, A, kappa, _ = get_geometric_props(PROPS)
    P = 0.05 * PROPS['E'] * Iz / length**3

    nodes = np.array([[0., 0.], [length, 0.]])
    edges = np.array([[0, 1]])
    opts = {'vectorize': False, 'matrix': 'dense', 'verbose': False,
            'n_elem_per_length': density,
            'fem_poly_order': poly_order,
            'fem_n_gauss': n_gauss}
    net = BeamNetwork(nodes, edges, beam_prop=PROPS, valid=True, options=opts)
    net.add_BC('fix', 'D', 'node', [0], [0., 0., 0.])
    net.add_BC('load', 'N', 'node', [1], [0., P, 0.])
    net.solve()
    return net.sol[4], P   # uy at tip, P


# ---------------------------------------------------------------------------
# Test 1: Single-element stiffness structure for 2D (p=1, n_gauss=1)
# ---------------------------------------------------------------------------

def test_fem_element_stiffness_2d_structure():
    """Linear Timoshenko element (p=1, n_gauss=1) has the expected entries."""
    l = 5.
    E, nu = PROPS['E'], PROPS['nu']
    G = E / (2 * (1 + nu))
    _, Iz, _, A, kappa, _ = get_geometric_props(PROPS)

    ea = E * A / l
    kg = kappa * G * A / l
    kg_h = kappa * G * A / 2.
    kg_l = kappa * G * A * l / 4.
    ei = E * Iz / l

    K = _fem_element_stiffness_2d(PROPS, l, n_nodes=2, n_gauss=1)

    # Symmetry
    np.testing.assert_allclose(K, K.T, atol=1e-14)

    # Diagonal entries
    assert K[0, 0] == pytest.approx(ea)
    assert K[1, 1] == pytest.approx(kg)
    assert K[2, 2] == pytest.approx(ei + kg_l)
    assert K[3, 3] == pytest.approx(ea)
    assert K[4, 4] == pytest.approx(kg)
    assert K[5, 5] == pytest.approx(ei + kg_l)

    # Coupling between transverse displacement and rotation (drives convergence)
    assert K[1, 2] == pytest.approx(kg_h)
    assert K[1, 5] == pytest.approx(kg_h)   # cross-node v–θ coupling

    # Off-diagonal rotation block
    assert K[2, 5] == pytest.approx(-ei + kg_l)


# ---------------------------------------------------------------------------
# Test 2: Single-element stiffness structure for 3D (p=1, n_gauss=1)
# ---------------------------------------------------------------------------

def test_fem_element_stiffness_3d_structure():
    """3D element (p=1, n_gauss=1) is symmetric with the expected coupling signs."""
    l = 5.
    E, nu = PROPS['E'], PROPS['nu']
    G = E / (2 * (1 + nu))
    Iy, Iz, J, A, kappa, _ = get_geometric_props(PROPS)

    kg_h = kappa * G * A / 2.
    gj = G * J / l

    K = _fem_element_stiffness_3d(PROPS, l, n_nodes=2, n_gauss=1)

    np.testing.assert_allclose(K, K.T, atol=1e-14)

    # Torsion diagonal
    assert K[3, 3] == pytest.approx(gj)

    # y-plane coupling K[v1, θz2] = +κGA/2  (Przemieniecki sign +)
    assert K[1, 11] == pytest.approx(kg_h)
    # z-plane coupling K[w1, θy1] = -κGA/2  (Przemieniecki sign -)
    assert K[2, 4] == pytest.approx(-kg_h)
    # z-plane cross coupling K[w2, θy1] = +κGA/2
    assert K[8, 4] == pytest.approx(kg_h)


# ---------------------------------------------------------------------------
# Test 3: Symmetry holds for all poly_order / n_gauss combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('poly_order,n_gauss', [
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 2), (3, 3), (3, 4),
])
def test_fem_element_stiffness_symmetry_2d(poly_order, n_gauss):
    """2D element stiffness is symmetric for any poly_order and n_gauss."""
    l = 3.7
    K = _fem_element_stiffness_2d(PROPS, l, n_nodes=poly_order + 1, n_gauss=n_gauss)
    np.testing.assert_allclose(K, K.T, atol=1e-13,
                               err_msg=f"Not symmetric: p={poly_order}, ng={n_gauss}")


@pytest.mark.parametrize('poly_order,n_gauss', [
    (1, 1), (1, 2),
    (2, 2), (2, 3),
    (3, 3),
])
def test_fem_element_stiffness_symmetry_3d(poly_order, n_gauss):
    """3D element stiffness is symmetric for any poly_order and n_gauss."""
    l = 3.7
    K = _fem_element_stiffness_3d(PROPS, l, n_nodes=poly_order + 1, n_gauss=n_gauss)
    np.testing.assert_allclose(K, K.T, atol=1e-13,
                               err_msg=f"Not symmetric: p={poly_order}, ng={n_gauss}")


# ---------------------------------------------------------------------------
# Test 4: Convergence — tip displacement error decreases monotonically
# ---------------------------------------------------------------------------

def test_convergence_tip_displacement():
    """More sub-elements → tip-displacement error converges toward analytic value."""
    length = 50.
    _, Iz, _, _, _, _ = get_geometric_props(PROPS)
    P = 0.05 * PROPS['E'] * Iz / length**3

    x_tip = np.array([length])
    _, uy_ref, _, _ = cantilever_analytic(x_tip, length, [0., P, 0.], 1.0, PROPS)
    uy_ref = uy_ref[0]

    errors = []
    # density 1/L, 0.2, 2.0 → n_elem ≈ 1, 10, 100
    for density in [1. / length, 0.2, 2.0]:
        uy_fem, _ = _tip_disp_fem(length, density)
        errors.append(abs(uy_fem - uy_ref))

    assert errors[0] > errors[1] > errors[2], (
        f"Expected monotone convergence, got errors={errors}")


# ---------------------------------------------------------------------------
# Test 5: High-refinement FEM ≈ exact Timoshenko (stiffness comparison)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('matrix', ['bsr', 'lil', 'dense'])
def test_large_n_matches_exact(matrix):
    """With many sub-elements the FEM stiffness converges to exact Timoshenko."""
    length = 10.
    num_nodes = 3   # 2 edges of length 5
    # density=20 → n_i = ceil(5*20) = 100 per edge
    opts_fem = {'vectorize': False, 'matrix': matrix, 'verbose': False,
                'n_elem_per_length': 20.0}
    opts_exact = {'vectorize': False, 'matrix': matrix, 'verbose': False}

    net_fem = _cantilever_2d(num_nodes, length, opts_fem)
    net_exact = _cantilever_2d(num_nodes, length, opts_exact)

    K_fem = net_fem.stiffness
    K_exact = net_exact.stiffness
    if hasattr(K_fem, 'toarray'):
        K_fem = K_fem.toarray()
    if hasattr(K_exact, 'toarray'):
        K_exact = K_exact.toarray()

    # 100 sub-elements per edge → h²-convergence, relative error ~ (1/100)²
    # atol guards against inf relative error on entries that are exactly zero
    np.testing.assert_allclose(K_fem, K_exact, rtol=1e-3, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 6: min_element_length constraint
# ---------------------------------------------------------------------------

def test_min_element_length():
    """No sub-element should be shorter than min_element_length."""
    min_len = 1.0
    density = 10.0   # without clamping: n_i = ceil(5*10) = 50

    nodes = np.array([[0., 0.], [5., 0.], [10., 0.]])
    edges = np.array([[0, 1], [1, 2]])
    opts = {'vectorize': False, 'matrix': 'dense', 'verbose': False,
            'n_elem_per_length': density, 'min_element_length': min_len}
    net = BeamNetwork(nodes, edges, beam_prop=PROPS, valid=True, options=opts)

    n_elems = net._compute_edge_discretization()
    assert n_elems is not None

    sub_lengths = net.bondlengths / n_elems
    assert np.all(sub_lengths >= min_len - 1e-12), (
        f"Sub-element lengths {sub_lengths} violate min_element_length={min_len}")


# ---------------------------------------------------------------------------
# Test 7: Backward compatibility — no new options → exact path unchanged
# ---------------------------------------------------------------------------

def test_backward_compatibility():
    """Omitting FEM options activates the exact path (_compute_edge_discretization → None)."""
    length = 10.
    num_nodes = 6
    opts = {'vectorize': True, 'matrix': 'bsr', 'verbose': False}

    net = _cantilever_2d(num_nodes, length, opts)

    assert net._compute_edge_discretization() is None

    net2 = _cantilever_2d(num_nodes, length, opts)
    np.testing.assert_allclose(net.stiffness.toarray(),
                               net2.stiffness.toarray(),
                               rtol=1e-14, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 8: FEM cantilever displacement matches analytic solution
# ---------------------------------------------------------------------------

def test_fem_cantilever_solution():
    """FEM tip solution matches analytic cantilever within O(h²) tolerance."""
    length = 50.
    _, Iz, _, _, _, _ = get_geometric_props(PROPS)
    P = 0.05 * PROPS['E'] * Iz / length**3

    # 2-node beam (1 edge), density=2.0 → n=100 sub-elements, h=0.5
    uy_fem, P = _tip_disp_fem(length, 2.0)

    x_tip = np.array([length])
    _, uy_ref, _, _ = cantilever_analytic(x_tip, length, [0., P, 0.], 1.0, PROPS)
    uy_ref = uy_ref[0]

    # h=0.5, L=50 → relative error O((h/L)²) = O(1e-4)
    np.testing.assert_allclose(uy_fem, uy_ref, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test 9: Higher poly_order converges faster (fewer elements for same accuracy)
# ---------------------------------------------------------------------------

def test_higher_poly_order_converges_faster():
    """Quadratic elements reach the analytic solution with fewer sub-elements."""
    length = 50.
    _, Iz, _, _, _, _ = get_geometric_props(PROPS)
    P = 0.05 * PROPS['E'] * Iz / length**3

    x_tip = np.array([length])
    _, uy_ref, _, _ = cantilever_analytic(x_tip, length, [0., P, 0.], 1.0, PROPS)
    uy_ref = uy_ref[0]

    # Use a small number of sub-elements where p=2 should clearly outperform p=1
    density = 1. / length   # n_elem = 1

    uy_p1, _ = _tip_disp_fem(length, density, poly_order=1, n_gauss=1)
    uy_p2, _ = _tip_disp_fem(length, density, poly_order=2, n_gauss=2)

    err_p1 = abs(uy_p1 - uy_ref)
    err_p2 = abs(uy_p2 - uy_ref)

    assert err_p2 < err_p1, (
        f"Expected p=2 ({err_p2:.3e}) to be more accurate than p=1 ({err_p1:.3e})")


# ---------------------------------------------------------------------------
# Test 10: Full integration is stiffer than reduced (shear locking indicator)
# ---------------------------------------------------------------------------

def test_full_integration_stiffer_than_reduced():
    """Full integration (n_gauss=p+1) produces a stiffer response than reduced (n_gauss=p)."""
    length = 50.   # slender beam: shear locking is pronounced
    density = 1. / length  # n_elem = 1, so h=L (coarsest possible mesh)

    # reduced integration (default)
    uy_red, _ = _tip_disp_fem(length, density, poly_order=1, n_gauss=1)
    # full integration
    uy_full, _ = _tip_disp_fem(length, density, poly_order=1, n_gauss=2)

    # Full integration over-constrains shear → smaller tip displacement
    assert uy_full < uy_red, (
        f"Expected full-integration tip disp ({uy_full:.3e}) < "
        f"reduced-integration ({uy_red:.3e})")


# ---------------------------------------------------------------------------
# Test 11: Condensed global stiffness is symmetric for p > 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('poly_order', [1, 2, 3])
def test_condensed_global_stiffness_symmetric(poly_order):
    """Condensed FEM element stiffness in global frame is symmetric."""
    d = np.array([3., 4.])   # 2D, L=5, non-axis-aligned
    n_elem = 3
    K = get_fem_element_stiffness_global(PROPS, d, n_elem=n_elem, poly_order=poly_order)
    np.testing.assert_allclose(K, K.T, atol=1e-12,
                               err_msg=f"Not symmetric for p={poly_order}")
