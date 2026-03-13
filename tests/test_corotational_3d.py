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
"""Tests for the 3D co-rotational beam element (Crisfield 1990).

Covers the local frame computation, element stiffness building blocks,
internal force assembly, and the nonlinear system assembler.
"""
import numpy as np
import pytest

from beam_networks.fem.corotational import (
    _rigid_body_rotation_3d,
    _local_stiffness_3d,
    _local_stiffness_2d,
    _b_matrix_3d,
    _b_matrix_2d,
    _geometric_stiffness_3d,
    _element_tangent_3d,
    assemble_nonlinear_system_3d,
    assemble_nonlinear_system_2d,
    compute_element_forces_3d,
    compute_element_forces_2d,
    _element_tangent_2d,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_element(x0, x1, ref_vec, d0=None, d1=None):
    """Minimal inputs for a single 3D element."""
    nodes = np.array([x0, x1], dtype=float)
    edges = np.array([[0, 1]])
    ref_vectors = np.array([ref_vec], dtype=float)
    d = np.zeros((2, 6))
    if d0 is not None:
        d[0] = d0
    if d1 is not None:
        d[1] = d1
    return nodes, edges, ref_vectors, d


_BEAM_PROP = {'b': 0.1, 'h': 0.1, 'E': 2.e11, 'nu': 0., 'name': 'rectangle'}
_L = np.array([10.0])


# ---------------------------------------------------------------------------
# Local frame (_rigid_body_rotation_3d)
# ---------------------------------------------------------------------------

class TestRigidBodyRotation3D:

    def test_straight_beam_x_identity_frame(self):
        """Beam along x with zero disp → e1=[1,0,0], e2=[0,1,0], e3=[0,0,1]."""
        nodes, edges, ref_vectors, d = _single_element(
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.])
        R, l0, ln = _rigid_body_rotation_3d(nodes, d, edges, ref_vectors)

        assert R.shape == (1, 3, 3)
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-14)
        np.testing.assert_allclose(l0, [1.0], atol=1e-14)
        np.testing.assert_allclose(ln, [1.0], atol=1e-14)

    def test_straight_beam_y_frame(self):
        """Beam along y with ref=[0,0,1] → e1=[0,1,0], e2=[0,0,1], e3=[1,0,0]."""
        nodes, edges, ref_vectors, d = _single_element(
            [0., 0., 0.], [0., 2., 0.], [0., 0., 1.])
        R, l0, ln = _rigid_body_rotation_3d(nodes, d, edges, ref_vectors)

        np.testing.assert_allclose(R[0, 0], [0., 1., 0.], atol=1e-14)  # e1
        np.testing.assert_allclose(R[0, 1], [0., 0., 1.], atol=1e-14)  # e2
        np.testing.assert_allclose(R[0, 2], [1., 0., 0.], atol=1e-14)  # e3
        np.testing.assert_allclose(l0, [2.0], atol=1e-14)
        np.testing.assert_allclose(ln, [2.0], atol=1e-14)

    def test_straight_beam_z_frame(self):
        """Beam along z with ref=[0,1,0] → e1=[0,0,1], e2=[0,1,0], e3=e1×e2."""
        nodes, edges, ref_vectors, d = _single_element(
            [0., 0., 0.], [0., 0., 3.], [0., 1., 0.])
        R, l0, ln = _rigid_body_rotation_3d(nodes, d, edges, ref_vectors)

        np.testing.assert_allclose(R[0, 0], [0., 0., 1.], atol=1e-14)
        np.testing.assert_allclose(R[0, 1], [0., 1., 0.], atol=1e-14)
        np.testing.assert_allclose(R[0, 2], np.cross([0., 0., 1.], [0., 1., 0.]), atol=1e-14)
        np.testing.assert_allclose(l0, [3.0], atol=1e-14)
        np.testing.assert_allclose(ln, [3.0], atol=1e-14)

    def test_pure_rigid_translation_preserves_lengths(self):
        """Equal translation of both nodes → ln = l0 (no deformation)."""
        nodes, edges, ref_vectors, d = _single_element(
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            d0=[5., -3., 7., 0., 0., 0.],
            d1=[5., -3., 7., 0., 0., 0.])
        R, l0, ln = _rigid_body_rotation_3d(nodes, d, edges, ref_vectors)

        np.testing.assert_allclose(ln, l0, atol=1e-14)

    def test_axial_stretch_changes_ln(self):
        """Node 1 displaced axially by δ → ln = l0 + δ."""
        delta = 0.25
        nodes, edges, ref_vectors, d = _single_element(
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            d1=[delta, 0., 0., 0., 0., 0.])
        R, l0, ln = _rigid_body_rotation_3d(nodes, d, edges, ref_vectors)

        np.testing.assert_allclose(ln, l0 + delta, atol=1e-14)

    def test_rigid_rotation_90deg_about_z(self):
        """Beam along x rigidly rotated 90° about z → e1 points along +y."""
        nodes, edges, ref_vectors, d = _single_element(
            [0., 0., 0.], [1., 0., 0.], [0., 0., 1.],
            d1=[-1., 1., 0., 0., 0., 0.])
        R, l0, ln = _rigid_body_rotation_3d(nodes, d, edges, ref_vectors)

        np.testing.assert_allclose(R[0, 0], [0., 1., 0.], atol=1e-14)  # e1 → +y
        np.testing.assert_allclose(ln, l0, atol=1e-14)                  # no stretching

    def test_R_is_orthogonal(self):
        """R must be a proper rotation matrix: R R^T = I, det(R) = +1."""
        nodes, edges, ref_vectors, d = _single_element(
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            d1=[0.3, 0.4, 0.2, 0., 0., 0.])
        R, _, _ = _rigid_body_rotation_3d(nodes, d, edges, ref_vectors)

        np.testing.assert_allclose(R[0] @ R[0].T, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(R[0]), 1.0, atol=1e-14)

    def test_e3_equals_e1_cross_e2(self):
        """e3 must equal e1 × e2 for all elements."""
        rng = np.random.default_rng(42)
        M = 10
        nodes_all = rng.standard_normal((M + 1, 3))
        edges_all = np.column_stack([np.arange(M), np.arange(1, M + 1)])
        ref_vectors = np.tile([0., 1., 0.], (M, 1))
        d = np.zeros((M + 1, 6))

        R, _, _ = _rigid_body_rotation_3d(nodes_all, d, edges_all, ref_vectors)

        e3_expected = np.cross(R[:, 0, :], R[:, 1, :])
        np.testing.assert_allclose(R[:, 2, :], e3_expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Local stiffness matrix (_local_stiffness_3d)
# ---------------------------------------------------------------------------

class TestLocalStiffness3D:

    def test_shape(self):
        """Returns (M, 7, 7) for M elements."""
        l0 = np.array([1.0, 2.0, 5.0])
        Kl = _local_stiffness_3d(_BEAM_PROP, l0)
        assert Kl.shape == (3, 7, 7)

    def test_symmetry(self):
        """Kl must be symmetric for each element."""
        l0 = np.array([1.0, 5.0, 10.0])
        Kl = _local_stiffness_3d(_BEAM_PROP, l0)
        np.testing.assert_allclose(Kl, Kl.transpose(0, 2, 1), atol=1e-20)

    def test_axial_entry(self):
        """Kl[0, 0] == EA / L."""
        from beam_networks.geometry.geo import get_geometric_props
        _, _, _, A, _, _ = get_geometric_props(_BEAM_PROP)
        E = _BEAM_PROP['E']
        expected = E * A / _L[0]
        Kl = _local_stiffness_3d(_BEAM_PROP, _L)
        np.testing.assert_allclose(Kl[0, 0, 0], expected, rtol=1e-12)

    def test_torsion_entries(self):
        """Torsion block: Kl[1,1] = Kl[4,4] = GJ/L, Kl[1,4] = Kl[4,1] = -GJ/L."""
        from beam_networks.geometry.geo import get_geometric_props
        _, _, J, _, kappa, _ = get_geometric_props(_BEAM_PROP)
        E = _BEAM_PROP['E']
        nu = _BEAM_PROP['nu']
        G = E / (2. * (1. + nu))
        alpha = G * J / _L[0]
        Kl = _local_stiffness_3d(_BEAM_PROP, _L)
        np.testing.assert_allclose(Kl[0, 1, 1],  alpha, rtol=1e-12)
        np.testing.assert_allclose(Kl[0, 4, 4],  alpha, rtol=1e-12)
        np.testing.assert_allclose(Kl[0, 1, 4], -alpha, rtol=1e-12)
        np.testing.assert_allclose(Kl[0, 4, 1], -alpha, rtol=1e-12)

    def test_bending_z_entries(self):
        """Bending-about-e3 block (indices 3,6) matches Timoshenko psi_y, xi_y."""
        from beam_networks.geometry.geo import get_geometric_props
        Iy, Iz, _, A, kappa, _ = get_geometric_props(_BEAM_PROP)
        E = _BEAM_PROP['E']
        nu = _BEAM_PROP['nu']
        G = E / (2. * (1. + nu))
        L = _L[0]
        kGA = kappa * G * A
        PhiY = 12. * E * Iz / (kGA * L**2)
        psi_y = (4. + PhiY) * E * Iz / (L * (1. + PhiY))
        xi_y = (2. - PhiY) * E * Iz / (L * (1. + PhiY))
        Kl = _local_stiffness_3d(_BEAM_PROP, _L)
        np.testing.assert_allclose(Kl[0, 3, 3], psi_y, rtol=1e-12)
        np.testing.assert_allclose(Kl[0, 6, 6], psi_y, rtol=1e-12)
        np.testing.assert_allclose(Kl[0, 3, 6], xi_y,  rtol=1e-12)
        np.testing.assert_allclose(Kl[0, 6, 3], xi_y,  rtol=1e-12)

    def test_bending_y_entries(self):
        """Bending-about-e2 block (indices 2,5) matches Timoshenko psi_z, xi_z."""
        from beam_networks.geometry.geo import get_geometric_props
        Iy, Iz, _, A, kappa, _ = get_geometric_props(_BEAM_PROP)
        E = _BEAM_PROP['E']
        nu = _BEAM_PROP['nu']
        G = E / (2. * (1. + nu))
        L = _L[0]
        kGA = kappa * G * A
        PhiZ = 12. * E * Iy / (kGA * L**2)
        psi_z = (4. + PhiZ) * E * Iy / (L * (1. + PhiZ))
        xi_z = (2. - PhiZ) * E * Iy / (L * (1. + PhiZ))
        Kl = _local_stiffness_3d(_BEAM_PROP, _L)
        np.testing.assert_allclose(Kl[0, 2, 2], psi_z, rtol=1e-12)
        np.testing.assert_allclose(Kl[0, 5, 5], psi_z, rtol=1e-12)
        np.testing.assert_allclose(Kl[0, 2, 5], xi_z,  rtol=1e-12)
        np.testing.assert_allclose(Kl[0, 5, 2], xi_z,  rtol=1e-12)

    def test_euler_bernoulli_limit(self):
        """With euler_bernoulli=True, psi=4EI/L and xi=2EI/L."""
        from beam_networks.geometry.geo import get_geometric_props
        bp = dict(_BEAM_PROP, euler_bernoulli=True)
        Iy, Iz, _, A, _, _ = get_geometric_props(bp)
        E = bp['E']
        L = _L[0]
        Kl = _local_stiffness_3d(bp, _L)
        np.testing.assert_allclose(Kl[0, 3, 3], 4. * E * Iz / L, rtol=1e-6)
        np.testing.assert_allclose(Kl[0, 3, 6], 2. * E * Iz / L, rtol=1e-6)
        np.testing.assert_allclose(Kl[0, 2, 2], 4. * E * Iy / L, rtol=1e-6)
        np.testing.assert_allclose(Kl[0, 2, 5], 2. * E * Iy / L, rtol=1e-6)

    def test_3d_bending_z_matches_2d(self):
        """3D Kl bending-z block (θz0,θz1) must match the 2D Kl (θ0,θ1) block."""
        Kl2 = _local_stiffness_2d(_BEAM_PROP, _L)   # (1, 3, 3)
        Kl3 = _local_stiffness_3d(_BEAM_PROP, _L)   # (1, 7, 7)
        np.testing.assert_allclose(Kl3[0, 3, 3], Kl2[0, 1, 1], rtol=1e-12)
        np.testing.assert_allclose(Kl3[0, 3, 6], Kl2[0, 1, 2], rtol=1e-12)
        np.testing.assert_allclose(Kl3[0, 6, 3], Kl2[0, 2, 1], rtol=1e-12)
        np.testing.assert_allclose(Kl3[0, 6, 6], Kl2[0, 2, 2], rtol=1e-12)


# ---------------------------------------------------------------------------
# B-matrix (_b_matrix_3d)
# ---------------------------------------------------------------------------

def _identity_frame(L=1.0):
    """Single element along x: R=I, ln=L."""
    R = np.eye(3)[None, :, :]     # (1, 3, 3)
    ln = np.array([L])
    return R, ln


class TestBMatrix3D:

    def test_shape(self):
        R, ln = _identity_frame()
        B = _b_matrix_3d(R, ln)
        assert B.shape == (1, 7, 12)

    def test_identity_frame_axial_row(self):
        """Row 0: B[0, 0:3]=-e1, B[0, 6:9]=e1, rest zero."""
        R, ln = _identity_frame(L=2.0)
        B = _b_matrix_3d(R, ln)[0]
        np.testing.assert_allclose(B[0, 0:3],  [-1., 0., 0.], atol=1e-14)
        np.testing.assert_allclose(B[0, 3:6],  [0., 0., 0.],  atol=1e-14)
        np.testing.assert_allclose(B[0, 6:9],  [1., 0., 0.],  atol=1e-14)
        np.testing.assert_allclose(B[0, 9:12], [0., 0., 0.],  atol=1e-14)

    def test_identity_frame_torsion_rows(self):
        """Rows 1,4: only e1 projection in rotation DOF slots."""
        R, ln = _identity_frame()
        B = _b_matrix_3d(R, ln)[0]
        # Row 1: node 0 torsion
        np.testing.assert_allclose(B[1, 0:3],  [0., 0., 0.], atol=1e-14)
        np.testing.assert_allclose(B[1, 3:6],  [1., 0., 0.], atol=1e-14)  # e1
        np.testing.assert_allclose(B[1, 6:9],  [0., 0., 0.], atol=1e-14)
        np.testing.assert_allclose(B[1, 9:12], [0., 0., 0.], atol=1e-14)
        # Row 4: node 1 torsion
        np.testing.assert_allclose(B[4, 0:3],  [0., 0., 0.], atol=1e-14)
        np.testing.assert_allclose(B[4, 3:6],  [0., 0., 0.], atol=1e-14)
        np.testing.assert_allclose(B[4, 6:9],  [0., 0., 0.], atol=1e-14)
        np.testing.assert_allclose(B[4, 9:12], [1., 0., 0.], atol=1e-14)  # e1

    def test_identity_frame_bending_y_rows(self):
        """Rows 2,5: -e3/ln and +e3/ln translation terms, e2 rotation projection."""
        L = 5.0
        R, ln = _identity_frame(L)
        B = _b_matrix_3d(R, ln)[0]
        # Row 2: node 0 θy_l  (e3=[0,0,1], e2=[0,1,0])
        np.testing.assert_allclose(B[2, 0:3],  [0., 0., -1./L], atol=1e-14)
        np.testing.assert_allclose(B[2, 3:6],  [0., 1., 0.],    atol=1e-14)  # e2
        np.testing.assert_allclose(B[2, 6:9],  [0., 0.,  1./L], atol=1e-14)
        np.testing.assert_allclose(B[2, 9:12], [0., 0., 0.],    atol=1e-14)
        # Row 5: node 1 θy_l
        np.testing.assert_allclose(B[5, 0:3],  [0., 0., -1./L], atol=1e-14)
        np.testing.assert_allclose(B[5, 3:6],  [0., 0., 0.],    atol=1e-14)
        np.testing.assert_allclose(B[5, 6:9],  [0., 0.,  1./L], atol=1e-14)
        np.testing.assert_allclose(B[5, 9:12], [0., 1., 0.],    atol=1e-14)  # e2

    def test_identity_frame_bending_z_rows(self):
        """Rows 3,6: +e2/ln and -e2/ln translation terms, e3 rotation projection."""
        L = 5.0
        R, ln = _identity_frame(L)
        B = _b_matrix_3d(R, ln)[0]
        # Row 3: node 0 θz_l  (e2=[0,1,0], e3=[0,0,1])
        np.testing.assert_allclose(B[3, 0:3],  [0., 1./L, 0.], atol=1e-14)
        np.testing.assert_allclose(B[3, 3:6],  [0., 0., 1.],   atol=1e-14)  # e3
        np.testing.assert_allclose(B[3, 6:9],  [0., -1./L, 0.], atol=1e-14)
        np.testing.assert_allclose(B[3, 9:12], [0., 0., 0.],   atol=1e-14)
        # Row 6: node 1 θz_l
        np.testing.assert_allclose(B[6, 0:3],  [0., 1./L, 0.], atol=1e-14)
        np.testing.assert_allclose(B[6, 3:6],  [0., 0., 0.],   atol=1e-14)
        np.testing.assert_allclose(B[6, 6:9],  [0., -1./L, 0.], atol=1e-14)
        np.testing.assert_allclose(B[6, 9:12], [0., 0., 1.],   atol=1e-14)  # e3

    def test_local_dofs_pure_axial(self):
        """Pure axial extension: only Δl non-zero."""
        R, ln = _identity_frame()
        B = _b_matrix_3d(R, ln)[0]
        d = np.zeros(12)
        d[6] = 0.5    # ux1
        ul = B @ d
        assert ul[0] == pytest.approx(0.5, abs=1e-14)
        np.testing.assert_allclose(ul[1:], 0., atol=1e-14)

    def test_local_dofs_pure_torsion_node0(self):
        """Pure rotation θx at node 0: only θx0_l non-zero."""
        R, ln = _identity_frame()
        B = _b_matrix_3d(R, ln)[0]
        d = np.zeros(12)
        d[3] = 1.2    # θx0
        ul = B @ d
        assert ul[1] == pytest.approx(1.2, abs=1e-14)
        np.testing.assert_allclose(np.delete(ul, 1), 0., atol=1e-14)

    def test_bending_z_consistency_with_2d(self):
        """In-plane (θz only) local DOFs match the 2D B-matrix result."""
        L = 4.0
        B2 = _b_matrix_2d(np.array([1.0]), np.array([0.0]), np.array([L]))[0]  # (3, 6)

        R = np.eye(3)[None, :, :]
        B3 = _b_matrix_3d(R, np.array([L]))[0]   # (7, 12)

        # In-plane 3D DOF indices: [ux0=0, uy0=1, θz0=5, ux1=6, uy1=7, θz1=11]
        # 3D local DOF rows for the 2D equivalent: axial=0, θz0=3, θz1=6
        inplane_3d = [0, 1, 5, 6, 7, 11]
        B3_inplane = B3[np.ix_([0, 3, 6], inplane_3d)]  # (3, 6)

        np.testing.assert_allclose(B3_inplane, B2, atol=1e-14)

    def test_energy_consistency(self):
        """fl^T ul == fg^T d_global (virtual work equivalence)."""
        L = 3.0
        R, ln = _identity_frame(L)
        B = _b_matrix_3d(R, ln)[0]            # (7, 12)
        Kl = _local_stiffness_3d(_BEAM_PROP, np.array([L]))[0]  # (7, 7)

        rng = np.random.default_rng(0)
        d = rng.standard_normal(12) * 0.01
        ul = B @ d
        fl = Kl @ ul
        fg = B.T @ fl

        np.testing.assert_allclose(fl @ ul, fg @ d, rtol=1e-12)


# ---------------------------------------------------------------------------
# Internal forces (compute_element_forces_3d)
# ---------------------------------------------------------------------------

def _cantilever_3d(L=10.0):
    """Single-element cantilever along x; returns (nodes, edges, ref_vectors)."""
    nodes = np.array([[0., 0., 0.], [L, 0., 0.]])
    edges = np.array([[0, 1]])
    ref_vectors = np.array([[0., 1., 0.]])
    return nodes, edges, ref_vectors


def _sol(n_nodes, **kwargs):
    """Build a 6*n_nodes zero solution vector, then set named DOFs."""
    slots = {'u0x': 0, 'u0y': 1, 'u0z': 2, 't0x': 3, 't0y': 4, 't0z': 5,
             'u1x': 6, 'u1y': 7, 'u1z': 8, 't1x': 9, 't1y': 10, 't1z': 11}
    sol = np.zeros(n_nodes * 6)
    for name, val in kwargs.items():
        sol[slots[name]] = val
    return sol


class TestComputeElementForces3D:

    def test_zero_displacement_zero_forces(self):
        """All forces must be zero at zero displacement."""
        nodes, edges, ref_vectors = _cantilever_3d()
        sol = np.zeros(12)
        fl = compute_element_forces_3d(nodes, edges, sol, _BEAM_PROP, ref_vectors)
        assert fl.shape == (1, 7)
        np.testing.assert_allclose(fl, 0., atol=1e-20)

    def test_pure_axial_extension(self):
        """Axial extension δ: N = EA*δ/L, all other forces = 0."""
        from beam_networks.geometry.geo import get_geometric_props
        L = 10.0
        nodes, edges, ref_vectors = _cantilever_3d(L)
        delta = 0.01
        sol = _sol(2, u1x=delta)
        fl = compute_element_forces_3d(nodes, edges, sol, _BEAM_PROP, ref_vectors)

        _, _, _, A, _, _ = get_geometric_props(_BEAM_PROP)
        N_expected = _BEAM_PROP['E'] * A * delta / L
        np.testing.assert_allclose(fl[0, 0], N_expected, rtol=1e-12)
        np.testing.assert_allclose(fl[0, 1:], 0., atol=1e-10)

    def test_pure_torsion_node0(self):
        """Rotation θx at node 0 only: Tx = GJ/L * θ, no bending forces."""
        from beam_networks.geometry.geo import get_geometric_props
        L = 10.0
        nodes, edges, ref_vectors = _cantilever_3d(L)
        theta = 0.05
        sol = _sol(2, t0x=theta)
        fl = compute_element_forces_3d(nodes, edges, sol, _BEAM_PROP, ref_vectors)

        _, _, J, _, _, _ = get_geometric_props(_BEAM_PROP)
        E = _BEAM_PROP['E']
        nu = _BEAM_PROP['nu']
        G = E / (2. * (1. + nu))
        Tx_expected = G * J / L * theta

        np.testing.assert_allclose(fl[0, 0], 0., atol=1e-10)
        np.testing.assert_allclose(fl[0, 1], Tx_expected, rtol=1e-12)
        np.testing.assert_allclose(fl[0, 2:4], 0., atol=1e-10)
        np.testing.assert_allclose(fl[0, 4], -Tx_expected, rtol=1e-12)
        np.testing.assert_allclose(fl[0, 5:], 0., atol=1e-10)

    def test_pure_bending_z_node0(self):
        """Rotation θz at node 0: Mz0 = psi_y*θ, Mz1 = xi_y*θ, others = 0."""
        from beam_networks.geometry.geo import get_geometric_props
        L = 10.0
        nodes, edges, ref_vectors = _cantilever_3d(L)
        theta = 0.01
        sol = _sol(2, t0z=theta)
        fl = compute_element_forces_3d(nodes, edges, sol, _BEAM_PROP, ref_vectors)

        _, Iz, _, A, kappa, _ = get_geometric_props(_BEAM_PROP)
        E = _BEAM_PROP['E']
        nu = _BEAM_PROP['nu']
        G = E / (2. * (1. + nu))
        PhiY = 12. * E * Iz / (kappa * G * A * L**2)
        psi_y = (4. + PhiY) * E * Iz / (L * (1. + PhiY))
        xi_y = (2. - PhiY) * E * Iz / (L * (1. + PhiY))

        np.testing.assert_allclose(fl[0, 0],  0.,          atol=1e-10)
        np.testing.assert_allclose(fl[0, 1],  0.,          atol=1e-10)
        np.testing.assert_allclose(fl[0, 2],  0.,          atol=1e-10)
        np.testing.assert_allclose(fl[0, 3],  psi_y*theta, rtol=1e-12)
        np.testing.assert_allclose(fl[0, 4],  0.,          atol=1e-10)
        np.testing.assert_allclose(fl[0, 5],  0.,          atol=1e-10)
        np.testing.assert_allclose(fl[0, 6],  xi_y*theta,  rtol=1e-12)

    def test_pure_bending_y_node0(self):
        """Rotation θy at node 0: My0 = psi_z*θ, My1 = xi_z*θ, others = 0."""
        from beam_networks.geometry.geo import get_geometric_props
        L = 10.0
        nodes, edges, ref_vectors = _cantilever_3d(L)
        theta = 0.01
        sol = _sol(2, t0y=theta)
        fl = compute_element_forces_3d(nodes, edges, sol, _BEAM_PROP, ref_vectors)

        Iy, _, _, A, kappa, _ = get_geometric_props(_BEAM_PROP)
        E = _BEAM_PROP['E']
        nu = _BEAM_PROP['nu']
        G = E / (2. * (1. + nu))
        PhiZ = 12. * E * Iy / (kappa * G * A * L**2)
        psi_z = (4. + PhiZ) * E * Iy / (L * (1. + PhiZ))
        xi_z = (2. - PhiZ) * E * Iy / (L * (1. + PhiZ))

        np.testing.assert_allclose(fl[0, 0], 0.,          atol=1e-10)
        np.testing.assert_allclose(fl[0, 1], 0.,          atol=1e-10)
        np.testing.assert_allclose(fl[0, 2], psi_z*theta, rtol=1e-12)
        np.testing.assert_allclose(fl[0, 3], 0.,          atol=1e-10)
        np.testing.assert_allclose(fl[0, 4], 0.,          atol=1e-10)
        np.testing.assert_allclose(fl[0, 5], xi_z*theta,  rtol=1e-12)
        np.testing.assert_allclose(fl[0, 6], 0.,          atol=1e-10)

    def test_rigid_body_rotation_zero_forces(self):
        """Pure rigid-body chord rotation must produce zero deformational forces."""
        L = 10.0
        nodes, edges, ref_vectors = _cantilever_3d(L)
        angle = np.deg2rad(30.)
        u1x = L * np.cos(angle) - L
        u1y = L * np.sin(angle)
        sol = _sol(2, u1x=u1x, u1y=u1y, t0z=angle, t1z=angle)
        fl = compute_element_forces_3d(nodes, edges, sol, _BEAM_PROP, ref_vectors)
        np.testing.assert_allclose(fl, 0., atol=1e-8)

    def test_reduces_to_2d_forces(self):
        """In-plane loading: 3D N/Mz0/Mz1 match compute_element_forces_2d."""
        L = 10.0
        nodes_2d = np.array([[0., 0.], [L, 0.]])
        edges = np.array([[0, 1]])
        uy1, tz0, tz1 = 0.05, 0.01, -0.01
        sol_2d = np.array([0., 0., tz0, 0., uy1, tz1])
        fl_2d = compute_element_forces_2d(nodes_2d, edges, sol_2d, _BEAM_PROP)

        nodes_3d = np.array([[0., 0., 0.], [L, 0., 0.]])
        ref_vectors = np.array([[0., 1., 0.]])
        sol_3d = np.array([0., 0., 0., 0., 0., tz0,
                           0., uy1, 0., 0., 0., tz1])
        fl_3d = compute_element_forces_3d(nodes_3d, edges, sol_3d, _BEAM_PROP, ref_vectors)

        np.testing.assert_allclose(fl_3d[0, 0], fl_2d[0, 0], rtol=1e-12)  # N
        np.testing.assert_allclose(fl_3d[0, 3], fl_2d[0, 1], rtol=1e-12)  # Mz0
        np.testing.assert_allclose(fl_3d[0, 6], fl_2d[0, 2], rtol=1e-12)  # Mz1


# ---------------------------------------------------------------------------
# Geometric stiffness (_geometric_stiffness_3d)
# ---------------------------------------------------------------------------

class TestGeometricStiffness3D:

    @staticmethod
    def _beam_along_x(L=10.0):
        nodes = np.array([[0., 0., 0.], [L, 0., 0.]])
        edges = np.array([[0, 1]])
        d = np.zeros((2, 6))
        R, l0, ln = _rigid_body_rotation_3d(nodes, d, edges,
                                            np.array([[0., 1., 0.]]))
        return R, l0, ln

    def test_zero_forces_zero_Kg(self):
        """fl = 0 → Kg = 0."""
        R, l0, ln = self._beam_along_x()
        fl = np.zeros((1, 7))
        Kg = _geometric_stiffness_3d(R, ln, fl)
        np.testing.assert_allclose(Kg, 0., atol=1e-15)

    def test_symmetry(self):
        """Kg is always symmetric."""
        L = 10.0
        R, l0, ln = self._beam_along_x(L)
        fl = np.array([[1e5, 0., 500., 300., 0., 400., 200.]])
        Kg = _geometric_stiffness_3d(R, ln, fl)[0]
        np.testing.assert_allclose(Kg, Kg.T, atol=1e-10)

    def test_axial_only_acts_on_translations(self):
        """With only N ≠ 0, Kg entries for rotation DOFs are zero."""
        L = 10.0
        R, l0, ln = self._beam_along_x(L)
        fl = np.zeros((1, 7))
        fl[0, 0] = 1e5
        Kg = _geometric_stiffness_3d(R, ln, fl)[0]
        rot_dofs = [3, 4, 5, 9, 10, 11]
        np.testing.assert_allclose(Kg[rot_dofs, :], 0., atol=1e-10)
        np.testing.assert_allclose(Kg[:, rot_dofs], 0., atol=1e-10)

    def test_axial_tension_string_stiffness(self):
        """Tension N stiffens transverse directions with string stiffness N/L."""
        L = 10.0
        N = 1e5
        R, l0, ln = self._beam_along_x(L)
        fl = np.zeros((1, 7))
        fl[0, 0] = N
        Kg = _geometric_stiffness_3d(R, ln, fl)[0]
        expected = N / L
        np.testing.assert_allclose(Kg[1, 1],  expected, rtol=1e-12)
        np.testing.assert_allclose(Kg[7, 7],  expected, rtol=1e-12)
        np.testing.assert_allclose(Kg[1, 7], -expected, rtol=1e-12)
        np.testing.assert_allclose(Kg[2, 2],  expected, rtol=1e-12)
        np.testing.assert_allclose(Kg[8, 8],  expected, rtol=1e-12)
        np.testing.assert_allclose(Kg[2, 8], -expected, rtol=1e-12)
        np.testing.assert_allclose(Kg[0, 0], 0., atol=1e-10)

    def test_Mz_moment_coupling(self):
        """Mz couples axial and transverse-y displacements (positive sign)."""
        L = 10.0
        Mz = 1e3
        R, l0, ln = self._beam_along_x(L)
        fl = np.zeros((1, 7))
        fl[0, 3] = Mz
        fl[0, 6] = Mz
        Kg = _geometric_stiffness_3d(R, ln, fl)[0]
        expected = 2 * Mz / L**2
        np.testing.assert_allclose(Kg[6, 7], expected, rtol=1e-12)
        np.testing.assert_allclose(Kg[7, 6], expected, rtol=1e-12)
        np.testing.assert_allclose(Kg[6, 8], 0., atol=1e-10)

    def test_My_moment_coupling(self):
        """My couples axial and transverse-z displacements (negative sign)."""
        L = 10.0
        My = 1e3
        R, l0, ln = self._beam_along_x(L)
        fl = np.zeros((1, 7))
        fl[0, 2] = My
        fl[0, 5] = My
        Kg = _geometric_stiffness_3d(R, ln, fl)[0]
        expected = -2 * My / L**2
        np.testing.assert_allclose(Kg[6, 8], expected, rtol=1e-12)
        np.testing.assert_allclose(Kg[8, 6], expected, rtol=1e-12)

    def test_reduces_to_2d_for_in_plane(self):
        """3D Kg restricted to in-plane DOFs matches 2D Kg for the same loading."""
        L = 10.0
        nodes2d = np.array([[0., 0.], [L, 0.]])
        nodes3d = np.array([[0., 0., 0.], [L, 0., 0.]])
        edges = np.array([[0, 1]])

        uy = 0.1
        sol2d = np.zeros(6)
        sol2d[4] = uy
        sol3d = np.zeros(12)
        sol3d[7] = uy

        Kt2d, fg2d = _element_tangent_2d(nodes2d, edges, sol2d, _BEAM_PROP)
        from beam_networks.fem.corotational import (
            _local_stiffness_2d, _b_matrix_2d, _rigid_body_rotation_2d,
        )
        d2d = sol2d.reshape(-1, 3)
        alpha, l0, ln2, c, s = _rigid_body_rotation_2d(nodes2d, d2d, edges)
        Kl2d = _local_stiffness_2d(_BEAM_PROP, l0)
        B2d = _b_matrix_2d(c, s, ln2)
        Km2d = (B2d.transpose(0, 2, 1) @ Kl2d @ B2d)[0]
        Kg2d = Kt2d[0] - Km2d

        d3d = sol3d.reshape(-1, 6)
        ref3d = np.array([[0., 0., 1.]])
        R, l0_3d, ln3d = _rigid_body_rotation_3d(nodes3d, d3d, edges, ref3d)
        from beam_networks.fem.corotational import _exact_local_dofs_3d
        ul3d = _exact_local_dofs_3d(nodes3d, d3d, edges, R, l0_3d, ln3d)
        Kl3d = _local_stiffness_3d(_BEAM_PROP, l0_3d)
        fl3d = np.einsum('mij,mj->mi', Kl3d, ul3d)
        Kg3d = _geometric_stiffness_3d(R, ln3d, fl3d)[0]

        ip3d = [0, 1, 5, 6, 7, 11]
        Kg3d_ip = Kg3d[np.ix_(ip3d, ip3d)]

        np.testing.assert_allclose(Kg3d_ip, Kg2d, atol=1e-6)


# ---------------------------------------------------------------------------
# Element tangent stiffness (_element_tangent_3d)
# ---------------------------------------------------------------------------

class TestElementTangent3D:

    @staticmethod
    def _beam_along_x(L=10.0, sol3d=None):
        nodes = np.array([[0., 0., 0.], [L, 0., 0.]])
        edges = np.array([[0, 1]])
        ref = np.array([[0., 1., 0.]])
        if sol3d is None:
            sol3d = np.zeros(12)
        return nodes, edges, ref, sol3d

    def test_output_shapes(self):
        """_element_tangent_3d returns (M,12,12) and (M,12) arrays."""
        nodes, edges, ref, sol = self._beam_along_x()
        Kt, fg = _element_tangent_3d(nodes, edges, sol, _BEAM_PROP, ref)
        assert Kt.shape == (1, 12, 12)
        assert fg.shape == (1, 12)

    def test_symmetry(self):
        """Kt is symmetric."""
        L = 10.0
        sol = np.zeros(12)
        sol[7] = 0.1
        nodes, edges, ref, _ = self._beam_along_x(L)
        Kt, _ = _element_tangent_3d(nodes, edges, sol, _BEAM_PROP, ref)
        np.testing.assert_allclose(Kt[0], Kt[0].T, atol=1e-8)

    def test_zero_displacement_gives_zero_forces(self):
        """At zero displacement fg = 0."""
        nodes, edges, ref, sol = self._beam_along_x()
        _, fg = _element_tangent_3d(nodes, edges, sol, _BEAM_PROP, ref)
        np.testing.assert_allclose(fg, 0., atol=1e-15)

    def test_reduces_to_2d_tangent(self):
        """3D Kt restricted to in-plane DOFs matches the 2D tangent stiffness."""
        L = 10.0
        uy = 0.5
        nodes2d = np.array([[0., 0.], [L, 0.]])
        nodes3d = np.array([[0., 0., 0.], [L, 0., 0.]])
        edges = np.array([[0, 1]])

        sol2d = np.zeros(6)
        sol2d[4] = uy
        sol3d = np.zeros(12)
        sol3d[7] = uy

        Kt2d, fg2d = _element_tangent_2d(nodes2d, edges, sol2d, _BEAM_PROP)
        Kt3d, fg3d = _element_tangent_3d(nodes3d, edges, sol3d, _BEAM_PROP,
                                         np.array([[0., 0., 1.]]))

        ip = [0, 1, 5, 6, 7, 11]
        np.testing.assert_allclose(Kt3d[0][np.ix_(ip, ip)], Kt2d[0], atol=1e-5)
        np.testing.assert_allclose(fg3d[0][ip], fg2d[0], atol=1e-5)

    def test_fg_equals_BT_fl(self):
        """fg = B^T fl (virtual work principle)."""
        L = 10.0
        sol3d = np.zeros(12)
        sol3d[7] = 0.1
        nodes, edges, ref, _ = self._beam_along_x(L)
        _, fg = _element_tangent_3d(nodes, edges, sol3d, _BEAM_PROP, ref)

        d = sol3d.reshape(-1, 6)
        R, l0, ln = _rigid_body_rotation_3d(nodes, d, edges, ref)
        from beam_networks.fem.corotational import _exact_local_dofs_3d
        ul = _exact_local_dofs_3d(nodes, d, edges, R, l0, ln)
        Kl = _local_stiffness_3d(_BEAM_PROP, l0)
        fl = np.einsum('mij,mj->mi', Kl, ul)
        B = _b_matrix_3d(R, ln)
        fg_expected = np.einsum('mji,mj->mi', B, fl)
        np.testing.assert_allclose(fg, fg_expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Global assembly (assemble_nonlinear_system_3d)
# ---------------------------------------------------------------------------

class TestAssembleNonlinear3D:

    @staticmethod
    def _cantilever_3d(ne=3, L=10.0):
        """Straight cantilever along x, ne elements."""
        x = np.linspace(0., L, ne + 1)
        nodes = np.column_stack([x, np.zeros(ne + 1), np.zeros(ne + 1)])
        edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])
        ref = np.tile([0., 1., 0.], (ne, 1))
        return nodes, edges, ref

    def test_output_shapes(self):
        """Assembled K and F_int have correct shapes."""
        nodes, edges, ref = self._cantilever_3d(ne=3)
        N = nodes.shape[0]
        sol = np.zeros(N * 6)
        K, F = assemble_nonlinear_system_3d(nodes, edges, sol, _BEAM_PROP, ref)
        assert K.shape == (N * 6, N * 6)
        assert F.shape == (N * 6,)

    def test_zero_displacement_zero_F_int(self):
        """At zero displacement F_int = 0 for both matrix formats."""
        nodes, edges, ref = self._cantilever_3d(ne=5)
        N = nodes.shape[0]
        sol = np.zeros(N * 6)
        for fmt in ('dense', 'bsr'):
            _, F = assemble_nonlinear_system_3d(
                nodes, edges, sol, _BEAM_PROP, ref, matrix=fmt)
            np.testing.assert_allclose(F, 0., atol=1e-15,
                                       err_msg=f"F_int ≠ 0 for matrix={fmt}")

    def test_symmetry_dense(self):
        """Dense K is symmetric."""
        nodes, edges, ref = self._cantilever_3d(ne=4)
        N = nodes.shape[0]
        sol = np.zeros(N * 6)
        sol[7] = 0.1
        K, _ = assemble_nonlinear_system_3d(
            nodes, edges, sol, _BEAM_PROP, ref, matrix='dense')
        np.testing.assert_allclose(K, K.T, atol=1e-8)

    def test_dense_bsr_agree(self):
        """Dense and BSR assemblers produce identical K and F_int."""
        nodes, edges, ref = self._cantilever_3d(ne=5)
        N = nodes.shape[0]
        sol = np.zeros(N * 6)
        sol[7] = 0.2
        K_d, F_d = assemble_nonlinear_system_3d(
            nodes, edges, sol, _BEAM_PROP, ref, matrix='dense')
        K_b, F_b = assemble_nonlinear_system_3d(
            nodes, edges, sol, _BEAM_PROP, ref, matrix='bsr')
        np.testing.assert_allclose(K_b.toarray(), K_d, atol=1e-8)
        np.testing.assert_allclose(F_b, F_d, atol=1e-12)

    def test_reduces_to_2d_system(self):
        """3D K restricted to in-plane DOFs matches the 2D assembled K."""
        ne = 4
        L = 10.0
        x = np.linspace(0., L, ne + 1)
        nodes2d = np.column_stack([x, np.zeros(ne + 1)])
        nodes3d = np.column_stack([x, np.zeros(ne + 1), np.zeros(ne + 1)])
        edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])
        ref = np.tile([0., 0., 1.], (ne, 1))

        N = ne + 1
        sol2d = np.zeros(N * 3)
        sol3d = np.zeros(N * 6)
        sol2d[N // 2 * 3 + 1] = 0.1
        sol3d[N // 2 * 6 + 1] = 0.1

        K2d, F2d = assemble_nonlinear_system_2d(
            nodes2d, edges, sol2d, _BEAM_PROP, matrix='dense')
        K3d, F3d = assemble_nonlinear_system_3d(
            nodes3d, edges, sol3d, _BEAM_PROP, ref, matrix='dense')

        ip_3d = np.array([n * 6 + k for n in range(N) for k in [0, 1, 5]])
        K3d_ip = K3d[np.ix_(ip_3d, ip_3d)]
        F3d_ip = F3d[ip_3d]

        np.testing.assert_allclose(K3d_ip, K2d, atol=1e-5)
        np.testing.assert_allclose(F3d_ip, F2d, atol=1e-8)

    def test_invalid_matrix_raises(self):
        """An unknown matrix format raises ValueError."""
        nodes, edges, ref = self._cantilever_3d(ne=2)
        sol = np.zeros(nodes.shape[0] * 6)
        with pytest.raises(ValueError):
            assemble_nonlinear_system_3d(
                nodes, edges, sol, _BEAM_PROP, ref, matrix='invalid')
