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
"""Tests for truss element support."""
import numpy as np

from beam_networks.fem.stiffness import (
    global_element_stiffness_truss_exact_single,
    global_element_stiffness_truss_exact_all)
from beam_networks.problem import ElasticNetwork


BEAM_PROP = {'name': 'circle', 'radius': 0.1, 'E': 1e6, 'nu': 0.3, 'truss': True}


class TestTrussStiffness:
    """Unit tests for truss element stiffness matrices."""

    def test_single_2d_shape(self):
        d = np.array([1., 0.])
        K = global_element_stiffness_truss_exact_single(BEAM_PROP, d)
        assert K.shape == (4, 4)

    def test_single_3d_shape(self):
        d = np.array([1., 0., 0.])
        K = global_element_stiffness_truss_exact_single(BEAM_PROP, d)
        assert K.shape == (6, 6)

    def test_single_2d_symmetric(self):
        d = np.array([3., 4.])
        K = global_element_stiffness_truss_exact_single(BEAM_PROP, d)
        np.testing.assert_allclose(K, K.T)

    def test_single_3d_symmetric(self):
        d = np.array([1., 2., 3.])
        K = global_element_stiffness_truss_exact_single(BEAM_PROP, d)
        np.testing.assert_allclose(K, K.T)

    def test_axial_alignment_2d(self):
        """Horizontal bar: global K[0,0] == EA/L."""
        from beam_networks.geometry.geo import get_geometric_props
        d = np.array([2., 0.])
        L = 2.
        E = BEAM_PROP['E']
        _, _, _, A, _, _ = get_geometric_props(BEAM_PROP)
        K = global_element_stiffness_truss_exact_single(BEAM_PROP, d)
        gamma = E * A / L
        np.testing.assert_allclose(K[0, 0], gamma)
        np.testing.assert_allclose(K[1, 1], 0.)
        np.testing.assert_allclose(K[0, 2], -gamma)

    def test_all_equals_single(self):
        d_vec = np.array([[1., 0.], [0., 1.], [1., 1.]])
        Ks = global_element_stiffness_truss_exact_all(BEAM_PROP, d_vec)
        for i, d in enumerate(d_vec):
            K_single = global_element_stiffness_truss_exact_single(BEAM_PROP, d)
            np.testing.assert_allclose(Ks[i], K_single)

    def test_null_space_rigid_body_2d(self):
        """Global K must be singular (rigid body modes in the nullspace)."""
        d = np.array([1., 0.])
        K = global_element_stiffness_truss_exact_single(BEAM_PROP, d)
        rank = np.linalg.matrix_rank(K)
        assert rank < 4  # at most 1 non-zero mode for a single bar


class TestTrussDofPerNode:
    """Tests for DOF count in ElasticNetwork with truss flag."""

    def _make_simple_truss(self):
        nodes = np.array([[0., 0.], [1., 0.]])
        edges = np.array([[0, 1]])
        return ElasticNetwork(
            nodes, edges, beam_prop=BEAM_PROP,
            options={'vectorize': True, 'matrix': 'bsr', 'verbose': False},
            outdir='/tmp')

    def test_dof_per_node_2d(self):
        bn = self._make_simple_truss()
        assert bn.dof_per_node == 2

    def test_dof_per_node_3d(self):
        nodes = np.array([[0., 0., 0.], [1., 0., 0.]])
        edges = np.array([[0, 1]])
        beam_prop = dict(BEAM_PROP)
        bn = ElasticNetwork(
            nodes, edges, beam_prop=beam_prop,
            options={'vectorize': True, 'matrix': 'bsr', 'verbose': False},
            outdir='/tmp')
        assert bn.dof_per_node == 3

    def test_stiffness_size_2d(self):
        bn = self._make_simple_truss()
        assert bn.stiffness.shape == (4, 4)  # 2 nodes * 2 dof

    def test_rotation_empty_for_truss(self):
        bn = self._make_simple_truss()
        bn.add_BC('fix', 'D', 'node', [0], [0., 0.])
        bn.add_BC('fix_t', 'D', 'node', [1], [None, 0.])
        bn.add_BC('load', 'N', 'node', [1], [1., None])
        bn.solve()
        assert bn.rotation.shape == (2, 0)

    def test_displacement_shape(self):
        bn = self._make_simple_truss()
        bn.add_BC('fix', 'D', 'node', [0], [0., 0.])
        bn.add_BC('fix_t', 'D', 'node', [1], [None, 0.])
        bn.add_BC('load', 'N', 'node', [1], [1., None])
        bn.solve()
        assert bn.displacement.shape == (2, 2)


class TestTrussAnalytical:
    """Test axial bar deformation against analytical solution."""

    def test_axial_bar_2d(self):
        """Single horizontal bar: u_tip = F*L / (E*A)."""
        from beam_networks.geometry.geo import get_geometric_props
        L = 2.
        F = 100.
        nodes = np.array([[0., 0.], [L, 0.]])
        edges = np.array([[0, 1]])
        beam_prop = {'name': 'circle', 'radius': 0.05, 'E': 1e6,
                     'nu': 0.3, 'truss': True}
        bn = ElasticNetwork(
            nodes, edges, beam_prop=beam_prop,
            options={'vectorize': True, 'matrix': 'bsr', 'verbose': False},
            outdir='/tmp')
        # Fix node 0 fully; fix transverse of node 1; apply axial force
        bn.add_BC('fix', 'D', 'node', [0], [0., 0.])
        bn.add_BC('fix_t', 'D', 'node', [1], [None, 0.])
        bn.add_BC('load', 'N', 'node', [1], [F, None])
        bn.solve()

        _, _, _, A, _, _ = get_geometric_props(beam_prop)
        E = beam_prop['E']
        u_tip_expected = F * L / (E * A)
        u_tip = bn.displacement[1, 0]
        np.testing.assert_allclose(u_tip, u_tip_expected, rtol=1e-10)

    def test_axial_stress(self):
        """Single horizontal bar: stress = F / A."""
        from beam_networks.geometry.geo import get_geometric_props
        L = 2.
        F = 100.
        nodes = np.array([[0., 0.], [L, 0.]])
        edges = np.array([[0, 1]])
        beam_prop = {'name': 'circle', 'radius': 0.05, 'E': 1e6,
                     'nu': 0.3, 'truss': True}
        bn = ElasticNetwork(
            nodes, edges, beam_prop=beam_prop,
            options={'vectorize': True, 'matrix': 'bsr', 'verbose': False},
            outdir='/tmp')
        bn.add_BC('fix', 'D', 'node', [0], [0., 0.])
        bn.add_BC('fix_t', 'D', 'node', [1], [None, 0.])
        bn.add_BC('load', 'N', 'node', [1], [F, None])
        bn.solve()

        _, _, _, A, _, _ = get_geometric_props(beam_prop)
        stress_expected = F / A
        np.testing.assert_allclose(bn._sVM[0], stress_expected, rtol=1e-8)

    def test_two_bars_series_2d(self):
        """Two bars in series along x: tip displacement = 2*F*L/(EA)."""
        from beam_networks.geometry.geo import get_geometric_props
        L = 1.
        F = 100.
        nodes = np.array([[0., 0.], [L, 0.], [2 * L, 0.]])
        edges = np.array([[0, 1], [1, 2]])
        beam_prop = {'name': 'circle', 'radius': 0.05, 'E': 1e6,
                     'nu': 0.3, 'truss': True}
        bn = ElasticNetwork(
            nodes, edges, beam_prop=beam_prop,
            options={'vectorize': True, 'matrix': 'bsr', 'verbose': False},
            outdir='/tmp')
        bn.add_BC('fix', 'D', 'node', [0], [0., 0.])
        bn.add_BC('fix_t1', 'D', 'node', [1], [None, 0.])
        bn.add_BC('fix_t2', 'D', 'node', [2], [None, 0.])
        bn.add_BC('load', 'N', 'node', [2], [F, None])
        bn.solve()

        _, _, _, A, _, _ = get_geometric_props(beam_prop)
        E = beam_prop['E']
        u_tip_expected = 2. * F * L / (E * A)
        np.testing.assert_allclose(bn.displacement[2, 0], u_tip_expected, rtol=1e-8)

    def test_3d_axial_bar(self):
        """Single bar along z-axis in 3D."""
        from beam_networks.geometry.geo import get_geometric_props
        L = 3.
        F = 50.
        nodes = np.array([[0., 0., 0.], [0., 0., L]])
        edges = np.array([[0, 1]])
        beam_prop = {'name': 'circle', 'radius': 0.05, 'E': 1e6,
                     'nu': 0.3, 'truss': True}
        bn = ElasticNetwork(
            nodes, edges, beam_prop=beam_prop,
            options={'vectorize': True, 'matrix': 'bsr', 'verbose': False},
            outdir='/tmp')
        # Fix node 0 fully; fix transverse DOFs at node 1; apply axial (z) force
        bn.add_BC('fix', 'D', 'node', [0], [0., 0., 0.])
        bn.add_BC('fix_t', 'D', 'node', [1], [0., 0., None])
        bn.add_BC('load', 'N', 'node', [1], [None, None, F])
        bn.solve()

        _, _, _, A, _, _ = get_geometric_props(beam_prop)
        E = beam_prop['E']
        u_tip_expected = F * L / (E * A)
        np.testing.assert_allclose(bn.displacement[1, 2], u_tip_expected, rtol=1e-10)
        assert bn.dof_per_node == 3
        assert bn.rotation.shape == (2, 0)


class TestTrussBCVector:
    """Test that BC vectors have the right length for trusses."""

    def test_2d_truss_bc_length(self):
        """2D truss BC vector should have length 2, not 3."""
        nodes = np.array([[0., 0.], [1., 0.]])
        edges = np.array([[0, 1]])
        bn = ElasticNetwork(
            nodes, edges, beam_prop=BEAM_PROP,
            options={'vectorize': True, 'matrix': 'bsr', 'verbose': False},
            outdir='/tmp')
        # vector of length 2 (no rotation DOF)
        bn.add_BC('fix', 'D', 'node', [0], [0., 0.])
        bn.add_BC('fix_t', 'D', 'node', [1], [None, 0.])
        bn.add_BC('load', 'N', 'node', [1], [1., None])
        bn.solve()
        assert bn.has_solution

    def test_2d_beam_bc_unchanged(self):
        """2D beam BC vector should still have length 3."""
        beam_prop = {'name': 'circle', 'radius': 0.1, 'E': 1e6, 'nu': 0.3}
        nodes = np.array([[0., 0.], [1., 0.]])
        edges = np.array([[0, 1]])
        bn = ElasticNetwork(
            nodes, edges, beam_prop=beam_prop,
            options={'vectorize': True, 'matrix': 'bsr', 'verbose': False},
            outdir='/tmp')
        bn.add_BC('fix', 'D', 'node', [0], [0., 0., 0.])
        bn.add_BC('load', 'N', 'node', [1], [1., 0., None])
        bn.solve()
        assert bn.has_solution
