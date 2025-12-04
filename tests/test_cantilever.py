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
from scipy.spatial.transform import Rotation

from beam_networks.problem import BeamNetwork
from beam_networks.geo import get_geometric_props
from beam_networks.reference_solutions.cantilever import cantilever_analytic


@pytest.mark.parametrize('phi_deg, a, num_nodes, assembly, vectorized, solver, preconditioner',
                         [(0., 0.5, 51, 'bsr', True, 'direct', None),
                          (0., 0.5, 51, 'bsr', False, 'cg', None),
                          (-45., 0.5, 51, 'lil', True, 'direct', None),
                          (30., 0.7, 101, 'lil', False, 'direct', None),
                          (30., 0.7, 101, 'dense', True, 'cg', None),
                          (-120., 0.9, 101, 'dense', False, 'direct', None),
                          (-120., 0.9, 101, 'bsr', True, 'cg', None),
                          (180., 0.3, 101, 'bsr', True, 'direct', None),
                          (180., 0.3, 101, 'bsr', True, 'direct', 'diagonal'),
                          (-120., 0.9, 101, 'bsr', True, 'pcg', 'diagonal')])
def test_cantilever_rot(phi_deg, a, num_nodes, assembly, vectorized,
                        solver, preconditioner):

    # Beam properties
    E = 1.
    nu = 0.3
    R = 0.05
    length = 50.
    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

    # Rotation
    phi = phi_deg / 180 * np.pi
    rot = Rotation.from_euler('z', phi).as_matrix()

    # External force and torque
    Iy, Iz, _, A, kappa, ymax = get_geometric_props(props)
    _Fext = np.array([0.05 * E * A / length, 0.05 * E * Iz / length**3])
    _Mext = np.array([-_Fext[1] * a / length])
    F_local = np.hstack([_Fext, _Mext])
    F_rot = rot.dot(F_local)

    ############################

    # Nodes positions (rotated)
    x = np.linspace(0., length, num_nodes)
    y = np.zeros(num_nodes)
    z = np.zeros(num_nodes)
    nodes_positions = np.vstack([x, y, z]).T
    nodes_positions = np.matmul(rot[None, :, :], nodes_positions.T)[0].T
    nodes_positions = nodes_positions[:, :2]

    # Adjacency
    num_elements = num_nodes - 1
    edges_indices_1 = np.arange(num_elements)
    edges_indices_2 = np.arange(1, num_elements + 1)
    edges_indices = np.vstack([edges_indices_1, edges_indices_2]).T

    # Force and torque location
    x_pos = rot.dot(np.array([a * length, 0., 0.]))[:2]
    dist = np.sqrt(np.sum((nodes_positions - x_pos)**2, axis=-1))
    node = np.argmin(dist)
    dist_n = np.sqrt(np.sum(nodes_positions[node]**2)) / length

    # System setup
    problem = BeamNetwork(nodes_positions,
                          edges_indices,
                          beam_prop=props,
                          valid=True,
                          options={'matrix': assembly, 'vectorize': vectorized, 'verbose': False})

    # Add BCs
    problem.add_BC('0', 'D', 'node', [0], [0., 0., 0.])
    problem.add_BC('1', 'N', 'node', [node], F_rot)

    # Solve the system
    problem.solve(solver=solver, preconditioner=preconditioner)
    d_num = problem.sol

    # Rotate solution back into beam frame
    u = np.vstack([d_num[0::3], d_num[1::3], d_num[2::3]])
    u_rot = rot.T.dot(u)

    # Reference solution
    xref_n = np.linspace(0., length, num_nodes)
    xref_e = (xref_n[1:] + xref_n[:-1]) / 2.

    uxref, uyref, tref, _ = cantilever_analytic(xref_n, length, F_local, dist_n, props)
    _, _, _, sref = cantilever_analytic(xref_e, length, F_local, dist_n, props)

    np.testing.assert_almost_equal(u_rot[0], uxref)
    np.testing.assert_almost_equal(u_rot[1], uyref)
    np.testing.assert_almost_equal(u_rot[2], tref)
    np.testing.assert_almost_equal(problem._sVM, sref)
