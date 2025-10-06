
import pytest
import numpy as np
from scipy.spatial.transform import Rotation

from beam_networks.problem import BeamNetwork
from beam_networks.geo import get_geometric_props
from beam_networks.reference_solutions.cantilever import cantilever_analytic


@pytest.mark.parametrize('phi_deg_z, phi_deg_y, a, num_nodes, assembly, vectorized,, solver',
                         [(0., 0., 0.5, 51, 'bsr', True, 'direct'),
                          (0., 45.,  0.5, 51, 'bsr', False, 'cg'),
                          (-45., 30., 0.5, 51, 'lil', True, 'direct'),
                          (30., 90., 0.7, 101, 'lil', False, 'direct'),
                          (30., 120., 0.7, 101, 'dense', True, 'cg'),
                          (-120., -30., 0.9, 101, 'dense', False, 'direct'),
                          (-120., -135., 0.9, 101, 'bsr', True, 'cg'),
                          (180., 0., 0.3, 101, 'bsr', True, 'direct')])
def test_cantilever_rot(phi_deg_z, phi_deg_y, a, num_nodes, assembly, vectorized, solver):

    # Beam properties
    E = 1.
    nu = 0.3
    R = 0.05
    length = 50.
    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

    # Rotation
    phi_z = phi_deg_z / 180 * np.pi
    phi_y = phi_deg_y / 180 * np.pi
    rot = Rotation.from_euler('zy', [phi_z, phi_y]).as_matrix()

    # External force and torque
    Iy, Iz, _, A, kappa, ymax = get_geometric_props(props)
    _Fext = np.array([0.05 * E * A / length, 0.05 * E * Iz / length**3, 0.])
    _Mext = np.array([0., 0., -_Fext[1] * a / length])
    F_local = np.hstack([_Fext, _Mext])
    F_rot = np.hstack([rot.dot(_Fext), rot.dot(_Mext)])

    ############################

    # Nodes positions (rotated)
    x = np.linspace(0., length, num_nodes)
    y = np.zeros(num_nodes)
    z = np.zeros(num_nodes)
    nodes_positions = np.vstack([x, y, z]).T
    nodes_positions = np.matmul(rot[None, :, :], nodes_positions.T)[0].T

    # Adjacency
    num_elements = num_nodes - 1
    edges_indices_1 = np.arange(num_elements)
    edges_indices_2 = np.arange(1, num_elements + 1)
    edges_indices = np.vstack([edges_indices_1, edges_indices_2]).T

    # Shuffle adjacency to test ordering
    shuffle_col = np.random.choice(np.arange(num_elements), num_elements, replace=False)
    shuffle_row = np.random.choice([0, 1], num_elements, replace=True)
    edges_indices = edges_indices[shuffle_col]
    edges_indices[shuffle_row] = edges_indices[shuffle_row, ::-1]

    # Force and torque location
    x_pos = rot.dot(np.array([a * length, 0., 0.]))
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
    problem.add_BC('0', 'D', 'node', [0], [0., 0., 0., 0., 0., 0.])
    problem.add_BC('1', 'N', 'node', [node], F_rot)

    # Solve the system
    problem.solve(solver=solver)

    # Rotate solution back into beam frame
    u = problem.displacement.T
    v = problem.rotation.T
    u_rot = rot.T.dot(u)
    v_rot = rot.T.dot(v)

    # Reference solution
    xref_n = np.linspace(0., length, num_nodes)
    xref_e = (xref_n[1:] + xref_n[:-1]) / 2.

    uxref, uyref, tref, _ = cantilever_analytic(xref_n, length, F_local[[0, 1, 5]], dist_n, props)
    _, _, _, sref = cantilever_analytic(xref_e, length, F_local[[0, 1, 5]], dist_n, props)

    np.testing.assert_almost_equal(u_rot[0], uxref)
    np.testing.assert_almost_equal(u_rot[1], uyref)
    np.testing.assert_almost_equal(v_rot[2], tref)
    np.testing.assert_almost_equal(problem._sVM, sref)
