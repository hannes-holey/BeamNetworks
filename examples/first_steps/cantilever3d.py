
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser

from beam_networks.problem import BeamNetwork
from beam_networks.geo import get_geometric_props
from beam_networks.viz import plot_solution_1D
from beam_networks.reference_solutions.cantilever import cantilever_analytic


def get_parser():

    parser = ArgumentParser()

    parser.add_argument('-E', type=float, default=2.1e11)
    parser.add_argument('-nu', type=float, default=0.3)
    parser.add_argument('-L', type=float, default=2.)
    parser.add_argument('-R', type=float, default=0.05)
    parser.add_argument('-a', type=float, default=0.3)

    parser.add_argument('-s', '--savefig', action='store_true', default=False)
    parser.add_argument('-p', '--plot', action='store_true', default=False)
    parser.add_argument('-o', '--outdir', type=str, default='data')

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    # Beam properties
    E = args.E
    nu = args.nu
    R = args.R
    length = args.L
    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}
    a = args.a

    phi_z_deg = 45.
    phi_y_deg = 0.
    phi_z = phi_z_deg / 180 * np.pi
    phi_y = phi_y_deg / 180 * np.pi
    rot = Rotation.from_euler('zy', [phi_z, phi_y]).as_matrix()

    Iy, Iz, _, A, kappa, ymax = get_geometric_props(props)
    _Fext = np.array([0.05 * E * A / length, 0.1 * E * Iz / length**3, 0.])
    _Mext = np.array([0., 0., -0.2 * _Fext[1] * a / length])
    F_local = np.hstack([_Fext, _Mext])
    F_rot = np.hstack([rot.dot(_Fext), rot.dot(_Mext)])

    # Discretization
    num_nodes = 51
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

    # Force and torque location
    x_pos = rot.dot(np.array([a * length, 0., 0.]))
    dist = np.sqrt(np.sum((nodes_positions - x_pos)**2, axis=-1))
    node = np.argmin(dist)
    dist_n = np.sqrt(np.sum(nodes_positions[node]**2)) / length

    # System setup
    problem = BeamNetwork(nodes_positions,
                          edges_indices,
                          beam_prop=props,
                          valid=True)

    # Add BCs
    problem.add_BC('0', 'D', 'node', [0], [0., 0., 0., 0., 0., 0.])
    problem.add_BC('1', 'N', 'node', [node], F_rot)

    # Solve the system
    problem.solve()
    d_num = problem.sol

    u = np.vstack([d_num[0::6], d_num[1::6], d_num[2::6]])
    v = np.vstack([d_num[3::6], d_num[4::6], d_num[5::6]])
    u_rot = rot.T.dot(u)
    v_rot = rot.T.dot(v)

    # Bring in same format as 2D for plotting
    d_num_red = np.hstack([u_rot[0], u_rot[1], v_rot[2]]).reshape(3, -1).T.flatten()

    sx, sy = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(4,
                           figsize=(sx, 2 * sy),
                           sharex=True,
                           constrained_layout=True)

    # Reference solution
    xref = np.linspace(0., length, 101)
    uxref, uref, tref, sref = cantilever_analytic(xref, length, F_local[[0, 1, 5]], dist_n, props)
    ax[0].plot(xref, uxref, '--', color='0.0', label='Analytical')
    ax[1].plot(xref, uref, '--', color='0.0')
    ax[2].plot(xref, tref, '--', color='0.0')
    ax[3].plot(xref, sref, '--', color='0.0')

    # Plot solution along beam axis
    plot_solution_1D(ax, d_num_red, problem._sVM, length)

    if args.savefig:
        fig.savefig('fig01_cantilever.pdf')

    if args.plot:
        plt.show()
