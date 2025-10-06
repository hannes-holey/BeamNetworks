import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from beam_networks.problem import BeamNetwork
from beam_networks.utils import _reflect
from beam_networks.viz import _plot_network


def get_parser():

    parser = ArgumentParser()

    parser.add_argument('-o', '--outdir', type=str, default='data')
    parser.add_argument('-p', '--plot', action='store_true', default=False)

    return parser


if __name__ == "__main__":

    name = 'triangular'
    resources = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'resources')
    nodes = np.loadtxt(os.path.join(resources, f'{name}.nodes'))
    edges = np.loadtxt(os.path.join(resources, f'{name}.edges')).astype(int)

    E = 2.1e11
    nu = 0.3
    R = 0.05
    Fext = -.75

    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

    actuator = BeamNetwork(nodes,
                           edges,
                           beam_prop=props,
                           valid=False)

    actuator.add_BC('0',                        # just a name
                    'D',                        # 'D': Dirichlet, 'N': Neumann
                    'box',                      # 'box', 'node', 'point'
                    [None, None, None, 0.05],   # box: [xlo, xhi, ylo, yhi] (box units)
                    [0., 0., 0.]                # 'D': [ux, uy, thetaz], 'N': [Fx, Fy, Mz]
                    )

    actuator.add_BC('1',
                    'D',
                    'box',
                    [None, None, 0.95, None],
                    [None, Fext, None])

    # symmetry BC
    actuator.add_BC('sym_x_min',
                    'D',
                    'box',
                    [None, 0.01, None, None],
                    [0, None, 0])

    actuator.solve()

    fig, ax = plt.subplots(1)
    actuator.plot(ax, node_ids=False, contour=None, lw=3)

    # reflect displaced nodes at x_min plane
    nodes, edges = _reflect(actuator.displaced_nodes, edges, plane='x_min')
    # reflect again at y_min plane
    nodes, edges = _reflect(nodes, edges, plane='y_min')

    _plot_network(ax, nodes, edges, nodes[edges[:, 1]] - nodes[edges[:, 0]], color='C1', lw=1)

    args = get_parser().parse_args()
    if args.plot:
        plt.show()
