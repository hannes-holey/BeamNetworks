
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from beam_networks.problem import BeamNetwork


def get_parser():

    parser = ArgumentParser()

    parser.add_argument('-n', '--name', type=str, default='jammed')
    parser.add_argument('-p', '--plot', action='store_true', default=False)

    return parser


if __name__ == "__main__":

    args, _ = get_parser().parse_known_args()
    name = args.name

    resources = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'resources')
    nodes_positions = np.loadtxt(os.path.join(resources, f'{name}.nodes'))
    edges_indices = np.loadtxt(os.path.join(resources, f'{name}.edges')).astype(int)

    E = 2.1e11
    nu = 0.3
    R = 0.05
    Fext = -0.02

    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

    actuator = BeamNetwork(nodes_positions,
                           edges_indices,
                           beam_prop=props,
                           valid=True)

    actuator.add_BC('0',                        # just a name
                    'D',                        # 'D': Dirichlet, 'N': Neumann
                    'box',                      # 'box', 'node', 'point'
                    [None, None, None, 0.05],   # box: [xlo, xhi, ylo, yhi] (box units)
                    [0., 0., 0.]                # 'D': [ux, uy, thetaz], 'N': [Fx, Fy, Mz]
                    )

    actuator.add_BC('1',
                    'N',
                    'box',
                    [None, None, 0.95, None],
                    [None, Fext, None])

    actuator.solve()

    if args.plot:
        fig, ax = plt.subplots(1)
        actuator.plot(ax, contour=actuator._sVM)

        plt.show()
