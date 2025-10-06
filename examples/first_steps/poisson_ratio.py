
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from beam_networks.problem import BeamNetwork
from beam_networks.utils import box_selection


def get_parser():

    parser = ArgumentParser()

    parser.add_argument('-n', '--name', type=str, default='jammed')
    parser.add_argument('-p', '--plot', action='store_true', default=False)

    return parser


def poisson_via_disp(actuator):

    left = box_selection(actuator._nodes, (None, 0.02, None, None,))
    right = box_selection(actuator._nodes, (0.98, None, None, None))
    top = box_selection(actuator._nodes, (None, None, 0.98, None,))
    bot = box_selection(actuator._nodes, (None, None, None, 0.02,))

    ux = actuator.sol[0::3]
    uy = actuator.sol[1::3]

    Lx = np.mean(actuator._nodes[right, 0] + ux[right]) - np.mean(actuator._nodes[left, 0] + ux[left])
    Lx0 = np.mean(actuator._nodes[right, 0]) - np.mean(actuator._nodes[left, 0])

    Ly = np.mean(actuator._nodes[top, 1] + uy[top]) - np.mean(actuator._nodes[bot, 1] + uy[bot])
    Ly0 = np.mean(actuator._nodes[top, 1]) - np.mean(actuator._nodes[bot, 1])

    poisson_ratio = -((Lx - Lx0) / Lx0) / ((Ly - Ly0) / Ly)

    return poisson_ratio


def poisson_via_grad(actuator, nbins=None):

    if nbins is None:
        nbins = actuator.num_nodes // 20

    ux = actuator.sol[0::3]
    uy = actuator.sol[1::3]

    _ux, _uy = [], []

    dx = actuator.Lx / nbins
    dy = actuator.Ly / nbins

    xbins = ((actuator._nodes[:, 0] - actuator.xlo) // dx).astype(int)
    ybins = ((actuator._nodes[:, 1] - actuator.ylo) // dy).astype(int)

    for i in range(nbins):
        _ux.append(np.mean(ux[xbins == i]))
        _uy.append(np.mean(uy[ybins == i]))

    dux = np.gradient(_ux, actuator.Lx / nbins)
    duy = np.gradient(_uy, actuator.Ly / nbins)
    pr = -np.nanmean(dux) / np.nanmean(duy)

    return pr


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
    dy = 1.

    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

    actuator = BeamNetwork(nodes_positions,
                           edges_indices,
                           beam_prop=props,
                           valid=True)

    actuator.add_BC('0',                        # just a name
                    'D',                        # 'D': Dirichlet, 'N': Neumann
                    'box',                      # 'box', 'node', 'point'
                    [None, None, None, 0.05],   # box: [xlo, xhi, ylo, yhi] (box units)
                    [None, 0., None]                # 'D': [ux, uy, thetaz], 'N': [Fx, Fy, Mz]
                    )

    actuator.add_BC('1',
                    'D',
                    'box',
                    [None, None, 0.95, None],
                    [None, dy, None])

    actuator.add_BC('2', 'D', 'point', [0., 0., ], [0., None, 0., ])

    actuator.solve()

    nu_d = poisson_via_disp(actuator)
    nu_g = poisson_via_grad(actuator)

    print(nu_d, nu_g)

    if args.plot:
        fig, ax = plt.subplots(1)

        disp = (actuator.sol[0::3][actuator.edges[:, 0]] + actuator.sol[0::3][actuator.edges[:, 1]]) / 2.
        actuator.plot(ax, contour=disp)

        plt.show()
