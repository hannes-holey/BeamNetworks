
import os
import numpy as np
from argparse import ArgumentParser

from beam_networks.problem import BeamNetwork


def get_parser():

    parser = ArgumentParser()

    parser.add_argument('-n', '--name', type=str, default='jammed')
    parser.add_argument('-o', '--outdir', type=str, default='data')
    parser.add_argument('-d', '--dim', type=int, default=2)

    return parser


if __name__ == "__main__":

    path = os.path.abspath(os.path.dirname(__file__))
    args, _ = get_parser().parse_known_args()
    name = args.name

    E = 2.1e11
    nu = 0.3
    R = 0.1
    Fext = -0.02

    if args.dim == 2:
        props = {'name': 'rectangle', 'b': 10 * R, 'h': R, 'E': E, 'nu': nu}
        resources = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'resources')
        nodes_positions = np.loadtxt(os.path.join(resources, f'{name}.nodes'))
        edges_indices = np.loadtxt(os.path.join(resources, f'{name}.edges')).astype(int)

        actuator = BeamNetwork(nodes_positions,
                               edges_indices,
                               beam_prop=props,
                               outdir=args.outdir,
                               valid=True)

        actuator.to_stl('mesh.stl')

    elif args.dim == 3:
        props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}
        lattice = BeamNetwork.generate_cubic_lattice(a=1., bbox=(3., 3., 3.), lattice_type='fcc')

        actuator = BeamNetwork(lattice.nodes,
                               lattice.edges,
                               beam_prop=props,
                               outdir=args.outdir,
                               valid=True)

        actuator.to_stl('mesh.stl')
