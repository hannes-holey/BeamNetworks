from argparse import ArgumentParser

from beam_networks.network import Network
from beam_networks.problem import BeamNetwork


def get_parser():

    parser = ArgumentParser()

    parser.add_argument('-o', '--outdir', type=str, default='data')

    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()

    # Example usage
    E = 2.1e11
    nu = 0.3
    R = 0.05
    length = 1.
    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

    lt = 'bcc'
    lattice = Network.generate_cubic_lattice(a=1., bbox=(20., 5., 5.), lattice_type=lt)

    problem = BeamNetwork(lattice._nodes,
                          lattice._edges,
                          beam_prop=props,
                          valid=True,
                          outdir=args.outdir)

    problem.add_BC('0', 'D', 'box',
                   [None, 0.01, None, None, None, None],
                   [0., 0., 0., 0., 0., 0.])

    problem.add_BC('1', 'N', 'point',
                   [20., 2.5, 2.5],
                   [None, -1, None, None, None, None])

    problem.solve()
    problem.to_vtk(f"{lt}_lattice.vtk")
