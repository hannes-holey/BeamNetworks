from argparse import ArgumentParser

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

    lt = 'fcc'

    problem = BeamNetwork.from_square_lattice(a=0.25,
                                              bbox=(20., 5.),
                                              lattice_type=lt,
                                              beam_prop=props,
                                              outdir=args.outdir)

    problem.add_BC('0', 'D', 'box',
                   [None, 0.01, None, None],
                   [0., 0., 0.])

    problem.add_BC('1', 'D', 'point',
                   [20., 2.5],
                   [None, -1, None])

    problem.solve()
    problem.to_vtk(f"{lt}_lattice.vtk")
