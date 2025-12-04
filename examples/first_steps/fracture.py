import os
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from beam_networks.fracture import FractureProblem


def get_parser():

    parser = ArgumentParser()

    parser.add_argument('-s', '--sign', type=int, default=-1)
    parser.add_argument('-m', '--mode', type=str, default='cascade')
    parser.add_argument('-o', '--outdir', type=str, default='data')
    parser.add_argument('-p', '--plot', action='store_true', default=False)

    return parser


def pdf_weibull(x, eta, beta):
    x_n = x / eta
    return beta / eta * x_n ** (beta - 1.) * np.exp(-x_n**beta)


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    sign = np.sign(args.sign)  # 1: tension, -1: compression
    mode = args.mode

    props = {'name': 'circle',
             'radius': 0.05,
             'E': 2.1e11,
             'nu': 0.3}

    name = 'jammed'
    resources = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'resources')

    nodes_positions = np.loadtxt(os.path.join(resources, f'{name}.nodes'))
    edges_indices = np.loadtxt(os.path.join(resources, f'{name}.edges')).astype(int)

    if args.plot:
        fig, ax = plt.subplots(1)

    mean = 0.1
    props['strength'] = mean * props['E']
    betas = [None, 4.]
    modes = ['cascade', 'adiabatic']

    size = edges_indices.shape[0]

    for i, beta in enumerate(betas):

        # failure distribution
        if beta is None:
            dist = beta
        else:
            dist = np.random.weibull(a=beta, size=size)

        for j, mode in enumerate(modes):

            fdist = f'weibull_shape{beta}_mean{mean}' if beta is not None else f'const{mean}'
            defo = 'tension' if sign > 0. else 'compression'
            outdir = os.path.join(args.outdir, f"fracture_{defo}_{fdist}")

            tic = datetime.now()
            problem = FractureProblem(nodes_positions,
                                      edges_indices,
                                      save_trajectory=True,
                                      beam_prop=props,
                                      valid=False,
                                      outdir=outdir)

            problem.add_BC('0', 'D', 'box', [None, None, None, .05], [0., 0., 0.])
            problem.add_BC('1', 'D', 'box', [None, None, 0.95, None], [None, 0., None])

            # Run
            problem.run(mode=mode, sign=sign, dist=dist)
            problem.write()

            t = datetime.now() - tic
            print('Elapsed time: ',  str(t).split('.')[0])

            if args.plot:
                buffer = np.array(problem._output['stress_strain'])
                color = f'C{i}'
                ls = '-' if mode == "cascade" else '--'
                ax.plot(*buffer.T, ls, color=color)

    if args.plot:
        ax.set_xlabel('Strain')
        ax.set_ylabel('Stress')
        plt.show()
