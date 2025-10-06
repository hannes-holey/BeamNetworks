import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    fig, ax = plt.subplots(1)

    versions = ['bsr-1', 'lil-1', 'dense-1', 'bsr-0', 'lil-0', 'dense-0']
    markers = ['-X', '-o', '-D', '--X', '--o', '--D']
    colors = 2 * ['C0', 'C1', 'C2']

    bench_path = os.path.dirname(os.path.abspath(__file__))

    for v, m, c in zip(versions, markers, colors):
        n, t = np.loadtxt(os.path.join(bench_path, f'time_{v}.txt'), unpack=True)
        ax.loglog(n, t, m, label=v, color=c)

    ax.set_aspect(1.)
    ax.set_xlabel('Num. elements')
    ax.set_ylabel('Assembly time (s)')

    ax.legend(ncol=2)

    plt.show()
