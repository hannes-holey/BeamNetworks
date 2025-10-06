import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    fig, ax = plt.subplots(1)

    versions = ['direct', 'pcg', 'cg']
    markers = ['-X', '-o', '-D']
    colors = ['C0', 'C1', 'C2']

    bench_path = os.path.dirname(os.path.abspath(__file__))

    for v, m, c in zip(versions, markers, colors):
        n, t = np.loadtxt(os.path.join(bench_path, f'time_{v}.txt'), unpack=True)
        ax.loglog(n, t, m, label=v, color=c)

    ax.loglog(n[:7], 1e-5 * n[:7], '--', color='0.7')
    ax.loglog(n[-5:], 1e-4 * n[-5:], '--', color='0.7')

    ax.set_aspect(1.)
    ax.set_xlabel('Num. DOFs')
    ax.set_ylabel('Solve time (s)')

    ax.legend()

    plt.show()
