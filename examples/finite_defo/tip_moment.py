import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from beam_networks.geo import get_geometric_props
from crisfield2d import solve_nonlin


def simple_plot(ax, nodes, edges, c=1., cmap=plt.cm.coolwarm):

    if cmap is not None:
        color = cmap(c)
    else:
        color = c

    xe = nodes[:, 0]
    ye = nodes[:, 1]

    ax.scatter(xe, ye, 10, color=color, marker='o')

    segments = [((nodes[e0, 0], nodes[e0, 1]),
                 (nodes[e1, 0], nodes[e1, 1])) for e0, e1 in edges]

    linecollection = LineCollection(segments=segments,
                                    linewidths=2.,
                                    colors=color)

    ax.add_collection(linecollection)


if __name__ == "__main__":

    plt.style.use('../paper.mplstyle')
    sx, sy = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(1, figsize=(sx, sx))

    ne = 10
    Lx = 10.
    beam_prop = {'b': 1., 'h': 1., 'E': 2.e11, 'nu': .0, 'name': 'rectangle'}
    E = beam_prop['E']
    Iz = get_geometric_props(beam_prop)[1]
    Mref = 2 * np.pi * E * Iz / Lx

    x = np.linspace(0., Lx, ne + 1)
    y = np.zeros_like(x)
    nodes = np.vstack([x, y]).T
    edges = np.vstack([np.arange(ne), np.arange(ne) + 1]).T
    num_dof = nodes.shape[0] * 3

    for Fmax in [Mref]:

        print('---')

        f_ext = np.zeros(num_dof)
        u = np.zeros(num_dof)

        bc_D = [0, 1, 2]
        bc_Dval = [0., 0., 0.]
        bc_N = [num_dof - 1]

        it = 100
        F_cum = 0.
        dF = Fmax / it
        for j in range(1, it + 1):

            F_cum += dF
            f_ext[bc_N] = dF

            u = solve_nonlin(nodes, edges, u, f_ext, beam_prop, bc_D)

            ux = u[0::3]
            uy = u[1::3]

            nodes[:, 0] += ux
            nodes[:, 1] += uy

            if j % (it//10) == 0:
                simple_plot(ax, nodes, edges, c=j / it)

        simple_plot(ax, nodes, edges, c=j / it)

    ax.set_aspect(1.)
    ax.set_box_aspect(1.)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                               norm=plt.Normalize(vmin=0, vmax=Fmax / Mref))

    fig.colorbar(sm, ax=ax, shrink=.5, label=r'$M_z/2\pi EI$')
    ax.set_title('Cantilever with tip moment')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()
