import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from beam_networks.problem import ElasticNetwork
from beam_networks.geometry.geo import get_geometric_props


# ---------------------------------------------------------------------------
# Common geometry and material
# ---------------------------------------------------------------------------
ne = 8
beam_prop = {'b': 1., 'h': 1., 'E': 1e7, 'nu': 0.3, 'name': 'rectangle'}
Iz = get_geometric_props(beam_prop)[1]
Iy = get_geometric_props(beam_prop)[0]

angle = np.linspace(0., np.pi / 4., ne + 1)
x = 100. * np.sin(angle)
y = -100. * (1. - np.cos(angle))
nodes = np.column_stack([x, y, np.zeros(ne + 1)])
edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])

n_steps = 8


def run(matrix='bsr', Fz=0.):
    net = ElasticNetwork(nodes, edges, beam_prop=beam_prop,
                         options={'vectorize': True,
                                  'matrix': matrix,
                                  'verbose': False},
                         assemble_on_init=True)

    net.add_BC('clamp', 'D', 'node', [0], [0., 0., 0., 0., 0., 0.])
    net.add_BC('load', 'N', 'node', [ne], [None, None, Fz, None, None, None])

    net.solve_nonlinear(n_steps=n_steps,
                        tol=1e-9,
                        verbose=True,
                        callback=None,
                        ref_vectors=None)

    return net


def draw_3d(ax, net, c='C0'):
    nodes_cur = net.displaced_nodes
    edges = net.edges

    segs = [(nodes_cur[e0], nodes_cur[e1]) for e0, e1 in edges]
    ax.add_collection(Line3DCollection(segs, linewidths=1.5, colors=c))
    ax.scatter(*nodes_cur[[-1]].T, s=20, color=c, zorder=3)


try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

fig, ax = plt.subplots(1, subplot_kw={'projection': '3d'}, figsize=(7, 5))

crisfield_ref = np.array([[70.71, -29.29, 0.],
                          [58.53, -22.16, 40.53],
                          [51.93, -18.43, 48.79],
                          [46.84, -15.61, 53.71]])


for i, Fz in enumerate([0., 300., 450., 600.]):

    print('---')
    print('Load: ', f'{Fz:.1f}')

    net_xy = run(Fz=Fz)
    tip_xy = net_xy.displaced_nodes[-1]

    print('Present: ', f"({tip_xy[0]:.2f}, {tip_xy[1]:.2f}, {tip_xy[2]:.2f})")
    print('Expected:', f"({crisfield_ref[i, 0]:.2f}, {crisfield_ref[i, 1]:.2f}, {crisfield_ref[i, 2]:.2f})")

    draw_3d(ax, net_xy, c=f'C{i}')
    ax.plot([], [], color=f'C{i}', label=f'{Fz:.0f}')

    ax.scatter(*crisfield_ref[i],
               marker='o',
               facecolor='none',
               linewidths=1.,
               edgecolors='0.0',
               label='Crisfield (1990)' if i == 3 else None,
               zorder=-10)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(title='Tip load (z)')
ax.set_aspect('equal')

plt.show()
