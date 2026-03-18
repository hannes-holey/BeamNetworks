# %% [markdown]
# # 2D Periodic Boundary Conditions
#
# This tutorial demonstrates three periodic-boundary-condition (PBC) configurations
# for a 2D disordered network built from a hard-disk packing:
#
# - **PBC in both $x$ and $y$** (`pbcx=True, pbcy=True`)
# - **PBC in $x$ only** (`pbcx=True, pbcy=False`)
# - **PBC in $y$ only** (`pbcx=False, pbcy=True`)
#
# Each problem is loaded from the same `hard_disks.txt` resource file and solved
# with an appropriate pair of point-displacement BCs.
#
# The figure shows a $3\times3$ grid of panels:
# rows — displacement magnitude, rotation, von Mises stress;
# columns — PBC-xy, PBC-x, PBC-y.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from beam_networks import ElasticNetwork
from beam_networks.postprocess.viz import _plot_network, _array_to_colors
from beam_networks.geometry.selection import get_edges_from_disks

# %% [markdown]
# ## Helper functions

# %%
def get_problem(pbcx, pbcy):
    nodes, edges, Lx, Ly = get_edges_from_disks('../resources/hard_disks.txt')

    problem = ElasticNetwork(
        nodes, edges,
        valid=False,
        periodic=[pbcx, pbcy],
        boxsize=(Lx, Ly),
        options={'vectorize': True, 'matrix': 'bsr', 'verbose': False},
    )

    if pbcx and pbcy:
        problem.add_BC('0', 'D', 'point', [Lx / 2., Ly / 2.], [0., .5, 0.])
        problem.add_BC('1', 'D', 'point', [Lx / 2., 0.],      [0., -.5, 0.])
    elif pbcx and not pbcy:
        problem.add_BC('0', 'D', 'point', [Lx / 2., 0.], [.5, 0., 0.])
        problem.add_BC('1', 'D', 'point', [0.,      0.], [-.5, 0., 0.])
    else:
        problem.add_BC('0', 'D', 'point', [0, Ly / 2.], [0., .5, 0.])
        problem.add_BC('1', 'D', 'point', [0, 0.],      [0., -.5, 0.])

    problem.solve()
    return problem


def get_disp_magnitude_edge(p):
    u_mag_0 = np.sqrt(np.sum(p.displacement[p.edges[:, 0]]**2, axis=-1))
    u_mag_1 = np.sqrt(np.sum(p.displacement[p.edges[:, 1]]**2, axis=-1))
    return (u_mag_0 + u_mag_1) / 2.


def get_rotation_edge(p):
    return (p.rotation[p.edges[:, 0]] + p.rotation[p.edges[:, 1]]) / 2.

# %% [markdown]
# ## Solve all three cases

# %%
pxy = get_problem(True,  True)
px  = get_problem(True,  False)
py  = get_problem(False, True)

# %% [markdown]
# ## Visualisation

# %%
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

sx, sy = plt.rcParams['figure.figsize']
fig, ax = plt.subplots(3, 3, constrained_layout=True, figsize=(3 * sx, 3 * sy))
repeat = 2

xx, yy = np.meshgrid(np.arange(repeat), np.arange(repeat))
uxy = get_disp_magnitude_edge(pxy)
txy = get_rotation_edge(pxy)

for _shift in zip(xx.flatten(), yy.flatten()):
    shift = np.array(_shift) * np.array(pxy.boxsize)
    _plot_network(ax[0, 0], (pxy.nodes + shift)[:, :2], pxy.edges,
                  pxy.edge_vectors, boxsize=pxy._boxsize, lw=2.5, edge_data=uxy)
    _plot_network(ax[1, 0], (pxy.nodes + shift)[:, :2], pxy.edges,
                  pxy.edge_vectors, boxsize=pxy._boxsize, lw=2.5, edge_data=txy, cmap='coolwarm')
    _plot_network(ax[2, 0], (pxy.nodes + shift)[:, :2], pxy.edges,
                  pxy.edge_vectors, boxsize=pxy._boxsize, lw=2.5, edge_data=pxy._sVM)

xx = np.arange(repeat);  yy = np.zeros_like(xx)
ux = get_disp_magnitude_edge(px);  tx = get_rotation_edge(px)

for _shift in zip(xx.flatten(), yy.flatten()):
    shift = np.array(_shift) * np.array(pxy.boxsize)
    _plot_network(ax[0, 1], (px.nodes + shift)[:, :2], px.edges,
                  px.edge_vectors, boxsize=px._boxsize, lw=2.5, edge_data=ux)
    _plot_network(ax[1, 1], (px.nodes + shift)[:, :2], px.edges,
                  px.edge_vectors, boxsize=px._boxsize, lw=2.5, edge_data=tx, cmap='coolwarm')
    _plot_network(ax[2, 1], (px.nodes + shift)[:, :2], px.edges,
                  px.edge_vectors, boxsize=px._boxsize, lw=2.5, edge_data=px._sVM)

yy = np.arange(repeat);  xx = np.zeros_like(yy)
uy = get_disp_magnitude_edge(py);  ty = get_rotation_edge(py)

for _shift in zip(xx.flatten(), yy.flatten()):
    shift = np.array(_shift) * np.array(pxy.boxsize)
    _plot_network(ax[0, 2], (py.nodes + shift)[:, :2], py.edges,
                  py.edge_vectors, boxsize=py._boxsize, lw=2.5, edge_data=uy)
    _plot_network(ax[1, 2], (py.nodes + shift)[:, :2], py.edges,
                  py.edge_vectors, boxsize=py._boxsize, lw=2.5, edge_data=ty, cmap='coolwarm')
    _plot_network(ax[2, 2], (py.nodes + shift)[:, :2], py.edges,
                  py.edge_vectors, boxsize=py._boxsize, lw=2.5, edge_data=py._sVM)

for col_title, col in zip(['PBC $xy$', 'PBC $x$', 'PBC $y$'], range(3)):
    ax[0, col].set_title(col_title)

sm, _ = _array_to_colors(np.hstack([uy, ux, uxy]), cmap='plasma')
plt.colorbar(sm, ax=ax[0, :], orientation='vertical',
             label='Displacement magnitude (edge avg.)')

sm, _ = _array_to_colors(np.vstack([ty, tx, txy]), cmap='coolwarm')
plt.colorbar(sm, ax=ax[1, :], orientation='vertical',
             label='Rotation (edge avg.)')

sm, _ = _array_to_colors(np.hstack([py._sVM, px._sVM, pxy._sVM]), cmap='plasma')
plt.colorbar(sm, ax=ax[2, :], orientation='vertical',
             label='Von Mises stress')

plt.show()
