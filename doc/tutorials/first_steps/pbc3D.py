# %% [markdown]
# # 3D Periodic Boundary Conditions
#
# This tutorial is the 3D counterpart of the 2D PBC example. The same
# hard-disk packing is extruded into 3D by appending a zero $z$-column to
# each node. Three PBC configurations are solved:
#
# - **PBC in $x$ and $y$** — compression along $y$.
# - **PBC in $x$ only** — shear in $x$.
# - **PBC in $y$ only** — compression along $y$.
#
# The visualisation projects the 3D results onto the $x$-$y$ plane for
# comparison with the 2D case.

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
    nodes = np.hstack([nodes, np.zeros(nodes.shape[0])[:, None]])

    problem = ElasticNetwork(
        nodes, edges,
        valid=False,
        periodic=[pbcx, pbcy, False],
        boxsize=(Lx, Ly, 1.),
        options={'vectorize': True, 'matrix': 'bsr', 'verbose': False},
    )

    if pbcx and pbcy:
        problem.add_BC('0', 'D', 'point', [Lx / 2., Ly / 2., 0.], [0., .5, 0., 0., 0., 0.])
        problem.add_BC('1', 'D', 'point', [Lx / 2., 0.,      0.], [0., -.5, 0., 0., 0., 0.])
    elif pbcx and not pbcy:
        problem.add_BC('0', 'D', 'point', [Lx / 2., 0., 0.], [.5, 0., 0., 0., 0., 0.])
        problem.add_BC('1', 'D', 'point', [0.,      0., 0.], [-.5, 0., 0., 0., 0., 0.])
    else:
        problem.add_BC('0', 'D', 'point', [0, Ly / 2., 0.], [0., .5, 0., 0., 0., 0.])
        problem.add_BC('1', 'D', 'point', [0, 0.,      0.], [0., -.5, 0., 0., 0., 0.])

    problem.solve()
    return problem


def get_disp_magnitude_edge(p):
    u_mag_0 = np.sqrt(np.sum(p.displacement[p.edges[:, 0]]**2, axis=-1))
    u_mag_1 = np.sqrt(np.sum(p.displacement[p.edges[:, 1]]**2, axis=-1))
    return (u_mag_0 + u_mag_1) / 2.


def get_rotation_edge(p):
    return (p.rotation[p.edges[:, 0], 2] + p.rotation[p.edges[:, 1], 2]) / 2.

# %% [markdown]
# ## Solve all three cases


# %%
pxy = get_problem(True,  True)
px = get_problem(True,  False)
py = get_problem(False, True)

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
zz = np.zeros_like(xx.flatten())
uxy = get_disp_magnitude_edge(pxy)
txy = get_rotation_edge(pxy)

for _shift in zip(xx.flatten(), yy.flatten(), zz):
    shift = np.array(_shift) * np.array(pxy.boxsize)
    _plot_network(ax[0, 0], (pxy.nodes + shift)[:, :2], pxy.edges,
                  pxy.edge_vectors[:, :2], boxsize=pxy._boxsize[:2], lw=2.5, edge_data=uxy)
    _plot_network(ax[1, 0], (pxy.nodes + shift)[:, :2], pxy.edges,
                  pxy.edge_vectors[:, :2], boxsize=pxy._boxsize[:2], lw=2.5, edge_data=txy, cmap='coolwarm')
    _plot_network(ax[2, 0], (pxy.nodes + shift)[:, :2], pxy.edges,
                  pxy.edge_vectors[:, :2], boxsize=pxy._boxsize[:2], lw=2.5, edge_data=pxy._sVM)

xx = np.arange(repeat)
yy = np.zeros_like(xx)
zz = np.zeros_like(xx)
ux = get_disp_magnitude_edge(px)
tx = get_rotation_edge(px)

for _shift in zip(xx, yy, zz):
    shift = np.array(_shift) * np.array(pxy.boxsize)
    _plot_network(ax[0, 1], (px.nodes + shift)[:, :2], px.edges,
                  px.edge_vectors[:, :2], boxsize=px._boxsize[:2], lw=2.5, edge_data=ux)
    _plot_network(ax[1, 1], (px.nodes + shift)[:, :2], px.edges,
                  px.edge_vectors[:, :2], boxsize=px._boxsize[:2], lw=2.5, edge_data=tx, cmap='coolwarm')
    _plot_network(ax[2, 1], (px.nodes + shift)[:, :2], px.edges,
                  px.edge_vectors[:, :2], boxsize=px._boxsize[:2], lw=2.5, edge_data=px._sVM)

yy = np.arange(repeat)
xx = np.zeros_like(yy)
zz = np.zeros_like(yy)
uy = get_disp_magnitude_edge(py)
ty = get_rotation_edge(py)

for _shift in zip(xx, yy, zz):
    shift = np.array(_shift) * np.array(pxy.boxsize)
    _plot_network(ax[0, 2], (py.nodes + shift)[:, :2], py.edges,
                  py.edge_vectors[:, :2], boxsize=py._boxsize[:2], lw=2.5, edge_data=uy)
    _plot_network(ax[1, 2], (py.nodes + shift)[:, :2], py.edges,
                  py.edge_vectors[:, :2], boxsize=py._boxsize[:2], lw=2.5, edge_data=ty, cmap='coolwarm')
    _plot_network(ax[2, 2], (py.nodes + shift)[:, :2], py.edges,
                  py.edge_vectors[:, :2], boxsize=py._boxsize[:2], lw=2.5, edge_data=py._sVM)

for col_title, col in zip(['PBC $xy$', 'PBC $x$', 'PBC $y$'], range(3)):
    ax[0, col].set_title(col_title)

sm, _ = _array_to_colors(np.hstack([uy, ux, uxy]), cmap='plasma')
plt.colorbar(sm, ax=ax[0, :], orientation='vertical',
             label='Displacement magnitude (edge avg.)')

sm, _ = _array_to_colors(np.hstack([ty, tx, txy]), cmap='coolwarm')
plt.colorbar(sm, ax=ax[1, :], orientation='vertical',
             label='Rotation (edge avg.)')

sm, _ = _array_to_colors(np.hstack([py._sVM, px._sVM, pxy._sVM]), cmap='plasma')
plt.colorbar(sm, ax=ax[2, :], orientation='vertical',
             label='Von Mises stress')

plt.show()
