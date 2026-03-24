# %% [markdown]
# # Poisson Ratio of a Jammed Network
#
# This tutorial computes the effective Poisson ratio of a 2D disordered fibre
# network under uniaxial compression. Two methods are compared:
#
# - **Displacement method** (`poisson_via_disp`): measures the relative change
#   in mean edge positions near the boundaries.
# - **Gradient method** (`poisson_via_grad`): bins the nodal displacements
#   spatially and takes the ratio of their gradients.
#
# **BCs** — bottom strip fixed in $y$ only; top strip displaced by $+1$ in $y$;
# the origin node pinned in $x$ to remove rigid-body translation.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from beam_networks import ElasticNetwork
from beam_networks.geometry.selection import box_selection

# %% [markdown]
# ## Helper functions

# %%


def poisson_via_disp(actuator):
    left = box_selection(actuator._nodes, (None, 0.02, None, None))
    right = box_selection(actuator._nodes, (0.98, None, None, None))
    top = box_selection(actuator._nodes, (None, None, 0.98, None))
    bot = box_selection(actuator._nodes, (None, None, None, 0.02))

    ux = actuator.sol[0::3]
    uy = actuator.sol[1::3]

    Lx = (np.mean(actuator._nodes[right, 0] + ux[right])
          - np.mean(actuator._nodes[left, 0] + ux[left]))
    Lx0 = np.mean(actuator._nodes[right, 0]) - np.mean(actuator._nodes[left, 0])

    Ly = (np.mean(actuator._nodes[top, 1] + uy[top])
          - np.mean(actuator._nodes[bot, 1] + uy[bot]))
    Ly0 = np.mean(actuator._nodes[top, 1]) - np.mean(actuator._nodes[bot, 1])

    return -((Lx - Lx0) / Lx0) / ((Ly - Ly0) / Ly)


def poisson_via_grad(actuator, nbins=None):
    if nbins is None:
        nbins = actuator.num_nodes // 20

    ux = actuator.sol[0::3]
    uy = actuator.sol[1::3]

    dx = actuator.Lx / nbins
    dy = actuator.Ly / nbins
    xbins = ((actuator._nodes[:, 0] - actuator.xlo) // dx).astype(int)
    ybins = ((actuator._nodes[:, 1] - actuator.ylo) // dy).astype(int)

    _ux, _uy = [], []
    for i in range(nbins):
        _ux.append(np.mean(ux[xbins == i]))
        _uy.append(np.mean(uy[ybins == i]))

    dux = np.gradient(_ux, actuator.Lx / nbins)
    duy = np.gradient(_uy, actuator.Ly / nbins)
    return -np.nanmean(dux) / np.nanmean(duy)

# %% [markdown]
# ## Load network and solve


# %%
nodes_positions = np.loadtxt('../resources/jammed.nodes')
edges_indices = np.loadtxt('../resources/jammed.edges').astype(int)

E = 2.1e11
nu = 0.3
R = 0.05
dy = 1.

props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

actuator = ElasticNetwork(nodes_positions, edges_indices, beam_prop=props, valid=True)

actuator.add_BC('0', 'D', 'box', [None, None, None, 0.05], [None, 0., None])
actuator.add_BC('1', 'D', 'box', [None, None, 0.95, None], [None, dy, None])
actuator.add_BC('2', 'D', 'point', [0., 0.], [0., None, 0.])

actuator.solve()

# %% [markdown]
# ## Poisson ratio

# %%
nu_d = poisson_via_disp(actuator)
nu_g = poisson_via_grad(actuator)

print(f'Poisson ratio (displacement method): {nu_d:.4f}')
print(f'Poisson ratio (gradient method):     {nu_g:.4f}')

# %% [markdown]
# ## Deformed network
#
# Coloured by the average nodal $u_x$ displacement per edge.

# %%
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

disp = (actuator.sol[0::3][actuator.edges[:, 0]]
        + actuator.sol[0::3][actuator.edges[:, 1]]) / 2.

fig, ax = plt.subplots(1)
actuator.plot(ax, contour=disp)
plt.show()
