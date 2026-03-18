# %% [markdown]
# # Geometric Reflection of a Deformed Network
#
# Many network geometries possess symmetry planes. This tutorial shows how to
# exploit symmetry by solving on a quarter domain and then reflecting the
# deformed shape to reconstruct the full network.
#
# **Network** — a triangular lattice loaded from the resource files.
#
# **BCs** — bottom strip clamped; top strip displaced downward; symmetry BC
# on the left face ($u_x = \theta_z = 0$).
#
# After solving, the deformed quarter domain is reflected:
# 1. at the $x_\mathrm{min}$ plane → the left half appears.
# 2. at the $y_\mathrm{min}$ plane → the bottom half appears.
#
# The original (quarter) deformed network is drawn in blue; the reflected
# parts in orange.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from beam_networks import ElasticNetwork
from beam_networks.geometry.selection import _reflect
from beam_networks.postprocess.viz import _plot_network

# %% [markdown]
# ## Load network and solve

# %%
nodes = np.loadtxt('../resources/triangular.nodes')
edges = np.loadtxt('../resources/triangular.edges').astype(int)

E = 2.1e11
nu = 0.3
R = 0.05
Fext = -.75

props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

actuator = ElasticNetwork(nodes, edges, beam_prop=props, valid=False)

actuator.add_BC('0', 'D', 'box', [None, None, None, 0.05], [0., 0., 0.])
actuator.add_BC('1', 'D', 'box', [None, None, 0.95, None], [None, Fext, None])
actuator.add_BC('sym_x_min', 'D', 'box', [None, 0.01, None, None], [0, None, 0])

actuator.solve()

# %% [markdown]
# ## Plot deformed quarter + reflections

# %%
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

fig, ax = plt.subplots(1)
actuator.plot(ax, node_ids=False, contour=None, lw=3)

# reflect at x_min plane, then y_min plane
nodes_r, edges_r = _reflect(actuator.displaced_nodes, edges, plane='x_min')
nodes_r, edges_r = _reflect(nodes_r, edges_r, plane='y_min')

_plot_network(ax, nodes_r, edges_r,
              nodes_r[edges_r[:, 1]] - nodes_r[edges_r[:, 0]],
              color='C1', lw=1)
plt.show()
