# %% [markdown]
# # Jammed Fibre Network: Box Selection BCs
#
# This tutorial shows how to load a pre-generated disordered (jammed) fibre
# network from the resource files and solve it with box-selection boundary
# conditions.
#
# **BCs** — the bottom strip ($y < 0.05$ in box units) is fully clamped
# ($u_x = u_y = \theta_z = 0$). A uniform downward force $F_y = -0.02$ is
# applied to all nodes in the top strip ($y > 0.95$).
#
# The deformed network is coloured by von Mises stress.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from beam_networks import ElasticNetwork

# %% [markdown]
# ## Load network and solve

# %%
nodes_positions = np.loadtxt('../resources/jammed.nodes')
edges_indices   = np.loadtxt('../resources/jammed.edges').astype(int)

E = 2.1e11
nu = 0.3
R = 0.05
Fext = -0.02

props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

actuator = ElasticNetwork(nodes_positions, edges_indices, beam_prop=props, valid=True)

actuator.add_BC('0',               # label
                'D',               # Dirichlet
                'box',             # select all nodes in a box
                [None, None, None, 0.05],  # [xlo, xhi, ylo, yhi] in box units
                [0., 0., 0.])      # prescribed [ux, uy, θz]

actuator.add_BC('1',
                'N',               # Neumann
                'box',
                [None, None, 0.95, None],
                [None, Fext, None])

actuator.solve()

# %% [markdown]
# ## Deformed network
#
# The network is coloured by the von Mises stress field.

# %%
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

fig, ax = plt.subplots(1)
actuator.plot(ax, contour=actuator._sVM)
plt.show()
