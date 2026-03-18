# %% [markdown]
# # Low-Level Assembly (No OOP)
#
# This tutorial shows how to assemble and solve a 3D beam network without using
# the `ElasticNetwork` high-level interface. The same steps happen internally;
# exposing them is useful for understanding the assembly pipeline or for writing
# custom workflows.
#
# **Steps:**
# 1. Generate an FCC lattice with `Network.generate_cubic_lattice`.
# 2. Assemble the global stiffness matrix with `assemble_global_system`.
# 3. Define BCs using `_get_bc_dof` / `_assemble_BCs`.
# 4. Solve with the direct linear solver.

# %% Imports
import numpy as np

from beam_networks.network import Network
from beam_networks.fem.assembly import assemble_global_system
from beam_networks.fem.bc import _get_bc_dof, _assemble_BCs
from beam_networks.solvers.linear import solve

# %% [markdown]
# ## Generate lattice

# %%
E = 2.1e11
nu = 0.3
R = 0.05

lattice = Network.generate_cubic_lattice(a=1., bbox=(5., 2., 2.), lattice_type='fcc')

radius = np.ones(lattice.num_edges)
props = {'name': 'circle', 'radius': radius, 'E': E, 'nu': nu}

# %% [markdown]
# ## Assemble stiffness matrix

# %%
K = assemble_global_system(lattice.nodes,
                           lattice.edges,
                           lattice.edge_vectors,
                           props,
                           sorted_edges=False,
                           vectorize=True, verbose=True,
                           matrix="dense")

# %% [markdown]
# ## Define and apply BCs

# %%
bc = {}

bc['0'] = {'type': 'D', 'active': True}
bc['0'].update(_get_bc_dof(lattice.nodes, 'box',
                            [None, 0.01, None, None, None, None],
                            [0., 0., 0., 0., 0., 0.]))

bc['1'] = {'type': 'N', 'active': True}
bc['1'].update(_get_bc_dof(lattice.nodes, 'point',
                            [5., 1.5, 1.5],
                            [None, -1, None, None, None, None]))

dof_D, val_D, dof_N, val_N = _assemble_BCs(bc)

# %% [markdown]
# ## Solve

# %%
u, F, info = solve(K, dof_D, val_D, dof_N, val_N)
print(f'Solved: {lattice.num_nodes} nodes, {lattice.num_edges} elements')
print(f'Max displacement: {np.max(np.abs(u)):.4e}')
