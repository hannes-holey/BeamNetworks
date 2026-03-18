# %% [markdown]
# # 3D BCC Lattice: `from_cubic_lattice`
#
# This tutorial uses the `ElasticNetwork.from_cubic_lattice` factory method
# to build a body-centred cubic (BCC) 3D lattice in a rectangular box.
#
# **Geometry** — lattice constant $a = 1$, bounding box $20 \times 5 \times 5$.
#
# **BCs** — left face clamped (all 6 DOFs); a point force $F_y = -1$ applied
# at $(20, 2.5, 2.5)$.
#
# The result is exported to a VTK file.

# %% Imports
from beam_networks import ElasticNetwork

# %% [markdown]
# ## Build lattice and solve

# %%
E = 2.1e11
nu = 0.3
R = 0.05
props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

lt = 'bcc'

problem = ElasticNetwork.from_cubic_lattice(
    a=1.,
    pbc=None,
    bbox=(20., 5., 5.),
    lattice_type=lt,
    beam_prop=props)

problem.add_BC('0', 'D', 'box',
               [None, 0.01, None, None, None, None],
               [0., 0., 0., 0., 0., 0.])

problem.add_BC('1', 'N', 'point',
               [20., 2.5, 2.5],
               [None, -1, None, None, None, None])

problem.solve()
problem.to_vtk(f'{lt}_lattice.vtk')

print(f'Network: {problem.num_nodes} nodes, {problem.num_edges} elements')
print(f'VTK written to {lt}_lattice.vtk')
