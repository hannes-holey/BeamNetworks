# %% [markdown]
# # 2D FCC Lattice: `from_square_lattice`
#
# This tutorial demonstrates the `ElasticNetwork.from_square_lattice` factory
# method for generating a face-centred cubic (FCC) 2D lattice in a rectangular
# bounding box.
#
# **Geometry** — lattice constant $a = 0.25$, bounding box $20 \times 5$.
#
# **BCs** — left edge clamped ($u_x = u_y = \theta_z = 0$);
# the point at $(20, 2.5)$ is displaced downward by 1 unit.
#
# The solved displacement field is exported to a VTK file for visualisation
# in ParaView.

# %% Imports
from beam_networks import ElasticNetwork

# %% [markdown]
# ## Build lattice and solve

# %%
E = 2.1e11
nu = 0.3
R = 0.05
props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

lt = 'fcc'

problem = ElasticNetwork.from_square_lattice(
    a=0.25,
    bbox=(20., 5.),
    lattice_type=lt,
    beam_prop=props)

problem.add_BC('0', 'D', 'box',
               [None, 0.01, None, None],
               [0., 0., 0.])

problem.add_BC('1', 'D', 'point',
               [20., 2.5],
               [None, -1, None])

problem.solve()
problem.to_vtk(f'{lt}_lattice.vtk')

print(f'Network: {problem.num_nodes} nodes, {problem.num_edges} elements')
print(f'VTK written to {lt}_lattice.vtk')
