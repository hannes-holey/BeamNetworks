# %% [markdown]
# # Exporting to STL
#
# This tutorial shows how to export a beam network mesh to an STL file using
# `ElasticNetwork.to_stl`. STL files can be opened in mesh-inspection tools or
# used as input for 3D printing.
#
# Two cases are demonstrated:
#
# - **2D** — a jammed network with rectangular cross-sections
#   ($b = 10R$, $h = R$) loaded from the resource files.
# - **3D** — a small FCC cubic lattice with circular cross-sections,
#   generated programmatically.

# %% Imports
import numpy as np
from beam_networks import ElasticNetwork

# %% [markdown]
# ## 2D jammed network

# %%
R = 0.1
E = 2.1e11
nu = 0.3

props_2d = {'name': 'rectangle', 'b': 10 * R, 'h': R, 'E': E, 'nu': nu}

nodes_positions = np.loadtxt('../resources/jammed.nodes')
edges_indices = np.loadtxt('../resources/jammed.edges').astype(int)

actuator_2d = ElasticNetwork(nodes_positions, edges_indices,
                             beam_prop=props_2d, valid=True)
actuator_2d.to_stl('mesh_2d.stl')
print('2D STL written to mesh_2d.stl')

# %% [markdown]
# ## 3D FCC lattice

# %%
props_3d = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

lattice = ElasticNetwork.generate_cubic_lattice(
    a=1., bbox=(3., 3., 3.), lattice_type='fcc')

actuator_3d = ElasticNetwork(lattice.nodes, lattice.edges,
                             beam_prop=props_3d, valid=True)
actuator_3d.to_stl('mesh_3d.stl')
print('3D STL written to mesh_3d.stl')
