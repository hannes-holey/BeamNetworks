# BeamNetworks

[![Tests](https://github.com/hannes-holey/BeamNetworks/actions/workflows/test.yaml/badge.svg)](https://github.com/hannes-holey/BeamNetworks/actions/workflows/test.yaml)
[![Coverage](maintenance/coverage.svg)](maintenance/coverage.svg)

Solver for the elastic deformation of Timoshenko beam networks.

## Installation
Install the package using pip.
```
pip install .[tests]
```

## Minimal example
```python
import numpy as np
from beam_networks.problem import BeamNetwork

# Generate nodes and edges
nodes = np.array([[0., 0., 0.],
                  [1., 1., 1.],
                  [2., 2., 2.]])


edges = np.array([[0, 1],
                  [1, 2]])

# Beam cross section and elastic properties
beam = {'name': 'circle', 'radius': 0.1, 'E': 2.1e11, 'nu': 0.3}

# Setup problem
problem = BeamNetwork(nodes, edges, beam_prop=beam)

# Add boundary conditions
problem.add_BC('0', 'D', 'node', [0, ], [0., 0., 0., 0., 0., 0.])
problem.add_BC('1', 'D', 'node', [2, ], [None, -0.1, None, None, None, None])

# Solve
problem.solve()

# Write output
problem.to_vtk('example.vtk')
problem.to_stl('example.stl')

# Plot (only 2D projection)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
problem.plot(ax)
ax.scatter(*problem.displaced_nodes[:, :2].T)
plt.show()

```

More examples can be found in the [examples](./examples) directory.

## Tests
We test against analytic solutions for a 1D cantilever beam both in 2D and 3D space with
```
pytest
```

## Funding
This project received funding from the European Union’s Horizon Europe research and 
innovation programme via the ARCHIBIOFOAM project under grant agreement No 101161052. 
Views and opinions expressed are however those of the author(s) only and do not necessarily 
reflect those of the European Union or European Innovation Council and SMEs Executive 
Agency (EISMEA). Neither the European Union nor the granting authority can be held responsible for them.
