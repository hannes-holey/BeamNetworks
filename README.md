# ElasticNetworks

[![Tests](https://github.com/hannes-holey/ElasticNetworks/actions/workflows/test.yaml/badge.svg)](https://github.com/hannes-holey/ElasticNetworks/actions/workflows/test.yaml)
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
from beam_networks.problem import ElasticNetwork

# Generate nodes and edges
nodes = np.array([[0., 0., 0.],
                  [1., 1., 1.],
                  [2., 2., 2.]])


edges = np.array([[0, 1],
                  [1, 2]])

# Beam cross section and elastic properties
beam = {'name': 'circle', 'radius': 0.1, 'E': 2.1e11, 'nu': 0.3}

# Setup problem
problem = ElasticNetwork(nodes, edges, beam_prop=beam)

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

More examples can be found in the [examples](./examples) directory, for instance for quasi-static problems
of 2D and 3D lattice structures, or for elastic-brittle fracture problems.

## Tests
We test against analytic solutions for a 1D cantilever beam both in 2D and 3D space with
```
pytest
```

## Dependencies

The core solver requires only:
- [__numpy__](https://numpy.org)
- [__scipy__](https://scipy.org)
- [__PyYAML__](https://pyyaml.org/)

All other dependencies are optional and can be installed via extras:

```
pip install .[viz]   # matplotlib + pyvista (2-D and 3-D plotting)
pip install .[io]    # meshio, trimesh, pandas (VTK/STL/HDF5 output)
pip install .[solvers]  # scikit-sparse, pyamg (alternative linear solvers)
pip install .[viz,io]   # everything needed to run the examples
```

| Extra | Packages | Purpose |
|-------|----------|---------|
| `viz` | [matplotlib](https://matplotlib.org), [pyvista](https://pyvista.org) | 2-D and 3-D network plots |
| `io` | [meshio](https://github.com/nschloe/meshio), [trimesh](https://trimesh.org), [pandas](https://pandas.pydata.org) | VTK, STL and HDF5 file output |
| `solvers` | [scikit-sparse](https://scikit-sparse.readthedocs.io), [pyamg](https://pyamg.readthedocs.io) | Sparse Cholesky and AMG linear solvers |


## Funding
This project received funding from the European Union’s Horizon Europe research and 
innovation programme via the [ARCHIBIOFOAM](https://archibiofoam.eu/) project 
under grant agreement No 101161052. Views and opinions expressed are however those of 
the author(s) only and do not necessarily reflect those of the European Union or European
Innovation Council and SMEs Executive Agency (EISMEA). Neither the European Union nor 
the granting authority can be held responsible for them.

![EIC Logo](doc/assets/EIC-logo-FundedBy.png)


## Disclaimer on AI Usage                                                                                                              
The initial version (0.0.1) was written by Hannes Holey with contributions from Stefan Hiemer.
Later versions were developed with the assistance of AI coding tools (Claude Code / Sonnet 4.6),                                       
with all generated code reviewed and validated by the authors.