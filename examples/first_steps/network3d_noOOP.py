import numpy as np

from beam_networks.network import Network
from beam_networks.assembly import assemble_global_system
from beam_networks.bc import _get_bc_dof, _assemble_BCs
from beam_networks.solve import solve


if __name__ == "__main__":

    # Example usage
    E = 2.1e11
    nu = 0.3
    R = 0.05

    # Generate fcc lattice
    lattice = Network.generate_cubic_lattice(a=1., bbox=(5., 2., 2.), lattice_type='fcc')

    # Vary radius by +/- 50%
    p = 0.5
    radius = np.ones(lattice.num_edges)

    # Collect beam properties
    props = {'name': 'circle', 'radius': radius, 'E': E, 'nu': nu}

    # Assemble stiffness matrix
    K = assemble_global_system(lattice.nodes,
                               lattice.edges,
                               lattice.edge_vectors,
                               props,
                               sorted_edges=False,
                               vectorize=True, verbose=True,
                               matrix="dense")
    # Define BCs
    bc = {}

    # Add BC 0 (Dirichlet)
    bc['0'] = {}
    bc['0'].update([('type', 'D')])
    bc['0'].update([('active', True)])
    bc['0'].update(_get_bc_dof(lattice.nodes, 'box', [None, 0.01, None, None, None, None], [0., 0., 0., 0., 0., 0.]))

    # Add BC 1 (Neumann)
    bc['1'] = {}
    bc['1'].update([('type', 'N')])
    bc['1'].update([('active', True)])
    bc['1'].update(_get_bc_dof(lattice.nodes, 'point', [5., 1.5, 1.5], [None, -1, None, None, None, None]))

    # Assemble BCs
    dof_D, val_D, dof_N, val_N = _assemble_BCs(bc)
    # Solve
    u, F, info = solve(K, dof_D, val_D, dof_N, val_N)
