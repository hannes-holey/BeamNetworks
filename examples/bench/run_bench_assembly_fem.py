"""Benchmark FEM assembly time vs. network size.

For each combination of poly_order and n_elem_per_length the script assembles
a BCC lattice of increasing size (BSR format, no vectorization) and records
the wall time.  Results are written to::

    time_fem-p{poly_order}-n{n_elem_per_length}.txt

with two columns: [num_edges, time_s].

Run with::

    cd examples/bench
    python run_bench_assembly_fem.py

then visualise with plot_timing_assembly.py.
"""

import numpy as np
import time
from beam_networks.network import Network
from beam_networks.problem import BeamNetwork


PROPS = {'name': 'circle', 'radius': 0.05, 'E': 2.1e11, 'nu': 0.3}

# BCC lattice with a=1: edge length ≈ 0.866.
# n_elem_per_length values chosen so that each edge gets O(1–10) sub-elements.
N_ELEM_PER_LENGTH = [1, 2, 5, 10]
POLY_ORDERS = [1, 2, 3]

# Fewer/smaller lattices than the exact bench because FEM is more expensive.
SIZES = 2 * np.logspace(0, 2, 20)[:8]


def run(s, poly_order, n_elem_per_length):
    lattice = Network.generate_cubic_lattice(a=1., bbox=(s, s, s), lattice_type='bcc')
    tic = time.time()
    problem = BeamNetwork(lattice._nodes, lattice._edges,
                          beam_prop=PROPS, valid=True,
                          options={'vectorize': False,
                                   'matrix': 'bsr',
                                   'verbose': True,
                                   'n_elem_per_length': n_elem_per_length,
                                   'fem_poly_order': poly_order,
                                   'fem_n_gauss': None})   # None → reduced integration

    return problem.num_dof, time.time() - tic


if __name__ == "__main__":

    for poly_order in POLY_ORDERS:
        for n_elem in N_ELEM_PER_LENGTH:
            buffer = []
            for s in SIZES:
                n, t = run(s, poly_order, n_elem)
                buffer.append([n, t])
            fname = f'time_fem-p{poly_order}-n{n_elem}.txt'
            np.savetxt(fname, np.array(buffer))
            print(f'Saved {fname}')
