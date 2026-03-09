"""Benchmark solve time and peak memory vs. network size for beam networks.

All solvers use symmetric Jacobi scaling internally.

Direct solvers
--------------
direct    -- sparse LU via spsolve
cholesky  -- sparse Cholesky via CHOLMOD  [requires scikit-sparse]

Iterative solvers (Jacobi-scaled)
----------------------------------
cg        -- unpreconditioned conjugate gradient
ilu       -- CG + incomplete LU
ssor      -- CG + SSOR (ω = 1)
amg       -- CG + smoothed-aggregation AMG  [requires pyamg]
amg_rs    -- CG + Ruge–Stüben AMG           [requires pyamg]

Output files
------------
time_solve_{solver}.txt   -- three columns: [num_dof, time_s, peak_mem_MiB]

Run with::

    cd examples/bench
    python run_bench_solve.py

then visualise with plot_timing_solve.py.
"""

import tracemalloc
import numpy as np
import time

from beam_networks import Network
from beam_networks import ElasticNetwork


PROPS = {'name': 'circle', 'radius': 0.05, 'E': 1., 'nu': 0.3}

SOLVERS_SLOW = ['direct', 'ilu']
SOLVERS_FAST = ['cg', 'ssor']
SOLVERS_OPTIONAL_DIRECT = ['cholesky']          # requires scikit-sparse
SOLVERS_OPTIONAL_ITERATIVE = ['amg', 'amg_rs']  # require pyamg

# FCC lattice; 5 sizes on a log scale
SIZES = 2 * np.logspace(0, 2, 20)[1:10:2]
SIZES_DIRECT = SIZES[:-1]
SIZES_ITERATIVE = SIZES[:]

N_REPEATS = 5


def build_problem(s):
    lattice = Network.generate_cubic_lattice(a=1., bbox=(s, s, s), lattice_type='fcc')
    problem = ElasticNetwork(
        lattice._nodes, lattice._edges,
        beam_prop=PROPS, valid=True,
        options={'vectorize': True, 'matrix': 'bsr', 'verbose': False})
    problem.add_BC('0', 'D', 'box',
                   [None, 0.01, None, None, None, None],
                   [0., 0., 0., 0., 0., 0.])
    problem.add_BC('1', 'D', 'point',
                   [s, s / 2., s / 2.],
                   [None, 0.01 * s, None, None, None, None])

    problem._assemble_global_system()
    problem.assemble_BCs()

    return problem


def run(s, solver):
    problem = build_problem(s)

    count = 0
    wall_time = 0.
    peak_mem_mib = 0.

    for _ in range(N_REPEATS):
        tracemalloc.start()
        tic = time.time()
        problem.solve(solver=solver)
        toc = time.time() - tic
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if problem.has_solution:
            wall_time += toc
            peak_mem_mib += peak / 1024**2
            count += 1
            print('.', end='', flush=True)

    wall_time /= count
    peak_mem_mib /= count

    print(f'solver={solver} dof={problem.num_dof} '
          f'ok={problem.has_solution} t={wall_time:.4f}s '
          f'mem={peak_mem_mib:.1f} MiB')

    return problem.num_dof, wall_time, peak_mem_mib


def bench(solvers, sizes):
    for solver in solvers:
        print(f'\n=== solver={solver} ===')
        buffer = []
        for s in sizes:
            n, t, m = run(s, solver)
            buffer.append([n, t, m])
        fname = f'time_solve_{solver}.txt'
        np.savetxt(fname, np.array(buffer))
        print(f'Saved {fname}')


if __name__ == '__main__':

    bench(SOLVERS_SLOW, SIZES_DIRECT)
    bench(SOLVERS_FAST, SIZES_ITERATIVE)

    try:
        import sksparse  # noqa: F401
        bench(SOLVERS_OPTIONAL_DIRECT, SIZES_ITERATIVE)
    except ImportError:
        print('\nSkipping cholesky benchmark (scikit-sparse not installed).')

    try:
        import pyamg  # noqa: F401
        bench(SOLVERS_OPTIONAL_ITERATIVE, SIZES_ITERATIVE)
    except ImportError:
        print('\nSkipping amg/amg_rs benchmark (pyamg not installed).')
