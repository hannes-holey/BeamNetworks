import numpy as np
import time

from beam_networks.network import Network
from beam_networks.problem import BeamNetwork


def run(s, v, N=1):

    # Example usage
    E = 1.
    nu = 0.3
    R = 0.05
    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}
    lt = 'fcc'

    mean_time = 0.

    c = 0
    for _ in range(N):
        lattice = Network.generate_cubic_lattice(a=1., bbox=(s, s, s), lattice_type=lt)

        problem = BeamNetwork(lattice._nodes,
                              lattice._edges,
                              beam_prop=props,
                              valid=True,
                              options={'vectorize': True, 'matrix': 'bsr', 'verbose': False})

        problem.add_BC('0', 'D', 'box',
                       [None, 0.01, None, None, None, None],
                       [0., 0., 0., 0., 0., 0.])

        problem.add_BC('1', 'D', 'point',
                       [s, s / 2., s / 2.],
                       [None, 0.01 * s, None, None, None, None])

        tic = time.time()
        problem.solve(solver=v)
        toc = time.time() - tic

        num_dofs = problem.num_dof

        print(problem.num_dof, problem.has_solution, toc)

        if problem.has_solution:
            mean_time += toc
            c += 1

    if c > 0.:
        mean_time /= c

    return num_dofs, mean_time


if __name__ == "__main__":

    for vv in [
            'direct',
            'pcg',
            'cg'
    ]:
        sizes = 2 * np.logspace(0, 2, 20)[:10]

        buffer = []
        for s in sizes:
            n, t = run(s, vv)
            buffer.append([n, t])

        np.savetxt(f'time_{vv}.txt', np.array(buffer))
