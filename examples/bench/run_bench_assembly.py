import numpy as np
import time
from beam_networks.network import Network
from beam_networks.problem import BeamNetwork


def run(s, v, vec, N=5):

    # Example usage
    E = 2.1e11
    nu = 0.3
    R = 0.05
    props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}
    lt = 'bcc'

    mean_time = 0.

    for _ in range(N):
        lattice = Network.generate_cubic_lattice(a=1., bbox=(s, s, s), lattice_type=lt)
        tic = time.time()

        problem = BeamNetwork(lattice._nodes,
                              lattice._edges,
                              beam_prop=props,
                              valid=True,
                              options={'vectorize': bool(vec), 'matrix': v, 'verbose': True})

        toc = time.time() - tic
        mean_time += toc

    mean_time /= N

    return problem.num_edges, mean_time


if __name__ == "__main__":

    for vv in ['bsr',
               'lil',
               'dense'
               ]:
        for vec in [0, 1]:
            if vv == 'bsr':
                sizes = 2 * np.logspace(0, 2, 20)[:12]
            else:
                sizes = 2 * np.logspace(0, 2, 20)[:8]

            buffer = []
            for s in sizes:
                n, t = run(s, vv, vec)
                buffer.append([n, t])

            np.savetxt(f'time_{vv}-{vec}.txt', np.array(buffer))
