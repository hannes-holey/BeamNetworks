import numpy as np
import time
from beam_networks.network import Network
from beam_networks.problem import ElasticNetwork


PROPS = {'name': 'circle', 'radius': 0.05, 'E': 2.1e11, 'nu': 0.3}


def run(s, matrix, vectorize):
    lattice = Network.generate_cubic_lattice(a=1., bbox=(s, s, s), lattice_type='bcc')
    tic = time.time()
    problem = ElasticNetwork(lattice._nodes, lattice._edges,
                          beam_prop=PROPS, valid=True,
                          options={'vectorize': bool(vectorize),
                                   'matrix': matrix,
                                   'verbose': True})
    return problem.num_dof, time.time() - tic


if __name__ == "__main__":

    for matrix in ['bsr', 'lil', 'dense']:
        for vec in [0, 1]:
            # sizes = (2 * np.logspace(0, 2, 20)[:12] if matrix == 'bsr'
            #          else 2 * np.logspace(0, 2, 20)[:8])
            sizes = 2 * np.logspace(0, 2, 20)[:8]

            buffer = []
            for s in sizes:
                n, t = run(s, matrix, vec)
                buffer.append([n, t])
            np.savetxt(f'time_{matrix}-{vec}.txt', np.array(buffer))
