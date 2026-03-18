# %% [markdown]
# # Fracture of a Disordered Fibre Network
#
# This tutorial runs a fracture simulation on a jammed network using the
# `FractureProblem` class. Two Weibull strength distributions are compared
# (uniform and shape parameter $\beta = 4$), each solved with two fracture
# modes:
#
# - **cascade** — the most stressed element is removed and the system is
#   immediately re-solved; stresses can concentrate further.
# - **adiabatic** — all elements that exceed their local strength threshold
#   at the current load are removed simultaneously.
#
# **Material** — circular cross-section, $R = 0.05$ m, mean strength
# $\sigma_c = 0.1 E$.
#
# The stress–strain curve is plotted for all four combinations.

# %% Imports
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from beam_networks import FractureProblem

# %% [markdown]
# ## Setup

# %%
props = {'name': 'circle', 'radius': 0.05, 'E': 2.1e11, 'nu': 0.3}
mean = 0.1
props['strength'] = mean * props['E']

nodes_positions = np.loadtxt('../resources/jammed.nodes')
edges_indices   = np.loadtxt('../resources/jammed.edges').astype(int)
size = edges_indices.shape[0]

sign = -1   # compression

betas = [None, 4.]
modes = ['cascade', 'adiabatic']

# %% [markdown]
# ## Run fracture simulations

# %%
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

fig, ax = plt.subplots(1)

orig_dir = os.getcwd()

for i, beta in enumerate(betas):
    dist = np.random.weibull(a=beta, size=size) if beta is not None else None

    for j, mode in enumerate(modes):
        problem = FractureProblem(nodes_positions, edges_indices,
                                  save_trajectory=False,
                                  beam_prop=props,
                                  valid=False)

        problem.add_BC('0', 'D', 'box', [None, None, None, .05], [0., 0., 0.])
        problem.add_BC('1', 'D', 'box', [None, None, 0.95, None], [None, 0., None])

        # run() always writes VTK files to cwd; redirect to a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                problem.run(mode=mode, sign=sign, dist=dist, solver='direct')
            finally:
                os.chdir(orig_dir)

        buffer = np.array(problem._output['stress_strain'][0])
        label = (f'$\\beta = {beta:.0f}$, {mode}' if beta is not None
                 else f'uniform, {mode}')
        ax.plot(*buffer.T, ls='-' if mode == 'cascade' else '--',
                color=f'C{i}', label=label)

ax.set_xlabel('Strain')
ax.set_ylabel('Stress')
ax.legend(fontsize=8)
plt.show()
