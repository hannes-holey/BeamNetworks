# %% [markdown]
# # Euler–Bernoulli vs Timoshenko Cantilever
#
# This tutorial compares the Euler–Bernoulli (EB) and Timoshenko beam theories
# for a cantilever loaded by a transverse tip force.
#
# Timoshenko theory includes transverse shear deformation, which matters for
# short, stocky beams. The shear contribution is quantified by the parameter
#
# $$\Phi = \frac{12 E I}{\kappa G A L^2}$$
#
# The ratio of tip deflections is $v_\mathrm{tip}^\mathrm{Timo} / v_\mathrm{tip}^\mathrm{EB} = 1 + \Phi/4$.
#
# **Material** — circular cross-section, $R = 0.05$ m, $E = 2.1\times10^{11}$ Pa, $\nu = 0.3$.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from beam_networks import ElasticNetwork
from beam_networks.geometry.geo import get_geometric_props
from beam_networks.reference_solutions.cantilever import cantilever_analytic

# %% [markdown]
# ## Setup

# %%
E = 2.1e11
nu = 0.3
R = 0.05
NUM_NODES = 51
PROPS = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}


def _solve_cantilever(length, euler_bernoulli=False):
    """Solve a horizontal end-loaded cantilever; return nodal solution and tip load."""
    _, Iz, _, _, _, _ = get_geometric_props(PROPS)
    P = 4. * E * Iz / length**3

    x = np.linspace(0., length, NUM_NODES)
    nodes = np.column_stack([x, np.zeros(NUM_NODES)])
    edges = np.column_stack([np.arange(NUM_NODES - 1), np.arange(1, NUM_NODES)])

    problem = ElasticNetwork(
        nodes, edges, beam_prop=PROPS, valid=True,
        options={'verbose': False, 'vectorize': True, 'matrix': 'bsr',
                 'euler_bernoulli': euler_bernoulli})

    problem.add_BC('fixed', 'D', 'node', [0], [0., 0., 0.])
    problem.add_BC('tip',   'N', 'node', [NUM_NODES - 1], [0., P, 0.])
    problem.solve(verbosity=0)
    return problem.sol, P


def _analytic(x, length, P, euler_bernoulli=False):
    _, uy, _, _ = cantilever_analytic(x, length, np.array([0., P, 0.]),
                                      a=1.0, beam_prop=PROPS,
                                      euler_bernoulli=euler_bernoulli)
    return uy

# %% [markdown]
# ## Left panel: deflection profile for a short beam ($L/R = 5$)
#
# Shear deformation is most visible at low slenderness. At $L/R = 5$ the
# Timoshenko tip deflection is noticeably larger than the EB prediction.

# %%
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

L_short = 5 * R

sol_timo, P = _solve_cantilever(L_short, euler_bernoulli=False)
sol_eb,   _ = _solve_cantilever(L_short, euler_bernoulli=True)

x_nodes = np.linspace(0., L_short, NUM_NODES)
x_ref = np.linspace(0., L_short, 300)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]

ax.plot(x_ref / L_short, _analytic(x_ref, L_short, P, euler_bernoulli=False),
        '-', color='C0', label='Timoshenko (analytic)')
ax.plot(x_ref / L_short, _analytic(x_ref, L_short, P, euler_bernoulli=True),
        '-', color='C1', label='Euler–Bernoulli (analytic)')
ax.plot(x_nodes / L_short, sol_timo[1::3],
        'o', color='C0', ms=3, label='Timoshenko (FEM)')
ax.plot(x_nodes / L_short, sol_eb[1::3],
        's', color='C1', ms=3, label='EB (FEM)')

ax.set_xlabel('$x / L$')
ax.set_ylabel('$v$ (m)')
ax.set_title(f'Deflection profile  ($L/R = {int(L_short / R)}$)')
ax.legend(fontsize=7)

# %% [markdown]
# ## Right panel: normalised tip deflection vs. slenderness
#
# Sweeping $L/R$ from 3 to 200 confirms the analytic formula $1 + \Phi/4$.
# The EB result ($\Phi = 0$) is recovered at high slenderness.

# %%
ax = axes[1]
slenderness = np.logspace(np.log10(3.), np.log10(200.), 20)

_, Iz, _, A, kappa, _ = get_geometric_props(PROPS)
G = E / (2. * (1. + nu))

ratio_num, ratio_ana = [], []
for slend in slenderness:
    L = slend * R
    v_timo = _solve_cantilever(L, euler_bernoulli=False)[0][1::3][-1]
    v_eb   = _solve_cantilever(L, euler_bernoulli=True)[0][1::3][-1]
    ratio_num.append(v_timo / v_eb)
    Phi = 12. * E * Iz / (kappa * G * A * L**2)
    ratio_ana.append(1. + Phi / 4.)

ax.semilogx(slenderness, ratio_ana, '-',  color='0.3', label='Analytic  $1 + \\Phi/4$')
ax.semilogx(slenderness, ratio_num, 'o',  color='C0', ms=4, label='FEM ratio')
ax.axhline(1., color='0.6', lw=0.8, ls='--')
ax.set_xlabel('$L / R$')
ax.set_ylabel(r'$v_\mathrm{tip}^\mathrm{Timo}\;/\;v_\mathrm{tip}^\mathrm{EB}$')
ax.set_title('Shear contribution vs. slenderness')
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
