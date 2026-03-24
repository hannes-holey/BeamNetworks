# %% [markdown]
# # FEM Convergence Studies
#
# This tutorial demonstrates $h$- and $p$-refinement convergence for a 2D
# Timoshenko cantilever. The error is measured as the absolute difference
# between the numerical and analytic tip deflection / rotation.
#
# **$h$-refinement** — polynomial order $p = 1$ is fixed; the number of
# sub-elements per element is increased from 1 to 100.
#
# **$p$-refinement** — a single element is used ($n_\mathrm{sub} = 1$);
# the polynomial order is swept from 1 to 3.
#
# For both studies, reduced integration (Gauss points = $p$) and exact
# integration (Gauss points = $p+1$) are compared.
#
# **Material** — circular cross-section, $R = 0.05$ m, $E = 1$ Pa, $\nu = 0.3$,
# beam length $L = 50$.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from beam_networks import ElasticNetwork
from beam_networks.geometry.geo import get_geometric_props
from beam_networks.reference_solutions.cantilever import cantilever_analytic

# %% [markdown]
# ## Setup

# %%
PROPS = {'name': 'circle', 'radius': 0.05, 'E': 1., 'nu': 0.3}


def _cantilever_2d(num_nodes, length, options):
    x = np.linspace(0., length, num_nodes)
    nodes = np.column_stack([x, np.zeros(num_nodes)])
    edges = np.column_stack([np.arange(num_nodes - 1), np.arange(1, num_nodes)])
    return ElasticNetwork(nodes, edges, beam_prop=PROPS, valid=True, options=options)


def _disp_fem(length, num_nodes, density, fem_poly_order, fem_n_gauss):
    _, Iz, _, A, kappa, _ = get_geometric_props(PROPS)
    P = 0.05 * PROPS['E'] * Iz / length**3

    opts = {'vectorize': True, 'matrix': 'dense', 'verbose': False,
            'n_elem_per_length': density,
            'fem_poly_order': fem_poly_order,
            'fem_n_gauss': fem_n_gauss}

    net = _cantilever_2d(num_nodes, length, opts)
    net.add_BC('fix',  'D', 'point', [0.,     0.], [0., 0., 0.])
    net.add_BC('load', 'N', 'point', [length, 0.], [0., P,  0.])
    net.solve()
    return net.nodes[:, 0], net.displacement, net.rotation

# %% [markdown]
# ## $p$-refinement


# %%
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

length = 50.
_, Iz, _, _, _, _ = get_geometric_props(PROPS)
P = 0.05 * PROPS['E'] * Iz / length**3
x = np.linspace(0., length, 100)
_, uy_ref, phi_ref, _ = cantilever_analytic(x, length, [0., P, 0.], 1.0, PROPS)

fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex='col')

ax[0, 0].plot(x, uy_ref,  '--', color='0.0')
ax[1, 0].plot(x, phi_ref, '--', color='0.0')

num_nodes = 2
n_sub = 1

for i, red in enumerate([True, False]):
    errors_u, errors_phi = [], []
    density = n_sub / length
    ps = [1, 2, 3]
    for p in ps:
        n = p if red else p + 1
        x_fem, u_fem, phi_fem = _disp_fem(length, num_nodes, density, p, n)
        uy_fem = u_fem[:, 1]
        if red:
            ax[0, 0].plot(x_fem, uy_fem)
            ax[1, 0].plot(x_fem, phi_fem)
        errors_u.append(np.abs(uy_ref[-1] - uy_fem[-1]))
        errors_phi.append(np.abs(phi_ref[-1] - phi_fem[-1]))
    ax[0, i + 1].plot(ps, errors_u)
    ax[1, i + 1].plot(ps, errors_phi)

fig.suptitle('$p$-refinement')
ax[0, 0].set_title('Cantilever')
ax[0, 0].set_ylabel(r'$u$')
ax[1, 0].set_ylabel(r'$\theta$')
ax[1, 0].set_xlabel(r'$x$')
ax[0, 1].set_title('Reduced integration ($p$)')
ax[0, 1].set_ylabel(r'$|\Delta u|$')
ax[0, 1].set_yscale('log')
ax[1, 1].set_ylabel(r'$|\Delta \theta|$')
ax[1, 1].set_yscale('log')
ax[1, 1].set_xlabel('p')
ax[0, 2].set_title('Exact integration ($p+1$)')
ax[0, 2].set_ylabel(r'$|\Delta u|$')
ax[0, 2].set_yscale('log')
ax[1, 2].set_ylabel(r'$|\Delta \theta|$')
ax[1, 2].set_yscale('log')
ax[1, 2].set_xlabel('p')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## $h$-refinement

# %%
fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex='col')

ax[0, 0].plot(x, uy_ref,  '--', color='0.0')
ax[1, 0].plot(x, phi_ref, '--', color='0.0')

num_nodes = 3
p = 1

for i, red in enumerate([True, False]):
    n = p if red else p + 1
    errors_u, errors_phi = [], []
    n_subs = [1, 5, 10, 100]
    for n_sub in n_subs:
        density = n_sub / length
        x_fem, u_fem, phi_fem = _disp_fem(length, num_nodes, density, p, n)
        uy_fem = u_fem[:, 1]
        if red:
            ax[0, 0].plot(x_fem, uy_fem, '-x')
            ax[1, 0].plot(x_fem, phi_fem, '-x')
        errors_u.append(np.abs(uy_ref[-1] - uy_fem[-1]))
        errors_phi.append(np.abs(phi_ref[-1] - phi_fem[-1]))
    ax[0, i + 1].plot(n_subs, errors_u)
    ax[1, i + 1].plot(n_subs, errors_phi)

fig.suptitle(f'$h$-refinement  ($p = {p}$)')
ax[0, 0].set_title('Cantilever')
ax[0, 0].set_ylabel(r'$u$')
ax[1, 0].set_ylabel(r'$\theta$')
ax[1, 0].set_xlabel(r'$x$')
ax[0, 1].set_title('Reduced integration ($p$)')
ax[0, 1].set_ylabel(r'$|\Delta u|$')
ax[0, 1].set_yscale('log')
ax[1, 1].set_ylabel(r'$|\Delta \theta|$')
ax[1, 1].set_yscale('log')
ax[1, 1].set_xlabel(r'$n_\mathrm{sub}$')
ax[0, 2].set_title('Exact integration ($p+1$)')
ax[0, 2].set_ylabel(r'$|\Delta u|$')
ax[0, 2].set_yscale('log')
ax[1, 2].set_ylabel(r'$|\Delta \theta|$')
ax[1, 2].set_yscale('log')
ax[1, 2].set_xlabel(r'$n_\mathrm{sub}$')

plt.tight_layout()
plt.show()
