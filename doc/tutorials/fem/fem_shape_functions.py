# %% [markdown]
# # FEM Shape Functions
#
# This tutorial visualises the shape functions used in the beam FEM
# implementation:
#
# - **Lagrange** ($p = 3$): standard polynomial interpolation of the axial DOFs.
# - **Hermite**: cubic $C^1$ basis used in the Euler–Bernoulli limit.
# - **Friedman–Kosmatka (FK)**: Timoshenko-consistent transverse/rotation shape
#   functions parameterised by the shear influence parameter
#   $\Phi = 12EI / (\kappa G A L^2)$.
#
# Dashed vertical lines mark the Gauss–Legendre quadrature points used for
# numerical integration.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from beam_networks.fem.basis import (
    _lagrange_basis, _hermite_basis, _timoshenko_basis_FK, _gauss_legendre,
)

# %% [markdown]
# ## Parameters

# %%
P = 3                       # Lagrange polynomial order
PHI_VALUES = [0., 1., 10.]  # FK shear-influence parameter values
PHI_LS = ['-', '--', ':']

# %% [markdown]
# ## Shape function plots

# %%
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

xi = np.linspace(-1., 1., 300)
xg, _ = _gauss_legendre(P)

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

# ── Lagrange ──────────────────────────────────────────────────────────────────
N_lag, dN_lag = [], []
for xi_i in xi:
    Ni, dNi = _lagrange_basis(P + 1, xi_i)
    N_lag.append(Ni); dN_lag.append(dNi)
N_lag  = np.array(N_lag).T
dN_lag = np.array(dN_lag).T

for i, (Ni, dNi) in enumerate(zip(N_lag, dN_lag)):
    axes[0, 0].plot(xi, Ni,  color=f'C{i}', label=f'$N_{i+1}$')
    axes[1, 0].plot(xi, dNi, color=f'C{i}')

axes[0, 0].set_title(f'Lagrange  ($p = {P}$)')
axes[0, 0].set_ylabel('$N_i(\\xi)$')
axes[1, 0].set_ylabel('$\\mathrm{d}N_i / \\mathrm{d}\\xi$')
axes[0, 0].legend(fontsize=7, ncol=2)

# ── Hermite ───────────────────────────────────────────────────────────────────
N_her, dN_her = [], []
for xi_i in xi:
    Ni, dNi = _hermite_basis(xi_i)
    N_her.append(Ni); dN_her.append(dNi)
N_her  = np.array(N_her).T
dN_her = np.array(dN_her).T

her_labels = ['$H_1$', '$H_2$', '$H_3$', '$H_4$']
for i, (Ni, dNi) in enumerate(zip(N_her, dN_her)):
    axes[0, 1].plot(xi, Ni,  color=f'C{i}', label=her_labels[i])
    axes[1, 1].plot(xi, dNi, color=f'C{i}')

axes[0, 1].set_title('Hermite')
axes[0, 1].set_ylabel('$H_i(\\xi)$')
axes[1, 1].set_ylabel('$\\mathrm{d}H_i / \\mathrm{d}\\xi$')
axes[0, 1].legend(fontsize=7, ncol=2)

# ── Friedman–Kosmatka ─────────────────────────────────────────────────────────
fk_labels = ['$N_{w,1}$', '$N_{w,2}$', '$N_{w,3}$', '$N_{w,4}$']
for Phi, ls in zip(PHI_VALUES, PHI_LS):
    Nw, Nt = _timoshenko_basis_FK(xi, L=1., Phi=Phi)
    for i in range(4):
        label_w = fk_labels[i] if ls == '-' else None
        label_p = f'$\\Phi = {Phi:.0f}$' if i == 0 else None
        axes[0, 2].plot(xi, Nw[i], ls, color=f'C{i}', label=label_w)
        axes[1, 2].plot(xi, Nt[i], ls, color=f'C{i}', label=label_p)

axes[0, 2].set_title('Friedman–Kosmatka  ($L = 1$)')
axes[0, 2].set_ylabel('$N_{w,i}(\\xi)$')
axes[1, 2].set_ylabel('$N_{\\theta,i}(\\xi)$')
axes[0, 2].legend(fontsize=7, ncol=2)
axes[1, 2].legend(fontsize=7)

# ── shared formatting ──────────────────────────────────────────────────────────
for ax in axes.flat:
    for xgi in xg:
        ax.axvline(xgi, ls='dashed', color='0.7', lw=0.8)
    ax.axhline(0., color='0.5', lw=0.5)
    ax.set_xlabel('$\\xi$')

plt.tight_layout()
plt.show()
