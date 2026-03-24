# %% [markdown]
# # Rolling Cantilever (2D): Prescribed Tip Rotation
#
# A clamped–free beam is subjected to a prescribed rotation at its free end.
# This is the classic *rolling cantilever* benchmark: a sufficiently flexible
# beam bends into an exact circular arc when driven by either a pure tip moment
# (Neumann BC) or a prescribed tip rotation (Dirichlet BC).
#
# Prescribing the rotation $\theta$ at the tip while leaving the translational
# DOFs free corresponds, by energy minimisation, to uniform curvature — the same
# shape produced by a pure tip moment. Two load levels are studied:
#
# - $\theta = \pi$: the beam forms a **semicircle**; tip reaches $(0,\;2L/\pi)$.
# - $\theta = 2\pi$: the beam forms a **full circle**; tip returns to the origin.
#
# The full-circle case driven by a tip moment is added as a third case for
# direct comparison with the prescribed-rotation result.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from beam_networks.problem import ElasticNetwork
from beam_networks.geometry.geo import get_geometric_props

# %% [markdown]
# ## Geometry and material
#
# A straight cantilever of length $L = 10$ is discretised with `ne = 10`
# Timoshenko beam elements. The cross-section is a $0.1\times0.1$ square of
# steel ($E = 2\times10^{11}$ Pa, $\nu = 0$).

# %%
ne = 10
Lx = 10.
beam_prop = {'b': 0.1, 'h': 0.1, 'E': 2.e11, 'nu': 0., 'name': 'rectangle'}
Iz = get_geometric_props(beam_prop)[1]

x = np.linspace(0., Lx, ne + 1)
nodes = np.column_stack([x, np.zeros_like(x)])
edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])

# %% [markdown]
# ## Solver setup
#
# A small factory function creates a fresh network and clamps the root node
# (node 0) in all three 2D DOFs ($u_x$, $u_y$, $\theta_z$).
# The load is applied incrementally over `n_steps` load steps; snapshots of the
# deformed shape are collected via a callback for later plotting.

# %%
n_steps = 10
n_snapshots = 5
plot_every = max(n_steps // n_snapshots, 1)
cmap = plt.cm.coolwarm

try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass


def make_net():
    net = ElasticNetwork(nodes, edges, beam_prop=beam_prop,
                         options={'vectorize': False, 'matrix': 'bsr',
                                  'verbose': False},
                         assemble_on_init=False)
    net.add_BC('clamp', 'D', 'node', [0], [0., 0., 0.])
    return net


def draw_snapshots(ax, edges, snapshots):
    for nodes_cur, color in snapshots:
        segs = [(nodes_cur[e0], nodes_cur[e1]) for e0, e1 in edges]
        ax.add_collection(LineCollection(segs, linewidths=1.5, colors=color))
        ax.scatter(*nodes_cur[[-1]].T, s=15, color=color, zorder=3)

# %% [markdown]
# ## Case 1 — Semicircle ($\theta = \pi$)
#
# The rotational DOF at the free tip (node `ne`) is prescribed to $\pi$; both
# translational DOFs are left free (`None` entries in the BC vector). The
# nonlinear solver ramps the rotation from 0 to $\pi$ over `n_steps` steps.
#
# For comparison the linear small-deformation solution is also computed.
# It substantially over-predicts the tip displacement because it ignores
# chord-rotation stiffening.


# %%
net_pi = make_net()
net_pi.add_BC('tip_rot', 'D', 'node', [ne], [None, None, np.pi])

snaps_pi = []


def cb_pi(step, nodes_cur, _):
    if step % plot_every == 0 or step == n_steps:
        snaps_pi.append((nodes_cur.copy(), cmap(step / n_steps / 2.)))


cb_pi(0, net_pi.nodes, None)   # store straight initial configuration
net_pi.solve_nonlinear(n_steps=n_steps, tol=1e-9, verbose=False, callback=cb_pi)

net_pi_lin = make_net()
net_pi_lin.add_BC('tip_rot', 'D', 'node', [ne], [None, None, np.pi])
net_pi_lin.solve()
snaps_pi_lin = [(net_pi_lin.displaced_nodes, '0.7')]

tip_pi = net_pi.displaced_nodes[-1]
print(f"θ = π  → tip = ({tip_pi[0]:.4f}, {tip_pi[1]:.4f})")
print(f"         expected (0.0000, {2*Lx/np.pi:.4f})")

# %%
fig, ax = plt.subplots(figsize=(5, 5))

draw_snapshots(ax, edges, snaps_pi)
draw_snapshots(ax, edges, snaps_pi_lin)
theta_ref = np.linspace(0, np.pi, 300)
R = Lx / np.pi
ax.plot(R * np.sin(theta_ref), R * (1 - np.cos(theta_ref)),
        'k--', lw=0.8, label='Exact arc')
ax.plot([], [], '-', lw=1.5, color='0.7', label='Linear')
ax.set_aspect('equal')
ax.set_xlim(-Lx / 4, 1.1 * Lx)
ax.set_title(r'Prescribed $\theta = \pi$  (semicircle)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 0.5))
fig.colorbar(sm, ax=ax, shrink=0.5, label=r'$\theta / 2\pi$')
plt.show()

# %% [markdown]
# ## Case 2 — Full circle ($\theta = 2\pi$)
#
# Prescribing a full $2\pi$ rotation drives the beam into a closed circle so the
# free tip must return to the clamped-end origin $(0, 0)$.

# %%
net_2pi = make_net()
net_2pi.add_BC('tip_rot', 'D', 'node', [ne], [None, None, 2. * np.pi])

snaps_2pi = []


def cb_2pi(step, nodes_cur, _):
    if step % plot_every == 0 or step == n_steps:
        snaps_2pi.append((nodes_cur.copy(), cmap(step / n_steps)))


cb_2pi(0, net_2pi.nodes, None)
net_2pi.solve_nonlinear(n_steps=n_steps, tol=1e-9, verbose=False, callback=cb_2pi)

tip_2pi = net_2pi.displaced_nodes[-1]
print(f"θ = 2π → tip = ({tip_2pi[0]:.4e}, {tip_2pi[1]:.4e})")
print("          expected (0, 0)")

# %%
fig, ax = plt.subplots(figsize=(5, 5))

draw_snapshots(ax, edges, snaps_2pi)
theta_ref = np.linspace(0, 2 * np.pi, 300)
R2 = Lx / (2 * np.pi)
ax.plot(R2 * np.sin(theta_ref), R2 * (1 - np.cos(theta_ref)),
        'k--', lw=0.8, label='Exact arc')
ax.set_aspect('equal')
ax.set_xlim(-Lx / 4, 1.1 * Lx)
ax.set_title(r'Prescribed $\theta = 2\pi$  (full circle)')
ax.set_xlabel('x')
ax.legend(fontsize=8)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1.))
fig.colorbar(sm, ax=ax, shrink=0.5, label=r'$\theta / 2\pi$')
plt.show()

# %% [markdown]
# ## Case 3 — Tip moment (Neumann BC)
#
# The reference moment that closes a beam of length $L$ into a circle is
# $M = 2\pi EI/L$. Applying this as a Neumann BC should produce the same
# deformed shape as the prescribed-rotation case above.

# %%
Mref = 2. * np.pi * beam_prop['E'] * Iz / Lx
net_mom = make_net()
net_mom.add_BC('load', 'N', 'node', [ne], [None, None, Mref])

snaps_mom = []


def cb_mom(step, nodes_cur, _):
    if step % plot_every == 0 or step == n_steps:
        snaps_mom.append((nodes_cur.copy(), cmap(step / n_steps)))


net_mom.solve_nonlinear(n_steps=n_steps, tol=1e-9, verbose=False, callback=cb_mom)

# %%
fig, ax = plt.subplots(figsize=(5, 5))

segs_mom = [(snaps_mom[-1][0][e0], snaps_mom[-1][0][e1]) for e0, e1 in edges]
ax.add_collection(LineCollection(segs_mom, linewidths=2.5, colors=cmap(1.),
                                 label='Tip moment', alpha=0.7))
segs_rot = [(snaps_2pi[-1][0][e0], snaps_2pi[-1][0][e1]) for e0, e1 in edges]
ax.add_collection(LineCollection(segs_rot, linewidths=1.5, colors='0.0',
                                 linestyles='dashed', label='Prescribed rotation'))
ax.set_aspect('equal')
ax.autoscale_view()
ax.set_title(r'Moment vs rotation  ($\theta = 2\pi$)')
ax.set_xlabel('x')
ax.legend(fontsize=8)
plt.show()
