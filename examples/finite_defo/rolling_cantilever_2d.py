"""Cantilever with prescribed tip rotation — geometrically nonlinear (2D).

Instead of applying a tip *moment* (Neumann BC), the rotational DOF at the
free end is prescribed directly (Dirichlet BC) while the two translational
DOFs remain free.  A prescribed end rotation with free translational DOFs
corresponds, by energy minimisation, to uniform curvature along the beam —
the same shape produced by a pure tip moment.

Three cases are shown side by side:

* **θ = π   (180°)** — semicircle; tip at (0, 2L/π).
* **θ = 2π  (360°)** — full circle; tip returns to the clamped-end origin.
* **Comparison** — both loading modes (tip moment vs prescribed rotation)
  plotted on top of each other for θ = 2π, demonstrating that the two
  approaches give identical deformed shapes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from beam_networks.problem import ElasticNetwork
from beam_networks.geometry.geo import get_geometric_props


# ---------------------------------------------------------------------------
# Common geometry and material
# ---------------------------------------------------------------------------
ne = 10
Lx = 10.
beam_prop = {'b': .1, 'h': .1, 'E': 2.e11, 'nu': 0., 'name': 'rectangle'}
Iz = get_geometric_props(beam_prop)[1]

x = np.linspace(0., Lx, ne + 1)
nodes = np.column_stack([x, np.zeros_like(x)])
edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])

n_steps = 10
n_snapshots = 5
plot_every = max(n_steps // n_snapshots, 1)

cmap = plt.cm.coolwarm


def make_net():
    net = ElasticNetwork(nodes, edges, beam_prop=beam_prop,
                         options={'vectorize': False, 'matrix': 'bsr',
                                  'verbose': False},
                         assemble_on_init=False)
    net.add_BC('clamp', 'D', 'node', [0], [0., 0., 0.])
    return net


def draw_snapshots(ax, edges, snapshots):
    """Draw a sequence of (nodes_current, color) snapshots."""
    for nodes_cur, color in snapshots:
        segs = [(nodes_cur[e0], nodes_cur[e1]) for e0, e1 in edges]
        ax.add_collection(LineCollection(segs, linewidths=1.5, colors=color))
        ax.scatter(*nodes_cur[[-1]].T, s=15, color=color, zorder=3)


# ---------------------------------------------------------------------------
# Case 1: θ_tip = π  (nonlinear)
# ---------------------------------------------------------------------------
net_pi = make_net()
net_pi.add_BC('tip_rot', 'D', 'node', [ne], [None, None, np.pi])

snaps_pi = []


def cb_pi(step, nodes_cur, _):
    if step % plot_every == 0 or step == n_steps:
        snaps_pi.append((nodes_cur.copy(), cmap(step / n_steps / 2.)))


cb_pi(0, net_pi.nodes, None)

net_pi.solve_nonlinear(n_steps=n_steps,
                       tol=1e-9,
                       verbose=True,
                       callback=cb_pi)

tip_pi = net_pi.displaced_nodes[-1]
print(f"θ = π  → tip = ({tip_pi[0]:.4f}, {tip_pi[1]:.4f}),  "
      f"expected (0, {2*Lx/np.pi:.4f})")

# ---------------------------------------------------------------------------
# Case 1a: θ_tip = π  (linear)
# ---------------------------------------------------------------------------
net_pi_lin = make_net()
net_pi_lin.add_BC('tip_rot', 'D', 'node', [ne], [None, None, np.pi])

snaps_pi_lin = []

net_pi_lin.solve()
snaps_pi_lin.append((net_pi_lin.displaced_nodes, '0.7'))

# ---------------------------------------------------------------------------
# Case 2: θ_tip = 2π  (full circle, prescribed rotation)
# ---------------------------------------------------------------------------
net_2pi = make_net()
net_2pi.add_BC('tip_rot', 'D', 'node', [ne], [None, None, 2. * np.pi])

snaps_2pi = []


def cb_2pi(step, nodes_cur, _):
    if step % plot_every == 0 or step == n_steps:
        snaps_2pi.append((nodes_cur.copy(), cmap(step / n_steps)))


cb_2pi(0, net_2pi.nodes, None)

net_2pi.solve_nonlinear(n_steps=n_steps,
                        tol=1e-9,
                        verbose=True,
                        callback=cb_2pi)
tip_2pi = net_2pi.displaced_nodes[-1]

print(f"θ = 2π → tip = ({tip_2pi[0]:.4e}, {tip_2pi[1]:.4e}),  "
      f"expected (0, 0)")

# ---------------------------------------------------------------------------
# Case 3: tip moment M = 2π EI/L  (Neumann, same circle for comparison)
# ---------------------------------------------------------------------------
Mref = 2. * np.pi * beam_prop['E'] * Iz / Lx
net_mom = make_net()
net_mom.add_BC('load', 'N', 'node', [ne], [None, None, Mref])

snaps_mom = []


def cb_mom(step, nodes_cur, _):
    if step % plot_every == 0 or step == n_steps:
        snaps_mom.append((nodes_cur.copy(), cmap(step / n_steps)))


net_mom.solve_nonlinear(n_steps=n_steps,
                        tol=1e-9,
                        verbose=True,
                        callback=cb_mom)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

fig, axes = plt.subplots(1, 3, figsize=(11, 5))

# --- left: semicircle ---
ax = axes[0]

# Nonlinear
draw_snapshots(ax, edges, snaps_pi)

# Linear
draw_snapshots(ax, edges, snaps_pi_lin)


# Semicircle
theta_ref = np.linspace(0, np.pi, 300)
R = Lx / np.pi
ax.plot(R * np.sin(theta_ref), R * (1 - np.cos(theta_ref)),
        'k--', lw=0.8, label='Exact arc')

ax.plot([], [], '-', lw=1.5, color='0.7', label='Linear')

ax.set_aspect('equal')
ax.set_xlim(-Lx/4, 1.1*Lx)
ax.set_title(r'Prescribed tip rotation $\theta = \pi$  (semicircle)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1.))
fig.colorbar(sm,
             ax=axes[:2],
             shrink=.33,
             label=r'$\theta / 2 \pi$',
             orientation='horizontal')

# --- centre: full circle (prescribed rotation) ---
ax = axes[1]
draw_snapshots(ax, edges, snaps_2pi)
theta_ref = np.linspace(0, 2 * np.pi, 300)
R2 = Lx / (2 * np.pi)
ax.plot(R2 * np.sin(theta_ref), R2 * (1 - np.cos(theta_ref)),
        'k--', lw=0.8, label='Exact arc')
ax.set_aspect('equal')
ax.set_xlim(*axes[0].get_xlim())
ax.set_ylim(*axes[0].get_ylim())
ax.set_title(r'Prescribed tip rotation $\theta = 2\pi$  (full circle)')
ax.set_xlabel('x')
ax.legend(fontsize=8)

# --- right: prescribed rotation vs tip moment (final shape only) ---
ax = axes[2]
# tip-moment snapshots in red
segs_mom = [(snaps_mom[-1][0][e0], snaps_mom[-1][0][e1]) for e0, e1 in edges]
ax.add_collection(LineCollection(segs_mom,
                                 linewidths=2.5,
                                 colors=cmap(1.),
                                 label='Tip moment',
                                 alpha=0.7))

# prescribed-rotation snapshots in black dashed
segs_rot = [(snaps_2pi[-1][0][e0], snaps_2pi[-1][0][e1]) for e0, e1 in edges]
ax.add_collection(LineCollection(segs_rot,
                                 linewidths=1.5,
                                 colors='0.0',
                                 linestyles='dashed',
                                 label='Prescribed rotation'))
ax.set_aspect('equal')
ax.autoscale_view()
ax.set_title(r'Tip moment vs prescribed rotation  ($\theta = 2\pi$)')
ax.set_xlabel('x')
ax.legend(fontsize=8)

fig.suptitle('Cantilever with prescribed tip rotation/moment',
             fontsize=13,
             fontweight='bold')


plt.show()
