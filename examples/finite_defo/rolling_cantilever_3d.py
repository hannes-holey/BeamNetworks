"""Rolling cantilever — geometrically nonlinear (3D).

Three cases are shown:

* **In-plane (x-y)**  — tip moment Mz = 2π EI/L bends the beam in the x-y
  plane; the tip completes a full circle and returns to the origin.

* **Out-of-plane (x-z)** — tip moment My = 2π EI/L bends the beam in the
  x-z plane; same full-circle geometry in the orthogonal plane.

* **3-D overlay** — both deformed shapes plotted together in a single
  perspective view to illustrate that the 3-D solver handles arbitrary
  bending planes.

Reference: Crisfield M.A. (1990). A consistent co-rotational formulation for
non-linear, three-dimensional, beam-elements. *CMAME*, 81(2), 131–150.
"""
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
Iy = get_geometric_props(beam_prop)[0]

x = np.linspace(0., Lx, ne + 1)
nodes = np.column_stack([x, np.zeros(ne + 1), np.zeros(ne + 1)])
edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])

n_steps = 20
n_snapshots = 6
plot_every = max(n_steps // n_snapshots, 1)

cmap = plt.cm.coolwarm


def make_net(matrix='bsr'):
    net = ElasticNetwork(nodes, edges, beam_prop=beam_prop,
                         options={'vectorize': False, 'matrix': matrix,
                                  'verbose': False},
                         assemble_on_init=False)
    net.add_BC('clamp', 'D', 'node', [0], [0., 0., 0., 0., 0., 0.])
    return net


# ---------------------------------------------------------------------------
# Case 1: tip moment Mz in the x-y plane
# ref = [0, 0, 1]: e2 = z, always ⊥ to any chord in the x-y plane
# ---------------------------------------------------------------------------
Mref_z = 2. * np.pi * beam_prop['E'] * Iz / Lx
net_xy = make_net()
net_xy.add_BC('load', 'N', 'node', [ne], [None, None, None, None, None, Mref_z])

ref_xy = np.tile([0., 0., 1.], (ne, 1))   # e2 = z for x-y bending
snaps_xy = []


def cb_xy(step, nodes_cur, _):
    if step % plot_every == 0 or step == n_steps:
        snaps_xy.append((nodes_cur.copy(), cmap(step / n_steps)))


cb_xy(0, net_xy.nodes, None)

net_xy.solve_nonlinear(n_steps=n_steps,
                       tol=1e-9,
                       verbose=True,
                       callback=cb_xy,
                       ref_vectors=None)

tip_xy = net_xy.displaced_nodes[-1]
print(f"x-y circle → tip = ({tip_xy[0]:.4e}, {tip_xy[1]:.4e}, {tip_xy[2]:.4e}), "
      f"expected (0, 0, 0)")

# ---------------------------------------------------------------------------
# Case 2: tip moment My in the x-z plane
# ref = [0, 1, 0]: e2 = y, always ⊥ to any chord in the x-z plane
# ---------------------------------------------------------------------------
Mref_y = 2. * np.pi * beam_prop['E'] * Iy / Lx
net_xz = make_net()
net_xz.add_BC('load', 'N', 'node', [ne], [None, None, None, None, Mref_y, None])

ref_xz = np.tile([0., 1., 0.], (ne, 1))   # e2 = y for x-z bending
snaps_xz = []


def cb_xz(step, nodes_cur, _):
    if step % plot_every == 0 or step == n_steps:
        snaps_xz.append((nodes_cur.copy(), cmap(step / n_steps)))


cb_xz(0, net_xz.nodes, None)

net_xz.solve_nonlinear(n_steps=n_steps,
                       tol=1e-9,
                       verbose=True,
                       callback=cb_xz,
                       ref_vectors=None)

tip_xz = net_xz.displaced_nodes[-1]
print(f"x-z circle → tip = ({tip_xz[0]:.4e}, {tip_xz[1]:.4e}, {tip_xz[2]:.4e}), "
      f"expected (0, 0, 0)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

fig = plt.figure(figsize=(14, 5))
ax_xy = fig.add_subplot(1, 3, 1)
ax_xz = fig.add_subplot(1, 3, 2)
ax_3d = fig.add_subplot(1, 3, 3, projection='3d')


def draw_2d(ax, snaps, i_horiz, i_vert):
    """Draw snapshots projected onto two coordinate axes."""
    for nodes_cur, color in snaps:
        segs = [(nodes_cur[e0, [i_horiz, i_vert]],
                 nodes_cur[e1, [i_horiz, i_vert]])
                for e0, e1 in edges]
        ax.add_collection(LineCollection(segs, linewidths=1.5, colors=color))
        ax.scatter(*nodes_cur[[-1], [i_horiz, i_vert]].T,
                   s=15, color=color, zorder=3)


def draw_3d(ax3, snaps, color_override=None):
    """Draw snapshots in 3D perspective (last snapshot only for clarity)."""
    nodes_cur, color = snaps[-1]
    c = color_override if color_override is not None else color
    segs = [(nodes_cur[e0], nodes_cur[e1]) for e0, e1 in edges]
    ax3.add_collection(Line3DCollection(segs, linewidths=1.5, colors=c))
    ax3.scatter(*nodes_cur[[-1]].T, s=20, color=c, zorder=3)


# --- x-y panel ---
draw_2d(ax_xy, snaps_xy, i_horiz=0, i_vert=1)
theta_ref = np.linspace(0, 2 * np.pi, 400)
R = Lx / (2 * np.pi)
ax_xy.plot(R * np.sin(theta_ref), R * (1 - np.cos(theta_ref)),
           'k--', lw=0.8, label='Exact arc')
ax_xy.set_aspect('equal')
ax_xy.autoscale_view()
ax_xy.set_title(r'In-plane bending  ($M_z$,  x-y plane)')
ax_xy.set_xlabel('x')
ax_xy.set_ylabel('y')
ax_xy.legend(fontsize=8)

# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1.))
# fig.colorbar(sm, ax=ax_xy, shrink=0.9, label=r'$M / M_\mathrm{ref}$')

# --- x-z panel ---
draw_2d(ax_xz, snaps_xz, i_horiz=0, i_vert=2)
ax_xz.plot(R * np.sin(theta_ref), -R * (1 - np.cos(theta_ref)),
           'k--', lw=0.8, label='Exact arc')
ax_xz.set_aspect('equal')
ax_xz.autoscale_view()
ax_xz.set_title(r'Out-of-plane bending  ($M_y$,  x-z plane)')
ax_xz.set_xlabel('x')
ax_xz.set_ylabel('z')
ax_xz.legend(fontsize=8)

# fig.colorbar(sm, ax=ax_xz, shrink=0.9, label=r'$M / M_\mathrm{ref}$')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1.))
fig.colorbar(sm,
             ax=[ax_xy, ax_xz],
             shrink=.33,
             label=r'$M / M_\mathrm{ref}$',
             orientation='horizontal')


# --- 3D overlay panel ---
draw_3d(ax_3d, snaps_xy, color_override=cmap(1.0))
draw_3d(ax_3d, snaps_xz, color_override=cmap(0.0))

# Exact circles for reference
ax_3d.plot(R * np.sin(theta_ref), R * (1 - np.cos(theta_ref)),
           np.zeros_like(theta_ref), 'k--', lw=0.6)
ax_3d.plot(R * np.sin(theta_ref), np.zeros_like(theta_ref),
           -R * (1 - np.cos(theta_ref)), 'k--', lw=0.6)

ax_3d.set_xlabel('x')
ax_3d.set_ylabel('y')
ax_3d.set_zlabel('z')
ax_3d.set_title('3-D overlay  (final shapes)')
# Equal-ish aspect
r_max = Lx / np.pi * 1.1
ax_3d.set_xlim(-r_max / 4, Lx * 1.1)
ax_3d.set_ylim(-r_max / 2, r_max * 1.1)  # y: x-y circle goes to +y
ax_3d.set_zlim(-r_max * 1.1, r_max / 2)  # z: x-z circle goes to -z

# Legend patches
legend_elements = [
    Line2D([0], [0], color=cmap(1.0), lw=2, label=r'$M_z$ (x-y)'),
    Line2D([0], [0], color=cmap(0.0), lw=2, label=r'$M_y$ (x-z)'),
    Line2D([0], [0], color='k', lw=0.8, ls='--', label='Exact arc'),
]
ax_3d.legend(handles=legend_elements, fontsize=8, loc='upper right')

fig.suptitle('3-D rolling cantilever  (co-rotational, Crisfield 1990)',
             fontsize=13, fontweight='bold')

plt.show()
