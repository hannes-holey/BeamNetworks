# %% [markdown]
# # Rolling Cantilever (3D): In-Plane and Out-of-Plane Bending
#
# This tutorial extends the 2D rolling-cantilever benchmark to three dimensions.
# A straight cantilever is loaded with a tip moment that drives it through a full
# $2\pi$ rotation. Both in-plane (x-y) and out-of-plane (x-z) bending planes are
# exercised, and their final shapes are overlaid in a single 3D perspective view.
#
# ## Reference vector
#
# The 3D co-rotational formulation requires one **reference vector** per element
# to define the local $\hat{\mathbf{e}}_2$ axis.
# The reference vector must not be parallel to the element chord at any point
# during the deformation; otherwise the Gram–Schmidt projection degenerates.
#
# - **x-y bending**: chord stays in the x-y plane, so $\hat{z} = (0,0,1)$ is
#   always perpendicular → use `ref_vec = (0, 0, 1)`.
# - **x-z bending**: chord stays in the x-z plane, so $\hat{y} = (0,1,0)$ is
#   always perpendicular → use `ref_vec = (0, 1, 0)`.

# %% Imports
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

from beam_networks.problem import ElasticNetwork
from beam_networks.geometry.geo import get_geometric_props

# %% [markdown]
# ## Geometry and material
#
# The 3D node array has a third (z) column; all nodes start at $z = 0$.
# The beam is clamped with a 6-DOF constraint at the root.

# %%
ne = 20
Lx = 10.
beam_prop = {'b': 0.1, 'h': 0.1, 'E': 2.e11, 'nu': 0., 'name': 'rectangle'}
Iz = get_geometric_props(beam_prop)[1]
Iy = get_geometric_props(beam_prop)[0]

x = np.linspace(0., Lx, ne + 1)
nodes = np.column_stack([x, np.zeros(ne + 1), np.zeros(ne + 1)])
edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])

n_steps = 50
n_snapshots = 5
plot_every = max(n_steps // n_snapshots, 1)
cmap = plt.cm.coolwarm

try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass


def make_net(matrix='bsr'):
    net = ElasticNetwork(nodes, edges, beam_prop=beam_prop,
                         options={'vectorize': True, 'matrix': matrix,
                                  'verbose': False},
                         assemble_on_init=False)
    net.add_BC('clamp', 'D', 'node', [0], [0., 0., 0., 0., 0., 0.])
    return net


def draw_2d(ax, snaps, i_horiz, i_vert):
    for nodes_cur, color in snaps:
        segs = [(nodes_cur[e0, [i_horiz, i_vert]],
                 nodes_cur[e1, [i_horiz, i_vert]])
                for e0, e1 in edges]
        ax.add_collection(LineCollection(segs, linewidths=1.5, colors=color))
        ax.scatter(*nodes_cur[[-1], [i_horiz, i_vert]].T,
                   s=15, color=color, zorder=3)


def draw_3d(ax3, snaps, color_override=None):
    nodes_cur, color = snaps[-1]
    c = color_override if color_override is not None else color
    segs = [(nodes_cur[e0], nodes_cur[e1]) for e0, e1 in edges]
    ax3.add_collection(Line3DCollection(segs, linewidths=1.5, colors=c))
    ax3.scatter(*nodes_cur[[-1]].T, s=20, color=c, zorder=3)

# %% [markdown]
# ## Case 1 — In-plane bending ($M_z$, x-y plane)
#
# The reference moment $M_z = 2\pi E I_z / L$ is applied as a Neumann BC on the
# tip node's $\theta_z$ DOF (index 5 in the 6-component vector).
# With `ref_vec = (0, 0, 1)` the local $\hat{\mathbf{e}}_2$ axis is fixed along
# $\hat{z}$, which is always perpendicular to any chord in the x-y plane.

# %%
Mref_z = 2. * np.pi * beam_prop['E'] * Iz / Lx
net_xy = make_net()
net_xy.add_BC('load', 'N', 'node', [ne], [None, None, None, None, None, Mref_z])

ref_xy = np.tile([0., 0., 1.], (ne, 1))
snaps_xy = []


def cb_xy(step, nodes_cur, _):
    if step % plot_every == 0 or step == n_steps:
        snaps_xy.append((nodes_cur.copy(), cmap(step / n_steps)))


cb_xy(0, net_xy.nodes, None)
net_xy.solve_nonlinear(n_steps=n_steps, tol=1e-9, verbose=False,
                       callback=cb_xy, ref_vectors=ref_xy)

tip_xy = net_xy.displaced_nodes[-1]
print(f"x-y circle → tip = ({tip_xy[0]:.4e}, {tip_xy[1]:.4e}, {tip_xy[2]:.4e})")
print(f"              expected (0, 0, 0)")

# %%
theta_ref = np.linspace(0, 2 * np.pi, 400)
R = Lx / (2 * np.pi)

fig, ax = plt.subplots(figsize=(5, 5))
draw_2d(ax, snaps_xy, i_horiz=0, i_vert=1)
ax.plot(R * np.sin(theta_ref), R * (1 - np.cos(theta_ref)),
        'k--', lw=0.8, label='Exact arc')
ax.set_aspect('equal')
ax.autoscale_view()
ax.set_title(r'In-plane bending  ($M_z$, x-y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1.))
fig.colorbar(sm, ax=ax, shrink=0.5, label=r'$M / M_\mathrm{ref}$')
plt.show()

# %% [markdown]
# ## Case 2 — Out-of-plane bending ($M_y$, x-z plane)
#
# The moment $M_y = 2\pi E I_y / L$ acts on the $\theta_y$ DOF (index 4).
# With `ref_vec = (0, 1, 0)` the local $\hat{\mathbf{e}}_2$ axis stays along
# $\hat{y}$, perpendicular to any chord in the x-z plane.
# Because the cross-section is square ($I_y = I_z$), the same reference moment
# drives both cases to identical circle geometries in their respective planes.

# %%
Mref_y = 2. * np.pi * beam_prop['E'] * Iy / Lx
net_xz = make_net()
net_xz.add_BC('load', 'N', 'node', [ne], [None, None, None, None, Mref_y, None])

ref_xz = np.tile([0., 1., 0.], (ne, 1))
snaps_xz = []


def cb_xz(step, nodes_cur, _):
    if step % plot_every == 0 or step == n_steps:
        snaps_xz.append((nodes_cur.copy(), cmap(step / n_steps)))


cb_xz(0, net_xz.nodes, None)
net_xz.solve_nonlinear(n_steps=n_steps, tol=1e-9, verbose=False,
                       callback=cb_xz, ref_vectors=ref_xz)

tip_xz = net_xz.displaced_nodes[-1]
print(f"x-z circle → tip = ({tip_xz[0]:.4e}, {tip_xz[1]:.4e}, {tip_xz[2]:.4e})")
print(f"              expected (0, 0, 0)")

# %%
fig, ax = plt.subplots(figsize=(5, 5))
draw_2d(ax, snaps_xz, i_horiz=0, i_vert=2)
ax.plot(R * np.sin(theta_ref), -R * (1 - np.cos(theta_ref)),
        'k--', lw=0.8, label='Exact arc')
ax.set_aspect('equal')
ax.autoscale_view()
ax.set_title(r'Out-of-plane bending  ($M_y$, x-z)')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.legend(fontsize=8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1.))
fig.colorbar(sm, ax=ax, shrink=0.5, label=r'$M / M_\mathrm{ref}$')
plt.show()

# %% [markdown]
# ## 3D overlay
#
# The final shapes of both cases are superimposed in a single perspective view.
# Each deformation is strictly confined to its intended bending plane, with
# out-of-plane displacement at machine-precision level.

# %%
fig = plt.figure(figsize=(6, 5))
ax_3d = fig.add_subplot(projection='3d')

draw_3d(ax_3d, snaps_xy, color_override=cmap(1.0))
draw_3d(ax_3d, snaps_xz, color_override=cmap(0.0))
ax_3d.plot(R * np.sin(theta_ref), R * (1 - np.cos(theta_ref)),
           np.zeros_like(theta_ref), 'k--', lw=0.6)
ax_3d.plot(R * np.sin(theta_ref), np.zeros_like(theta_ref),
           -R * (1 - np.cos(theta_ref)), 'k--', lw=0.6)

ax_3d.set_xlabel('x')
ax_3d.set_ylabel('y')
ax_3d.set_zlabel('z')
ax_3d.set_title('3-D overlay  (final shapes)')

r_max = Lx / np.pi * 1.1
ax_3d.set_xlim(-r_max / 4, Lx * 1.1)
ax_3d.set_ylim(-r_max / 2, r_max * 1.1)
ax_3d.set_zlim(-r_max * 1.1, r_max / 2)

legend_elements = [
    Line2D([0], [0], color=cmap(1.0), lw=2, label=r'$M_z$ (x-y)'),
    Line2D([0], [0], color=cmap(0.0), lw=2, label=r'$M_y$ (x-z)'),
    Line2D([0], [0], color='k', lw=0.8, ls='--', label='Exact arc'),
]
ax_3d.legend(handles=legend_elements, fontsize=8, loc='upper right')
fig.suptitle('3-D rolling cantilever  (co-rotational, Crisfield 1990)',
             fontsize=12, fontweight='bold')
plt.show()
