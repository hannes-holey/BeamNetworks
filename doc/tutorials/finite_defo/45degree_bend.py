# %% [markdown]
# # 45-Degree Cantilever: Out-of-Plane Tip Load
#
# This tutorial reproduces the 45-degree bent cantilever benchmark from
# Crisfield (1990), one of the most widely used validation cases for 3D
# geometrically nonlinear beam elements.
#
# ## Problem description
#
# A slender cantilever lies initially in the x-y plane, curved through 45°
# from the clamped root at the origin to the free tip.
# An out-of-plane tip load $F_z$ is applied at the free end.
# As $F_z$ increases, the beam twists and deflects out of the x-y plane,
# producing a fully 3D deformed shape.
#
# **Geometry** — arc of radius 100, subtending 45°:
#
# $$x(\varphi) = 100\sin\varphi, \quad y(\varphi) = -100(1-\cos\varphi),
# \quad \varphi \in [0,\,\pi/4]$$
#
# **Material** — $E = 10^7$, $\nu = 0.3$, square cross-section $b = h = 1$.
#
# **Reference tip positions** (Crisfield 1990, Table 3):
#
# | $F_z$ | $x$ | $y$ | $z$ |
# |------:|----:|----:|----:|
# |   0   | 70.71 | −29.29 |  0.00 |
# | 300   | 58.53 | −22.16 | 40.53 |
# | 450   | 51.93 | −18.43 | 48.79 |
# | 600   | 46.84 | −15.61 | 53.71 |

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from beam_networks.problem import ElasticNetwork
from beam_networks.geometry.geo import get_geometric_props

# %% [markdown]
# ## Geometry and material

# %%
ne = 8
beam_prop = {'b': 1., 'h': 1., 'E': 1e7, 'nu': 0.3, 'name': 'rectangle'}

angle = np.linspace(0., np.pi / 4., ne + 1)
x = 100. * np.sin(angle)
y = -100. * (1. - np.cos(angle))
nodes = np.column_stack([x, y, np.zeros(ne + 1)])
edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])

n_steps = 8

crisfield_ref = np.array([[70.71, -29.29,  0.  ],
                           [58.53, -22.16, 40.53],
                           [51.93, -18.43, 48.79],
                           [46.84, -15.61, 53.71]])

# %% [markdown]
# ## Solving for each load level
#
# The `run` function creates a fresh network, clamps node 0 with all 6 DOFs
# fixed, applies the out-of-plane tip load $F_z$, and runs the nonlinear solver.
# The load is incremented over `n_steps = 8` steps — the same discretisation
# used in the original Crisfield paper.

# %%
def run(Fz=0.):
    net = ElasticNetwork(nodes, edges, beam_prop=beam_prop,
                         options={'vectorize': True,
                                  'matrix': 'bsr',
                                  'verbose': False},
                         assemble_on_init=True)
    net.add_BC('clamp', 'D', 'node', [0], [0., 0., 0., 0., 0., 0.])
    net.add_BC('load', 'N', 'node', [ne], [None, None, Fz, None, None, None])
    net.solve_nonlinear(n_steps=n_steps, tol=1e-9, verbose=False)
    return net


# %%
print(f"{'Fz':>6}  {'x':>7} {'y':>7} {'z':>7}    "
      f"{'x_ref':>7} {'y_ref':>7} {'z_ref':>7}")
print("-" * 58)

nets = []
for i, Fz in enumerate([0., 300., 450., 600.]):
    net = run(Fz=Fz)
    nets.append(net)
    tip = net.displaced_nodes[-1]
    ref = crisfield_ref[i]
    print(f"{Fz:>6.0f}  {tip[0]:>7.2f} {tip[1]:>7.2f} {tip[2]:>7.2f}    "
          f"{ref[0]:>7.2f} {ref[1]:>7.2f} {ref[2]:>7.2f}")

# %% [markdown]
# The computed tip positions agree with the Crisfield reference values to within
# the discretisation error of the 8-element mesh (below 1% for all load levels).

# %% [markdown]
# ## Visualisation
#
# The deformed shapes for all four load levels are shown together. Open circles
# mark the Crisfield reference tip positions.

# %%
def draw_3d(ax, net, c='C0'):
    nodes_cur = net.displaced_nodes
    segs = [(nodes_cur[e0], nodes_cur[e1]) for e0, e1 in net.edges]
    ax.add_collection(Line3DCollection(segs, linewidths=1.5, colors=c))
    ax.scatter(*nodes_cur[[-1]].T, s=20, color=c, zorder=3)


try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

fig, ax = plt.subplots(1, subplot_kw={'projection': '3d'}, figsize=(7, 5))

for i, (net, Fz) in enumerate(zip(nets, [0., 300., 450., 600.])):
    draw_3d(ax, net, c=f'C{i}')
    ax.plot([], [], color=f'C{i}', label=f'$F_z = {Fz:.0f}$')
    ax.scatter(*crisfield_ref[i],
               marker='o', facecolor='none', linewidths=1., edgecolors='0.0',
               label='Crisfield (1990)' if i == 3 else None, zorder=-10)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(title='Tip load')
ax.set_aspect('equal')
fig.suptitle('45° cantilever — out-of-plane tip load',
             fontsize=12, fontweight='bold')
plt.show()
