# %% [markdown]
# # 2D Cantilever: Point Load and Moment
#
# This tutorial demonstrates the basic workflow for setting up and solving a
# 2D cantilever beam with a combined point force and torque applied at an
# intermediate location. The beam is rotated 180° in the x-y plane to
# exercise the rotation handling.
#
# **Geometry** — circular cross-section, radius $R = 0.05$ m, length $L = 2$ m.
#
# **Material** — $E = 2.1\times10^{11}$ Pa, $\nu = 0.3$.
#
# **Loading** — a point force $(F_x, F_y)$ and torque $M_z$ applied at $x = 0.3 L$,
# with the beam rotated 180° around $\hat{z}$.
#
# The numerical solution is compared with the Timoshenko analytic reference.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from beam_networks import ElasticNetwork
from beam_networks.geometry.geo import get_geometric_props
from beam_networks.postprocess.viz import plot_solution_1D
from beam_networks.reference_solutions.cantilever import cantilever_analytic

# %% [markdown]
# ## Parameters

# %%
E = 2.1e11
nu = 0.3
R = 0.05
length = 2.
a = 0.3   # load location as fraction of L

props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

phi_deg = 180.
phi = phi_deg / 180 * np.pi
rot = Rotation.from_euler('z', phi).as_matrix()

# %% [markdown]
# ## Geometry
#
# 101 nodes are placed along the beam axis and then rotated by the chosen angle.

# %%
Iy, Iz, _, A, kappa, ymax = get_geometric_props(props)
_Fext = np.array([0.05 * E * A / length, 4. * E * Iz / length**3])
_Mext = np.array([-2. * _Fext[1] * a / length])
F_local = np.hstack([_Fext, _Mext])
F_rot = rot.dot(F_local)

num_nodes = 101
x = np.linspace(0., length, num_nodes)
y = np.zeros(num_nodes)
z = np.zeros(num_nodes)
nodes_positions = np.vstack([x, y, z]).T
nodes_positions = np.matmul(rot[None, :, :], nodes_positions.T)[0].T
nodes_positions = nodes_positions[:, :2]

num_elements = num_nodes - 1
edges_indices = np.column_stack([np.arange(num_elements), np.arange(1, num_elements + 1)])

# Force application node (closest to a * L along the beam axis)
x_pos = rot.dot(np.array([a * length, 0., 0.]))[:2]
dist = np.sqrt(np.sum((nodes_positions - x_pos)**2, axis=-1))
node = np.argmin(dist)
dist_n = np.sqrt(np.sum(nodes_positions[node]**2)) / length

# %% [markdown]
# ## Solve

# %%
problem = ElasticNetwork(nodes_positions, edges_indices, beam_prop=props, valid=True)
problem.add_BC('0', 'D', 'node', [0], [0., 0., 0.])
problem.add_BC('1', 'N', 'node', [node], F_rot)
problem.solve()
d_num = problem.sol

# Rotate solution back to beam frame
u = np.vstack([d_num[0::3], d_num[1::3], d_num[2::3]])
u_rot = rot.T.dot(u)
d_num_red = np.hstack([u_rot[0], u_rot[1], u_rot[2]]).reshape(3, -1).T.flatten()

seq = problem._sVM

# %% [markdown]
# ## Displacement and stress profiles
#
# The four panels show axial displacement $u_x$, transverse displacement $u_y$,
# rotation $\theta_z$, and von Mises stress, all plotted along the beam axis
# in the beam frame. Dashed lines are the analytic Timoshenko solution.

# %%
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

xref = np.linspace(0., length, 101)
uxref, uref, tref, sref = cantilever_analytic(xref, length, F_local, dist_n, props)

sx, sy = plt.rcParams['figure.figsize']
fig, ax = plt.subplots(4, figsize=(sx, 2 * sy), sharex=True, constrained_layout=True)

ax[0].plot(xref, uxref, '--', color='0.0', label='Analytical')
ax[1].plot(xref, uref,  '--', color='0.0')
ax[2].plot(xref, tref,  '--', color='0.0')
ax[3].plot(xref, sref,  '--', color='0.0')

plot_solution_1D(ax, d_num_red, seq, length)
ax[0].legend()
plt.show()

# %% [markdown]
# ## Deformed shape
#
# The network is plotted in the rotated (global) frame, coloured by von Mises stress.

# %%
fig1, ax1 = plt.subplots(1)
problem.plot(ax1, contour=seq)
plt.show()
