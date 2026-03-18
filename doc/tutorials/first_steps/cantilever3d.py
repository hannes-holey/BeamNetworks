# %% [markdown]
# # 3D Cantilever: Point Load and Moment
#
# This tutorial extends the 2D cantilever to full 3D. A circular-section beam
# is rotated 45° around the $z$-axis and an additional 0° around $y$. A combined
# force and torque is applied at $x = 0.3 L$ in the beam frame.
#
# **Geometry** — circular cross-section, radius $R = 0.05$ m, length $L = 2$ m.
#
# **Material** — $E = 2.1\times10^{11}$ Pa, $\nu = 0.3$.
#
# The numerical solution (rotated back to the beam frame) is compared with the
# Timoshenko analytic reference, confirming that the 3D solver correctly handles
# arbitrary beam orientations.

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
a = 0.3

props = {'name': 'circle', 'radius': R, 'E': E, 'nu': nu}

phi_z_deg = 45.
phi_y_deg = 0.
rot = Rotation.from_euler('zy', [phi_z_deg / 180 * np.pi,
                                  phi_y_deg / 180 * np.pi]).as_matrix()

Iy, Iz, _, A, kappa, ymax = get_geometric_props(props)
_Fext = np.array([0.05 * E * A / length, 0.1 * E * Iz / length**3, 0.])
_Mext = np.array([0., 0., -0.2 * _Fext[1] * a / length])
F_local = np.hstack([_Fext, _Mext])
F_rot = np.hstack([rot.dot(_Fext), rot.dot(_Mext)])

# %% [markdown]
# ## Geometry

# %%
num_nodes = 51
x = np.linspace(0., length, num_nodes)
nodes_positions = np.vstack([x, np.zeros(num_nodes), np.zeros(num_nodes)]).T
nodes_positions = np.matmul(rot[None, :, :], nodes_positions.T)[0].T

num_elements = num_nodes - 1
edges_indices = np.column_stack([np.arange(num_elements), np.arange(1, num_elements + 1)])

x_pos = rot.dot(np.array([a * length, 0., 0.]))
dist = np.sqrt(np.sum((nodes_positions - x_pos)**2, axis=-1))
node = np.argmin(dist)
dist_n = np.sqrt(np.sum(nodes_positions[node]**2)) / length

# %% [markdown]
# ## Solve

# %%
problem = ElasticNetwork(nodes_positions, edges_indices, beam_prop=props, valid=True)
problem.add_BC('0', 'D', 'node', [0], [0., 0., 0., 0., 0., 0.])
problem.add_BC('1', 'N', 'node', [node], F_rot)
problem.solve()
d_num = problem.sol

u = np.vstack([d_num[0::6], d_num[1::6], d_num[2::6]])
v = np.vstack([d_num[3::6], d_num[4::6], d_num[5::6]])
u_rot = rot.T.dot(u)
v_rot = rot.T.dot(v)
d_num_red = np.hstack([u_rot[0], u_rot[1], v_rot[2]]).reshape(3, -1).T.flatten()

# %% [markdown]
# ## Displacement and stress profiles

# %%
try:
    plt.style.use('../beams.mplstyle')
except OSError:
    pass

xref = np.linspace(0., length, 101)
uxref, uref, tref, sref = cantilever_analytic(xref, length, F_local[[0, 1, 5]], dist_n, props)

sx, sy = plt.rcParams['figure.figsize']
fig, ax = plt.subplots(4, figsize=(sx, 2 * sy), sharex=True, constrained_layout=True)

ax[0].plot(xref, uxref, '--', color='0.0', label='Analytical')
ax[1].plot(xref, uref,  '--', color='0.0')
ax[2].plot(xref, tref,  '--', color='0.0')
ax[3].plot(xref, sref,  '--', color='0.0')

plot_solution_1D(ax, d_num_red, problem._sVM, length)
ax[0].legend()
plt.show()
