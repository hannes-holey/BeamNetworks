# %% [markdown]
# # Lattice Zoo
#
# This tutorial gives an overview of all built-in lattice generators and their
# key topological properties.
#
# ## 2-D lattices
#
# | Type | Key | Coordination | Bond length |
# |------|-----|:---:|:---:|
# | Simple square | `'sc'` | 4 | $a$ |
# | Rotated square (legacy) | `'fcc'` | 4 | $a/\sqrt{2}$ |
# | Triangular / hexagonal | `'triangular'` or `'hex'` | 6 | $a$ |
# | Kagome | `'kagome'` | 4 | $a$ |
#
# ## 3-D lattices
#
# ### Primitive lattices
#
# | Type | Key | Coordination | Bond length |
# |------|-----|:---:|:---:|
# | Simple cubic | `'sc'` | 6 | $a$ |
# | Body-centred cubic | `'bcc'` | 8 | $a\sqrt{3}/2$ |
# | Face-centred cubic | `'fcc'` | 12 | $a/\sqrt{2}$ |
# | Diamond | `'dia'` | 4 | $a\sqrt{3}/4$ |
#
# ### Compound lattices (`'A-B'` syntax)
#
# Compound types are specified as dash-separated primitive names.  Each
# sublattice is generated independently and the node/edge sets are merged,
# so no spurious cross-sublattice bonds appear at the SC bond length.
#
# | Type | Key | Coordination | Bond lengths |
# |------|-----|:---:|:---:|
# | SC + BCC bonds | `'sc-bcc'` | 14 | $a$ and $a\sqrt{3}/2$ |
# | SC + FCC bonds | `'sc-fcc'` | 18 | $a$ and $a/\sqrt{2}$ |
#
# ### Directionally reinforced lattices (`'bcc[xyz]'` / `'fcc[xyz]'`)
#
# These add axis-aligned bonds of length $a$ along one coordinate direction
# on top of the underlying BCC or FCC bond network.  Both corner nodes *and*
# body/face-centre nodes are connected along the chosen axis.
#
# | Type | Key | Coordination | Bond lengths |
# |------|-----|:---:|:---:|
# | BCC + x bonds | `'bccx'` | 10 | $a$ and $a\sqrt{3}/2$ |
# | BCC + y bonds | `'bccy'` | 10 | $a$ and $a\sqrt{3}/2$ |
# | BCC + z bonds | `'bccz'` | 10 | $a$ and $a\sqrt{3}/2$ |
# | FCC + x bonds | `'fccx'` | 14 | $a$ and $a/\sqrt{2}$ |
# | FCC + y bonds | `'fccy'` | 14 | $a$ and $a/\sqrt{2}$ |
# | FCC + z bonds | `'fccz'` | 14 | $a$ and $a/\sqrt{2}$ |

# %%
import matplotlib.pyplot as plt
import numpy as np

from beam_networks import ElasticNetwork

props = {'name': 'circle', 'radius': 0.05, 'E': 1.0, 'nu': 0.3}

# %% [markdown]
# ## 2-D lattice gallery

# %%
LATTICE_TYPES_2D = ['sc', 'triangular', 'kagome']
LABELS_2D = ['Simple square (sc)', 'Triangular (hex)', 'Kagome']

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, lt, label in zip(axes, LATTICE_TYPES_2D, LABELS_2D):
    net = ElasticNetwork.from_square_lattice(
        a=1., bbox=(8., 8.), lattice_type=lt, beam_prop=props)
    net.plot(ax=ax)
    ax.set_title(f'{label}\n{net.num_nodes} nodes, {net.num_edges} edges')
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3-D lattice gallery
#
# PyVista is used for 3-D rendering (requires the `viz` extra:
# `pip install beam_networks[viz]`).

# %%
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("pyvista not installed — skipping 3-D plot.\n"
          "Install with:  pip install beam_networks[viz]")

# %%
LATTICE_TYPES_3D = [
    'sc', 'bcc', 'fcc', 'dia',
    'sc-bcc', 'sc-fcc',
    'bccz', 'fccz',
]
LABELS_3D = [
    'Simple cubic (sc)', 'BCC', 'FCC', 'Diamond (dia)',
    'SC-BCC', 'SC-FCC',
    'BCC+z (bccz)', 'FCC+z (fccz)',
]

if HAS_PYVISTA:
    # For an interactive in-notebook view uncomment:
    #   pv.set_jupyter_backend('trame')
    pl = pv.Plotter(shape=(2, 4), window_size=(1800, 800), off_screen=True)

    for idx, (lt, label) in enumerate(zip(LATTICE_TYPES_3D, LABELS_3D)):
        row, col = divmod(idx, 4)
        net = ElasticNetwork.from_cubic_lattice(
            a=1., bbox=(2., 2., 2.), lattice_type=lt, beam_prop=props)
        pl.subplot(row, col)
        net.plot(plotter=pl, scalar_bar_args=None)
        pl.add_title(f'{label}\n{net.num_nodes}N / {net.num_edges}E', font_size=10)

    pl.show(screenshot='lattice_zoo_3d.png')

    from IPython.display import Image, display
    display(Image('lattice_zoo_3d.png'))

# %% [markdown]
# ## Coordination numbers
#
# For a network in bulk (away from boundaries), the degree of each node equals
# the lattice coordination number.

# %%
print(f"{'Lattice':<20} {'Nodes':>6} {'Edges':>6} {'max coord':>10}")
print('-' * 47)

for lt, label in zip(LATTICE_TYPES_2D, LABELS_2D):
    net = ElasticNetwork.from_square_lattice(
        a=1., bbox=(8., 8.), lattice_type=lt, beam_prop=props)
    deg = np.bincount(net.edges.ravel(), minlength=net.num_nodes)
    print(f'{label:<20} {net.num_nodes:>6} {net.num_edges:>6} {deg.max():>10}')

for lt, label in zip(LATTICE_TYPES_3D, LABELS_3D):
    net = ElasticNetwork.from_cubic_lattice(
        a=1., bbox=(4., 4., 4.), lattice_type=lt, beam_prop=props)
    deg = np.bincount(net.edges.ravel(), minlength=net.num_nodes)
    print(f'{label:<20} {net.num_nodes:>6} {net.num_edges:>6} {deg.max():>10}')
