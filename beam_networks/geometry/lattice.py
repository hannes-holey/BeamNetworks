#
# Copyright 2026 Hannes Holey
#
# This file is part of beam_networks. beam_networks is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. beam_networks is distributed in
# the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# beam_networks. If not, see <https://www.gnu.org/licenses/>.
#
import numpy as np
from scipy.spatial import cKDTree


def generate_cubic_lattice(a=1.,
                           pbc=None,
                           bbox=(1.0, 1.0, 1.0),
                           lattice_type="sc"):
    """
    Generate coordinates and connectivity for a 3D cubic lattice.

    Parameters
    ----------
    a : float, optional
        Lattice constant (distance between nearest neighbors)
    pbc: iterable, optional
        Periodic boundary conditions (the default is None, which means no periodic BCs).
        Example: In a 3D system [True, False, False] means that only x is a periodic direction.
    bbox : array-like
        Size of the bounding box, filled with repeated unit cells
    lattice_type : str, optional
        Name of the lattice type: ``'sc'``, ``'bcc'``, ``'fcc'``, ``'dia'``
        (diamond), or ``'sc-bcc'`` (simple-cubic + BCC bonds).
        The default is ``'sc'``.

    Returns
    -------
    np.ndarray
        Node coordinates
    np.ndarray
        Edge connectivity
    np.ndarray
        Periodic boundary flags
    tuple
        Bounding box
    bool
        Valid flag indicating whether further preprocessing is required
    """

    if pbc is None:
        # set all directions to False
        pbc = np.zeros(3, dtype=bool)
    else:
        pbc = np.array(pbc).astype(bool)

    _dir_map = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1]),
    }
    _bcc_directional = {'bccx', 'bccy', 'bccz'}
    _fcc_directional = {'fccx', 'fccy', 'fccz'}

    # Basis and nearest-neighbour distance for each primitive lattice type.
    _bases = {
        'sc': np.array([[0, 0, 0]]) * a,
        'fcc': np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a,
        'bcc': np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) * a,
        'dia': np.array([[0, 0, 0],
                         [0.5, 0.5, 0],
                         [0.5, 0, 0.5],
                         [0, 0.5, 0.5],
                         [0.25, 0.25, 0.25],
                         [0.75, 0.75, 0.25],
                         [0.75, 0.25, 0.75],
                         [0.25, 0.75, 0.75]]) * a,
    }
    # Directional variants share basis/a0 with their base type.
    for _d in 'xyz':
        _bases[f'bcc{_d}'] = _bases['bcc']
        _bases[f'fcc{_d}'] = _bases['fcc']

    _a0s = {
        'sc':  a,
        'fcc': a * np.sqrt(2) / 2,
        'bcc': a * np.sqrt(3) / 2,
        'dia': a * np.sqrt(3) / 4,
    }
    for _d in 'xyz':
        _a0s[f'bcc{_d}'] = _a0s['bcc']
        _a0s[f'fcc{_d}'] = _a0s['fcc']

    _valid_types = set(_bases.keys())

    bbox = tuple(bbox)

    def _make_coords(basis):
        """Fill bbox with one sublattice and return deduplicated node coords."""
        nx, ny, nz = (
            np.array(bbox) / np.maximum(np.ones(3) * a, np.amax(basis, axis=0))
        ).astype(int) + 1
        grid = (
            np.stack(
                np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"),
                axis=-1,
            ).reshape(-1, 3)
            * a
        )
        coords = (grid[:, None, :] + basis).reshape(-1, 3)
        mask = np.all(coords >= 0.0, axis=1) & np.all(coords <= bbox, axis=1)
        return np.unique(coords[mask], axis=0)

    # ------------------------------------------------------------------ #
    # Parse lattice type: split on '-' to support compound specifications #
    # ------------------------------------------------------------------ #
    subtypes = lattice_type.split('-')

    unknown = [s for s in subtypes if s not in _valid_types]
    if unknown:
        raise RuntimeError(
            f"Unknown sublattice type(s): {unknown}. "
            "Each component must be one of: "
            + str(sorted(_valid_types))
        )

    if len(subtypes) == 1:
        # ---- Single lattice (including directional variants) ----
        basis = _bases[lattice_type]
        a0 = _a0s[lattice_type]
        lattice_coords = _make_coords(basis)

        if lattice_type in _bcc_directional | _fcc_directional:
            # Directional reinforcement: bonds of length a along one axis only.
            # Includes both corner-to-corner and body/face-center-to-body/face-center
            # bonds along that direction.
            dir_vec = _dir_map[lattice_type[-1]]
            all_a_edges = _get_connections(lattice_coords, a)
            if len(all_a_edges):
                edge_vecs = (lattice_coords[all_a_edges[:, 1]]
                             - lattice_coords[all_a_edges[:, 0]])
                dir_mask = np.isclose(np.abs(edge_vecs @ dir_vec), a)
                dir_edges = all_a_edges[dir_mask]
            else:
                dir_edges = np.empty((0, 2), dtype=int)
            connections = np.vstack([dir_edges, _get_connections(lattice_coords, a0)])

        else:
            connections = _get_connections(lattice_coords, a0)

    else:
        # ---- Compound lattice: iterate over sublattices ----
        # Generate each sublattice's nodes independently, then merge.
        sub_coords_list = [_make_coords(_bases[s]) for s in subtypes]

        # Merge all sublattice coords into a single deduplicated array.
        lattice_coords = np.unique(np.vstack(sub_coords_list), axis=0)

        # For each sublattice, find its local edges and remap to merged indices.
        tree = cKDTree(lattice_coords)
        all_edges = []
        for sub_type, sub_coords in zip(subtypes, sub_coords_list):
            _, idx = tree.query(sub_coords)          # exact match (distance ≈ 0)
            local_edges = _get_connections(sub_coords, _a0s[sub_type])
            if len(local_edges):
                all_edges.append(idx[local_edges])
        connections = (
            np.vstack(all_edges) if all_edges else np.empty((0, 2), dtype=int)
        )

    for d in np.arange(3)[pbc]:

        pbcx_mask_left = np.isclose(lattice_coords[:, d], 0.0)
        pbcx_mask_right = np.isclose(lattice_coords[:, d], bbox[d])

        pbcx_nodes_left = np.arange(lattice_coords.shape[0])[pbcx_mask_left]
        pbcx_nodes_right = np.arange(lattice_coords.shape[0])[pbcx_mask_right]

        new_connections = np.copy(connections)

        for pl, pr in zip(pbcx_nodes_left, pbcx_nodes_right):

            new_connections[connections[:, 0] == pl, 0] = pr
            new_connections[connections[:, 1] == pl, 1] = pr

        connections = new_connections

    connections = np.unique(connections, axis=0)

    # Lattice types with a multi-atom basis can produce boundary nodes whose
    # all neighbours fall outside the bounding box — prune them now.
    # This applies to: diamond, all directional variants, and any compound type.
    _needs_prune = (
        lattice_type == "dia"
        or lattice_type in _bcc_directional | _fcc_directional
        or len(subtypes) > 1
    )
    if _needs_prune:
        lattice_coords, connections = _remove_isolated(lattice_coords, connections)

    # Need to delete obsolete nodes for pbc, thus set valid to false
    if np.any(pbc):
        valid = False
    else:
        valid = True

    return lattice_coords, connections, pbc, bbox, valid


def generate_square_lattice(a=1.0,
                            bbox=(1.0, 1.0),
                            lattice_type="sc"):
    """
    Generate coordinates and connectivity for a 2D square lattice.

    Parameters
    ----------
    a : float, optional
        Lattice constant (distance between nearest neighbors)
    bbox : array-like
        Size of the bounding box ``(Lx, Ly)``, filled with repeated unit cells.
        The default is ``(1.0, 1.0)``.
    lattice_type : str, optional
        Name of the lattice type:

        * ``'sc'`` – simple square (coordination 4)
        * ``'fcc'`` – rotated-square (coordination 4, kept for compatibility)
        * ``'triangular'`` / ``'hex'`` – true triangular / hexagonal (coordination 6)
        * ``'kagome'`` – corner-sharing triangles (coordination 4)

        The default is ``'sc'``.

    Returns
    -------
    np.ndarray
        Node coordinates
    np.ndarray
        Edge connectivity
    """

    if lattice_type == "sc":
        cell = np.array([a, a])
        basis = np.array([[0., 0.]])
        a0 = a
    elif lattice_type == "fcc":
        # Rotated-square lattice (kept for backward compatibility).
        cell = np.array([a, a])
        basis = np.array([[0., 0.], [0.5 * a, 0.5 * a]])
        a0 = a * np.sqrt(2) / 2
    elif lattice_type in ("triangular", "hex"):
        # True triangular (hexagonal close-packed) lattice, coordination 6.
        # Rectangular supercell of height a*sqrt(3) contains 2 atoms.
        cell = np.array([a, a * np.sqrt(3)])
        basis = np.array([[0., 0.], [0.5 * a, 0.5 * a * np.sqrt(3)]])
        a0 = a
    elif lattice_type == "kagome":
        # Kagome lattice: corner-sharing triangles, coordination 4.
        # Uses a non-orthogonal primitive cell with lattice vectors
        #   A1 = (2a, 0),  A2 = (a, a*sqrt(3))
        # and three basis atoms at the midpoints of the cell's triangle edges.
        bbox = tuple(bbox)
        A1 = np.array([2. * a, 0.])
        A2 = np.array([a, a * np.sqrt(3)])
        basis = np.array([[a, 0.],
                          [0.5 * a, 0.5 * a * np.sqrt(3)],
                          [1.5 * a, 0.5 * a * np.sqrt(3)]])
        nj = int(bbox[1] / (a * np.sqrt(3))) + 2
        ni = int(bbox[0] / (2. * a)) + 2
        # A2 has an x-component (a), so cells with negative i can still
        # contribute nodes inside the bbox; start at i = -nj to be safe.
        gi, gj = np.meshgrid(np.arange(-nj, ni + nj), np.arange(nj), indexing="ij")
        grid_origins = (gi[..., None] * A1 + gj[..., None] * A2).reshape(-1, 2)
        lattice_coords = (grid_origins[:, None, :] + basis).reshape(-1, 2)
        mask = np.logical_and(
            np.all(lattice_coords <= bbox, axis=-1),
            np.all(lattice_coords >= 0.0, axis=-1),
        )
        lattice_coords = lattice_coords[mask]
        connections = _get_connections(lattice_coords, a)
        return lattice_coords, connections
    else:
        raise RuntimeError(
            "Lattice must be one of ['sc', 'fcc', 'triangular', 'hex', 'kagome']"
        )

    bbox = tuple(bbox)

    nx, ny = (np.array(bbox) / cell).astype(int) + 1

    # Generate the grid of unit cells
    grid_x, grid_y = np.meshgrid(
        np.arange(nx), np.arange(ny), indexing="ij"
    )

    # Stack the grid into vectors of unit cell origins and scale by cell
    grid_coords = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2) * cell

    # Add the basis atoms to the grid
    lattice_coords = grid_coords[:, None, :] + basis
    lattice_coords = lattice_coords.reshape(
        -1, 2
    )  # Reshape into list of 2D coordinates

    # Select only nodes within the box
    mask = np.logical_and(
        np.all(lattice_coords <= bbox, axis=-1),
        np.all(lattice_coords >= 0.0, axis=-1),
    )
    lattice_coords = lattice_coords[mask]

    # Now we find connections between neighboring atoms using a KD-Tree
    connections = _get_connections(lattice_coords, a0)

    return lattice_coords, connections


def generate_bowtie_lattice(a=1.0,
                            w=0.1,
                            bbox=(1.0, 1.0)):
    """
    Generate coordinates and connectivity for a 2D bowtie lattice.

    Parameters
    ----------
    a : float, optional
        Lattice constant (distance between nearest neighbors)
    w : float, optional
        Offset parameter controlling the bowtie shape
    bbox : array-like
        Size of the bounding box, filled with repeated unit cells

    Returns
    -------
    np.ndarray
        Node coordinates
    np.ndarray
        Edge connectivity
    """

    basis = (
        np.array(
            [
                [w, 0.0],
                [1.0 - w, 0.0],
                [0.5 - w, 0.5],
                [0.5 + w, 0.5],
            ]
        )
        * a
    )

    bbox = tuple(bbox)

    nx, ny = (
        np.array(bbox) / np.maximum(np.ones(2) * a, np.amax(basis, axis=0))
    ).astype(int) + 1

    # Generate the grid of unit cells
    grid_x, grid_y = np.meshgrid(
        np.arange(nx), np.arange(ny), indexing="ij"
    )

    # Stack the grid into vectors of unit cell origins and scale by a
    grid_coords = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2) * a

    # Add the basis atoms to the grid
    lattice_coords = grid_coords[:, None, :] + basis
    lattice_coords = lattice_coords.reshape(
        -1, 2
    )  # Reshape into list of 2D coordinates

    # Select only nodes within the box
    mask = np.logical_and(
        np.all(lattice_coords <= bbox, axis=-1),
        np.all(lattice_coords >= 0.0, axis=-1),
    )
    lattice_coords = lattice_coords[mask]

    d0 = np.linalg.norm(basis[1] - basis[0])
    d1 = np.linalg.norm(basis[2] - basis[0])

    connections_1 = _get_connections(lattice_coords * np.array([1.0, 1.0]), d0)
    connections_2 = _get_connections(lattice_coords * np.array([1.0, 1.0]), d1)

    connections = np.vstack(
        [
            connections_1,
            connections_2,
        ]
    )

    return lattice_coords, connections


def _remove_isolated(coords, edges):
    """Remove nodes not referenced by any edge and re-index edge array.

    Parameters
    ----------
    coords : np.ndarray
        Node coordinates, shape (N, dim).
    edges : np.ndarray
        Edge connectivity, shape (M, 2).

    Returns
    -------
    np.ndarray
        Pruned node coordinates.
    np.ndarray
        Re-indexed edge connectivity.
    """
    used = np.unique(edges)
    if len(used) == len(coords):
        return coords, edges
    remap = np.full(len(coords), -1, dtype=int)
    remap[used] = np.arange(len(used))
    return coords[used], remap[edges]


def _get_connections(coords, d):
    """Find all node pairs separated by distance *d* using a KD-tree.

    Parameters
    ----------
    coords : np.ndarray
        Node coordinates, shape (num_nodes, dim).
    d : float
        Target distance (nearest-neighbour distance for the lattice).

    Returns
    -------
    np.ndarray
        Edge connectivity array of shape (num_edges, 2).
    """
    hi = d + 1e-8
    lo = d - 1e-8

    kdtree = cKDTree(coords)
    connections_hi = kdtree.query_pairs(
        r=hi, output_type="ndarray"
    )  # Neighbors within hi

    connections = []
    for e0, e1 in connections_hi:
        dd = np.linalg.norm(coords[e1] - coords[e0])
        if dd > lo:
            connections.append([e0, e1])

    return np.array(connections)
