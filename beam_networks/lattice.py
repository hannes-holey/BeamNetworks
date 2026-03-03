#
# Copyright 2025 Hannes Holey
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
        Name of the lattice type ['sc', 'bcc', 'fcc'] (the default is 'sc')

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

    a0 = a

    if lattice_type == "sc":
        basis = np.array(
            [
                [0, 0, 0],  # Corner of the cube
            ]
        ) * a
    elif lattice_type == "fcc":
        basis = (
            np.array(
                [
                    [0, 0, 0],  # Corner of the cube
                    [0.5, 0.5, 0],  # Face centers
                    [0.5, 0, 0.5],
                    [0, 0.5, 0.5],
                ]
            )
            * a
        )
        a0 = a * np.sqrt(2) / 2
    elif lattice_type == "bcc":
        basis = (
            np.array(
                [
                    [0, 0, 0],  # Corner of the cube
                    [0.5, 0.5, 0.5],  # Center of the cube
                ]
            )
            * a
        )
        a0 = a * np.sqrt(3) / 2
    else:
        raise RuntimeError("Lattice must be one of ['sc', 'fcc', 'bcc']")

    bbox = tuple(bbox)

    nx, ny, nz = (
        np.array(bbox) / np.maximum(np.ones(3) * a, np.amax(basis, axis=0))
    ).astype(int) + 1

    # Generate the grid of unit cells
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )

    # Stack the grid into vectors of unit cell origins and scale by a
    grid_coords = (
        np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3) * a
    )

    # Add the basis atoms to the grid
    lattice_coords = grid_coords[:, None, :] + basis
    lattice_coords = lattice_coords.reshape(
        -1, 3
    )  # Reshape into list of 3D coordinates

    # Select only nodes within the box
    mask = np.logical_and(
        np.all(lattice_coords <= bbox, axis=-1),
        np.all(lattice_coords >= 0.0, axis=-1),
    )
    lattice_coords = lattice_coords[mask]

    connections = _get_connections(lattice_coords, a0)

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

    # Need to delete obsolete nodes for pbc, thus set valid to false
    if np.any(pbc):
        valid = False
    else:
        valid = True

    print(valid)

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
        Size of the bounding box, filled with repeated unit cells
    lattice_type : str, optional
        Name of the lattice type ['sc', 'bcc', 'fcc'] (the default is 'sc')

    Returns
    -------
    np.ndarray
        Node coordinates
    np.ndarray
        Edge connectivity
    """

    a0 = a

    if lattice_type == "sc":
        basis = np.array([[0, 0]]) * a
    elif lattice_type == "fcc":
        basis = (
            np.array(
                [
                    [0, 0],  # Corner of the cube
                    [0.5, 0],  # Face centers
                    [0, 0.5],
                ]
            )
            * a
        )
        a0 = a * np.sqrt(2) / 2
    elif lattice_type == "bcc":
        basis = (
            np.array(
                [
                    [0, 0],  # Corner of the cube
                    [0.5, 0.5],  # Center of the cube
                ]
            )
            * a
        )
        a0 = a * np.sqrt(3) / 2
    else:
        raise RuntimeError("Lattice must be one of ['sc', 'bcc', 'fcc']")

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


def _get_connections(coords, d):

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

