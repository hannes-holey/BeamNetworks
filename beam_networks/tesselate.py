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
from itertools import product

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def tesselate(coords, adj, rep,
              cell=None, pbc=False,
              overlap_tol=1e-8,
              debug=False):
    """Repeat unit cell in each direction. Currently implemented by copying
    coordinates and adjacency cell repeatedly and then detecting overlapping
    nodes via a cKDTree. The overlapping nodes are reduced to one node which
    overtakes all the connections from the overlapping nodes. Then the
    overlapping nodes are deleted and the adjacency matrix renumbered to
    account for deleted nodes.

    Parameters
    ----------
    coords : np.ndarray
        coordinates of shape (n_nodes,ndim).
    adj : np.ndarray
        adjacency list of shape (n_bonds,2).
    rep : list or np.ndarray
        number of repetitions in each direction
    cell : None or np.ndarray, optional
        cell lengths of shape (ndim). If None infer from coordinates (max-min).
    pbc : bool or list, optional
        Collect all element stiffness matrices in a large array before assembly.
        Faster, but may cause memory issues for very large systems (the default is True).
    overlap_tol : float, optional
        tolerance for detecting overlaps.
    debug : bool, optional
        print out information for debugging.

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (n_nodes_tesselate,ndim).
    adj : np.ndarray
        adjacency list of shape (n_bonds_tesselate,2).
    cell : np.ndarray
        cell lengths of shape (ndim).

    """

    #
    n_nodes, ndim = coords.shape
    # transform coordinates to start from zero
    shift = coords.min(axis=0)
    coords = coords - shift[None, :]
    # infer cell lengths if necessary
    if cell is None:
        cell = coords.max(axis=0)
    # copy adjacency matrices
    if ndim == 2:
        adj_new = [adj+n_nodes*(i+j*rep[0])
                   for i, j in product(np.arange(rep[0]),
                                       np.arange(rep[1]))]
    elif ndim == 3:
        adj_new = [adj+n_nodes*(i+j*rep[0]+k*rep[0]*rep[1])
                   for i, j, k in product(np.arange(rep[0]),
                                          np.arange(rep[1]),
                                          np.arange(rep[2]))]
    # copy coordinates
    if ndim == 2:
        coords_t = [coords + cell[:ndim]*np.array([i, j])
                    for j, i in product(np.arange(rep[1]),
                                        np.arange(rep[0]))]

    elif ndim == 3:
        coords_t = [coords + cell[:ndim]*np.array([i, j, k])
                    for k, j, i in product(np.arange(rep[2]),
                                           np.arange(rep[1]),
                                           np.arange(rep[0]))]
    # convert to np arrays
    coords_t, adj_new = np.vstack(coords_t), np.vstack(adj_new)
    # find overlapping nodes/coordinates
    overlaps = cKDTree(coords_t).query_pairs(r=overlap_tol,
                                             output_type='ndarray')
    # sort that first column greater than second
    overlaps = np.sort(overlaps, axis=1)
    # sort by first column
    overlaps = overlaps[np.argsort(overlaps[:, 0])]
    if debug:
        print(overlaps)
    # replace entries of overlapping nodes in adj
    _, first_ind = np.unique(overlaps[:, 1],
                             return_index=True)
    # second column redundant node, first column is the node with whom it should
    # be replaced
    node_dictionary = overlaps[first_ind, :]
    node_inds_new = np.arange(coords_t.shape[0])
    node_inds_new[node_dictionary[:, 1]] = node_dictionary[:, 0].copy()
    if debug:
        print(node_dictionary)
    for i, connect in enumerate(adj_new):
        mask = np.isin(connect, node_dictionary[:, 1])
        if mask.any():
            connect[mask] = node_inds_new[connect[mask]]
            adj_new[i] = connect
    # get unique connections
    adj_new = np.unique(adj_new, axis=0)
    # renumber adjacency matrix
    adj_new = renumber_adjacency(adj_new, node_dictionary[:, 1])
    if debug:
        print(adj_new.shape)
    # delete overlapping coordinates. Of the overlapping ones keep only the one
    # with the lowest index
    if debug:
        print(coords_t.shape)
    coords_t = np.delete(coords_t, node_dictionary[:, 1], axis=0)
    if debug:
        print(coords_t.shape)
    #
    cell[:ndim] = cell[:ndim] * rep[:ndim]
    return coords_t+shift[None, :], adj_new, cell


def renumber_adjacency(adj, indices_delete):
    """
    Renumber the indices by adjusting the counting for deleted nodes.

    Parameters
    ----------
    adj : np.array
        adjacency matrix.
    indices_delete : np.array
        indices to be deleted.

    Returns
    -------
    adj : np.array
        updated adjacency.

    """

    adj_mask = ~(np.isin(adj[:, 0], indices_delete) |
                 np.isin(adj[:, 1], indices_delete))

    val, ind = np.unique(adj, return_inverse=True)

    _mask = ~np.isin(val, indices_delete)
    val[_mask] = np.arange(_mask.sum())

    return val[ind].reshape(adj.shape)[adj_mask]


def sanity_checks(coords, adj, cell,
                  overlap=True, overlap_tol=1e-5,
                  connected=True,
                  duplicate_bonds=True):
    """Check a given network of nodes for consistency.

    Parameters
    ----------
    coords : np.ndarray
        coordinates of shape (n_nodes,ndim).
    adj : np.ndarray
        adjacency list of shape (n_bonds,2).
    cell : None or np.ndarray, optional
        cell lengths of shape (ndim). If None infer from coordinates (max-min).
    overlap : bool, optional
        check for overlapping nodes via cKDTree.
    overlap_tol : float, optional
        tolerance for detecting overlaps.
    connected : bool, optional
        check whether all nodes are connected.
    duplicate_bonds : bool, optional
        check for duplicate bonds.

    Raises
    -------
    AssertionError
        if any discrepancies (overlapping nodes, disconnected network,
        duplicate bonds) detected.
    """
    # check for overlaps
    if overlap:
        overlaps = cKDTree(coords).query_pairs(r=overlap_tol,
                                               output_type='ndarray')
        try:
            assert overlaps.size == 0
        except AssertionError:
            raise AssertionError("Overlapping nodes detected.")
    # check that lattice is connected
    if connected:

        graph = csr_matrix((np.ones(adj.shape[0]), (adj[:, 0], adj[:, 1])),
                           shape=(coords.shape[0], coords.shape[0]))
        graph = graph + graph.T
        n_comp, labels = connected_components(graph)
        try:
            assert n_comp == 1
        except AssertionError:
            print(coords.shape)
            print(adj.shape)
            raise AssertionError(f"Network disconnected with {n_comp} connected components")

    # check for doubled bonds
    if duplicate_bonds:
        # sort that first column greater than second
        adj = np.sort(adj, axis=1)
        # sort by first column
        adj = adj[np.argsort(adj[:, 0])]
        try:
            n_dupls = int(np.unique(adj, axis=0).shape[0] - adj.shape[0])
            assert n_dupls == 0
        except AssertionError:
            raise AssertionError(f"{n_dupls} duplicate bonds detected.")
    return
