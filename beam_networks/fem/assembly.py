#
# Copyright 2025-2026 Hannes Holey
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
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp

from beam_networks.fem.stiffness import (
    global_element_stiffness_timoshenko_exact_single,
    global_element_stiffness_timoshenko_exact_all,
    global_element_stiffness_euler_exact_single,
    global_element_stiffness_euler_exact_all,
    global_element_stiffness_euler_numeric_single,
    global_element_stiffness_euler_numeric_all,
    global_element_stiffness_timoshenko_numeric_single,
    global_element_stiffness_timoshenko_numeric_all,
    global_element_stiffness_truss_exact_single,
    global_element_stiffness_truss_exact_all)


def _dof_per_node(ndim, beam_prop):
    """Return the number of DOFs per node.

    Parameters
    ----------
    ndim : int
        Spatial dimension (2 or 3).
    beam_prop : dict
        Beam properties; the ``'truss'`` key selects pin-jointed bar elements.

    Returns
    -------
    int
        ``ndim`` for trusses, ``3 * (ndim - 1)`` for beams.
    """
    if beam_prop.get('truss', False):
        return ndim
    return 3 * (ndim - 1)


def assemble_global_system(nodes_positions: np.ndarray,
                           edges_indices: np.ndarray,
                           dr: np.ndarray,
                           beam_prop: dict,
                           sorted_edges: bool = True,
                           vectorize: bool = True,
                           matrix: str = 'bsr',
                           verbose: bool = False,
                           n_elems: np.ndarray | None = None,
                           fem_poly_order: int = 3,
                           fem_n_gauss: int | None = None):
    """Assemble the global stiffness matrix for a Timoshenko beam network.

    Parameters
    ----------
    nodes_positions : np.ndarray
        Nodal coordinates
    edges_indices : np.ndarray (of ints)
        Edge connectivity
    dr : np.ndarray
        Edge vectors
    beam_prop : dict
        Beam properties (cross section and elastic properties)
    sorted_edges : bool, optional
        Edge indices are sorted (the default is True)
    vectorize : bool, optional
        Collect all element stiffness matrices in a large array before assembly.
        Faster, but may cause memory issues for very large systems (the default is True).
    matrix : str, optional
        Format of the global stiffness matrix during assembly.
        Choose from ['bsr', 'lil', 'dense'].
        'bsr' : Block sparse row matrix (Default)
        'lil' : List of lists
        'dense': Only for small matrices
    verbose : bool, optional
        Verbosity (the default is False which hides the progress bar)
    n_elems : np.ndarray or None, optional
        Per-edge number of FEM sub-elements, shape (num_edges,).
        When None (default), uses the exact Timoshenko stiffness matrix.
        When provided, each beam is discretized into the given number of
        sub-elements and the interior DOFs are eliminated via static
        condensation before assembly.
    fem_poly_order : int, optional
        Polynomial degree of the Lagrange shape functions used in each
        FEM sub-element (default 3 = cubic).  Ignored when ``n_elems``
        is None.
    fem_n_gauss : int or None, optional
        Number of Gauss-Legendre quadrature points per sub-element.
        None (default) selects ``fem_poly_order`` (reduced integration),
        which avoids shear locking for slender Timoshenko beams.
        Ignored when ``n_elems`` is None.

    Returns
    -------
    np.ndarray or scipy.sparse.bsr_array
        The global matrix
    """

    if not sorted_edges:
        edges_indices = np.sort(edges_indices, axis=1)
        edges_indices = edges_indices[np.lexsort((edges_indices[:, 1], edges_indices[:, 0]))]

    edges_indices = np.array(edges_indices)

    # Stiffness matrix assembled via numerical integration
    if n_elems is not None:
        # Vectorized versions compute all element stiffness matrices at once before assembly
        if vectorize:
            if matrix == 'bsr':
                K_global = _assemble_sparse_bsr_fem_vec(nodes_positions,
                                                        edges_indices,
                                                        dr,
                                                        beam_prop,
                                                        n_elems,
                                                        fem_poly_order=fem_poly_order,
                                                        fem_n_gauss=fem_n_gauss,
                                                        verbose=verbose)
            elif matrix == 'lil':
                K_global = _assemble_sparse_lil_fem_vec(nodes_positions,
                                                        edges_indices,
                                                        dr,
                                                        beam_prop,
                                                        n_elems,
                                                        fem_poly_order=fem_poly_order,
                                                        fem_n_gauss=fem_n_gauss,
                                                        verbose=verbose)
            elif matrix == 'dense':
                K_global = _assemble_dense_fem_vec(nodes_positions,
                                                   edges_indices,
                                                   dr,
                                                   beam_prop,
                                                   n_elems,
                                                   fem_poly_order=fem_poly_order,
                                                   fem_n_gauss=fem_n_gauss,
                                                   verbose=verbose)
            else:
                raise ValueError

        # 'Loop'-assembly (slower)
        elif matrix == 'bsr':
            K_global = _assemble_sparse_bsr_fem(nodes_positions,
                                                edges_indices,
                                                dr,
                                                beam_prop,
                                                n_elems,
                                                fem_poly_order=fem_poly_order,
                                                fem_n_gauss=fem_n_gauss,
                                                verbose=verbose)
        elif matrix == 'lil':
            K_global = _assemble_sparse_lil_fem(nodes_positions,
                                                edges_indices,
                                                dr,
                                                beam_prop,
                                                n_elems,
                                                fem_poly_order=fem_poly_order,
                                                fem_n_gauss=fem_n_gauss,
                                                verbose=verbose)
        elif matrix == 'dense':
            K_global = _assemble_dense_fem(nodes_positions,
                                           edges_indices,
                                           dr,
                                           beam_prop,
                                           n_elems,
                                           fem_poly_order=fem_poly_order,
                                           fem_n_gauss=fem_n_gauss,
                                           verbose=verbose)
        else:
            raise ValueError

    # Exact stiffness matrix for two node elements
    elif vectorize:
        if matrix == 'bsr':
            K_global = _assemble_sparse_bsr_vec(nodes_positions,
                                                edges_indices,
                                                dr,
                                                beam_prop,
                                                verbose=verbose)
        elif matrix == 'lil':
            K_global = _assemble_sparse_lil_vec(nodes_positions,
                                                edges_indices,
                                                dr,
                                                beam_prop,
                                                verbose=verbose)
        elif matrix == 'dense':
            K_global = _assemble_dense_vec(nodes_positions,
                                           edges_indices,
                                           dr,
                                           beam_prop,
                                           verbose=verbose)
        else:
            raise ValueError
    else:
        if matrix == 'bsr':
            K_global = _assemble_sparse_bsr(nodes_positions,
                                            edges_indices,
                                            dr,
                                            beam_prop,
                                            verbose=verbose)
        elif matrix == 'lil':
            K_global = _assemble_sparse_lil(nodes_positions,
                                            edges_indices,
                                            dr,
                                            beam_prop,
                                            verbose=verbose)
        elif matrix == 'dense':
            K_global = _assemble_dense(nodes_positions,
                                       edges_indices,
                                       dr,
                                       beam_prop,
                                       verbose=verbose)
        else:
            raise ValueError

    return K_global


def _assemble_sparse_bsr(nodes, edges, dr, beam_prop, verbose=True):
    """Assemble global stiffness matrix in BSR format.

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray (of ints)
        Edge connectivity
    dr : np.ndarray
        Edge vectors
    beam_prop : dict
        Beam properties (cross section and elastic properties)
    verbose : bool, optional
        Verbosity (the default is False which hides the progress bar)

    Returns
    -------
    scipy.sparse.bsr_array
        The global stiffness matrix
    """

    if verbose:
        pbar = tqdm(desc="Assemble BSR matrix", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    aux = sp.csr_array((np.ones_like(edges[:, 0]),
                        (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
    aux = aux + sp.eye_array(num_nodes)

    indices = aux.indices
    indptr = aux.indptr
    data = np.zeros((len(indices), num_dof_per_node, num_dof_per_node))

    i = 0
    n0s, c0s = np.unique(edges[:, 0], return_counts=True)

    for n0, c0 in zip(n0s, c0s):
        for k, n1 in enumerate(edges[i + np.arange(c0), 1]):
            if beam_prop.get('truss', False):
                Ke = global_element_stiffness_truss_exact_single(beam_prop, dr[i])
            elif beam_prop.get('euler_bernoulli', False):
                Ke = global_element_stiffness_euler_exact_single(beam_prop, dr[i])
            else:
                Ke = global_element_stiffness_timoshenko_exact_single(beam_prop, dr[i])

            Ke00 = Ke[:num_dof_per_node, :num_dof_per_node]
            Ke01 = Ke[:num_dof_per_node, num_dof_per_node:]
            Ke11 = Ke[num_dof_per_node:, num_dof_per_node:]

            # factor 1/2 to make symmetric later
            data[indptr[n0]] += Ke00 / 2.
            data[indptr[n1]] += Ke11 / 2.
            data[indptr[n0] + k + 1] += Ke01
            i += 1
            if verbose:
                pbar.update(1)

    # Create block sparse array (upper triangular)
    K_global = sp.bsr_array((data, indices, indptr),
                            shape=(num_dof, num_dof),
                            blocksize=(num_dof_per_node, num_dof_per_node)
                            )
    # Make symmetric
    K_global = K_global + K_global.T

    return K_global


def _assemble_sparse_bsr_vec(nodes, edges, dr, beam_prop, verbose=True):
    """Assemble global stiffness matrix in BSR format.

    All element stiffness matrices are computed at once, then scattered into
    the BSR data array using ``np.add.at``.

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray of int
        Edge connectivity (sorted, ``edges[:, 0] < edges[:, 1]``)
    dr : np.ndarray
        Edge vectors
    beam_prop : dict
        Beam properties (cross section and elastic properties)
    verbose : bool, optional
        Unused; kept for API compatibility.

    Returns
    -------
    scipy.sparse.bsr_array
        The global stiffness matrix
    """
    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    aux = sp.csr_array((np.ones_like(edges[:, 0]),
                        (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
    aux = aux + sp.eye_array(num_nodes)

    indices = aux.indices
    indptr = aux.indptr
    data = np.zeros((len(indices), num_dof_per_node, num_dof_per_node))

    if beam_prop.get('truss', False):
        Ke = global_element_stiffness_truss_exact_all(beam_prop, dr)
    elif beam_prop.get('euler_bernoulli', False):
        Ke = global_element_stiffness_euler_exact_all(beam_prop, dr)
    else:
        Ke = global_element_stiffness_timoshenko_exact_all(beam_prop, dr)

    e0s, e1s = edges[:, 0], edges[:, 1]

    # Diagonal block positions (diagonal is always the first entry in each row
    # because e0 < e1 for all edges, making the diagonal the smallest column)
    diag_pos_n0 = indptr[e0s]
    diag_pos_n1 = indptr[e1s]

    # Off-diagonal block positions: rank of each edge within its source row
    _, first_occ, counts = np.unique(e0s, return_index=True, return_counts=True)
    k_per_edge = np.arange(len(e0s)) - np.repeat(first_occ, counts)
    offdiag_pos = indptr[e0s] + 1 + k_per_edge

    ndpn = num_dof_per_node
    np.add.at(data, diag_pos_n0, Ke[:, :ndpn, :ndpn] / 2.)
    np.add.at(data, diag_pos_n1, Ke[:, ndpn:, ndpn:] / 2.)
    np.add.at(data, offdiag_pos, Ke[:, :ndpn, ndpn:])

    K_global = sp.bsr_array((data, indices, indptr),
                            shape=(num_dof, num_dof),
                            blocksize=(num_dof_per_node, num_dof_per_node))
    return K_global + K_global.T


def _assemble_sparse_lil(nodes, edges, dr, beam_prop, verbose=True):
    """Assemble global stiffness matrix in LIL format.

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray (of ints)
        Edge connectivity
    dr : np.ndarray
        Edge vectors
    beam_prop : dict
        Beam properties (cross section and elastic properties)
    verbose : bool, optional
        Verbosity (the default is False which hides the progress bar)

    Returns
    -------
    scipy.sparse.bsr_array
        The global stiffness matrix
    """

    if verbose:
        pbar = tqdm(desc="Assemble LIL matrix", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    K_global = sp.lil_array((num_dof, num_dof))

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        if beam_prop.get('truss', False):
            Ke = global_element_stiffness_truss_exact_single(beam_prop, dr[i])
        else:
            Ke = global_element_stiffness_timoshenko_exact_single(beam_prop, dr[i])

        K_global[s1, s1] += Ke[:num_dof_per_node, :num_dof_per_node] / 2.
        K_global[s1, s2] += Ke[:num_dof_per_node, num_dof_per_node:]
        K_global[s2, s2] += Ke[num_dof_per_node:, num_dof_per_node:] / 2.
        if verbose:
            pbar.update(1)

    K_global = K_global + K_global.T

    return K_global.tobsr()


def _assemble_sparse_lil_vec(nodes, edges, dr, beam_prop, verbose=True):
    """Assemble global stiffness matrix in LIL format.

    Vectorized version

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray (of ints)
        Edge connectivity
    dr : np.ndarray
        Edge vectors
    beam_prop : dict
        Beam properties (cross section and elastic properties)
    verbose : bool, optional
        Verbosity (the default is False which hides the progress bar)

    Returns
    -------
    scipy.sparse.bsr_array
        The global stiffness matrix
    """

    if verbose:
        pbar = tqdm(desc="Assemble LIL matrix (vectorized)", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    K_global = sp.lil_array((num_dof, num_dof))

    if beam_prop.get('truss', False):
        Ke = global_element_stiffness_truss_exact_all(beam_prop, dr)
    elif beam_prop.get('euler_bernoulli', False):
        Ke = global_element_stiffness_euler_exact_all(beam_prop, dr)
    else:
        Ke = global_element_stiffness_timoshenko_exact_all(beam_prop, dr)

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        K_global[s1, s1] += Ke[i, :num_dof_per_node, :num_dof_per_node] / 2.
        K_global[s1, s2] += Ke[i, :num_dof_per_node, num_dof_per_node:]
        K_global[s2, s2] += Ke[i, num_dof_per_node:, num_dof_per_node:] / 2.
        if verbose:
            pbar.update(1)

    K_global = K_global + K_global.T

    return K_global.tobsr()


def _assemble_dense(nodes, edges, dr, beam_prop, verbose=True):
    """Assemble global stiffness matrix as dense array.

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray (of ints)
        Edge connectivity
    dr : np.ndarray
        Edge vectors
    beam_prop : dict
        Beam properties (cross section and elastic properties)
    verbose : bool, optional
        Verbosity (the default is False which hides the progress bar)

    Returns
    -------
    np.ndrarray
        The global stiffness matrix
    """

    if verbose:
        pbar = tqdm(desc="Assemble dense matrix", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    K_global = np.zeros((num_dof, num_dof))

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        if beam_prop.get('truss', False):
            Ke = global_element_stiffness_truss_exact_single(beam_prop, dr[i])
        else:
            Ke = global_element_stiffness_timoshenko_exact_single(beam_prop, dr[i])

        K_global[s1, s1] += Ke[:num_dof_per_node, :num_dof_per_node]
        K_global[s1, s2] += Ke[:num_dof_per_node, num_dof_per_node:]
        K_global[s2, s1] += Ke[num_dof_per_node:, :num_dof_per_node]
        K_global[s2, s2] += Ke[num_dof_per_node:, num_dof_per_node:]
        if verbose:
            pbar.update(1)

    return K_global


def _assemble_dense_vec(nodes, edges, dr, beam_prop, verbose=True):
    """Assemble global stiffness matrix as dense array.

    Vectorized version.

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray (of ints)
        Edge connectivity
    dr : np.ndarray
        Edge vectors
    beam_prop : dict
        Beam properties (cross section and elastic properties)
    verbose : bool, optional
        Verbosity (the default is False which hides the progress bar)

    Returns
    -------
    np.ndrarray
        The global stiffness matrix
    """

    if verbose:
        pbar = tqdm(desc="Assemble dense matrix (vectorized)", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    K_global = np.zeros((num_dof, num_dof))
    if beam_prop.get('truss', False):
        Ke = global_element_stiffness_truss_exact_all(beam_prop, dr)
    elif beam_prop.get('euler_bernoulli', False):
        Ke = global_element_stiffness_euler_exact_all(beam_prop, dr)
    else:
        Ke = global_element_stiffness_timoshenko_exact_all(beam_prop, dr)

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        K_global[s1, s1] += Ke[i, :num_dof_per_node, :num_dof_per_node]
        K_global[s1, s2] += Ke[i, :num_dof_per_node, num_dof_per_node:]
        K_global[s2, s1] += Ke[i, num_dof_per_node:, :num_dof_per_node]
        K_global[s2, s2] += Ke[i, num_dof_per_node:, num_dof_per_node:]
        if verbose:
            pbar.update(1)

    return K_global


def _assemble_sparse_bsr_fem(nodes, edges, dr, beam_prop, n_elems,
                             fem_poly_order=1, fem_n_gauss=None, verbose=True):
    """Assemble global stiffness matrix in BSR format using FEM sub-elements."""

    if verbose:
        pbar = tqdm(desc="Assemble BSR matrix (FEM)", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    aux = sp.csr_array((np.ones_like(edges[:, 0]),
                        (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
    aux = aux + sp.eye_array(num_nodes)

    indices = aux.indices
    indptr = aux.indptr
    data = np.zeros((len(indices), num_dof_per_node, num_dof_per_node))

    i = 0
    n0s, c0s = np.unique(edges[:, 0], return_counts=True)

    for n0, c0 in zip(n0s, c0s):
        for k, n1 in enumerate(edges[i + np.arange(c0), 1]):
            if beam_prop.get('euler_bernoulli', False):
                Ke = global_element_stiffness_euler_numeric_single(
                    beam_prop, dr[i], n_elems[i])
            else:
                Ke = global_element_stiffness_timoshenko_numeric_single(
                    beam_prop, dr[i], n_elems[i],
                    poly_order=fem_poly_order,
                    n_gauss=fem_n_gauss)

            Ke00 = Ke[:num_dof_per_node, :num_dof_per_node]
            Ke01 = Ke[:num_dof_per_node, num_dof_per_node:]
            Ke11 = Ke[num_dof_per_node:, num_dof_per_node:]

            data[indptr[n0]] += Ke00 / 2.
            data[indptr[n1]] += Ke11 / 2.
            data[indptr[n0] + k + 1] += Ke01
            i += 1
            if verbose:
                pbar.update(1)

    K_global = sp.bsr_array((data, indices, indptr),
                            shape=(num_dof, num_dof),
                            blocksize=(num_dof_per_node, num_dof_per_node))
    K_global = K_global + K_global.T

    return K_global


def _assemble_sparse_lil_fem(nodes, edges, dr, beam_prop, n_elems,
                             fem_poly_order=1, fem_n_gauss=None, verbose=True):
    """Assemble global stiffness matrix in LIL format using FEM sub-elements."""

    if verbose:
        pbar = tqdm(desc="Assemble LIL matrix (FEM)", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    K_global = sp.lil_array((num_dof, num_dof))

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        Ke = global_element_stiffness_timoshenko_numeric_single(
            beam_prop, dr[i], n_elems[i],
            poly_order=fem_poly_order,
            n_gauss=fem_n_gauss)

        K_global[s1, s1] += Ke[:num_dof_per_node, :num_dof_per_node] / 2.
        K_global[s1, s2] += Ke[:num_dof_per_node, num_dof_per_node:]
        K_global[s2, s2] += Ke[num_dof_per_node:, num_dof_per_node:] / 2.
        if verbose:
            pbar.update(1)

    K_global = K_global + K_global.T

    return K_global.tobsr()


def _assemble_dense_fem(nodes, edges, dr, beam_prop, n_elems,
                        fem_poly_order=1, fem_n_gauss=None, verbose=True):
    """Assemble global stiffness matrix as dense array using FEM sub-elements."""

    if verbose:
        pbar = tqdm(desc="Assemble dense matrix (FEM)", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    K_global = np.zeros((num_dof, num_dof))

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        Ke = global_element_stiffness_timoshenko_numeric_single(
            beam_prop, dr[i], n_elems[i],
            poly_order=fem_poly_order,
            n_gauss=fem_n_gauss)

        K_global[s1, s1] += Ke[:num_dof_per_node, :num_dof_per_node]
        K_global[s1, s2] += Ke[:num_dof_per_node, num_dof_per_node:]
        K_global[s2, s1] += Ke[num_dof_per_node:, :num_dof_per_node]
        K_global[s2, s2] += Ke[num_dof_per_node:, num_dof_per_node:]
        if verbose:
            pbar.update(1)

    return K_global


def _assemble_sparse_bsr_fem_vec(nodes, edges, dr, beam_prop, n_elems,
                                 fem_poly_order=1, fem_n_gauss=None, verbose=True):
    """Assemble BSR stiffness (FEM sub-elements): batch computation + scatter via np.add.at."""
    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    aux = sp.csr_array((np.ones_like(edges[:, 0]),
                        (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
    aux = aux + sp.eye_array(num_nodes)

    indices = aux.indices
    indptr = aux.indptr
    data = np.zeros((len(indices), num_dof_per_node, num_dof_per_node))

    if beam_prop.get('euler_bernoulli', False):
        Ke = global_element_stiffness_euler_numeric_all(beam_prop, dr, n_elems)
    else:
        Ke = global_element_stiffness_timoshenko_numeric_all(
            beam_prop, dr, n_elems,
            poly_order=fem_poly_order,
            n_gauss=fem_n_gauss)

    e0s, e1s = edges[:, 0], edges[:, 1]

    diag_pos_n0 = indptr[e0s]
    diag_pos_n1 = indptr[e1s]

    _, first_occ, counts = np.unique(e0s, return_index=True, return_counts=True)
    k_per_edge = np.arange(len(e0s)) - np.repeat(first_occ, counts)
    offdiag_pos = indptr[e0s] + 1 + k_per_edge

    ndpn = num_dof_per_node
    np.add.at(data, diag_pos_n0, Ke[:, :ndpn, :ndpn] / 2.)
    np.add.at(data, diag_pos_n1, Ke[:, ndpn:, ndpn:] / 2.)
    np.add.at(data, offdiag_pos, Ke[:, :ndpn, ndpn:])

    K_global = sp.bsr_array((data, indices, indptr),
                            shape=(num_dof, num_dof),
                            blocksize=(num_dof_per_node, num_dof_per_node))
    return K_global + K_global.T


def _assemble_sparse_lil_fem_vec(nodes, edges, dr, beam_prop, n_elems,
                                 fem_poly_order=1, fem_n_gauss=None, verbose=True):
    """Assemble global stiffness matrix in LIL format using FEM sub-elements.

    Vectorized version.
    """

    if verbose:
        pbar = tqdm(desc="Assemble LIL matrix (FEM, vectorized)", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    K_global = sp.lil_array((num_dof, num_dof))

    if beam_prop.get('euler_bernoulli', False):
        Ke = global_element_stiffness_euler_numeric_all(beam_prop, dr, n_elems)
    else:
        Ke = global_element_stiffness_timoshenko_numeric_all(
            beam_prop, dr, n_elems,
            poly_order=fem_poly_order,
            n_gauss=fem_n_gauss)

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        K_global[s1, s1] += Ke[i, :num_dof_per_node, :num_dof_per_node] / 2.
        K_global[s1, s2] += Ke[i, :num_dof_per_node, num_dof_per_node:]
        K_global[s2, s2] += Ke[i, num_dof_per_node:, num_dof_per_node:] / 2.
        if verbose:
            pbar.update(1)

    K_global = K_global + K_global.T

    return K_global.tobsr()


def _assemble_dense_fem_vec(nodes, edges, dr, beam_prop, n_elems,
                            fem_poly_order=1, fem_n_gauss=None, verbose=True):
    """Assemble global stiffness matrix as dense array using FEM sub-elements.

    Vectorized version.
    """

    if verbose:
        pbar = tqdm(desc="Assemble dense matrix (FEM, vectorized)", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = _dof_per_node(ndim, beam_prop)
    num_dof = num_nodes * num_dof_per_node

    K_global = np.zeros((num_dof, num_dof))

    if beam_prop.get('euler_bernoulli', False):
        Ke = global_element_stiffness_euler_numeric_all(beam_prop, dr, n_elems)
    else:
        Ke = global_element_stiffness_timoshenko_numeric_all(
            beam_prop, dr, n_elems,
            poly_order=fem_poly_order,
            n_gauss=fem_n_gauss)

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        K_global[s1, s1] += Ke[i, :num_dof_per_node, :num_dof_per_node]
        K_global[s1, s2] += Ke[i, :num_dof_per_node, num_dof_per_node:]
        K_global[s2, s1] += Ke[i, num_dof_per_node:, :num_dof_per_node]
        K_global[s2, s2] += Ke[i, num_dof_per_node:, num_dof_per_node:]
        if verbose:
            pbar.update(1)

    return K_global
