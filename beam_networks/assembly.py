from tqdm import tqdm
import numpy as np
import scipy.sparse as sp

from beam_networks.stiffness import get_element_stiffness_global, get_element_stiffness_global_vec


def assemble_global_system(nodes_positions, edges_indices, dr, beam_prop, sorted_edges=True,
                           vectorize=True, matrix='bsr', verbose=False):
    """Assembly of the global stiffness matrix for a Timoshenko beam network (wrapper)

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

    Returns
    -------
    np.ndarray or scipy.sparse.bsr_array
        The global matrix
    """

    if not sorted_edges:
        edges_indices = np.sort(edges_indices, axis=1)
        edges_indices = edges_indices[np.lexsort((edges_indices[:, 1], edges_indices[:, 0]))]

    edges_indices = np.array(edges_indices)

    if vectorize:
        if matrix == 'bsr':
            K_global = _assemble_sparse_bsr_vec(nodes_positions, edges_indices, dr, beam_prop, verbose=verbose)
        elif matrix == 'lil':
            K_global = _assemble_sparse_lil_vec(nodes_positions, edges_indices, dr, beam_prop, verbose=verbose)
        elif matrix == 'dense':
            K_global = _assemble_dense_vec(nodes_positions, edges_indices, dr,  beam_prop, verbose=verbose)
        else:
            raise ValueError
    else:
        if matrix == 'bsr':
            K_global = _assemble_sparse_bsr(nodes_positions, edges_indices, dr, beam_prop, verbose=verbose)
        elif matrix == 'lil':
            K_global = _assemble_sparse_lil(nodes_positions, edges_indices, dr, beam_prop, verbose=verbose)
        elif matrix == 'dense':
            K_global = _assemble_dense(nodes_positions, edges_indices, dr, beam_prop, verbose=verbose)
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
    num_dof_per_node = 3 * (ndim - 1)
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
            Ke = get_element_stiffness_global(beam_prop, dr[i])

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
    scipy.sparse.bsr_array
        The global stiffness matrix
    """

    if verbose:
        pbar = tqdm(desc="Assemble BSR matrix (vectorized)", total=edges.shape[0], ncols=100)

    num_nodes, ndim = nodes.shape
    num_dof_per_node = 3 * (ndim - 1)
    num_dof = num_nodes * num_dof_per_node

    aux = sp.csr_array((np.ones_like(edges[:, 0]),
                        (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
    aux = aux + sp.eye_array(num_nodes)

    indices = aux.indices
    indptr = aux.indptr
    data = np.zeros((len(indices), num_dof_per_node, num_dof_per_node))

    Ke = get_element_stiffness_global_vec(beam_prop, dr)

    i = 0
    n0s, c0s = np.unique(edges[:, 0], return_counts=True)

    for n0, c0 in zip(n0s, c0s):
        for k, n1 in enumerate(edges[i + np.arange(c0), 1]):
            Ke00 = Ke[i, :num_dof_per_node, :num_dof_per_node]
            Ke01 = Ke[i, :num_dof_per_node, num_dof_per_node:]
            Ke11 = Ke[i, num_dof_per_node:, num_dof_per_node:]

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
    num_dof_per_node = 3 * (ndim - 1)
    num_dof = num_nodes * num_dof_per_node

    K_global = sp.lil_array((num_dof, num_dof))

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        Ke = get_element_stiffness_global(beam_prop, dr[i])

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
    num_dof_per_node = 3 * (ndim - 1)
    num_dof = num_nodes * num_dof_per_node

    K_global = sp.lil_array((num_dof, num_dof))

    Ke = get_element_stiffness_global_vec(beam_prop, dr)

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
    num_dof_per_node = 3 * (ndim - 1)
    num_dof = num_nodes * num_dof_per_node

    K_global = np.zeros((num_dof, num_dof))

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        Ke = get_element_stiffness_global(beam_prop, dr[i])

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
    num_dof_per_node = 3 * (ndim - 1)
    num_dof = num_nodes * num_dof_per_node

    K_global = np.zeros((num_dof, num_dof))
    Ke = get_element_stiffness_global_vec(beam_prop, dr)

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
