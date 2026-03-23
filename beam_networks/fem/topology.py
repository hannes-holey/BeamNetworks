#
# Copyright 2026 Hannes Holey
#
# This file is part of beam_networks. beam_networks is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. beam_networks is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# beam_networks. If not, see <https://www.gnu.org/licenses/>.
#
"""Mesh-topology utilities for sparse matrix assembly.

These functions depend only on node count and edge connectivity — not on
spatial dimension, DOF count, or element formulation — and are shared by
both the linear (``fem.assembly``) and nonlinear (``fem.corotational``)
assembly routines.
"""
import numpy as np
import scipy.sparse as sp


def _build_bsr_node_sparsity(
        nodes: np.ndarray,
        edges: np.ndarray,
) -> tuple:
    """Build a full-symmetric BSR node-level sparsity pattern.

    Computes the node-level CSR structure (one entry per node pair that
    shares an edge, plus diagonal) and the per-element scatter positions
    needed to fill all four block types — diagonal n0, diagonal n1, upper
    off-diagonal (n0, n1) and lower off-diagonal (n1, n0) — directly
    without a subsequent ``K + K.T`` symmetrisation step.

    The result is independent of the number of DOFs per node and of the
    element formulation, so it can be reused for any block size.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, d)
        Nodal coordinates (only the row count N is used).
    edges : np.ndarray, shape (M, 2)
        Edge connectivity (integer node-index pairs).

    Returns
    -------
    edges_sorted : np.ndarray, shape (M, 2)
        Edges with ``n0 < n1``, sorted lexicographically by ``(n0, n1)``.
    indices : np.ndarray, shape (nnz,)
        CSR column indices for the full-symmetric node-level pattern.
    indptr : np.ndarray, shape (N+1,)
        CSR row pointer.
    diag_pos_n0 : np.ndarray of intp, shape (M,)
        Index into the BSR data array for the n0 diagonal block of each edge.
    diag_pos_n1 : np.ndarray of intp, shape (M,)
        Index into the BSR data array for the n1 diagonal block of each edge.
    offdiag_pos_upper : np.ndarray of intp, shape (M,)
        Index into the BSR data array for the upper off-diagonal (n0, n1)
        block of each edge.
    offdiag_pos_lower : np.ndarray of intp, shape (M,)
        Index into the BSR data array for the lower off-diagonal (n1, n0)
        block of each edge.
    """
    num_nodes = nodes.shape[0]
    edges_sorted = np.sort(edges, axis=1)
    edges_sorted = edges_sorted[np.lexsort((edges_sorted[:, 1], edges_sorted[:, 0]))]
    e0s, e1s = edges_sorted[:, 0], edges_sorted[:, 1]

    # Full symmetric pattern: both off-diagonal directions + diagonal
    aux = sp.csr_array(
        (np.ones(2 * len(edges_sorted)),
         (np.r_[e0s, e1s], np.r_[e1s, e0s])),
        shape=(num_nodes, num_nodes),
    )
    aux = aux + sp.eye_array(num_nodes)
    indices = aux.indices
    indptr = aux.indptr

    # Build (row, col) → data-index lookup (topology-setup cost only)
    row_arr = np.repeat(np.arange(num_nodes), np.diff(indptr))
    pos_of = dict(zip(zip(row_arr.tolist(), indices.tolist()), range(len(indices))))

    e0 = e0s.tolist()
    e1 = e1s.tolist()
    diag_pos_n0 = np.array([pos_of[n, n] for n in e0],              dtype=np.intp)
    diag_pos_n1 = np.array([pos_of[n, n] for n in e1],              dtype=np.intp)
    offdiag_pos_upper = np.array([pos_of[n0, n1] for n0, n1 in zip(e0, e1)], dtype=np.intp)
    offdiag_pos_lower = np.array([pos_of[n1, n0] for n0, n1 in zip(e0, e1)], dtype=np.intp)

    return (edges_sorted, indices, indptr,
            diag_pos_n0, diag_pos_n1,
            offdiag_pos_upper, offdiag_pos_lower)
