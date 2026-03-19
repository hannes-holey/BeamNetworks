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
"""Stiffness-matrix partitioning into free (F) and essential (E) DOF blocks.

The partitioned system arises from applying Dirichlet BCs: given the global
stiffness K and the free/prescribed DOF index sets, we need KFF, KFE and the
corresponding RHS contributions.

Two implementations are available via the *method* parameter of
:func:`partition_stiffness`:

``'csr_slice'`` (default)
    Convert BSR→CSR once and extract submatrices by direct row/column index
    slicing.  Faster than ``'sel_mat'`` for both small and large systems;
    ``'sel_mat'`` only wins in a narrow ~5k–20k DOF band.
    See ``examples/bench/bench_partitioning.py`` for the comparison.

``'sel_mat'``
    Build explicit sparse selection matrices L_E, L_F and compute
    KFF = LFs.T @ K @ LFs, KFE = LFs.T @ K @ LEs via SpMM.
    Kept for benchmarking purposes.
"""
import numpy as np
import scipy.sparse as sp

_PARTITION_METHODS = frozenset({'csr_slice', 'sel_mat'})


def partition_stiffness(K, bc_D, f, method='csr_slice'):
    """Partition a sparse stiffness matrix and load vector by DOF type.

    Parameters
    ----------
    K : scipy.sparse matrix
        Global stiffness matrix, shape (num_dof, num_dof).
    bc_D : array-like of int
        Indices of Dirichlet-constrained (essential) DOFs.
    f : np.ndarray
        Global load vector, shape (num_dof,).
    method : {'csr_slice', 'sel_mat'}, optional
        Partitioning back-end.  ``'csr_slice'`` converts BSR→CSR once and
        uses index slicing; ``'sel_mat'`` builds explicit sparse selection
        matrices and uses SpMM.  The default is ``'csr_slice'``.

    Returns
    -------
    KFF : scipy.sparse.csr_array
        Free–free stiffness block.
    KFE : scipy.sparse.csr_array
        Free–essential stiffness block.
    f_free : np.ndarray
        Load vector entries at free DOFs (``f[free_dofs]``).
    free_dofs : np.ndarray of int
        Indices of unconstrained DOFs.
    """
    if method not in _PARTITION_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {sorted(_PARTITION_METHODS)}")

    num_dof = K.shape[0]
    bc_D = np.asarray(bc_D)
    free_dofs = np.delete(np.arange(num_dof), bc_D)

    if method == 'csr_slice':
        K_csr = K.tocsr()
        K_free_rows = K_csr[free_dofs, :]
        KFF = K_free_rows[:, free_dofs]
        KFE = K_free_rows[:, bc_D]
    else:  # sel_mat
        n_D = len(bc_D)
        n_F = num_dof - n_D
        LEs = sp.bsr_array((np.ones(n_D), (bc_D, np.arange(n_D))), shape=(num_dof, n_D))
        LFs = sp.bsr_array((np.ones(n_F), (free_dofs, np.arange(n_F))), shape=(num_dof, n_F))
        KFF = LFs.T.dot(K.dot(LFs))
        KFE = LFs.T.dot(K.dot(LEs))

    return KFF, KFE, f[free_dofs], free_dofs


def scatter_solution(dF, bc_D, d_D, num_dof, free_dofs):
    """Assemble the full DOF solution vector from the free-DOF solution.

    Parameters
    ----------
    dF : np.ndarray
        Solution at free DOFs.
    bc_D : array-like of int
        Indices of Dirichlet-constrained DOFs.
    d_D : array-like of float
        Prescribed values at *bc_D*.
    num_dof : int
        Total number of DOFs.
    free_dofs : np.ndarray of int
        Indices of unconstrained DOFs (as returned by :func:`partition_stiffness`).

    Returns
    -------
    d : np.ndarray
        Global displacement vector, shape (num_dof,).
    """
    d = np.zeros(num_dof)
    d[np.asarray(bc_D)] = d_D
    d[free_dofs] = dF
    return d
