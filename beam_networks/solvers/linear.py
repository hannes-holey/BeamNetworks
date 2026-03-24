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
import numpy as np
import scipy.sparse as sp

from beam_networks.log import get_logger
from beam_networks.fem.partitioning import partition_stiffness, scatter_solution

_logger = get_logger("solvers.linear")


# Solvers that apply Jacobi scaling and use an iterative Krylov method.
_ITERATIVE_SOLVERS = frozenset({'cg', 'ilu', 'ssor', 'amg', 'amg_rs'})

# All valid solver names.
_ALL_SOLVERS = frozenset({'direct', 'cholesky'}) | _ITERATIVE_SOLVERS


def solve(K, bc_D: list, d_D: list, bc_N: list, F_N: list,
          solver: str = 'direct', verbosity: int = 0,
          tol: float = 1e-10,
          scale: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve the partitioned linear elastic system K·d = f.

    Dispatches to a sparse or dense backend depending on the type of *K*.

    Parameters
    ----------
    K : np.ndarray or scipy.sparse matrix
        Global stiffness matrix, shape (num_dof, num_dof).
    bc_D : list of int
        DOF indices with prescribed displacements (Dirichlet BCs).
    d_D : list of float
        Prescribed displacement values corresponding to *bc_D*.
    bc_N : list of int
        DOF indices with applied forces (Neumann BCs).
    F_N : list of float
        Applied force values corresponding to *bc_N*.
    solver : str, optional
        Linear solver.  Available options:

        * ``'direct'``    — sparse LU via ``spsolve`` (default)
        * ``'cholesky'``  — sparse Cholesky via CHOLMOD (requires
          ``scikit-sparse``); best for repeated solves with the same
          sparsity pattern (e.g. fracture simulations)
        * ``'cg'``        — unpreconditioned conjugate gradient
        * ``'ilu'``       — CG preconditioned with incomplete LU (ILU)
        * ``'ssor'``      — CG preconditioned with SSOR (ω = 1)
        * ``'amg'``       — CG preconditioned with smoothed-aggregation
          algebraic multigrid (requires ``pyamg``); optimal O(N) for
          homogeneous lattices
        * ``'amg_rs'``    — CG preconditioned with Ruge–Stüben AMG
          (requires ``pyamg``); better for heterogeneous / diluted networks

        The default is ``'direct'``.
    verbosity : int, optional
        Diagnostic output level (sparse solver only); see
        :meth:`~beam_networks.problem.ElasticNetwork.solve` for details.
        The default is 0.
    tol : float, optional
        Relative convergence tolerance for iterative solvers.  Convergence is
        declared when ``‖r‖ / ‖b‖ < tol`` in the scaled system.
        The default is 1e-10.
    scale : bool, optional
        Apply symmetric Jacobi (diagonal) scaling to the reduced system before
        solving.  This normalises all diagonal entries to 1, which improves
        convergence for iterative solvers on ill-conditioned systems.
        The default is ``True``.

    Returns
    -------
    d : np.ndarray
        Global displacement solution vector, shape (num_dof,).
    F : np.ndarray
        Global force vector (includes reaction forces at constrained DOFs),
        shape (num_dof,).
    info : int
        Solver status: 0 on success, non-zero on failure (e.g. singular matrix
        for the direct solver or non-convergence for CG).
    """

    if sp.issparse(K):
        d, F, info = _solve_sparse(K, bc_D, d_D, bc_N, F_N, solver=solver,
                                   verbosity=verbosity, tol=tol, scale=scale)
    else:
        d, F, info = _solve_dense(K, bc_D, d_D, bc_N, F_N)

    return d, F, info


def _solve_sparse(K_global, bc_D, d_D, bc_N, F_N,
                  solver='direct', verbosity=0, tol=1e-10, scale=True):
    """Solve sparse system.

    Parameters
    ----------
    K_global : scipy.sparse matrix
        Global stiffness matrix.
    bc_D : iterable
        List of constraint DOFs (Dirichlet BCs).
    d_D : iterable
        List of constraint DOF values (Dirichlet BCs).
    bc_N : iterable
        List of DOFs with nonzero loads (Neumann BCs).
    F_N : iterable
        List of DOF values with nonzero loads (Neumann BCs).
    solver : str, optional
        See :func:`solve` for the full list of options.
    verbosity : int, optional
        Verbosity level (default 0).
    tol : float, optional
        Relative convergence tolerance for iterative solvers (default 1e-10).

    Returns
    -------
    d : np.ndarray
        Global solution vector.
    F : np.ndarray
        Global load vector.
    info : int
        0 on success, non-zero on failure.
    """

    if solver not in _ALL_SOLVERS:
        raise ValueError(
            f"Unknown solver '{solver}'. Choose from: {sorted(_ALL_SOLVERS)}")

    num_dof = K_global.shape[0]

    bc_D = np.asarray(bc_D)
    d_D = np.asarray(d_D, dtype=float)

    # Global force vector
    f = np.zeros(num_dof)
    if len(F_N) > 0:
        f[bc_N] = F_N

    if verbosity >= 50 and verbosity < 100:
        _logger.debug("Global stiffness matrix statistics")
        _logger.debug("min(|K|): %s", K_global.data.min())
        _logger.debug("median(|K|): %s", np.median(K_global.data))
        _logger.debug("max(|K|): %s", K_global.data.max())

    KFF, KFE, f_free, free_dofs = partition_stiffness(K_global, bc_D, f)

    if verbosity >= 50 and verbosity < 100:
        _logger.debug("Free stiffness matrix statistics")
        _logger.debug("min(|KFF|): %s", KFF.data.min())
        _logger.debug("median(|KFF|): %s", np.median(KFF.data))
        _logger.debug("max(|KFF|): %s", KFF.data.max())
        _logger.debug("Essential stiffness matrix statistics")
        _logger.debug("min(|KFE|): %s", KFE.data.min())
        _logger.debug("median(|KFE|): %s", np.median(KFE.data))
        _logger.debug("max(|KFE|): %s", KFE.data.max())

    # Right-hand-side
    rhs = -KFE.dot(d_D) + f_free

    # Regularize isolated DOFs (zero-diagonal rows/columns).
    # These arise when all bonds at a node are removed (e.g. fracture)
    # Adding 1.0 to their diagonal gives the trivial equation 1·dF_i = 0
    zero_mask = KFF.diagonal() == 0.0
    if zero_mask.any():
        KFF = KFF + sp.diags(zero_mask.astype(float))

    # Symmetric Jacobi scaling for iterative solvers:
    # K̃ = S K_FF S,  b̃ = S b,  s = 1/sqrt(diag(K_FF))
    # After solving K̃ x̃ = b̃, recover dF = S x̃.
    # Makes all diagonal entries 1 and the convergence criterion ‖r̃‖/‖b̃‖
    # scale-invariant across arbitrary material and geometric parameters.
    if scale:
        diag = KFF.diagonal()
        s = 1.0 / np.sqrt(np.maximum(diag, 1e-300))
        S = sp.diags(s)
        KFF = S @ KFF @ S
        rhs = s * rhs

    # print condition number
    if verbosity >= 100:
        smallest_eigenvalue = sp.linalg.eigsh(KFF, k=1,
                                              which='SM',
                                              maxiter=1e7)[0][-1]
        largest_eigenvalue = sp.linalg.eigsh(KFF, k=1,
                                             which='LM',
                                             maxiter=1e7)[0][-1]
        diag_ratio = np.abs(KFF.diagonal())/np.abs(KFF.max(axis=1).todense())
        _logger.debug("number/ratio of diagonal weak rows in free stiffness matrix: %s %s",
                      np.sum(diag_ratio < 1e-4),
                      np.sum(diag_ratio < 1e-4)/diag_ratio.shape[0])
        _logger.debug("number/ratio of diagonal dominant rows in free stiffness matrix: %s %s",
                      np.sum(diag_ratio > 1e-4),
                      np.sum(diag_ratio > 1e-4)/diag_ratio.shape[0])
        _logger.debug("condition number: %s", largest_eigenvalue/smallest_eigenvalue)

    # ------------------------------------------------------------------
    # Solve reduced system
    # ------------------------------------------------------------------

    if solver == 'direct':
        try:
            dF = sp.linalg.spsolve(KFF.tocsr(), rhs)
            info = 0
        except sp.linalg.MatrixRankWarning:
            dF = np.zeros_like(rhs)
            info = 1

    elif solver == 'cholesky':
        # Sparse Cholesky via CHOLMOD (scikit-sparse).
        # Ideal for repeated solves with the same sparsity pattern.
        try:
            from sksparse.cholmod import cho_factor
        except ImportError:
            raise ImportError(
                "solver='cholesky' requires scikit-sparse.\n"
                "Install with:  pip install scikit-sparse")
        try:
            factor = cho_factor(KFF.tocsc())
            dF = factor.solve(rhs)
            info = 0
        except Exception as e:
            _logger.error(str(e))
            dF = np.zeros_like(rhs)
            info = 1

    elif solver == 'cg':
        dF, info = sp.linalg.cg(KFF, rhs, rtol=tol, atol=0., maxiter=10000)

    elif solver == 'ilu':
        # CG preconditioned with incomplete LU (ILU).
        ilu = sp.linalg.spilu(KFF.tocsc(), fill_factor=100., drop_tol=1e-5)
        M = sp.linalg.LinearOperator(KFF.shape, ilu.solve)
        dF, info = sp.linalg.cg(KFF, rhs, M=M, rtol=tol, atol=0.)

    elif solver == 'ssor':
        # CG preconditioned with SSOR (ω = 1).
        # M⁻¹ r = (D + U)⁻¹ D (D + L)⁻¹ r
        L_mat = sp.tril(KFF).tocsc()
        U_mat = sp.triu(KFF).tocsc()
        d_vec = KFF.diagonal()

        def _ssor_apply(r):
            y = sp.linalg.spsolve_triangular(L_mat, r, lower=True)
            return sp.linalg.spsolve_triangular(U_mat, d_vec * y, lower=False)

        M = sp.linalg.LinearOperator(KFF.shape, _ssor_apply)
        dF, info = sp.linalg.cg(KFF, rhs, M=M, rtol=tol, atol=0.)

    elif solver in ('amg', 'amg_rs'):
        # CG preconditioned with algebraic multigrid (pyamg).
        try:
            import pyamg
        except ImportError:
            raise ImportError(
                f"solver='{solver}' requires pyamg.\n"
                "Install with:  pip install pyamg")
        # pyamg requires int32 CSR indices; newer scipy defaults to int64.
        KFF_csr = KFF.tocsr()
        KFF_csr.indptr = KFF_csr.indptr.astype(np.int32)
        KFF_csr.indices = KFF_csr.indices.astype(np.int32)
        try:
            if solver == 'amg':
                # Smoothed aggregation: optimal for homogeneous lattices.
                ml = pyamg.smoothed_aggregation_solver(KFF_csr)
            else:
                # Ruge–Stüben: better for heterogeneous / diluted networks.
                # Can degenerate on ill-scaled (unscaled) matrices — caught below.
                ml = pyamg.ruge_stuben_solver(KFF_csr)
            M = ml.aspreconditioner()
            dF, info = sp.linalg.cg(KFF, rhs, M=M, rtol=tol, atol=0.)
        except (ValueError, RuntimeError):
            # AMG hierarchy failed (e.g. NaN/Inf in coarse levels due to
            # extreme stiffness ratios in unscaled beam matrices).
            dF = np.zeros_like(rhs)
            info = 1

    # Unscale solution
    if scale:
        dF = s * dF

    if verbosity >= 25 and verbosity < 50:
        _logger.debug("Free displacements dF statistics")
        _logger.debug("min(|dF|): %s", dF.min())
        _logger.debug("median(|dF|): %s", np.median(dF))
        _logger.debug("max(|dF|): %s", dF.max())

    # Solution all DOFs
    d = scatter_solution(dF, bc_D, d_D, num_dof, free_dofs)

    # Reaction forces
    F = K_global.dot(d)

    if verbosity >= 25 and verbosity < 50:
        _logger.debug("Reaction force F statistics")
        _logger.debug("min(|F|): %s", F.min())
        _logger.debug("median(|F|): %s", np.median(F))
        _logger.debug("max(|F|): %s", F.max())

    return d, F, info


def _solve_dense(K_global, bc_D, d_D, bc_N, F_N):
    """Solve dense system.


    Parameters
    ----------
    K : np.ndarry
        Stiffness matrix
    bc_D : iterable
        List of constraint DOFs (Dirichlet BCs)
    d_D : iterable
        List of constraint DOF values (Dirichlet BCs)
    bc_N : iterable
        List of DOFs with nonzero loads (Neumann BCs)
    F_N : iterable
        List of DOF values with nonzero loads (Neumann BCs)

    Returns
    -------
    d : np.ndarray
        Global solution vector
    F : np.ndarray
        Global load vector
    info : int
        Info about numerical solution, 0 if successful
    """

    num_dof = K_global.shape[0]

    # Partition DOF vector
    LE = np.zeros((num_dof, len(d_D)))
    LF = np.zeros((num_dof, num_dof - len(d_D)))

    dE_mask = np.zeros(num_dof, dtype=bool)
    dE_mask[bc_D] = True

    LE[dE_mask] = np.eye(dE_mask.sum())
    LF[np.invert(dE_mask)] = np.eye(num_dof - dE_mask.sum())

    # Global force vector
    f = np.zeros(num_dof)
    f[bc_N] = F_N

    KFF = np.dot(LF.T, np.dot(K_global, LF))
    KFE = np.dot(LF.T, np.dot(K_global, LE))

    # Right-hand-side
    rhs = -np.dot(KFE, d_D) + np.dot(LF.T, f)

    # Solve reduced system
    dF = np.linalg.solve(KFF, rhs)

    # Solution all DOFs
    d = np.dot(LE, d_D) + np.dot(LF, dF)

    # Reaction forces (full DOF vector, consistent with the sparse path)
    F = K_global.dot(d)

    info = 0

    return d, F, info
