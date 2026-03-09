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

    LE_rows = bc_D
    LE_cols = np.arange(len(d_D))
    LEs = sp.bsr_array((np.ones_like(LE_cols), (LE_rows, LE_cols)), shape=(num_dof, len(d_D)))
    LF_rows = np.delete(np.arange(num_dof), bc_D)

    LF_cols = np.arange(num_dof - len(d_D))
    LFs = sp.bsr_array((np.ones_like(LF_cols), (LF_rows, LF_cols)), shape=(num_dof, num_dof - len(d_D)))

    # Global force vector
    f = np.zeros(num_dof)
    if len(F_N) > 0:
        f[bc_N] = F_N

    if verbosity >= 50 and verbosity < 100:
        print("Global stiffness matrix statistics")
        print("min(|K|): ", K_global.data.min())
        print("median(|K|): ", np.median(K_global.data))
        print("max(|K|): ", K_global.data.max())

    KFF = LFs.T.dot(K_global.dot(LFs))
    KFE = LFs.T.dot(K_global.dot(LEs))

    if verbosity >= 50 and verbosity < 100:
        print("Free stiffness matrix statistics")
        print("min(|KFF|): ", KFF.data.min())
        print("median(|KFF|): ", np.median(KFF.data))
        print("max(|KFF|): ", KFF.data.max())
        print("Essential stiffness matrix statistics")
        print("min(|KFE|): ", KFE.data.min())
        print("median(|KFE|): ", np.median(KFE.data))
        print("max(|KFE|): ", KFE.data.max())

    # Right-hand-side
    rhs = -KFE.dot(d_D) + LFs.T.dot(f)

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
        print("number/ratio of diagonal weak rows in free stiffness matrix: ",
              np.sum(diag_ratio < 1e-4),
              np.sum(diag_ratio < 1e-4)/diag_ratio.shape[0])
        print("number/ratio of diagonal dominant rows in free stiffness matrix: ",
              np.sum(diag_ratio > 1e-4),
              np.sum(diag_ratio > 1e-4)/diag_ratio.shape[0])
        print("condition number: ",
              largest_eigenvalue/smallest_eigenvalue)

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
            print(e)
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
        print("Free displacements dF statistics")
        print("min(|dF|): ", dF.min())
        print("median(|dF|): ", np.median(dF))
        print("max(|dF|): ", dF.max())

    # Solution all DOFs
    d = LEs.dot(d_D) + LFs.dot(dF)

    # Reaction forces
    F = K_global.dot(d)

    if verbosity >= 25 and verbosity < 50:
        print("Reaction force F statistics")
        print("min(|F|): ", F.min())
        print("median(|F|): ", np.median(F))
        print("max(|F|): ", F.max())

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

    # Reaction forces
    F = LE.T.dot(K_global.dot(d))

    info = 0

    return d, F, info
