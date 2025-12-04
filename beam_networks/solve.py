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
import scipy.sparse as sp


def solve(K, bc_D, d_D, bc_N, F_N,
          solver='direct', preconditioner=None,
          verbosity=0):
    """Solve the linear system.

    Parameters
    ----------
    K : np.ndarry or scipy.sparse.csr_matrix
        Stiffness matrix
    bc_D : iterable
        List of constraint DOFs (Dirichlet BCs)
    d_D : iterable
        List of constraint DOF values (Dirichlet BCs)
    bc_N : iterable
        List of DOFs with nonzero loads (Neumann BCs)
    F_N : iterable
        List of DOF values with nonzero loads (Neumann BCs)
    solver : str, optional
        Type of solver, 'direct' or 'cg' (the default is 'direct')
    preconditioner : str, optional
        Type of preconditioner, 'diagonal' or None (the default is None). Only 
        active for sparse matrices.
    verbosity : int, optional
        level of verbosity, if between 25/50 print information about 
        displacements and reaction forces, if between 50/100 print 
        information about stiffness matrix, if greater equal 100 print 
        condition number (default is 0). Only active for sparse matrices.

    Returns
    -------
    d : np.ndarray
        Global solution vector
    F : np.ndarray
        Global load vector
    info : int
        Info about numerical solution, 0 if successful
    """

    if sp.issparse(K):
        d, F, info = _solve_sparse(K, bc_D, d_D, bc_N, F_N, solver=solver,
                                   preconditioner=preconditioner,
                                   verbosity=verbosity)
    else:
        d, F, info = _solve_dense(K, bc_D, d_D, bc_N, F_N)

    return d, F, info


def _solve_sparse(K_global, bc_D, d_D, bc_N, F_N,
                  solver='direct', preconditioner=None,
                  verbosity=0):
    """Solve sparse system.


    Parameters
    ----------
    K : scipy.sparse.csr_matrix
        Stiffness matrix
    bc_D : iterable
        List of constraint DOFs (Dirichlet BCs)
    d_D : iterable
        List of constraint DOF values (Dirichlet BCs)
    bc_N : iterable
        List of DOFs with nonzero loads (Neumann BCs)
    F_N : iterable
        List of DOF values with nonzero loads (Neumann BCs)
    solver : str, optional
        Type of solver, 'direct' or 'cg' (the default is 'direct')
    preconditioner : str, optional
        Type of preconditioner, 'diagonal' or None (the default is None)
    verbosity : int, optional
        level of verbosity, if between 25/50 print information about 
        displacements and reaction forces, if between 50/100 print 
        information about stiffness matrix, if greater equal 100 print 
        condition number (default is 0). 


    Returns
    -------
    d : np.ndarray
        Global solution vector
    F : np.ndarray
        Global load vector
    info : int
        Info about numerical solution, 0 if successful
    """

    assert solver in ['direct', 'cg', 'pcg']

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

    # apply diagonal preconditioner
    if preconditioner == "diagonal":
        P = sp.diags(1/KFF.diagonal(), format=KFF._format)
        P_inv = sp.diags(KFF.diagonal(), format=KFF._format)
        KFF = P@KFF
        rhs = P@rhs
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

    # Solve reduced system
    if solver == 'direct':
        try:
            dF = sp.linalg.spsolve(KFF.tocsr(), rhs)
            info = 0
        except sp.linalg.MatrixRankWarning:
            dF = np.zeros_like(rhs)
            info = 1
    elif solver == 'cg':
        dF, info = sp.linalg.cg(KFF, rhs, rtol=1e-10, maxiter=10000)
    elif solver == 'pcg':
        ilu = sp.linalg.spilu(KFF.tocsc(), fill_factor=100., drop_tol=1e-5)
        M = sp.linalg.LinearOperator(KFF.shape, ilu.solve)
        dF, info = sp.linalg.cg(KFF, rhs, M=M)
    else:
        raise RuntimeError(f'Solver {solver} not available.')

    if verbosity >= 25 and verbosity < 50:
        print("Free displacements dF statistics")
        print("min(|dF|): ", dF.min())
        print("median(|dF|): ", np.median(dF))
        print("max(|dF|): ", dF.max())

    # apply diagonal preconditioner
    if preconditioner == "diagonal":
        rhs = P_inv@rhs

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
