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
"""Load-stepped Newton–Raphson solver for geometrically nonlinear beam networks.

The co-rotational formulation (Crisfield 1990) is used at the element level;
see :mod:`beam_networks.fem.corotational` for the element routines.  An
Updated Lagrangian approach advances the reference configuration after each
converged load step.
"""
import numpy as np
import scipy.sparse as sp

from beam_networks.fem.corotational import assemble_nonlinear_system_2d


def solve_nonlinear(
        nodes: np.ndarray,
        edges: np.ndarray,
        beam_prop: dict,
        dof_D: list,
        dof_N: list,
        val_N: list,
        val_D=None,
        n_steps: int = 100,
        max_iter: int = 10000,
        tol: float = 1e-9,
        verbose: bool = True,
        callback=None,
        matrix: str = 'dense',
        out: dict | None = None,
) -> np.ndarray:
    """Load-stepped Newton–Raphson solver for a 2D geometrically nonlinear network.

    The total Neumann load is divided into *n_steps* equal increments.  Within
    each increment the nonlinear equilibrium is solved by Newton–Raphson
    iteration using the co-rotational tangent stiffness.  The reference
    configuration is updated after each converged step (Updated Lagrangian).

    Dirichlet boundary conditions support both fixed supports (zero
    displacement) and prescribed non-zero displacements via *val_D*.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 2)
        Original (undeformed) nodal coordinates.
    edges : np.ndarray, shape (M, 2)
        Edge connectivity (integer node-index pairs).
    beam_prop : dict
        Beam cross-section and elastic properties.
    dof_D : list of int
        DOF indices with prescribed displacement (Dirichlet BCs).
    dof_N : list of int
        DOF indices with applied loads (Neumann BCs).
    val_N : list of float
        Total applied load values corresponding to *dof_N*.  Each value is
        divided equally across all load steps.
    val_D : array-like of float or None, optional
        Total prescribed displacement values for *dof_D*.  Each value is
        ramped linearly from 0 to ``val_D[i]`` over *n_steps* steps.  If
        ``None`` (default), all *dof_D* DOFs are fixed at zero.
    n_steps : int, optional
        Number of load increments.  The default is 100.
    max_iter : int, optional
        Maximum Newton–Raphson iterations per load step.  The default is
        10 000.
    tol : float, optional
        Convergence tolerance on the Euclidean norm of the incremental
        displacement correction ``|Δu|``.  The default is 1e-9.
    verbose : bool, optional
        Print per-step convergence information.  The default is True.
    callback : callable or None, optional
        If provided, called after each converged load step as
        ``callback(step, nodes_current, sol_total)`` where *nodes_current* is
        the reference nodal array after committing the step (shape ``(N, 2)``)
        and *sol_total* is the accumulated displacement from the original nodes
        (shape ``(3*N,)``).  The default is None.
    matrix : {'dense', 'bsr'}, optional
        Assembly and solve format.  ``'dense'`` uses ``numpy.linalg.solve``;
        ``'bsr'`` assembles a ``scipy.sparse.bsr_array`` and uses
        ``scipy.sparse.linalg.spsolve``.  The default is ``'dense'``.
    out : dict or None, optional
        If a dict is provided it is populated with auxiliary results after the
        final load step:

        * ``'F_int'`` — global internal force vector at the last converged NR
          state (shape ``(3*N,)``).  The entries at *dof_D* equal the
          reaction forces needed to maintain the prescribed displacement.
        * ``'nodes_ref'`` — reference nodal coordinates after the final
          committed step (shape ``(N, 2)``).

        The default is None.

    Returns
    -------
    sol_total : np.ndarray, shape (3*N,)
        Total displacement vector measured from the original *nodes*.
    """
    ndof = nodes.shape[0] * 3

    dof_D = np.asarray(dof_D, dtype=int)
    dof_N = np.asarray(dof_N, dtype=int)
    val_N = np.asarray(val_N, dtype=float)

    # Per-step prescribed Dirichlet increment (zero for fixed supports)
    d_D_step = np.zeros(ndof)
    if val_D is not None and dof_D.size > 0:
        val_D = np.asarray(val_D, dtype=float)
        d_D_step[dof_D] = val_D / n_steps

    # Per-step incremental Neumann load
    f_step = np.zeros(ndof)
    if dof_N.size > 0:
        f_step[dof_N] = val_N / n_steps

    # Free DOF indices (not Dirichlet-constrained)
    mask_F = np.ones(ndof, dtype=bool)
    mask_F[dof_D] = False
    dof_F = np.where(mask_F)[0]

    # Partition matrix L_F (ndof × n_free): precomputed once for sparse path
    if matrix == 'bsr':
        LFs = sp.bsr_array(
            (np.ones(len(dof_F), dtype=float),
             (dof_F, np.arange(len(dof_F)))),
            shape=(ndof, len(dof_F)),
        )

    # Updated Lagrangian: reference nodes advanced after each load step
    nodes_ref = nodes.copy()

    # Incremental displacement from current reference (reset each step)
    sol_step = np.zeros(ndof)

    # Cumulative displacement from the original nodes
    sol_total = np.zeros(ndof)

    for step in range(n_steps):
        # Initialise prescribed DOFs for this step (fixed stay 0, driven get increment)
        sol_step[dof_D] = d_D_step[dof_D]

        for it in range(max_iter):
            K, F_int = assemble_nonlinear_system_2d(
                nodes_ref, edges, sol_step, beam_prop, matrix=matrix)

            if matrix == 'bsr':
                K_FF = LFs.T.dot(K.dot(LFs))
                rhs = LFs.T.dot(f_step - F_int)
                du_F = sp.linalg.spsolve(K_FF.tocsr(), rhs)
            else:
                res_F = (F_int - f_step)[dof_F]
                K_FF = K[np.ix_(dof_F, dof_F)]
                du_F = np.linalg.solve(K_FF, -res_F)

            sol_step[dof_F] += du_F

            norm = np.linalg.norm(du_F)
            if norm < tol:
                break

        if verbose:
            print(f"Step {step + 1:4d}/{n_steps}: converged in {it + 1:4d} iter,"
                  f" |Δu| = {norm:.3e}")

        # Commit: advance reference nodes with the translational increment
        nodes_ref[:, 0] += sol_step[0::3]
        nodes_ref[:, 1] += sol_step[1::3]

        # Accumulate total displacement from original nodes
        sol_total[0::3] += sol_step[0::3]
        sol_total[1::3] += sol_step[1::3]
        sol_total[2::3] += sol_step[2::3]

        sol_step = np.zeros(ndof)

        if callback is not None:
            callback(step + 1, nodes_ref.copy(), sol_total.copy())

    if out is not None:
        # Compute reaction forces via Total Lagrangian: original nodes + total
        # displacement.  The UL incremental F_int from the last NR step only
        # reflects the final load-step increment, not the accumulated load.
        _, F_int_tl = assemble_nonlinear_system_2d(
            nodes, edges, sol_total, beam_prop, matrix=matrix)
        out['F_int'] = F_int_tl
        out['nodes_ref'] = nodes_ref.copy()

    return sol_total
