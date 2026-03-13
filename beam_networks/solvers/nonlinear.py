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

from beam_networks.fem.corotational import (
    assemble_nonlinear_system_2d,
    assemble_nonlinear_system_3d,
)
from beam_networks.fem.partitioning import partition_stiffness


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
        ndim: int = 2,
        ref_vectors: np.ndarray | None = None,
) -> np.ndarray:
    """Load-stepped Newton–Raphson solver for geometrically nonlinear beam networks.

    Supports both 2D (*ndim* = 2) and 3D (*ndim* = 3) co-rotational beam
    elements.  The total Neumann load is divided into *n_steps* equal
    increments.  Within each increment the nonlinear equilibrium is solved by
    Newton–Raphson iteration using the co-rotational tangent stiffness.  The
    reference configuration is updated after each converged step (Updated
    Lagrangian).

    Dirichlet boundary conditions support both fixed supports (zero
    displacement) and prescribed non-zero displacements via *val_D*.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 2) or (N, 3)
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
        the reference nodal array after committing the step and *sol_total* is
        the accumulated displacement from the original nodes.  The default is
        None.
    matrix : {'dense', 'bsr'}, optional
        Assembly and solve format.  ``'dense'`` uses ``numpy.linalg.solve``;
        ``'bsr'`` assembles a ``scipy.sparse.bsr_array`` and uses
        ``scipy.sparse.linalg.spsolve``.  The default is ``'dense'``.
    out : dict or None, optional
        If a dict is provided it is populated with auxiliary results after the
        final load step:

        * ``'F_int'`` — global internal force vector computed via a Total
          Lagrangian pass (original nodes + sol_total).  The entries at
          *dof_D* equal the reaction forces needed to maintain the prescribed
          displacement.
        * ``'nodes_ref'`` — reference nodal coordinates after the final
          committed step.

        The default is None.
    ndim : {2, 3}, optional
        Problem dimension.  ``2`` uses the 2D co-rotational formulation
        (3 DOFs per node); ``3`` uses the 3D formulation (6 DOFs per node).
        The default is ``2``.
    ref_vectors : np.ndarray, shape (M, 3), or None, optional
        Reference vectors defining the local e2 axis per element.  Required
        when *ndim* = 3; ignored for *ndim* = 2.

    Returns
    -------
    sol_total : np.ndarray, shape (ndof_per_node * N,)
        Total displacement vector measured from the original *nodes*.
    """
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim}.")
    if ndim == 3 and ref_vectors is None:
        raise ValueError("ref_vectors must be provided when ndim=3.")

    ndof_per_node = 3 if ndim == 2 else 6
    ndof = nodes.shape[0] * ndof_per_node

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

    # Updated Lagrangian: reference nodes advanced after each load step
    nodes_ref = nodes.copy()

    # Incremental displacement from current reference (reset each step)
    sol_step = np.zeros(ndof)

    # Cumulative displacement from the original nodes
    sol_total = np.zeros(ndof)

    def _assemble(n_ref, s):
        if ndim == 2:
            return assemble_nonlinear_system_2d(
                n_ref, edges, s, beam_prop, matrix=matrix)
        return assemble_nonlinear_system_3d(
            n_ref, edges, s, beam_prop, ref_vectors, matrix=matrix)

    for step in range(n_steps):
        # Initialise prescribed DOFs for this step (fixed stay 0, driven get increment)
        sol_step[dof_D] = d_D_step[dof_D]

        for it in range(max_iter):
            K, F_int = _assemble(nodes_ref, sol_step)

            if matrix == 'bsr':
                K_FF, _, rhs, _ = partition_stiffness(K, dof_D, f_step - F_int)
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

        # Commit: advance reference nodes with the translational increments
        for k in range(ndim):
            nodes_ref[:, k] += sol_step[k::ndof_per_node]

        # Accumulate total displacement from original nodes
        for k in range(ndof_per_node):
            sol_total[k::ndof_per_node] += sol_step[k::ndof_per_node]

        sol_step = np.zeros(ndof)

        if callback is not None:
            callback(step + 1, nodes_ref.copy(), sol_total.copy())

    if out is not None:
        # Compute reaction forces via Total Lagrangian: original nodes + total
        # displacement.  The UL incremental F_int from the last NR step only
        # reflects the final load-step increment, not the accumulated load.
        _, F_int_tl = _assemble(nodes, sol_total)
        out['F_int'] = F_int_tl
        out['nodes_ref'] = nodes_ref.copy()

    return sol_total
