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
see :mod:`beam_networks.fem.corotational` for the element routines.  A Total
Lagrangian approach is used: the assembly always references the original
(undeformed) node positions, keeping the natural element length l0 constant
across all load steps.
"""
import numpy as np
import scipy.sparse as sp

from beam_networks.fem.corotational import (
    assemble_nonlinear_system_2d,
    assemble_nonlinear_system_3d,
    _build_bsr_sparsity_3d,
    _build_kff_sparsity_2d,
    _assemble_kff_nonlinear_2d,
    _build_kff_sparsity_3d,
    _assemble_kff_nonlinear_3d,
)
from beam_networks.fem.topology import _build_bsr_node_sparsity
from beam_networks.fem.partitioning import partition_stiffness
from beam_networks.geometry.geo import get_geometric_props


def solve_nonlinear(
        nodes: np.ndarray,
        edges: np.ndarray,
        beam_prop: dict,
        dof_D: list,
        dof_N: list,
        val_N: list,
        val_D=None,
        n_steps: int = 100,
        max_iter: int = 100,
        tol: float = 1e-9,
        verbose: bool = True,
        callback=None,
        matrix: str = 'bsr',
        out: dict | None = None,
        ndim: int = 2,
        ref_vectors: np.ndarray | None = None,
) -> np.ndarray:
    """Load-stepped Newton–Raphson solver for geometrically nonlinear beam networks.

    Supports both 2D (*ndim* = 2) and 3D (*ndim* = 3) co-rotational beam
    elements.  The total Neumann load is divided into *n_steps* equal
    increments.  Within each increment the nonlinear equilibrium is solved by
    Newton–Raphson iteration using the co-rotational tangent stiffness.  A
    Total Lagrangian approach is used throughout: the assembly always
    references the original undeformed nodes, so the natural element length
    l0 remains constant and results are path-independent with respect to the
    number of load steps.

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
        100.
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
        Assembly and linear-solve format.  ``'dense'`` uses
        ``numpy.linalg.solve``; ``'bsr'`` assembles a
        ``scipy.sparse.bsr_array`` and uses
        ``scipy.sparse.linalg.spsolve``.  The default is ``'bsr'``.
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

    # Per-step Dirichlet increment (zero for fixed supports)
    d_D_step = np.zeros(ndof)
    val_D = np.asarray(val_D, dtype=float)
    if val_D is not None and dof_D.size > 0:
        d_D_step[dof_D] = val_D / n_steps

    # Per-step Neumann load increment
    f_step = np.zeros(ndof)
    if dof_N.size > 0:
        f_step[dof_N] = val_N / n_steps

    # Free DOF indices (not Dirichlet-constrained)
    mask_F = np.ones(ndof, dtype=bool)
    mask_F[dof_D] = False
    dof_F = np.where(mask_F)[0]

    # Total Lagrangian: always assemble from original nodes + sol_total.
    # l0 (natural element length) is therefore constant across all steps,
    # eliminating the path-dependence of the reference length that arises in
    # the Updated Lagrangian scheme for non-pure-bending problems.
    sol_total = np.zeros(ndof)

    # Pre-compute topology-invariant quantities once before the Newton loop.
    # -----------------------------------------------------------------------
    # (a) Geometric cross-section properties are constant throughout the solve;
    #     store them in a local copy of beam_prop so the local-stiffness
    #     routines can read them without repeating the dict-lookup arithmetic.
    beam_prop = dict(beam_prop)
    beam_prop['_geom_props'] = get_geometric_props(beam_prop)

    # (b) For BSR format, the sparsity pattern (edge sorting, CSR indices/
    #     indptr, and scatter-position arrays) depends only on mesh topology
    #     and never changes during the solve.  Build it once here.
    # (c) For the 2-D BSR path, also pre-build the K_FF sparsity so that the
    #     free–free stiffness block can be assembled directly without a full-K
    #     build followed by BSR→CSR conversion and index slicing.
    bsr_pattern = None
    kff_pattern = None
    if matrix == 'bsr':
        if ndim == 2:
            bsr_pattern = _build_bsr_node_sparsity(nodes, edges)
            kff_pattern = _build_kff_sparsity_2d(nodes, edges, dof_F)
        else:
            bsr_pattern = _build_bsr_sparsity_3d(nodes, edges, ref_vectors)
            kff_pattern = _build_kff_sparsity_3d(nodes, edges, ref_vectors, dof_F)

    def _assemble(s):
        if ndim == 2:
            return assemble_nonlinear_system_2d(
                nodes, edges, s, beam_prop, matrix=matrix,
                bsr_pattern=bsr_pattern)
        return assemble_nonlinear_system_3d(
            nodes, edges, s, beam_prop, ref_vectors, matrix=matrix,
            bsr_pattern=bsr_pattern)

    # Tangent predictor (3-D only): linear extrapolation of the free DOFs from
    # the previous step keeps Newton on the correct solution branch near
    # co-rotational singularities (chord angle ≈ 180°).  The 2-D formulation
    # uses scalar arctan2 angles and has no such singularity, so the predictor
    # is not applied there.
    prev_sol_F = sol_total[dof_F].copy()   # zero at start

    for step in range(n_steps):
        # Cumulative prescribed Dirichlet displacements (linearly ramped)
        if dof_D.size > 0:
            sol_total[dof_D] = (step + 1) * d_D_step[dof_D]

        # Predictor: extrapolate free DOFs by the previous step's increment
        if ndim == 3:
            curr_sol_F = sol_total[dof_F].copy()
            sol_total[dof_F] += curr_sol_F - prev_sol_F
            prev_sol_F = curr_sol_F

        # Cumulative target Neumann load at free DOFs
        f_target_F = f_step[dof_F] * (step + 1)

        # Newton iteration
        for it in range(max_iter):
            if kff_pattern is not None:
                if ndim == 2:
                    K_FF, F_int_F = _assemble_kff_nonlinear_2d(
                        nodes, sol_total, beam_prop, kff_pattern)
                else:
                    K_FF, F_int_F = _assemble_kff_nonlinear_3d(
                        nodes, sol_total, beam_prop, kff_pattern)
                du_F = sp.linalg.spsolve(K_FF, f_target_F - F_int_F)
            elif matrix == 'bsr':
                K, F_int = _assemble(sol_total)
                K_FF, _, rhs, _ = partition_stiffness(
                    K, dof_D, f_step * (step + 1) - F_int)
                du_F = sp.linalg.spsolve(K_FF.tocsr(), rhs)
            else:
                K, F_int = _assemble(sol_total)
                res_F = (F_int - f_step * (step + 1))[dof_F]
                K_FF = K[np.ix_(dof_F, dof_F)]
                du_F = np.linalg.solve(K_FF, -res_F)

            sol_total[dof_F] += du_F

            norm = np.linalg.norm(du_F)
            if norm < tol:
                break

        if verbose:
            print(f"Step {step + 1:4d}/{n_steps}: converged in {it + 1:4d} iter,"
                  f" |Δu| = {norm:.3e}")

        if callback is not None:
            nodes_current = nodes.copy()
            for k in range(ndim):
                nodes_current[:, k] += sol_total[k::ndof_per_node]
            callback(step + 1, nodes_current, sol_total.copy())

    if out is not None:
        _, F_int_tl = _assemble(sol_total)
        out['F_int'] = F_int_tl
        nodes_final = nodes.copy()
        for k in range(ndim):
            nodes_final[:, k] += sol_total[k::ndof_per_node]
        out['nodes_ref'] = nodes_final

    return sol_total
