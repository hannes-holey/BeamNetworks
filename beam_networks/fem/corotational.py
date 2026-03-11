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

"""Co-rotational beam element formulation (Crisfield 1990).

Implements the geometrically nonlinear co-rotational method for 2D beam
networks. Small strains are assumed in the local (co-rotating) frame, so only
the rigid-body rotation is treated in a geometrically exact manner.

References
----------
Crisfield, M. A. (1990). A consistent co-rotational formulation for non-linear,
three-dimensional, beam-elements. *Computer Methods in Applied Mechanics and
Engineering*, 81(2), 131–150.
"""
import numpy as np
import scipy.sparse as sp

from beam_networks.geometry.geo import get_geometric_props


# ---------------------------------------------------------------------------
# Helpers (batch over M elements)
# ---------------------------------------------------------------------------

def _rigid_body_rotation_2d(
        nodes: np.ndarray,
        d: np.ndarray,
        edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised rigid-body rotation for all M elements.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 2)
        Reference (undeformed) nodal coordinates.
    d : np.ndarray, shape (N, 3)
        Current nodal DOFs ``[ux, uy, θ_z]`` per node.
    edges : np.ndarray, shape (M, 2)
        Edge connectivity.

    Returns
    -------
    alpha, l0, ln, c, s : np.ndarray, each shape (M,)
    """
    e0, e1 = edges[:, 0], edges[:, 1]

    dr = nodes[e1] - nodes[e0]            # (M, 2)
    l0 = np.linalg.norm(dr, axis=1)       # (M,)

    rn0 = nodes[e0] + d[e0, :2]          # (M, 2)
    rn1 = nodes[e1] + d[e1, :2]          # (M, 2)
    drn = rn1 - rn0                        # (M, 2)
    ln = np.linalg.norm(drn, axis=1)       # (M,)

    c0 = dr[:, 0] / l0
    s0 = dr[:, 1] / l0
    c = drn[:, 0] / ln
    s = drn[:, 1] / ln

    sin_a = c0 * s - s0 * c
    cos_a = c0 * c + s0 * s
    # arctan2(sin α, cos α) is equivalent to the quadrant-aware scalar logic
    alpha = np.arctan2(sin_a, cos_a)       # (M,)

    return alpha, l0, ln, c, s


def _local_stiffness_2d(beam_prop: dict, l0: np.ndarray) -> np.ndarray:
    """Vectorised 3×3 local stiffness for all M elements.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    l0 : np.ndarray, shape (M,)
        Reference (undeformed) element lengths.

    Returns
    -------
    Kl : np.ndarray, shape (M, 3, 3)
    """
    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2. * (1. + nu))
    _, Iz, _, A, kappa, _ = get_geometric_props(beam_prop)

    EA = E * A
    EI = E * Iz
    kGA = kappa * G * A
    Phi = 12. * EI / (kGA * l0**2)        # (M,)
    f = 1. / (1. + Phi)                    # (M,)

    M = len(l0)
    Kl = np.zeros((M, 3, 3))
    Kl[:, 0, 0] = EA / l0
    Kl[:, 1, 1] = EI * (4. + Phi) * f / l0
    Kl[:, 1, 2] = EI * (2. - Phi) * f / l0
    Kl[:, 2, 1] = EI * (2. - Phi) * f / l0
    Kl[:, 2, 2] = EI * (4. + Phi) * f / l0

    return Kl


def _b_matrix_2d(c: np.ndarray, s: np.ndarray, ln: np.ndarray) -> np.ndarray:
    """Vectorised 3×6 kinematic (B) matrix for all M elements.

    Parameters
    ----------
    c, s, ln : np.ndarray, shape (M,)
        Direction cosines and deformed lengths.

    Returns
    -------
    B : np.ndarray, shape (M, 3, 6)
    """
    M = len(c)
    B = np.zeros((M, 3, 6))
    B[:, 0, 0] = -c
    B[:, 0, 1] = -s
    B[:, 0, 3] = c
    B[:, 0, 4] = s
    B[:, 1, 0] = -s / ln
    B[:, 1, 1] = c / ln
    B[:, 1, 2] = 1.
    B[:, 1, 3] = s / ln
    B[:, 1, 4] = -c / ln
    B[:, 2, 0] = -s / ln
    B[:, 2, 1] = c / ln
    B[:, 2, 3] = s / ln
    B[:, 2, 4] = -c / ln
    B[:, 2, 5] = 1.
    return B


def _element_tangent_2d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised tangent stiffness and internal forces for all M elements.

    Computes element tangent stiffness and internal forces for the entire
    network in a single vectorised pass, without a Python loop over elements.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 2)
        Reference (undeformed) nodal coordinates.
    edges : np.ndarray, shape (M, 2)
        Edge connectivity.
    sol : np.ndarray, shape (3*N,)
        Current nodal DOFs ``[ux, uy, θ_z, ...]``.
    beam_prop : dict
        Beam cross-section and elastic properties.

    Returns
    -------
    Kt : np.ndarray, shape (M, 6, 6)
        Element tangent stiffness matrices in the global frame.
    fg : np.ndarray, shape (M, 6)
        Element internal force vectors in the global frame.
    """
    e0, e1 = edges[:, 0], edges[:, 1]
    d = sol.reshape(-1, 3)                         # (N, 3)

    alpha, l0, ln, c, s = _rigid_body_rotation_2d(nodes, d, edges)
    Kl = _local_stiffness_2d(beam_prop, l0)    # (M, 3, 3)

    # Local deformational DOFs: ul = [ln-l0, θ0-α, θ1-α], shape (M, 3)
    ul = np.stack([ln - l0, d[e0, 2] - alpha, d[e1, 2] - alpha], axis=1)

    # Local internal forces: fl = Kl @ ul, shape (M, 3)
    fl = np.einsum('mij,mj->mi', Kl, ul)

    # Kinematic (B) matrix: (M, 3, 6)
    B = _b_matrix_2d(c, s, ln)

    # Global internal forces: fg = B^T fl, shape (M, 6)
    fg = np.einsum('mji,mj->mi', B, fl)

    # Material stiffness: Km = B^T Kl B, shape (M, 6, 6)
    BT = B.transpose(0, 2, 1)                      # (M, 6, 3)
    Km = BT @ Kl @ B                               # (M, 6, 6)

    # Geometric stiffness (Crisfield 1990, eqs. 3.28–3.30)
    # chord and perpendicular unit vectors, shape (M, 6)
    zeros = np.zeros(len(c))
    r = np.column_stack([-c, -s, zeros, c, s, zeros])
    z = np.column_stack([s, -c, zeros, -s, c, zeros])

    zz = np.einsum('mi,mj->mij', z, z)            # (M, 6, 6)
    rz = np.einsum('mi,mj->mij', r, z)            # (M, 6, 6)

    Kg = (zz * (fl[:, 0] / ln)[:, None, None]
          + (rz + rz.transpose(0, 2, 1))
          * ((fl[:, 1] + fl[:, 2]) / ln**2)[:, None, None])

    return Km + Kg, fg


# ---------------------------------------------------------------------------
# Global assembly
# ---------------------------------------------------------------------------

def _assemble_dense_nonlinear_2d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense assembly of global tangent stiffness and internal force."""
    ndof = nodes.shape[0] * 3
    K = np.zeros((ndof, ndof))
    F_int = np.zeros(ndof)

    e0, e1 = edges[:, 0], edges[:, 1]
    Kt_all, fg_all = _element_tangent_2d(nodes, edges, sol, beam_prop)

    # Scatter internal forces
    for i in range(3):
        np.add.at(F_int, e0 * 3 + i, fg_all[:, i])
        np.add.at(F_int, e1 * 3 + i, fg_all[:, 3 + i])

    # Scatter 3×3 stiffness blocks
    for i in range(3):
        for j in range(3):
            np.add.at(K, (e0 * 3 + i, e0 * 3 + j), Kt_all[:, i,     j])
            np.add.at(K, (e0 * 3 + i, e1 * 3 + j), Kt_all[:, i,     3 + j])
            np.add.at(K, (e1 * 3 + i, e0 * 3 + j), Kt_all[:, 3 + i, j])
            np.add.at(K, (e1 * 3 + i, e1 * 3 + j), Kt_all[:, 3 + i, 3 + j])

    return K, F_int


def _assemble_bsr_nonlinear_2d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
) -> tuple[sp.bsr_array, np.ndarray]:
    """BSR sparse assembly of global tangent stiffness and internal force.

    Mirrors the structure of the linear BSR assembler in
    ``beam_networks.fem.assembly``. Edges must be sorted so that
    ``edges[:, 0] < edges[:, 1]`` and rows are ordered by the first column;
    this is the convention used throughout ``ElasticNetwork``. The symmetric
    tangent stiffness is built as upper-triangle + diagonal blocks and then
    symmetrised with ``K + K.T``.
    """
    num_nodes = nodes.shape[0]
    ndof_per_node = 3
    ndof = num_nodes * ndof_per_node

    # Ensure n0 < n1 and rows are sorted by n0 (matches the linear assembler)
    edges_sorted = np.sort(edges, axis=1)
    edges_sorted = edges_sorted[np.lexsort((edges_sorted[:, 1], edges_sorted[:, 0]))]
    e0s, e1s = edges_sorted[:, 0], edges_sorted[:, 1]

    # Build sparsity pattern: upper triangle (off-diagonal edges) + diagonal
    aux = sp.csr_array(
        (np.ones(len(edges_sorted)), (e0s, e1s)),
        shape=(num_nodes, num_nodes),
    )
    aux = aux + sp.eye_array(num_nodes)
    indices = aux.indices
    indptr = aux.indptr

    data = np.zeros((len(indices), ndof_per_node, ndof_per_node))
    F_int = np.zeros(ndof)

    # Compute all element tangent stiffnesses and internal forces at once
    Kt_all, fg_all = _element_tangent_2d(nodes, edges_sorted, sol, beam_prop)

    # Precompute BSR data-array positions for each edge
    # Diagonal block of n0 is always the first entry in its CSR row (n0 < n1)
    diag_pos_n0 = indptr[e0s]                       # (M,)
    diag_pos_n1 = indptr[e1s]                       # (M,)

    # Off-diagonal block (n0, n1): within row n0 it is the (k+1)-th entry,
    # where k is the 0-indexed rank of this edge among all edges from n0.
    _, first_occ, c0s = np.unique(e0s, return_index=True, return_counts=True)
    k_per_edge = np.arange(len(e0s)) - np.repeat(first_occ, c0s)
    offdiag_pos = indptr[e0s] + 1 + k_per_edge      # (M,)

    # Scatter diagonal blocks (factor 1/2; symmetrised by K + K.T below)
    np.add.at(data, diag_pos_n0, Kt_all[:, :3, :3] / 2.)
    np.add.at(data, diag_pos_n1, Kt_all[:, 3:, 3:] / 2.)

    # Scatter off-diagonal blocks
    np.add.at(data, offdiag_pos, Kt_all[:, :3, 3:])

    # Scatter internal forces
    for i in range(3):
        np.add.at(F_int, e0s * 3 + i, fg_all[:, i])
        np.add.at(F_int, e1s * 3 + i, fg_all[:, 3 + i])

    K = sp.bsr_array(
        (data, indices, indptr),
        shape=(ndof, ndof),
        blocksize=(ndof_per_node, ndof_per_node),
    )
    K = K + K.T

    return K, F_int


# ---------------------------------------------------------------------------
# Element stress computation (Total Lagrangian, for use after NR convergence)
# ---------------------------------------------------------------------------

def compute_element_forces_2d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
) -> np.ndarray:
    """Local element forces for all 2D co-rotational beam elements.

    Uses a Total Lagrangian interpretation: *nodes* are the original
    (undeformed) coordinates and *sol* is the total displacement from those
    coordinates.  This is the correct approach for computing stress after a
    converged NR solve when the caller holds the original node array and the
    accumulated displacement vector.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 2)
        Original (undeformed) nodal coordinates.
    edges : np.ndarray, shape (M, 2)
        Edge connectivity.
    sol : np.ndarray, shape (3*N,)
        Total displacement from *nodes*.
    beam_prop : dict
        Beam cross-section and elastic properties.

    Returns
    -------
    forces : np.ndarray, shape (M, 3)
        Local element forces ``[N, M0, M1]`` for each element, where *N* is
        the axial force, *M0* the moment at node 0, and *M1* the moment at
        node 1.
    """
    e0, e1 = edges[:, 0], edges[:, 1]
    d = sol.reshape(-1, 3)                          # (N, 3)

    alpha, l0, ln, _, _ = _rigid_body_rotation_2d(nodes, d, edges)
    Kl = _local_stiffness_2d(beam_prop, l0)    # (M, 3, 3)

    ul = np.stack([ln - l0, d[e0, 2] - alpha, d[e1, 2] - alpha], axis=1)
    return np.einsum('mij,mj->mi', Kl, ul)


def element_mises_stress_2d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
) -> np.ndarray:
    """Von Mises equivalent stress per element from co-rotational forces.

    The stress is evaluated at both element ends and the maximum is returned
    (equivalent to ``mode='max'``).  For a 2D beam under combined axial and
    bending loading the max-fibre stress at end *i* is::

        σ_i = |N / A| + |M_i| * y_max / Iz

    and von Mises reduces to this scalar for a uniaxial stress state.

    Parameters
    ----------
    nodes, edges, sol, beam_prop :
        Same as :func:`compute_element_forces_2d`.

    Returns
    -------
    svm : np.ndarray, shape (M,)
        Von Mises equivalent stress per element.
    """
    from beam_networks.geometry.geo import get_geometric_props
    _, Iz, _, A, _, ymax = get_geometric_props(beam_prop)
    forces = compute_element_forces_2d(nodes, edges, sol, beam_prop)
    N = forces[:, 0]
    sigma_a = N / A
    sigma_b0 = np.abs(forces[:, 1]) * ymax / Iz   # bending at node 0
    sigma_b1 = np.abs(forces[:, 2]) * ymax / Iz   # bending at node 1
    return np.abs(sigma_a) + np.maximum(sigma_b0, sigma_b1)


def assemble_nonlinear_system_2d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
        matrix: str = 'dense',
) -> tuple[np.ndarray | sp.bsr_array, np.ndarray]:
    """Assemble global tangent stiffness and internal force for a 2D network.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 2)
        Current reference nodal coordinates (updated each load step in the
        Updated Lagrangian sense).
    edges : np.ndarray, shape (M, 2)
        Edge connectivity (integer node-index pairs). Must be sorted
        (``edges[:, 0] < edges[:, 1]``) when *matrix* is ``'bsr'``.
    sol : np.ndarray, shape (3*N,)
        Current incremental displacement vector ``[ux, uy, θ_z, ...]``
        measured from *nodes*.
    beam_prop : dict
        Beam cross-section and elastic properties.
    matrix : {'dense', 'bsr'}, optional
        Storage format for the global stiffness matrix.  ``'dense'`` returns
        a plain ``np.ndarray``; ``'bsr'`` returns a
        ``scipy.sparse.bsr_array``.  The default is ``'dense'``.

    Returns
    -------
    K : np.ndarray or scipy.sparse.bsr_array, shape (3*N, 3*N)
        Global tangent stiffness matrix.
    F_int : np.ndarray, shape (3*N,)
        Global internal force vector.
    """
    if matrix == 'dense':
        return _assemble_dense_nonlinear_2d(nodes, edges, sol, beam_prop)
    elif matrix == 'bsr':
        return _assemble_bsr_nonlinear_2d(nodes, edges, sol, beam_prop)
    else:
        raise ValueError(f"Unknown matrix format '{matrix}'. Choose 'dense' or 'bsr'.")
