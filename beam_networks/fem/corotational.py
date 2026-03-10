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
# Local stiffness
# ---------------------------------------------------------------------------

def _local_stiffness_2d(beam_prop: dict, L: float) -> np.ndarray:
    """3×3 local stiffness for a 2D co-rotational beam element.

    The three reduced local DOFs after applying the chord constraint are::

        ul = [elongation,  θ_0 - α,  θ_1 - α]

    with corresponding local forces ``[N, M_0, M_1]``.

    The bending block is the exact condensed Timoshenko stiffness (Przemieniecki
    1968). For a Timoshenko beam the shear flexibility parameter is
    ``Φ = 12 EI / (κGA L²)``; setting ``Φ → 0`` recovers Euler–Bernoulli::

        Kl = [[EA/L,              0,                   0          ],
              [0,    EI(4+Φ)/(L(1+Φ)),   EI(2-Φ)/(L(1+Φ))      ],
              [0,    EI(2-Φ)/(L(1+Φ)),   EI(4+Φ)/(L(1+Φ))      ]]

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    L : float
        Reference (undeformed) element length.

    Returns
    -------
    Kl : np.ndarray, shape (3, 3)
        Symmetric local stiffness matrix.
    """
    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2. * (1. + nu))
    _, Iz, _, A, kappa, _ = get_geometric_props(beam_prop)

    EA = E * A
    EI = E * Iz
    kGA = kappa * G * A
    Phi = 12. * EI / (kGA * L**2)
    f = 1. / (1. + Phi)

    Kl = np.array([
        [EA, 0., 0.],
        [0., EI * (4. + Phi) * f,  EI * (2. - Phi) * f],
        [0., EI * (2. - Phi) * f,  EI * (4. + Phi) * f],
    ]) / L

    return Kl


# ---------------------------------------------------------------------------
# Corotational geometry
# ---------------------------------------------------------------------------

def _rigid_body_rotation_2d(r0: np.ndarray,
                            r1: np.ndarray,
                            d0: np.ndarray,
                            d1: np.ndarray,
                            ) -> tuple[float, float, float, float, float]:
    """Rigid-body rotation angle and deformed geometry of a 2D element.

    Parameters
    ----------
    r0, r1 : np.ndarray, shape (2,)
        Reference (undeformed) nodal coordinates.
    d0, d1 : np.ndarray, shape (3,)
        Current nodal DOFs ``[ux, uy, θ_z]`` for nodes 0 and 1.

    Returns
    -------
    alpha : float
        Rigid-body rotation angle of the deformed chord (radians).
    l0 : float
        Undeformed element length.
    ln : float
        Deformed element length.
    c, s : float
        Direction cosines of the deformed chord (cos α_global, sin α_global).
    """
    l0 = np.linalg.norm(r1 - r0)
    rn0 = r0 + d0[:2]
    rn1 = r1 + d1[:2]
    ln = np.linalg.norm(rn1 - rn0)

    # Unit vectors of reference and deformed chords
    c0 = (r1[0] - r0[0]) / l0
    s0 = (r1[1] - r0[1]) / l0
    c = (rn1[0] - rn0[0]) / ln
    s = (rn1[1] - rn0[1]) / ln

    sin_a = c0 * s - s0 * c
    cos_a = c0 * c + s0 * s

    if sin_a >= 0. and cos_a >= 0.:
        alpha = np.arcsin(sin_a)
    elif sin_a >= 0. and cos_a < 0.:
        alpha = np.arccos(cos_a)
    elif sin_a < 0. and cos_a >= 0.:
        alpha = np.arcsin(sin_a)
    else:
        alpha = -np.arccos(cos_a)

    return alpha, l0, ln, c, s


def _b_matrix_2d(c: float, s: float, ln: float) -> np.ndarray:
    """3×6 kinematic (B) matrix for a 2D co-rotational element.

    Linearised relationship ``δu_local = B · δu_global`` between the three
    local deformational DOFs and the six global element DOFs.

    Parameters
    ----------
    c, s : float
        Direction cosines of the deformed chord.
    ln : float
        Deformed element length.

    Returns
    -------
    B : np.ndarray, shape (3, 6)
    """
    return np.array([
        [-c,    -s,    0., c,     s,     0.],
        [-s/ln,  c/ln, 1., s/ln, -c/ln,  0.],
        [-s/ln,  c/ln, 0., s/ln, -c/ln,  1.],
    ])


# ---------------------------------------------------------------------------
# Element tangent stiffness and internal forces
# ---------------------------------------------------------------------------

def element_tangent_2d(r0: np.ndarray,
                       r1: np.ndarray,
                       d0: np.ndarray,
                       d1: np.ndarray,
                       beam_prop: dict,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Tangent stiffness and internal forces for a 2D co-rotational beam.

    Implements the Crisfield (1990) co-rotational formulation: large
    rigid-body rotations are handled exactly through the corotational frame
    while small strains in the local frame are treated with linear elasticity.

    Parameters
    ----------
    r0, r1 : np.ndarray, shape (2,)
        Reference (undeformed) nodal coordinates.
    d0, d1 : np.ndarray, shape (3,)
        Current nodal DOFs ``[ux, uy, θ_z]`` for nodes 0 and 1.
    beam_prop : dict
        Beam cross-section and elastic properties.

    Returns
    -------
    Kt : np.ndarray, shape (6, 6)
        Element tangent stiffness matrix in the global frame.
        Equals the material tangent ``B'·K_l·B`` plus geometric stiffness
        terms from the axial force and bending moments.
    fg : np.ndarray, shape (6,)
        Element internal force vector in the global frame.
    """
    alpha, l0, ln, c, s = _rigid_body_rotation_2d(r0, r1, d0, d1)
    Kl = _local_stiffness_2d(beam_prop, l0)

    # Local deformational DOFs and corresponding internal forces
    ul = np.array([ln - l0, d0[2] - alpha, d1[2] - alpha])
    fl = Kl @ ul

    # Kinematic transformation and global internal force
    B = _b_matrix_2d(c, s, ln)
    fg = B.T @ fl

    # Geometric stiffness contributions (Crisfield 1990, eqs. 3.28–3.30)
    r = np.array([-c, -s, 0., c, s, 0.])   # chord direction
    z = np.array([s, -c, 0., -s, c, 0.])   # perpendicular direction
    zz = z[:, None] @ z[None, :]
    rz = r[:, None] @ z[None, :]

    Kt = (B.T @ Kl @ B
          + zz * fl[0] / ln
          + (rz + rz.T) / ln**2 * (fl[1] + fl[2]))

    return Kt, fg


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

    for e0, e1 in edges:
        s0 = slice(e0 * 3, (e0 + 1) * 3)
        s1 = slice(e1 * 3, (e1 + 1) * 3)

        Kt, fg = element_tangent_2d(nodes[e0], nodes[e1], sol[s0], sol[s1], beam_prop)

        K[s0, s0] += Kt[:3, :3]
        K[s0, s1] += Kt[:3, 3:]
        K[s1, s0] += Kt[3:, :3]
        K[s1, s1] += Kt[3:, 3:]

        F_int[s0] += fg[:3]
        F_int[s1] += fg[3:]

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

    # Build sparsity pattern: upper triangle (off-diagonal edges) + diagonal
    aux = sp.csr_array(
        (np.ones(len(edges_sorted)), (edges_sorted[:, 0], edges_sorted[:, 1])),
        shape=(num_nodes, num_nodes),
    )
    aux = aux + sp.eye_array(num_nodes)
    indices = aux.indices
    indptr = aux.indptr

    data = np.zeros((len(indices), ndof_per_node, ndof_per_node))
    F_int = np.zeros(ndof)

    i = 0
    n0s, c0s = np.unique(edges_sorted[:, 0], return_counts=True)

    for n0, c0 in zip(n0s, c0s):
        for k, n1 in enumerate(edges_sorted[i + np.arange(c0), 1]):
            s0 = slice(n0 * ndof_per_node, (n0 + 1) * ndof_per_node)
            s1 = slice(n1 * ndof_per_node, (n1 + 1) * ndof_per_node)

            Kt, fg = element_tangent_2d(nodes[n0], nodes[n1], sol[s0], sol[s1], beam_prop)

            # Factor 1/2 on diagonal blocks; symmetrised by K + K.T below
            data[indptr[n0]] += Kt[:3, :3] / 2.
            data[indptr[n1]] += Kt[3:, 3:] / 2.
            data[indptr[n0] + k + 1] += Kt[:3, 3:]

            F_int[s0] += fg[:3]
            F_int[s1] += fg[3:]

            i += 1

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
    forces = np.zeros((len(edges), 3))
    for i, (e0, e1) in enumerate(edges):
        s0 = slice(e0 * 3, (e0 + 1) * 3)
        s1 = slice(e1 * 3, (e1 + 1) * 3)
        alpha, l0, ln, _, _ = _rigid_body_rotation_2d(
            nodes[e0], nodes[e1], sol[s0], sol[s1])
        Kl = _local_stiffness_2d(beam_prop, l0)
        ul = np.array([ln - l0, sol[s0][2] - alpha, sol[s1][2] - alpha])
        forces[i] = Kl @ ul
    return forces


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
