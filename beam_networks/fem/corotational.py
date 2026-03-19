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

Implements the geometrically nonlinear co-rotational method for 2D and 3D beam
networks.  Small strains are assumed in the local (co-rotating) frame, so only
the rigid-body rotation is treated in a geometrically exact manner.

The 3D formulation follows Crisfield (1990) with one simplification: the local
transverse axes e2 and e3 are defined by projecting a fixed reference vector
perpendicular to the deformed chord (Gram-Schmidt) and normalising, rather than
constructing them from the average nodal rotation matrix R_av as in the paper
(eqs. (32)–(37)).  This avoids tracking nodal rotation matrices entirely and
yields a simpler, explicit B matrix.  The trade-off is that the geometric
stiffness includes only the contributions from the chord-direction terms of B
(N-term and moment terms); the paper's additional K_σ2 … K_σ5 contributions
from the variation of the rotation matrices are absent.  For the conservative
problems considered here (moment loading, tip rotations) the missing terms
vanish at convergence and do not affect accuracy.

References
----------
Crisfield, M. A. (1990). A consistent co-rotational formulation for non-linear,
three-dimensional, beam-elements. *Computer Methods in Applied Mechanics and
Engineering*, 81(2), 131–150.  [cited as 'C90' in inline comments]
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
    alpha, l0, ln, c, s, sin_a, cos_a : np.ndarray, each shape (M,)
        ``sin_a`` and ``cos_a`` are the sine and cosine of the chord rotation
        angle *α*; they are returned to allow callers to form wrap-safe
        differences ``θ − α`` using the angle-difference identities.
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

    return alpha, l0, ln, c, s, sin_a, cos_a


def _rigid_body_rotation_3d(
        nodes: np.ndarray,
        d: np.ndarray,
        edges: np.ndarray,
        ref_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised co-rotating frame for all M elements in 3D.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 3)
        Reference (undeformed) nodal coordinates.
    d : np.ndarray, shape (N, 6)
        Current nodal DOFs ``[ux, uy, uz, θx, θy, θz]`` per node.
    edges : np.ndarray, shape (M, 2)
        Edge connectivity.
    ref_vectors : np.ndarray, shape (M, 3)
        One reference vector per element used to define the local y-axis.
        Must not be parallel to the deformed chord of any element.

    Returns
    -------
    R : np.ndarray, shape (M, 3, 3)
        Co-rotating frame for each element.  ``R[m, i, :]`` is the i-th
        local axis expressed in global coordinates:
        row 0 → e1 (axial / chord direction),
        row 1 → e2 (local y, perpendicular to chord in the ref-vector plane),
        row 2 → e3 = e1 × e2.
    l0 : np.ndarray, shape (M,)
        Reference (undeformed) element lengths.
    ln : np.ndarray, shape (M,)
        Deformed element lengths.
    """
    e0, e1 = edges[:, 0], edges[:, 1]

    dr = nodes[e1] - nodes[e0]               # (M, 3) reference chord
    l0 = np.linalg.norm(dr, axis=1)          # (M,)

    rn0 = nodes[e0] + d[e0, :3]             # (M, 3) deformed position node 0
    rn1 = nodes[e1] + d[e1, :3]             # (M, 3) deformed position node 1
    drn = rn1 - rn0                           # (M, 3) deformed chord
    ln = np.linalg.norm(drn, axis=1)          # (M,)

    # e1: deformed chord unit vector — C90 eq. (23)
    ax1 = drn / ln[:, None]                   # (M, 3)

    # e2: ref_vector projected perpendicular to ax1, then normalised.
    # This is the Gram-Schmidt alternative to the paper's average-rotation-matrix
    # construction (C90 eqs. (32)–(37)); see module docstring.
    # Crisfield book eq. (17.57).
    dot = np.sum(ref_vectors * ax1, axis=1, keepdims=True)   # (M, 1)
    perp = ref_vectors - dot * ax1                            # (M, 3)
    ax2 = perp / np.linalg.norm(perp, axis=1, keepdims=True)  # (M, 3)

    ax3 = np.cross(ax1, ax2)                  # (M, 3) e3 = e1 × e2

    R = np.stack([ax1, ax2, ax3], axis=1)     # (M, 3, 3)

    return R, l0, ln


def _local_stiffness_3d(beam_prop: dict, l0: np.ndarray) -> np.ndarray:
    """Vectorised 7×7 local stiffness for all M elements in 3D.

    Local deformational DOFs (7 per element)::

        [Δl, θx0, θy0, θz0, θx1, θy1, θz1]

    where θx is torsion (about the local axial axis e1), θy is bending
    about the local e2 axis (uses EIy), and θz is bending about the local
    e3 axis (uses EIz).

    This ordering follows C90: the axial DOF u_l (eq. (25)) and six
    rotational DOFs θ_i (eq. (28)) together form the vector of local
    'strains'.  The 6×6 rotational sub-block corresponds to the matrix D
    (eq. (31)), here extended to Timoshenko shear deformation.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    l0 : np.ndarray, shape (M,)
        Reference element lengths.

    Returns
    -------
    Kl : np.ndarray, shape (M, 7, 7)
    """
    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2. * (1. + nu))
    Iy, Iz, J, A, kappa, _ = get_geometric_props(beam_prop)

    kGA = kappa * G * A

    # Timoshenko shear parameters (zeroed for Euler-Bernoulli elements)
    if beam_prop.get('euler_bernoulli', False):
        PhiY = np.zeros_like(l0)
        PhiZ = np.zeros_like(l0)
    else:
        PhiY = 12. * E * Iz / (kGA * l0**2)   # for bending about e3 (uses Iz)  (M,)
        PhiZ = 12. * E * Iy / (kGA * l0**2)   # for bending about e2 (uses Iy)  (M,)

    gamma = E * A / l0                                          # axial        (M,)
    alpha = G * J / l0                                          # torsion      (M,)
    psi_y = (4. + PhiY) * E * Iz / (l0 * (1. + PhiY))         # bend-z diag  (M,)
    xi_y = (2. - PhiY) * E * Iz / (l0 * (1. + PhiY))         # bend-z cross (M,)
    psi_z = (4. + PhiZ) * E * Iy / (l0 * (1. + PhiZ))         # bend-y diag  (M,)
    xi_z = (2. - PhiZ) * E * Iy / (l0 * (1. + PhiZ))         # bend-y cross (M,)

    M = len(l0)
    Kl = np.zeros((M, 7, 7))

    Kl[:, 0, 0] = gamma                    # axial

    Kl[:, 1, 1] = alpha                    # torsion, node 0
    Kl[:, 4, 4] = alpha                    # torsion, node 1
    Kl[:, 1, 4] = -alpha                   # torsion coupling
    Kl[:, 4, 1] = -alpha

    Kl[:, 2, 2] = psi_z                    # bending about e2, node 0
    Kl[:, 5, 5] = psi_z                    # bending about e2, node 1
    Kl[:, 2, 5] = xi_z                     # bending-y cross coupling
    Kl[:, 5, 2] = xi_z

    Kl[:, 3, 3] = psi_y                    # bending about e3, node 0
    Kl[:, 6, 6] = psi_y                    # bending about e3, node 1
    Kl[:, 3, 6] = xi_y                     # bending-z cross coupling
    Kl[:, 6, 3] = xi_y

    return Kl


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


def _b_matrix_3d(R: np.ndarray, ln: np.ndarray) -> np.ndarray:
    """Vectorised 7×12 kinematic (B) matrix for all M elements in 3D.

    Maps the 12 global element DOFs
    ``[u0(0:3), θ0(3:6), u1(6:9), θ1(9:12)]``
    to the 7 local deformational DOFs
    ``[Δl, θx0_l, θy0_l, θz0_l, θx1_l, θy1_l, θz1_l]``.

    This is the linearised kinematic operator δp_l = B δp, where δp_l
    contains the variations of the local DOFs defined in C90 eqs. (25) and
    (28).  The axial row is the g-vector of C90 eq. (57).  Because e2 and e3
    are defined here by Gram-Schmidt projection of a fixed reference vector
    (rather than by the average nodal rotation matrix of C90 eqs. (32)–(37)),
    the bending rows take a simpler form than the paper's F matrix
    (eqs. (54)–(55)): the chord-rotation contribution to each bending DOF is
    ±e3/ln (for θy) or ±e2/ln (for θz), and the nodal rotation contribution
    is a direct projection onto e2 or e3.

    The translation rows follow the same "chord-rotation" logic as in 2D:
    moving a node in the e3 direction rotates the chord about e2, which
    contributes to θy_local; moving it in e2 rotates about −e3, contributing
    to θz_local.  Torsion (θx_l) has no translation contribution.

    Parameters
    ----------
    R : np.ndarray, shape (M, 3, 3)
        Co-rotating frame.  ``R[m, i, :]`` is the i-th local axis in global
        coordinates: row 0 → e1 (chord), row 1 → e2, row 2 → e3.
    ln : np.ndarray, shape (M,)
        Deformed element lengths.

    Returns
    -------
    B : np.ndarray, shape (M, 7, 12)
    """
    M = len(ln)
    B = np.zeros((M, 7, 12))

    e1 = R[:, 0, :]                    # (M, 3)
    e2 = R[:, 1, :]                    # (M, 3)
    e3 = R[:, 2, :]                    # (M, 3)

    e2_over_ln = e2 / ln[:, None]      # (M, 3)
    e3_over_ln = e3 / ln[:, None]      # (M, 3)

    # Row 0: axial elongation Δl — C90 eq. (57): g = [-e1, 0, e1, 0]
    B[:, 0, 0:3] = -e1
    B[:, 0, 6:9] = e1

    # Rows 1, 4: torsion θx_l = e1 · θ_global (projection onto chord)
    B[:, 1, 3:6] = R[:, 0, :]         # node 0: e1
    B[:, 4, 9:12] = R[:, 0, :]        # node 1: e1

    # Rows 2, 5: bending about e2 / local y (θy_l)
    # Translation: moving a node in the e3 direction rotates the chord about e2.
    # Rotation: direct projection of the nodal rotation onto e2.
    B[:, 2, 0:3] = -e3_over_ln        # node 0 translation (−e3/ln)
    B[:, 2, 3:6] = R[:, 1, :]         # node 0 rotation: e2
    B[:, 2, 6:9] = e3_over_ln         # node 1 translation (+e3/ln)
    B[:, 5, 0:3] = -e3_over_ln        # node 0 translation (−e3/ln)
    B[:, 5, 6:9] = e3_over_ln         # node 1 translation (+e3/ln)
    B[:, 5, 9:12] = R[:, 1, :]        # node 1 rotation: e2

    # Rows 3, 6: bending about e3 / local z (θz_l)
    # Translation: moving a node in the e2 direction rotates the chord about −e3.
    # Rotation: direct projection of the nodal rotation onto e3.
    B[:, 3, 0:3] = e2_over_ln         # node 0 translation (+e2/ln)
    B[:, 3, 3:6] = R[:, 2, :]         # node 0 rotation: e3
    B[:, 3, 6:9] = -e2_over_ln        # node 1 translation (−e2/ln)
    B[:, 6, 0:3] = e2_over_ln         # node 0 translation (+e2/ln)
    B[:, 6, 6:9] = -e2_over_ln        # node 1 translation (−e2/ln)
    B[:, 6, 9:12] = R[:, 2, :]        # node 1 rotation: e3

    return B


def _material_stiffness_3d(
        R: np.ndarray,
        ln: np.ndarray,
        beam_prop: dict,
        l0: np.ndarray,
) -> np.ndarray:
    """Vectorised material (elastic) stiffness ``Km = B^T Kl B`` for all M elements.

    Parameters
    ----------
    R : np.ndarray, shape (M, 3, 3)
        Co-rotating frame from :func:`_rigid_body_rotation_3d`.
    ln : np.ndarray, shape (M,)
        Deformed element lengths.
    beam_prop : dict
        Beam cross-section and elastic properties.
    l0 : np.ndarray, shape (M,)
        Reference element lengths (used to build *Kl*).

    Returns
    -------
    Km : np.ndarray, shape (M, 12, 12)
    """
    Kl = _local_stiffness_3d(beam_prop, l0)   # (M, 7, 7)
    B = _b_matrix_3d(R, ln)                  # (M, 7, 12)
    BT = B.transpose(0, 2, 1)                 # (M, 12, 7)
    return BT @ Kl @ B                        # (M, 12, 12)


def _geometric_stiffness_3d(
        R: np.ndarray,
        ln: np.ndarray,
        fl: np.ndarray,
) -> np.ndarray:
    """Vectorised geometric stiffness Kg for all M 3D beam elements.

    Includes contributions from the axial force N (string stiffness in both
    transverse directions), the bending moments Mz0/Mz1 (about e3), and the
    bending moments My0/My1 (about e2).  Torsional geometric stiffness is
    neglected.

    This is the portion of the full geometric stiffness K_σ (C90 eq. (62))
    that arises from the chord-direction terms of B: specifically the g-vector
    (axial row, eq. (57)) and the ±e2/ln, ±e3/ln translation entries of the
    bending rows.  The remaining K_σ2 … K_σ5 contributions in eq. (62), which
    stem from the variation of the average nodal rotation matrix R_av, are
    absent because we use a fixed reference vector for e2, e3.

    Derivation: ``Kg = fl^T ∂B/∂p``, retaining only the chord-dependent terms.
    Defining translation-only 12-vectors::

        E_x = [-x, 0_rot, +x, 0_rot]  (x ∈ {e1, e2, e3})

    the three contributions are (C90 eq. (64) for the N-term)::

        Kg_N  =  N/ln  * (E2⊗E2 + E3⊗E3)          — C90 eq. (64): K₁₁ = NA
        Kg_Mz = +(Mz0+Mz1)/ln² * (E1⊗E2 + E2⊗E1)
        Kg_My = -(My0+My1)/ln² * (E1⊗E3 + E3⊗E1)

    The negative sign of Kg_My reflects that the e3-transverse term in the
    θy row of B carries a minus sign whereas the e2-transverse term in the
    θz row carries a plus sign (see B rows 2 and 3 above).

    Parameters
    ----------
    R : np.ndarray, shape (M, 3, 3)
        Co-rotating frame from :func:`_rigid_body_rotation_3d`.
    ln : np.ndarray, shape (M,)
        Deformed element lengths.
    fl : np.ndarray, shape (M, 7)
        Local element forces ``[N, Tx0, My0, Mz0, Tx1, My1, Mz1]``.

    Returns
    -------
    Kg : np.ndarray, shape (M, 12, 12)
    """
    M_el = len(ln)
    e1 = R[:, 0, :]   # (M, 3)
    e2 = R[:, 1, :]   # (M, 3)
    e3 = R[:, 2, :]   # (M, 3)

    N = fl[:, 0]
    My0 = fl[:, 2]
    Mz0 = fl[:, 3]
    My1 = fl[:, 5]
    Mz1 = fl[:, 6]

    zeros3 = np.zeros((M_el, 3))

    # 12-component translation-only vectors (rotation slots are zero):
    #   E_x = [-x, 0, +x, 0]  shape (M, 12)
    E1 = np.concatenate([-e1, zeros3,  e1, zeros3], axis=1)
    E2 = np.concatenate([-e2, zeros3,  e2, zeros3], axis=1)
    E3 = np.concatenate([-e3, zeros3,  e3, zeros3], axis=1)

    inv_ln = 1. / ln
    inv_ln2 = inv_ln ** 2

    # Axial force term: N/ln * (E2⊗E2 + E3⊗E3)
    Kg = (N * inv_ln)[:, None, None] * (
        np.einsum('mi,mj->mij', E2, E2) + np.einsum('mi,mj->mij', E3, E3)
    )

    # Mz bending term: +(Mz0+Mz1)/ln² * (E1⊗E2 + E2⊗E1)
    E1E2 = np.einsum('mi,mj->mij', E1, E2)
    Kg += ((Mz0 + Mz1) * inv_ln2)[:, None, None] * (E1E2 + E1E2.transpose(0, 2, 1))

    # My bending term: -(My0+My1)/ln² * (E1⊗E3 + E3⊗E1)
    E1E3 = np.einsum('mi,mj->mij', E1, E3)
    Kg -= ((My0 + My1) * inv_ln2)[:, None, None] * (E1E3 + E1E3.transpose(0, 2, 1))

    return Kg


def _element_tangent_3d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
        ref_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised tangent stiffness and internal forces for all M 3D elements.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 3)
        Reference nodal coordinates (updated each load step, UL sense).
    edges : np.ndarray, shape (M, 2)
        Edge connectivity.
    sol : np.ndarray, shape (6*N,)
        Current incremental displacement vector ``[ux, uy, uz, θx, θy, θz, …]``
        measured from *nodes*.
    beam_prop : dict
        Beam cross-section and elastic properties.
    ref_vectors : np.ndarray, shape (M, 3)
        Reference vectors defining the local e2 axis per element.

    Returns
    -------
    Kt : np.ndarray, shape (M, 12, 12)
        Element tangent stiffness matrices in the global frame.
    fg : np.ndarray, shape (M, 12)
        Element internal force vectors in the global frame.
    """

    d = sol.reshape(-1, 6)                                         # (N, 6)

    R, l0, ln = _rigid_body_rotation_3d(nodes, d, edges, ref_vectors)

    Kl = _local_stiffness_3d(beam_prop, l0)                       # (M, 7, 7)

    # Exact local deformational DOFs — C90 eqs. (25) and (28)
    ul = _exact_local_dofs_3d(nodes, d, edges, R, l0, ln)         # (M, 7)
    # Local element forces fl = Kl ul — C90 eqs. (27) and (30)
    fl = np.einsum('mij,mj->mi', Kl, ul)                          # (M, 7)

    # Global internal force vector: fg = B^T fl — C90 eq. (58) / virtual work (39)
    B = _b_matrix_3d(R, ln)                                      # (M, 7, 12)
    fg = np.einsum('mji,mj->mi', B, fl)                           # (M, 12)

    # Material stiffness Km = B^T Kl B — C90 eq. (60): K_t1 = (EA/l0) gg^t + FDF^t
    BT = B.transpose(0, 2, 1)
    Km = BT @ Kl @ B                                              # (M, 12, 12)

    # Geometric stiffness Kg — C90 eq. (59): K = K_t1 + K_σ  (simplified K_σ)
    Kg = _geometric_stiffness_3d(R, ln, fl)                       # (M, 12, 12)

    return Km + Kg, fg


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

    alpha, l0, ln, c, s, sin_a, cos_a = _rigid_body_rotation_2d(nodes, d, edges)
    Kl = _local_stiffness_2d(beam_prop, l0)    # (M, 3, 3)

    # Local deformational DOFs: ul = [ln-l0, θ0-α, θ1-α], shape (M, 3).
    # The bending DOFs use a wrap-safe angle-difference formula so that
    # accumulated nodal rotations > π (Total Lagrangian) are handled
    # correctly: arctan2(sin(θ−α), cos(θ−α)) via the trig-difference identity.
    theta0 = d[e0, 2]
    theta1 = d[e1, 2]
    ul_b0 = np.arctan2(np.sin(theta0) * cos_a - np.cos(theta0) * sin_a,
                       np.cos(theta0) * cos_a + np.sin(theta0) * sin_a)
    ul_b1 = np.arctan2(np.sin(theta1) * cos_a - np.cos(theta1) * sin_a,
                       np.cos(theta1) * cos_a + np.sin(theta1) * sin_a)
    ul = np.stack([ln - l0, ul_b0, ul_b1], axis=1)

    # Local internal forces: fl = Kl @ ul, shape (M, 3)
    fl = np.einsum('mij,mj->mi', Kl, ul)

    # Kinematic (B) matrix: (M, 3, 6)
    B = _b_matrix_2d(c, s, ln)

    # Global internal forces: fg = B^T fl, shape (M, 6)
    fg = np.einsum('mji,mj->mi', B, fl)

    # Material stiffness: Km = B^T Kl B, shape (M, 6, 6)
    BT = B.transpose(0, 2, 1)                      # (M, 6, 3)
    Km = BT @ Kl @ B                               # (M, 6, 6)

    # Geometric stiffness — Crisfield, Non-linear FEA Vol. 1, eqs. (3.28)–(3.30).
    # r: chord unit vector (6-DOF form); z: perpendicular unit vector.
    # Kg = (N/ln) z⊗z + ((M0+M1)/ln²) (r⊗z + z⊗r)
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

    _, l0, ln, _, _, sin_a, cos_a = _rigid_body_rotation_2d(nodes, d, edges)
    Kl = _local_stiffness_2d(beam_prop, l0)    # (M, 3, 3)

    theta0 = d[e0, 2]
    theta1 = d[e1, 2]

    ul_b0 = np.arctan2(np.sin(theta0) * cos_a - np.cos(theta0) * sin_a,
                       np.cos(theta0) * cos_a + np.sin(theta0) * sin_a)
    ul_b1 = np.arctan2(np.sin(theta1) * cos_a - np.cos(theta1) * sin_a,
                       np.cos(theta1) * cos_a + np.sin(theta1) * sin_a)
    ul = np.stack([ln - l0, ul_b0, ul_b1], axis=1)
    return np.einsum('mij,mj->mi', Kl, ul)


def _exact_local_dofs_3d(
        nodes: np.ndarray,
        d: np.ndarray,
        edges: np.ndarray,
        R: np.ndarray,
        l0: np.ndarray,
        ln: np.ndarray,
) -> np.ndarray:
    """Exact 3D local deformational DOFs (non-linearised).

    The axial DOF is Δl = ln − l0 (C90 eq. (25)).  The rotational DOFs
    replace the paper's 2 sin θ relations (C90 eq. (28)) with an equivalent
    chord-rotation angle formula: the chord rotation pseudo-vector
    w = e0 × e1 (proportional to sin of the chord-rotation angle) is
    projected onto e2 and e3 to obtain the generalised chord-rotation angles
    α_e2 and α_e3.  These are then subtracted from the nodal rotation
    projections, mirroring the 2D formula θ_local = θ_nodal − α.

    The resulting six bending/torsion DOFs correspond to the θ_i of C90
    eq. (28) and, together with Δl, to the full local 'strain' vector.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 3)
        Reference nodal coordinates.
    d : np.ndarray, shape (N, 6)
        Current nodal DOFs ``[ux, uy, uz, θx, θy, θz]``.
    edges : np.ndarray, shape (M, 2)
    R : np.ndarray, shape (M, 3, 3)
        Co-rotating frame from :func:`_rigid_body_rotation_3d`.
    l0, ln : np.ndarray, shape (M,)
        Reference and deformed element lengths.

    Returns
    -------
    ul : np.ndarray, shape (M, 7)
        ``[Δl, θx0_l, θy0_l, θz0_l, θx1_l, θy1_l, θz1_l]``
    """
    e0i, e1i = edges[:, 0], edges[:, 1]

    # Reference chord unit vector (not available directly from R)
    dr = nodes[e1i] - nodes[e0i]                   # (M, 3)
    e0_hat = dr / l0[:, None]                       # (M, 3)

    e1_hat = R[:, 0, :]                             # (M, 3) deformed chord
    e2_hat = R[:, 1, :]                             # (M, 3)
    e3_hat = R[:, 2, :]                             # (M, 3)

    # Chord rotation pseudo-vector w = e0 × e1 (proportional to sin of angle)
    w = np.cross(e0_hat, e1_hat)                    # (M, 3)
    s = np.linalg.norm(w, axis=1)                   # (M,) = sin(chord angle)
    c = np.einsum('mi,mi->m', e0_hat, e1_hat)       # (M,) = cos(chord angle)

    # Scale = angle / sin(angle); guard against exact division by zero only.
    # For s near machine epsilon (chord ≈ 180°), the ratio phi/s * we2 is still
    # well-conditioned because |we2| ≤ s, so the product is bounded by phi ≤ π.
    # Using s > 0 (rather than a larger threshold) avoids the discontinuity at
    # the old threshold that occurred when a Newton iterate landed at s ≈ 1e-15.
    safe_s = np.where(s > 0, s, 1.)
    scale = np.where(s > 0, np.arctan2(s, c) / safe_s, 1.)  # (M,)

    # Chord rotation components about e2 and e3 (generalised α)
    alpha_e2 = scale * np.einsum('mi,mi->m', w, e2_hat)    # (M,)
    alpha_e3 = scale * np.einsum('mi,mi->m', w, e3_hat)    # (M,)

    theta0 = d[e0i, 3:6]                            # (M, 3)
    theta1 = d[e1i, 3:6]                            # (M, 3)

    # Bending DOFs use a wrap-safe angle-difference formula analogous to 2D:
    #   arctan2(sin(θ_proj − α), cos(θ_proj − α))
    # This handles accumulated nodal rotations > π correctly (TL mode).
    cos_ae2 = np.cos(alpha_e2)
    sin_ae2 = np.sin(alpha_e2)
    cos_ae3 = np.cos(alpha_e3)
    sin_ae3 = np.sin(alpha_e3)

    te2_0 = np.einsum('mi,mi->m', e2_hat, theta0)
    te3_0 = np.einsum('mi,mi->m', e3_hat, theta0)
    te2_1 = np.einsum('mi,mi->m', e2_hat, theta1)
    te3_1 = np.einsum('mi,mi->m', e3_hat, theta1)

    def _wdiff(tp, ca, sa):
        return np.arctan2(np.sin(tp) * ca - np.cos(tp) * sa,
                          np.cos(tp) * ca + np.sin(tp) * sa)

    return np.stack([
        ln - l0,                                               # Δl (exact)
        np.einsum('mi,mi->m', e1_hat, theta0),                # θx0_l (torsion)
        _wdiff(te2_0, cos_ae2, sin_ae2),                      # θy0_l
        _wdiff(te3_0, cos_ae3, sin_ae3),                      # θz0_l
        np.einsum('mi,mi->m', e1_hat, theta1),                # θx1_l (torsion)
        _wdiff(te2_1, cos_ae2, sin_ae2),                      # θy1_l
        _wdiff(te3_1, cos_ae3, sin_ae3),                      # θz1_l
    ], axis=1)  # (M, 7)


def compute_element_forces_3d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
        ref_vectors: np.ndarray,
) -> np.ndarray:
    """Local element forces for all 3D co-rotational beam elements.

    Uses a Total Lagrangian interpretation: *nodes* are the original
    (undeformed) coordinates and *sol* is the total displacement from those
    coordinates.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 3)
        Original (undeformed) nodal coordinates.
    edges : np.ndarray, shape (M, 2)
        Edge connectivity.
    sol : np.ndarray, shape (6*N,)
        Total displacement ``[ux, uy, uz, θx, θy, θz, ...]`` from *nodes*.
    beam_prop : dict
        Beam cross-section and elastic properties.
    ref_vectors : np.ndarray, shape (M, 3)
        Reference vectors defining the local e2 axis per element.

    Returns
    -------
    forces : np.ndarray, shape (M, 7)
        Local element forces ``[N, Tx, My0, Mz0, Tx1, My1, Mz1]`` for each
        element, where *N* is the axial force, *Tx* the torsional moment at
        node 0, *My0*/*Mz0* the bending moments at node 0 about the local e2
        and e3 axes respectively, and similarly for node 1.
    """
    d = sol.reshape(-1, 6)                                     # (N, 6)
    R, l0, ln = _rigid_body_rotation_3d(nodes, d, edges, ref_vectors)
    Kl = _local_stiffness_3d(beam_prop, l0)                    # (M, 7, 7)
    ul = _exact_local_dofs_3d(nodes, d, edges, R, l0, ln)      # (M, 7)
    return np.einsum('mij,mj->mi', Kl, ul)                     # (M, 7)


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


def element_mises_stress_3d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
        ref_vectors: np.ndarray,
) -> np.ndarray:
    """Von Mises equivalent stress per element from 3D co-rotational forces.

    Evaluates the stress at both element ends and returns the maximum.

    At each end the normal stress is the superposition of axial and both
    bending contributions evaluated at the extreme fibre::

        σ_i = |N / A| + |My_i| * y_max / Iy + |Mz_i| * y_max / Iz

    The torsional shear stress at the extreme fibre is::

        τ_i = |Tx_i| * y_max / Ip

    and von Mises reduces to::

        σ_VM,i = sqrt(σ_i² + 3 τ_i²)

    Local force ordering (output of :func:`compute_element_forces_3d`)::

        [N, Tx0, My0, Mz0, Tx1, My1, Mz1]

    where *My* is the moment about the local e2 axis (uses Iy) and *Mz* is
    the moment about the local e3 axis (uses Iz).

    Parameters
    ----------
    nodes, edges, sol, beam_prop, ref_vectors :
        Same as :func:`compute_element_forces_3d`.

    Returns
    -------
    svm : np.ndarray, shape (M,)
        Von Mises equivalent stress per element.
    """
    from beam_networks.geometry.geo import get_geometric_props
    Iy, Iz, Ip, A, _, ymax = get_geometric_props(beam_prop)
    forces = compute_element_forces_3d(nodes, edges, sol, beam_prop, ref_vectors)
    N = forces[:, 0]
    Tx0 = forces[:, 1]
    My0 = forces[:, 2]
    Mz0 = forces[:, 3]
    Tx1 = forces[:, 4]
    My1 = forces[:, 5]
    Mz1 = forces[:, 6]

    sigma_a = np.abs(N) / A
    sigma_0 = sigma_a + np.abs(My0) * ymax / Iy + np.abs(Mz0) * ymax / Iz
    sigma_1 = sigma_a + np.abs(My1) * ymax / Iy + np.abs(Mz1) * ymax / Iz
    tau_0 = np.abs(Tx0) * ymax / Ip
    tau_1 = np.abs(Tx1) * ymax / Ip

    svm_0 = np.sqrt(sigma_0**2 + 3. * tau_0**2)
    svm_1 = np.sqrt(sigma_1**2 + 3. * tau_1**2)
    return np.maximum(svm_0, svm_1)


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
        Reference nodal coordinates.  In a Total Lagrangian scheme these are
        the original (undeformed) coordinates; in an Updated Lagrangian scheme
        they are the committed reference from the previous load step.
    edges : np.ndarray, shape (M, 2)
        Edge connectivity (integer node-index pairs). Must be sorted
        (``edges[:, 0] < edges[:, 1]``) when *matrix* is ``'bsr'``.
    sol : np.ndarray, shape (3*N,)
        Current displacement vector ``[ux, uy, θ_z, ...]`` measured from
        *nodes*.
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


# ---------------------------------------------------------------------------
# 3D assembly
# ---------------------------------------------------------------------------

def _assemble_dense_nonlinear_3d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
        ref_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense assembly of global tangent stiffness and internal force (3D)."""
    ndof = nodes.shape[0] * 6
    K = np.zeros((ndof, ndof))
    F_int = np.zeros(ndof)

    e0, e1 = edges[:, 0], edges[:, 1]
    Kt_all, fg_all = _element_tangent_3d(nodes, edges, sol, beam_prop, ref_vectors)

    # Scatter internal forces (6 DOFs per node)
    for i in range(6):
        np.add.at(F_int, e0 * 6 + i, fg_all[:, i])
        np.add.at(F_int, e1 * 6 + i, fg_all[:, 6 + i])

    # Scatter 6×6 stiffness blocks
    for i in range(6):
        for j in range(6):
            np.add.at(K, (e0 * 6 + i, e0 * 6 + j), Kt_all[:, i,     j])
            np.add.at(K, (e0 * 6 + i, e1 * 6 + j), Kt_all[:, i,     6 + j])
            np.add.at(K, (e1 * 6 + i, e0 * 6 + j), Kt_all[:, 6 + i, j])
            np.add.at(K, (e1 * 6 + i, e1 * 6 + j), Kt_all[:, 6 + i, 6 + j])

    return K, F_int


def _assemble_bsr_nonlinear_3d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
        ref_vectors: np.ndarray,
) -> tuple[sp.bsr_array, np.ndarray]:
    """BSR sparse assembly of global tangent stiffness and internal force (3D).

    Mirrors the 2D BSR assembler with 6×6 blocks instead of 3×3.
    """
    num_nodes = nodes.shape[0]
    ndof_per_node = 6
    ndof = num_nodes * ndof_per_node

    # Ensure n0 < n1 and rows are sorted by n0
    edges_sorted = np.sort(edges, axis=1)
    edges_sorted = edges_sorted[np.lexsort((edges_sorted[:, 1], edges_sorted[:, 0]))]
    e0s, e1s = edges_sorted[:, 0], edges_sorted[:, 1]

    # Sort ref_vectors to match the re-ordered edges
    orig_order = np.lexsort((np.sort(edges, axis=1)[:, 1],
                             np.sort(edges, axis=1)[:, 0]))
    ref_sorted = ref_vectors[orig_order]

    # Build sparsity pattern: upper triangle + diagonal
    aux = sp.csr_array(
        (np.ones(len(edges_sorted)), (e0s, e1s)),
        shape=(num_nodes, num_nodes),
    )
    aux = aux + sp.eye_array(num_nodes)
    indices = aux.indices
    indptr = aux.indptr

    data = np.zeros((len(indices), ndof_per_node, ndof_per_node))
    F_int = np.zeros(ndof)

    Kt_all, fg_all = _element_tangent_3d(
        nodes, edges_sorted, sol, beam_prop, ref_sorted)

    diag_pos_n0 = indptr[e0s]
    diag_pos_n1 = indptr[e1s]

    _, first_occ, c0s = np.unique(e0s, return_index=True, return_counts=True)
    k_per_edge = np.arange(len(e0s)) - np.repeat(first_occ, c0s)
    offdiag_pos = indptr[e0s] + 1 + k_per_edge

    np.add.at(data, diag_pos_n0, Kt_all[:, :6, :6] / 2.)
    np.add.at(data, diag_pos_n1, Kt_all[:, 6:, 6:] / 2.)
    np.add.at(data, offdiag_pos, Kt_all[:, :6, 6:])

    for i in range(6):
        np.add.at(F_int, e0s * 6 + i, fg_all[:, i])
        np.add.at(F_int, e1s * 6 + i, fg_all[:, 6 + i])

    K = sp.bsr_array(
        (data, indices, indptr),
        shape=(ndof, ndof),
        blocksize=(ndof_per_node, ndof_per_node),
    )
    K = K + K.T

    return K, F_int


def assemble_nonlinear_system_3d(
        nodes: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
        beam_prop: dict,
        ref_vectors: np.ndarray,
        matrix: str = 'dense',
) -> tuple[np.ndarray | sp.bsr_array, np.ndarray]:
    """Assemble global tangent stiffness and internal force for a 3D network.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 3)
        Current reference nodal coordinates (Updated Lagrangian).
    edges : np.ndarray, shape (M, 2)
        Edge connectivity. Must be sorted (``edges[:, 0] < edges[:, 1]``)
        when *matrix* is ``'bsr'``.
    sol : np.ndarray, shape (6*N,)
        Current incremental displacement ``[ux, uy, uz, θx, θy, θz, …]``
        measured from *nodes*.
    beam_prop : dict
        Beam cross-section and elastic properties.
    ref_vectors : np.ndarray, shape (M, 3)
        Reference vectors defining the local e2 axis per element.  Must not
        be parallel to the chord of any element.
    matrix : {'dense', 'bsr'}, optional
        Storage format for the global stiffness matrix.  The default is
        ``'dense'``.

    Returns
    -------
    K : np.ndarray or scipy.sparse.bsr_array, shape (6*N, 6*N)
        Global tangent stiffness matrix.
    F_int : np.ndarray, shape (6*N,)
        Global internal force vector.
    """
    if matrix == 'dense':
        return _assemble_dense_nonlinear_3d(nodes, edges, sol, beam_prop,
                                            ref_vectors)
    elif matrix == 'bsr':
        return _assemble_bsr_nonlinear_3d(nodes, edges, sol, beam_prop,
                                          ref_vectors)
    else:
        raise ValueError(
            f"Unknown matrix format '{matrix}'. Choose 'dense' or 'bsr'."
        )
