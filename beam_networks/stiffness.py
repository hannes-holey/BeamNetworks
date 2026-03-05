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

from beam_networks.geo import get_geometric_props, get_geometric_props_derivative
from beam_networks.fem_utils import _gauss_legendre, _lagrange_basis


def _beam_stiffness_2d(beam_prop, L, derivative=None):
    """Element stiffness matrix 2D (local frame).

    Parameters
    ----------
    beam_prop : dict
        Beam properties (cross section and elastic constants)
    L : float or np.ndrarray
        Length of beam(s)
    derivative : None or int
        if not None returns the derivative with respect to a shape parameter.
        The shape parameter is determined by the value of derivative which
        depends on the shape of your beam.

    Returns
    -------
    list
        Entries of the stiffness matrix
    list
        Row indices
    list
        Column indices
    """

    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2 * (1. + nu))

    Iy, Iz, _, A, kappa, _ = get_geometric_props(beam_prop)

    PhiY = 12 * E * Iz / (kappa * G * A * L**2)

    if derivative is None:

        gamma = E * A / L
        zeta = 12. * E * Iz / (L**3 * (1. + PhiY))
        lamb = 6. * E * Iz / (L**2 * (1. + PhiY))
        psi = (4. + PhiY) * E * Iz / (L * (1. + PhiY))
        xi = (2. - PhiY) * E * Iz / (L * (1. + PhiY))
    else:
        dIy, dIz, _, dA, _, _ = get_geometric_props_derivative(beam_prop,
                                                               derivative)

        dPhiY = ((12 * E * dIz) * (kappa * G * A * L**2) -
                 (12 * E * Iz) * (kappa * G * dA * L**2)) / \
                (kappa * G * A * L**2)**2
        gamma = E * dA / L
        zeta = ((12. * E * dIz) * (L**3 * (1. + PhiY)) -
                (12. * E * Iz) * (L**3 * dPhiY)) / \
               (L**3 * (1. + PhiY))**2
        lamb = ((6. * E * dIz) * (L**2 * (1. + PhiY)) -
                (6. * E * Iz) * (L**2 * dPhiY)) /\
            (L**2 * (1. + PhiY))**2
        psi = (E * ((4. + PhiY) * dIz + dPhiY * Iz) * (L * (1. + PhiY)) -
               ((4. + PhiY) * E * Iz) * (L * dPhiY)) /\
              (L * (1. + PhiY))**2
        xi = (E*((2. - PhiY) * dIz - dPhiY * Iz) * (L * (1. + PhiY)) -
              ((2. - PhiY) * E * Iz) * (L * dPhiY)) / (L * (1. + PhiY))**2

    rows = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    cols = [0, 3, 1, 2, 4, 5, 1, 2, 4, 5, 0, 3, 1, 2, 4, 5, 1, 2, 4, 5]
    data = [gamma, -gamma, zeta, lamb, -zeta, lamb, lamb, psi, -lamb, xi,
            -gamma, gamma, -zeta, -lamb, zeta, -lamb, lamb, xi, -lamb, psi]

    return data, rows, cols


def _beam_stiffness_3d(beam_prop, L, derivative=None):
    """Element stiffness matrix 3D (local frame).

    See e.g.
    Przemieniecki J.S., Theory of Matrix Structural Analysis, McGraw-Hill 1968

    Parameters
    ----------
    beam_prop : dict
        Beam properties (cross section and elastic constants)
    L : float or np.ndrarray
        Length of beam(s)
    derivative : None or int
        if not None returns the derivative with respect to a shape parameter.
        The shape parameter is determined by the value of derivative which
        depends on the shape of your beam.

    Returns
    -------
    list
        Entries of the stiffness matrix
    list
        Row indices
    list
        Column indices
    """

    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2 * (1. + nu))

    Iy, Iz, J, A, kappa, _ = get_geometric_props(beam_prop)

    PhiY = 12 * E * Iz / (kappa * G * A * L**2)
    PhiZ = 12 * E * Iy / (kappa * G * A * L**2)

    if derivative is None:
        gamma = E * A / L
        alpha = G * J / L
        zeta_y = 12. * E * Iz / (L**3 * (1. + PhiY))
        zeta_z = 12. * E * Iy / (L**3 * (1. + PhiZ))
        lamb_y = 6. * E * Iz / (L**2 * (1. + PhiY))
        lamb_z = 6. * E * Iy / (L**2 * (1. + PhiZ))
        psi_y = (4. + PhiY) * E * Iz / (L * (1. + PhiY))
        psi_z = (4. + PhiZ) * E * Iy / (L * (1. + PhiZ))
        xi_y = (2. - PhiY) * E * Iz / (L * (1. + PhiY))
        xi_z = (2. - PhiZ) * E * Iy / (L * (1. + PhiZ))
    else:
        dIy, dIz, dJ, dA, _, _ = get_geometric_props_derivative(beam_prop,
                                                                derivative)

        dPhiY = ((12 * E * dIz) * (kappa * G * A * L**2) -
                 (12 * E * Iz) * (kappa * G * dA * L**2)) / \
                (kappa * G * A * L**2)**2
        dPhiZ = ((12 * E * dIy) * (kappa * G * A * L**2) -
                 (12 * E * Iy) * (kappa * G * dA * L**2)) / \
                (kappa * G * A * L**2)**2
        #
        gamma = E * dA / L
        alpha = G * dJ / L
        zeta_y = ((12. * E * dIz) * (L**3 * (1. + PhiY)) -
                  (12. * E * Iz) * (L**3 * dPhiY)) / \
            (L**3 * (1. + PhiY))**2
        zeta_z = ((12. * E * dIy) * (L**3 * (1. + PhiZ)) -
                  (12. * E * Iy) * (L**3 * dPhiZ)) / \
            (L**3 * (1. + PhiZ))**2
        lamb_y = ((6. * E * dIz) * (L**2 * (1. + PhiY)) -
                  (6. * E * Iz) * (L**2 * dPhiY)) /\
            (L**2 * (1. + PhiY))**2
        lamb_z = ((6. * E * dIy) * (L**2 * (1. + PhiZ)) -
                  (6. * E * Iy) * (L**2 * dPhiZ)) /\
            (L**2 * (1. + PhiZ))**2
        psi_y = (E * ((4. + PhiY) * dIz + dPhiY * Iz) * (L * (1. + PhiY)) -
                 ((4. + PhiY) * E * Iz) * (L * dPhiY)) /\
            (L * (1. + PhiY))**2
        psi_z = (E * ((4. + PhiZ) * dIy + dPhiZ * Iy) * (L * (1. + PhiZ)) -
                 ((4. + PhiZ) * E * Iy) * (L * dPhiZ)) /\
            (L * (1. + PhiZ))**2
        xi_y = (E*((2. - PhiY) * dIz - dPhiY * Iz) * (L * (1. + PhiY)) -
                ((2. - PhiY) * E * Iz) * (L * dPhiY)) / (L * (1. + PhiY))**2
        xi_z = (E*((2. - PhiZ) * dIy - dPhiZ * Iy) * (L * (1. + PhiZ)) -
                ((2. - PhiZ) * E * Iy) * (L * dPhiZ)) / (L * (1. + PhiZ))**2

    row_col_diag = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    data_diag = [gamma, zeta_y, zeta_z, alpha, psi_z, psi_y, gamma, zeta_y, zeta_z, alpha, psi_z, psi_y]

    rows_offdiag = [4, 5, 6, 7, 7, 8, 8, 9, 10, 10, 10, 11, 11, 11]
    cols_offdiag = [2, 1, 0, 1, 5, 2, 4, 3, 2, 4, 8, 1, 5, 7]

    data_offdiag = [
        -lamb_z,
        lamb_y,
        -gamma,
        -zeta_y, -lamb_y,
        -zeta_z, lamb_z,
        -alpha,
        -lamb_z, xi_z, lamb_z,
        lamb_y, xi_y, -lamb_y]

    rows = row_col_diag + rows_offdiag + cols_offdiag
    cols = row_col_diag + cols_offdiag + rows_offdiag
    data = data_diag + data_offdiag + data_offdiag

    return data, rows, cols


def _fem_element_stiffness_2d(beam_prop, l, n_nodes, n_gauss):
    """2D Timoshenko beam element stiffness via Gauss quadrature.

    Builds K = ∫₀ˡ (EA Bε^T Bε + EIz Bκ^T Bκ + κGA Bγ^T Bγ) dx using
    *n_gauss* Gauss-Legendre points and Lagrange shape functions of degree
    *n_nodes - 1*.

    DOF ordering per node: [u, v, θ].  Total DOFs: 3 * n_nodes.

    Parameters
    ----------
    beam_prop : dict
        Beam properties.
    l : float
        Element length.
    n_nodes : int
        Nodes per element (= poly_order + 1).
    n_gauss : int
        Number of Gauss-Legendre integration points.

    Returns
    -------
    np.ndarray, shape (3*n_nodes, 3*n_nodes)
    """
    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2. * (1. + nu))
    _, Iz, _, A, kappa, _ = get_geometric_props(beam_prop)

    EA = E * A
    EI = E * Iz
    kGA = kappa * G * A

    n_dof = 3 * n_nodes
    K = np.zeros((n_dof, n_dof))

    xi_g, w_g = _gauss_legendre(n_gauss)
    jac = l / 2.

    for xi, w in zip(xi_g, w_g):
        N, dN_dxi = _lagrange_basis(n_nodes, xi)
        dN_dx = dN_dxi / jac        # chain rule: dN/dx = dN/dξ * dξ/dx = dN/dξ * 2/l

        Be = np.zeros(n_dof)        # axial:   ε  = du/dx
        Bk = np.zeros(n_dof)        # bending: κ  = dθ/dx
        Bs = np.zeros(n_dof)        # shear:   γ  = dv/dx − θ

        for i in range(n_nodes):
            Be[3 * i] = dN_dx[i]
            Bk[3 * i + 2] = dN_dx[i]
            Bs[3 * i + 1] = dN_dx[i]
            Bs[3 * i + 2] = -N[i]

        fac = w * jac
        K += fac * (EA * np.outer(Be, Be) + EI * np.outer(Bk, Bk) + kGA * np.outer(Bs, Bs))

    return K


def _fem_element_stiffness_3d(beam_prop, l, n_nodes, n_gauss):
    """3D Timoshenko beam element stiffness via Gauss quadrature.

    Builds K = ∫₀ˡ (EA Bε^T Bε + EIy Bκy^T Bκy + EIz Bκz^T Bκz
                    + κGA Bγxy^T Bγxy + κGA Bγxz^T Bγxz + GJ Bτ^T Bτ) dx.

    DOF ordering per node: [u, v, w, θx, θy, θz].  Total DOFs: 6 * n_nodes.

    Sign conventions (Przemieniecki):
      γ_xy = dv/dx − θz,   γ_xz = dw/dx + θy.

    Parameters
    ----------
    beam_prop : dict
        Beam properties.
    l : float
        Element length.
    n_nodes : int
        Nodes per element (= poly_order + 1).
    n_gauss : int
        Number of Gauss-Legendre integration points.

    Returns
    -------
    np.ndarray, shape (6*n_nodes, 6*n_nodes)
    """
    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2. * (1. + nu))
    Iy, Iz, J, A, kappa, _ = get_geometric_props(beam_prop)

    EA = E * A
    EIy = E * Iy
    EIz = E * Iz
    kGA = kappa * G * A
    GJ = G * J

    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))

    xi_g, w_g = _gauss_legendre(n_gauss)
    jac = l / 2.

    for xi, w in zip(xi_g, w_g):
        N, dN_dxi = _lagrange_basis(n_nodes, xi)
        dN_dx = dN_dxi / jac

        Be = np.zeros(n_dof)   # axial:    ε    = du/dx
        Bt = np.zeros(n_dof)   # torsion:  χ    = dθx/dx
        Bky = np.zeros(n_dof)   # bending:  κy   = dθy/dx
        Bkz = np.zeros(n_dof)   # bending:  κz   = dθz/dx
        Bsy = np.zeros(n_dof)   # shear xy: γ_xy = dv/dx  − θz
        Bsz = np.zeros(n_dof)   # shear xz: γ_xz = dw/dx  + θy

        for i in range(n_nodes):
            Be[6 * i] = dN_dx[i]   # u
            Bt[6 * i + 3] = dN_dx[i]   # θx
            Bky[6 * i + 4] = dN_dx[i]  # θy
            Bkz[6 * i + 5] = dN_dx[i]  # θz
            Bsy[6 * i + 1] = dN_dx[i]  # v
            Bsy[6 * i + 5] = -N[i]     # −θz
            Bsz[6 * i + 2] = dN_dx[i]  # w
            Bsz[6 * i + 4] = N[i]      # +θy

        fac = w * jac
        K += fac * (EA * np.outer(Be, Be) + GJ * np.outer(Bt, Bt) +
                    EIy * np.outer(Bky, Bky) + EIz * np.outer(Bkz, Bkz) +
                    kGA * np.outer(Bsy, Bsy) + kGA * np.outer(Bsz, Bsz))

    return K


def _fem_condensed_stiffness_local(beam_prop: dict, L: float,
                                   n_elem: int, ndim: int,
                                   poly_order: int = 1,
                                   n_gauss: int | None = None) -> np.ndarray:
    """Assemble and statically condense a chain of FEM sub-elements.

    Each sub-element is built via variational integration
    (:func:`_fem_element_stiffness_2d` / :func:`_fem_element_stiffness_3d`)
    using Lagrange shape functions of degree *poly_order* and *n_gauss*
    Gauss-Legendre quadrature points.  The interior DOFs (all but the first
    and last global nodes) are eliminated by Guyan (static) condensation so
    that the result has the same (2·dof × 2·dof) shape as a single exact
    element.

    As *n_elem* → ∞ the condensed matrix converges to the exact Timoshenko
    stiffness regardless of *poly_order* and *n_gauss* (as long as n_gauss ≥ 1).
    The convergence rate and the behaviour for small *n_elem* depend on the
    chosen integration rule.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    L : float
        Total beam length.
    n_elem : int
        Number of sub-elements (>= 1).
    ndim : int
        Spatial dimension (2 or 3).
    poly_order : int, optional
        Polynomial degree of the Lagrange shape functions (default 1 = linear).
        Each sub-element has poly_order + 1 nodes.
    n_gauss : int or None, optional
        Number of Gauss-Legendre integration points per sub-element.
        If None (default) uses ``poly_order`` (reduced integration), which
        under-integrates the shear term and thereby avoids shear locking
        for slender Timoshenko beams.  Use ``poly_order + 1`` for full
        integration (exact for all terms; note: can exhibit shear locking).

    Returns
    -------
    np.ndarray
        Condensed stiffness matrix, shape (2*dof, 2*dof), local frame.
    """

    if n_gauss is None:
        n_gauss = poly_order     # reduced integration: avoids shear locking

    dof = 3 * (ndim - 1)           # DOF per node
    n_nodes_per_elem = poly_order + 1
    l = L / n_elem                  # sub-element length

    # Total global nodes along the chain: each element contributes poly_order
    # new nodes, plus the single shared starting node.
    n_total_nodes = n_elem * poly_order + 1
    n_total = n_total_nodes * dof

    K_loc = np.zeros((n_total, n_total))

    for i in range(n_elem):
        if ndim == 2:
            Ke = _fem_element_stiffness_2d(beam_prop, l, n_nodes_per_elem, n_gauss)
        else:
            Ke = _fem_element_stiffness_3d(beam_prop, l, n_nodes_per_elem, n_gauss)

        start = i * poly_order * dof
        end = start + n_nodes_per_elem * dof
        K_loc[start:end, start:end] += Ke

    # Boundary DOFs: first node + last node
    b_dof = list(range(dof)) + list(range(n_total - dof, n_total))
    # Interior DOFs: all intermediate global nodes
    i_dof = list(range(dof, n_total - dof))

    if len(i_dof) == 0:
        # n_elem == 1, poly_order == 1: no interior DOFs
        return K_loc

    K_bb = K_loc[np.ix_(b_dof, b_dof)]
    K_bi = K_loc[np.ix_(b_dof, i_dof)]
    K_ib = K_loc[np.ix_(i_dof, b_dof)]
    K_ii = K_loc[np.ix_(i_dof, i_dof)]

    return K_bb - K_bi @ np.linalg.solve(K_ii, K_ib)


def get_fem_element_stiffness_global(beam_prop: dict, d: np.ndarray,
                                     n_elem: int,
                                     poly_order: int = 1,
                                     n_gauss: int | None = None) -> np.ndarray:
    """FEM element stiffness matrix for a single beam in the global frame.

    Discretizes the beam into *n_elem* equal sub-elements using Lagrange
    shape functions of degree *poly_order*, integrates with *n_gauss*
    Gauss-Legendre points, performs static condensation to eliminate interior
    DOFs, and rotates the result into the global coordinate frame.
    The output has the same shape and contract as
    :func:`get_element_stiffness_global`.  The result converges to the exact
    Timoshenko stiffness as *n_elem* increases.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    d : np.ndarray
        Vector from one beam endpoint to the other, shape (dim,).  Its
        Euclidean norm is the beam length.
    n_elem : int
        Number of sub-elements (>= 1).
    poly_order : int, optional
        Polynomial degree of the Lagrange shape functions (default 1).
    n_gauss : int or None, optional
        Number of Gauss-Legendre integration points (default None → poly_order,
        i.e. reduced integration to avoid shear locking).

    Returns
    -------
    np.ndarray
        Element stiffness matrix in the global frame,
        shape (num_elem_dof, num_elem_dof) where num_elem_dof is 6 (2D)
        or 12 (3D).
    """

    ndim = len(d)
    L = np.linalg.norm(d)

    K_cond_local = _fem_condensed_stiffness_local(beam_prop, L, n_elem, ndim,
                                                  poly_order=poly_order,
                                                  n_gauss=n_gauss)
    T = _get_transformation_matrix(d)

    return T.T @ K_cond_local @ T


def get_element_stiffness_global(beam_prop: dict, d: np.ndarray,
                                 derivative: int | None = None) -> np.ndarray:
    """Element stiffness matrix for a single beam in the global frame.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties (see
        :func:`~beam_networks.geo.get_geometric_props`).
    d : np.ndarray
        Vector pointing from one beam endpoint to the other, shape (dim,).
        Its magnitude is the beam length.
    derivative : int or None, optional
        If not None, return the derivative of the stiffness matrix with
        respect to a cross-section shape parameter instead of the matrix
        itself. For rectangular cross-sections: ``0`` → derivative w.r.t.
        height *h*, ``1`` → derivative w.r.t. width *b*. The default is None.

    Returns
    -------
    np.ndarray
        Element stiffness matrix in the global frame,
        shape (num_elem_dof, num_elem_dof) where num_elem_dof is 6 (2D)
        or 12 (3D).
    """

    ndim = len(d)
    num_dof_elem = 2 * 3 * (ndim - 1)

    L = np.linalg.norm(d, axis=-1)

    if ndim == 3:
        data, rows, cols = _beam_stiffness_3d(beam_prop, L, derivative)
    else:
        data, rows, cols = _beam_stiffness_2d(beam_prop, L, derivative)

    K_elem = np.zeros(shape=(num_dof_elem, num_dof_elem))

    K_elem[rows, cols] = data

    T_elem = _get_transformation_matrix(d)

    # transform element stiffness in global frame
    Ke_global = T_elem.T.dot(K_elem.dot(T_elem))

    return Ke_global


def _get_transformation_matrix(d):
    """Compute the transformation matrix from local to global frame for a single element.

    Parameters
    ----------
    d : np.ndarray
        Vector pointing from one beam endpoint to the other

    Returns
    -------
    np.ndarray
        Transformation matrix
    """

    ndim = len(d)
    if ndim == 2:
        d = np.hstack([d, [0.]])
        v_ref = np.array([0., 0., 1.])
    else:
        # Arbitrary reference vector (for circular beams)
        v_ref = np.random.rand(3)

    x0 = np.array([1., 0., 0.])
    y0 = np.array([0., 1., 0.])
    z0 = np.array([0., 0., 1.])

    # local axial direction
    l0 = d / np.linalg.norm(d)

    m0 = np.cross(v_ref, l0)
    m0 /= np.linalg.norm(m0)

    n0 = np.cross(l0, m0)

    if ndim == 3:
        data_R = [np.dot(l0, x0), np.dot(l0, y0), np.dot(l0, z0),
                  np.dot(m0, x0), np.dot(m0, y0), np.dot(m0, z0),
                  np.dot(n0, x0), np.dot(n0, y0), np.dot(n0, z0)]
        rows_R = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int)
        cols_R = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=int)

        rows = np.hstack([rows_R, rows_R + 3, rows_R + 6, rows_R + 9])
        cols = np.hstack([cols_R, cols_R + 3, cols_R + 6, cols_R + 9])

        data = data_R * 4
    else:
        data_R = [np.dot(l0, x0), np.dot(l0, y0),
                  np.dot(m0, x0), np.dot(m0, y0), 1.]
        rows_R = np.array([0, 0, 1, 1, 2], dtype=int)
        cols_R = np.array([0, 1, 0, 1, 2], dtype=int)

        rows = np.hstack([rows_R, rows_R + 3])
        cols = np.hstack([cols_R, cols_R + 3])
        data = data_R * 2

    T = np.zeros(shape=(3 * (ndim - 1) * 2, 3 * (ndim - 1) * 2))
    T[rows, cols] = data

    return T


def get_element_stiffness_local_vec(beam_prop: dict, d: np.ndarray,
                                    ndim: int,
                                    derivative: int | None = None) -> np.ndarray:
    """Element stiffness matrices for all beams in the local frame.

    The local frame has the beam axis along its first coordinate direction.
    No coordinate transformation to the global frame is applied.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    d : np.ndarray
        Edge lengths, shape (num_elements,).
    ndim : int
        Spatial dimension of the problem (2 or 3).
    derivative : int or None, optional
        If not None, return derivatives of the stiffness matrices with
        respect to a cross-section shape parameter (see
        :func:`get_element_stiffness_global`). The default is None.

    Returns
    -------
    np.ndarray
        Array of local element stiffness matrices,
        shape (num_elements, num_elem_dof, num_elem_dof).
    """

    num_e = len(d)
    num_dof_e = 2 * 3 * (ndim - 1)

    if ndim == 2:
        data, rows, cols = _beam_stiffness_2d(beam_prop, d, derivative)
    else:
        data, rows, cols = _beam_stiffness_3d(beam_prop, d, derivative)

    data = np.array(data)

    K_elem = np.zeros((num_e, num_dof_e, num_dof_e))
    K_elem[:, rows, cols] = data.T

    return K_elem


def _get_transformation_matrix_vec(d):
    """Compute the transformation matrices from local to global frame for all elements.

    Parameters
    ----------
    d : np.ndarray
        Vectors pointing from one beam endpoint to the other

    Returns
    -------
    np.ndarray
        Array of transformation matrices
    """

    nelem, ndim = d.shape

    if ndim == 2:
        d = np.vstack([d.T, np.zeros(nelem)]).T
        v_ref = np.array([0., 0., 1.])
        nr = 2
    else:
        # Arbitrary reference vector (for circular beams)
        v_ref = np.random.rand(3)
        nr = 4

    x0 = np.array([1., 0., 0.])
    y0 = np.array([0., 1., 0.])
    z0 = np.array([0., 0., 1.])

    # local axial direction
    l0 = d / np.linalg.norm(d, axis=1)[:, None]

    m0 = np.cross(v_ref[None, :], l0)

    m0 /= np.linalg.norm(m0, axis=1)[:, None]

    n0 = np.cross(l0, m0)

    if ndim == 3:
        data_R = np.array([np.einsum('...j,j', l0, x0), np.einsum('...j,j', l0, y0), np.einsum('...j,j', l0, z0),
                           np.einsum('...j,j', m0, x0), np.einsum('...j,j', m0, y0), np.einsum('...j,j', m0, z0),
                           np.einsum('...j,j', n0, x0), np.einsum('...j,j', n0, y0), np.einsum('...j,j', n0, z0)])
    else:
        data_R = np.array([np.dot(l0, x0), np.dot(l0, y0), np.zeros(nelem),
                           np.dot(m0, x0), np.dot(m0, y0), np.zeros(nelem),
                           np.zeros(nelem), np.zeros(nelem), np.ones(nelem)])

    data = data_R.T.reshape(-1, 3, 3)

    T = np.zeros((nelem, 3 * nr, 3 * nr))

    T[:, :3, :3] = data
    T[:, 3:6, 3:6] = data
    if ndim == 3:
        T[:, 6:9, 6:9] = data
        T[:, 9:, 9:] = data

    return T


def get_element_stiffness_global_vec(beam_prop: dict, d_vec: np.ndarray,
                                     derivative: int | None = None) -> np.ndarray:
    """Element stiffness matrices for all beams in the global frame.

    Vectorised counterpart of :func:`get_element_stiffness_global`.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    d_vec : np.ndarray
        Edge vectors (tail → head), shape (num_elements, dim). The Euclidean
        norm of each row is the beam length.
    derivative : int or None, optional
        If not None, return derivatives of the stiffness matrices with
        respect to a cross-section shape parameter (see
        :func:`get_element_stiffness_global`). The default is None.

    Returns
    -------
    np.ndarray
        Array of global element stiffness matrices,
        shape (num_elements, num_elem_dof, num_elem_dof).
    """

    nelem, ndim = d_vec.shape

    d = np.linalg.norm(d_vec, axis=-1)

    num_dof_e = 2 * 3 * (ndim - 1)

    if ndim == 2:
        data, rows, cols = _beam_stiffness_2d(beam_prop, d, derivative)
    else:
        data, rows, cols = _beam_stiffness_3d(beam_prop, d, derivative)

    data = np.array(data)

    T = _get_transformation_matrix_vec(d_vec)

    K_elem = np.zeros((nelem, num_dof_e, num_dof_e))
    K_elem[:, rows, cols] = data.T

    K = np.einsum('...ni,...ij,...jk->...nk', np.transpose(T, axes=(0, 2, 1)), K_elem, T)

    return K
