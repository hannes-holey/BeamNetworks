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

from beam_networks.geometry.geo import get_geometric_props, get_geometric_props_derivative
from beam_networks.fem.basis import _gauss_legendre, _lagrange_basis


def _local_element_stiffness_timoshenko_exact_2d_single(beam_prop, L, derivative=None):
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

    PhiY = 0. if beam_prop.get('euler_bernoulli', False) else 12 * E * Iz / (kappa * G * A * L**2)

    if derivative is None:

        gamma = E * A / L
        zeta = 12. * E * Iz / (L**3 * (1. + PhiY))
        lamb = 6. * E * Iz / (L**2 * (1. + PhiY))
        psi = (4. + PhiY) * E * Iz / (L * (1. + PhiY))
        xi = (2. - PhiY) * E * Iz / (L * (1. + PhiY))
    else:
        dIy, dIz, _, dA, _, _ = get_geometric_props_derivative(beam_prop,
                                                               derivative)

        dPhiY = (0. if beam_prop.get('euler_bernoulli', False) else
                 ((12 * E * dIz) * (kappa * G * A * L**2) -
                  (12 * E * Iz) * (kappa * G * dA * L**2)) /
                 (kappa * G * A * L**2)**2)
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


def _local_element_stiffness_timoshenko_exact_3d_single(beam_prop, L, derivative=None):
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

    eb = beam_prop.get('euler_bernoulli', False)
    PhiY = 0. if eb else 12 * E * Iz / (kappa * G * A * L**2)
    PhiZ = 0. if eb else 12 * E * Iy / (kappa * G * A * L**2)

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

        dPhiY = (0. if eb else
                 ((12 * E * dIz) * (kappa * G * A * L**2) -
                  (12 * E * Iz) * (kappa * G * dA * L**2)) /
                 (kappa * G * A * L**2)**2)
        dPhiZ = (0. if eb else
                 ((12 * E * dIy) * (kappa * G * A * L**2) -
                  (12 * E * Iy) * (kappa * G * dA * L**2)) /
                 (kappa * G * A * L**2)**2)
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


def _local_element_stiffness_timoshenko_numeric_2d_single(beam_prop, le, n_nodes, n_gauss):
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
    kGA = 0. if beam_prop.get('euler_bernoulli', False) else kappa * G * A

    n_dof = 3 * n_nodes
    K = np.zeros((n_dof, n_dof))

    xi_g, w_g = _gauss_legendre(n_gauss)
    jac = le / 2.

    for xi, w in zip(xi_g, w_g):
        N, dN_dxi = _lagrange_basis(n_nodes, xi)
        dN_dx = dN_dxi / jac        # chain rule: dN/dx = dN/dξ * dξ/dx = dN/dξ * 2/le

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


def _local_element_stiffness_timoshenko_numeric_3d_single(beam_prop, le, n_nodes, n_gauss):
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
    kGA = 0. if beam_prop.get('euler_bernoulli', False) else kappa * G * A
    GJ = G * J

    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))

    xi_g, w_g = _gauss_legendre(n_gauss)
    jac = le / 2.

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


def _local_element_stiffness_timoshenko_condensed_single(
        beam_prop: dict, L: float,
        n_elem: int, ndim: int,
        poly_order: int = 1,
        n_gauss: int | None = None) -> np.ndarray:
    """Assemble and statically condense a chain of FEM sub-elements.

    Each sub-element is built via variational integration
    (:func:`_local_element_stiffness_timoshenko_numeric_2d_single` /
    :func:`_local_element_stiffness_timoshenko_numeric_3d_single`)
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
    le = L / n_elem                 # sub-element length

    # Total global nodes along the chain: each element contributes poly_order
    # new nodes, plus the single shared starting node.
    n_total_nodes = n_elem * poly_order + 1
    n_total = n_total_nodes * dof

    K_loc = np.zeros((n_total, n_total))

    for i in range(n_elem):
        if ndim == 2:
            Ke = _local_element_stiffness_timoshenko_numeric_2d_single(beam_prop, le, n_nodes_per_elem, n_gauss)
        else:
            Ke = _local_element_stiffness_timoshenko_numeric_3d_single(beam_prop, le, n_nodes_per_elem, n_gauss)

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


def global_element_stiffness_timoshenko_numeric_single(
        beam_prop: dict, d: np.ndarray,
        n_elem: int,
        poly_order: int = 1,
        n_gauss: int | None = None) -> np.ndarray:
    """FEM element stiffness matrix for a single beam in the global frame.

    Discretizes the beam into *n_elem* equal sub-elements using Lagrange
    shape functions of degree *poly_order*, integrates with *n_gauss*
    Gauss-Legendre points, performs static condensation to eliminate interior
    DOFs, and rotates the result into the global coordinate frame.
    The output has the same shape and contract as
    :func:`global_element_stiffness_timoshenko_exact_single`.  The result converges to the exact
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

    K_cond_local = _local_element_stiffness_timoshenko_condensed_single(
        beam_prop, L, n_elem, ndim,
        poly_order=poly_order,
        n_gauss=n_gauss)
    T = _transformation_matrix_single(d)

    return T.T @ K_cond_local @ T


def _local_element_stiffness_timoshenko_numeric_2d_all(
        beam_prop: dict, le: np.ndarray,
        n_nodes: int, n_gauss: int) -> np.ndarray:
    """Vectorised 2D Timoshenko element stiffness for an array of element lengths.

    Parameters
    ----------
    beam_prop : dict
        Beam properties.
    le : np.ndarray, shape (n,)
        Element lengths.
    n_nodes : int
        Nodes per element (= poly_order + 1).
    n_gauss : int
        Number of Gauss-Legendre integration points.

    Returns
    -------
    np.ndarray, shape (n, 3*n_nodes, 3*n_nodes)
    """
    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2. * (1. + nu))
    _, Iz, _, A, kappa, _ = get_geometric_props(beam_prop)

    EA = E * A
    EI = E * Iz
    kGA = 0. if beam_prop.get('euler_bernoulli', False) else kappa * G * A

    le = np.asarray(le)
    n = len(le)
    n_dof = 3 * n_nodes
    K = np.zeros((n, n_dof, n_dof))

    xi_g, w_g = _gauss_legendre(n_gauss)
    jac = le / 2.                           # (n,)

    for xi, w in zip(xi_g, w_g):
        N, dN_dxi = _lagrange_basis(n_nodes, xi)
        dN_dx = dN_dxi[None, :] / jac[:, None]   # (n, n_nodes)

        Be = np.zeros((n, n_dof))
        Bk = np.zeros((n, n_dof))
        Bs = np.zeros((n, n_dof))

        for idx in range(n_nodes):
            Be[:, 3 * idx] = dN_dx[:, idx]
            Bk[:, 3 * idx + 2] = dN_dx[:, idx]
            Bs[:, 3 * idx + 1] = dN_dx[:, idx]
            Bs[:, 3 * idx + 2] = -N[idx]

        fac = w * jac                        # (n,)
        K += fac[:, None, None] * (
            EA * np.einsum('ni,nj->nij', Be, Be) +
            EI * np.einsum('ni,nj->nij', Bk, Bk) +
            kGA * np.einsum('ni,nj->nij', Bs, Bs)
        )

    return K


def _local_element_stiffness_timoshenko_numeric_3d_all(
        beam_prop: dict, le: np.ndarray,
        n_nodes: int, n_gauss: int) -> np.ndarray:
    """Vectorised 3D Timoshenko element stiffness for an array of element lengths.

    Parameters
    ----------
    beam_prop : dict
        Beam properties.
    le : np.ndarray, shape (n,)
        Element lengths.
    n_nodes : int
        Nodes per element (= poly_order + 1).
    n_gauss : int
        Number of Gauss-Legendre integration points.

    Returns
    -------
    np.ndarray, shape (n, 6*n_nodes, 6*n_nodes)
    """
    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2. * (1. + nu))
    Iy, Iz, J, A, kappa, _ = get_geometric_props(beam_prop)

    EA = E * A
    EIy = E * Iy
    EIz = E * Iz
    kGA = 0. if beam_prop.get('euler_bernoulli', False) else kappa * G * A
    GJ = G * J

    le = np.asarray(le)
    n = len(le)
    n_dof = 6 * n_nodes
    K = np.zeros((n, n_dof, n_dof))

    xi_g, w_g = _gauss_legendre(n_gauss)
    jac = le / 2.                           # (n,)

    for xi, w in zip(xi_g, w_g):
        N, dN_dxi = _lagrange_basis(n_nodes, xi)
        dN_dx = dN_dxi[None, :] / jac[:, None]   # (n, n_nodes)

        Be = np.zeros((n, n_dof))
        Bt = np.zeros((n, n_dof))
        Bky = np.zeros((n, n_dof))
        Bkz = np.zeros((n, n_dof))
        Bsy = np.zeros((n, n_dof))
        Bsz = np.zeros((n, n_dof))

        for idx in range(n_nodes):
            Be[:, 6 * idx] = dN_dx[:, idx]
            Bt[:, 6 * idx + 3] = dN_dx[:, idx]
            Bky[:, 6 * idx + 4] = dN_dx[:, idx]
            Bkz[:, 6 * idx + 5] = dN_dx[:, idx]
            Bsy[:, 6 * idx + 1] = dN_dx[:, idx]
            Bsy[:, 6 * idx + 5] = -N[idx]
            Bsz[:, 6 * idx + 2] = dN_dx[:, idx]
            Bsz[:, 6 * idx + 4] = N[idx]

        fac = w * jac                        # (n,)
        K += fac[:, None, None] * (
            EA * np.einsum('ni,nj->nij', Be,  Be) +
            GJ * np.einsum('ni,nj->nij', Bt,  Bt) +
            EIy * np.einsum('ni,nj->nij', Bky, Bky) +
            EIz * np.einsum('ni,nj->nij', Bkz, Bkz) +
            kGA * np.einsum('ni,nj->nij', Bsy, Bsy) +
            kGA * np.einsum('ni,nj->nij', Bsz, Bsz)
        )

    return K


def _local_element_stiffness_timoshenko_condensed_all(
        beam_prop: dict, L: np.ndarray,
        n_elem: int, ndim: int,
        poly_order: int = 1,
        n_gauss: int | None = None) -> np.ndarray:
    """Vectorised static condensation for a batch of beams with the same n_elem.

    All beams in the batch share the same sub-element count *n_elem* and
    polynomial order, but may have different lengths.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    L : np.ndarray, shape (n,)
        Beam lengths.
    n_elem : int
        Number of sub-elements (same for every beam in the batch).
    ndim : int
        Spatial dimension (2 or 3).
    poly_order : int, optional
        Lagrange polynomial degree (default 1).
    n_gauss : int or None, optional
        Gauss points per sub-element (default None → poly_order).

    Returns
    -------
    np.ndarray, shape (n, 2*dof, 2*dof)
        Condensed local stiffness matrices.
    """
    if n_gauss is None:
        n_gauss = poly_order

    dof = 3 * (ndim - 1)
    n_nodes_per_elem = poly_order + 1
    L = np.asarray(L)
    n = len(L)
    le = L / n_elem                         # (n,) sub-element lengths

    n_total_nodes = n_elem * poly_order + 1
    n_total = n_total_nodes * dof

    K_loc = np.zeros((n, n_total, n_total))

    # All sub-elements of a beam have the same length le[i], so one vectorised
    # call gives the stiffness for every (beam, sub-element) pair.
    if ndim == 2:
        Ke = _local_element_stiffness_timoshenko_numeric_2d_all(beam_prop, le, n_nodes_per_elem, n_gauss)
    else:
        Ke = _local_element_stiffness_timoshenko_numeric_3d_all(beam_prop, le, n_nodes_per_elem, n_gauss)
    # Ke: (n, elem_dof, elem_dof)

    elem_dof = n_nodes_per_elem * dof
    for i in range(n_elem):
        start = i * poly_order * dof
        end = start + elem_dof
        K_loc[:, start:end, start:end] += Ke

    b_dof = list(range(dof)) + list(range(n_total - dof, n_total))
    i_dof = list(range(dof, n_total - dof))

    if len(i_dof) == 0:
        return K_loc

    K_bb = K_loc[:, b_dof, :][:, :, b_dof]   # (n, 2*dof, 2*dof)
    K_bi = K_loc[:, b_dof, :][:, :, i_dof]   # (n, 2*dof, n_int)
    K_ib = K_loc[:, i_dof, :][:, :, b_dof]   # (n, n_int, 2*dof)
    K_ii = K_loc[:, i_dof, :][:, :, i_dof]   # (n, n_int, n_int)

    return K_bb - K_bi @ np.linalg.solve(K_ii, K_ib)


def global_element_stiffness_timoshenko_numeric_all(
        beam_prop: dict, d_vec: np.ndarray,
        n_elems: np.ndarray,
        poly_order: int = 1,
        n_gauss: int | None = None) -> np.ndarray:
    """FEM element stiffness matrices for all beams in the global frame.

    Vectorised counterpart of :func:`global_element_stiffness_timoshenko_numeric_single`.
    Beams are grouped by their sub-element count so that each group is
    processed with a single batched call.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    d_vec : np.ndarray, shape (num_edges, dim)
        Edge vectors (tail → head).
    n_elems : np.ndarray, shape (num_edges,)
        Per-edge sub-element counts.
    poly_order : int, optional
        Lagrange polynomial degree (default 1).
    n_gauss : int or None, optional
        Gauss points per sub-element (default None → poly_order).

    Returns
    -------
    np.ndarray, shape (num_edges, num_elem_dof, num_elem_dof)
        Global-frame element stiffness matrices.
    """
    n_edges, ndim = d_vec.shape
    num_dof_e = 2 * 3 * (ndim - 1)
    L = np.linalg.norm(d_vec, axis=-1)

    K_cond = np.zeros((n_edges, num_dof_e, num_dof_e))
    for n_elem in np.unique(n_elems):
        mask = n_elems == n_elem
        K_cond[mask] = _local_element_stiffness_timoshenko_condensed_all(
            beam_prop, L[mask], int(n_elem), ndim,
            poly_order=poly_order, n_gauss=n_gauss)

    T = _transformation_matrix_all(d_vec)
    return np.einsum('nki,nij,njl->nkl', T, K_cond, T)


def _local_element_stiffness_euler_exact_2d_single(beam_prop: dict, le: float) -> np.ndarray:
    """Euler-Bernoulli 2D element stiffness via Hermite curvature B-matrices (local frame).

    DOF order: [u₁, v₁, θ₁, u₂, v₂, θ₂]  where θ = dv/dx.

    Two-point Gauss integration is exact for the cubic Hermite curvature field.
    The result is identical to the analytical Timoshenko stiffness at Φ = 0,
    which equals the Friedman-Kosmatka shape functions at Φ = 0.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    le : float
        Element length.

    Returns
    -------
    np.ndarray, shape (6, 6)
        Local-frame stiffness matrix.
    """
    E = beam_prop['E']
    _, Iz, _, A, _, _ = get_geometric_props(beam_prop)

    n_dof = 6
    K = np.zeros((n_dof, n_dof))

    # Axial (constant strain — analytical)
    ea = E * A / le
    K[0, 0] = K[3, 3] = ea
    K[0, 3] = K[3, 0] = -ea

    # Bending via 2-point Gauss integration (exact for cubic Hermite)
    xi_g, w_g = _gauss_legendre(2)
    jac = le / 2.
    # d²(physical shape fn)/dx²: scale = [4/le², 2/le, 4/le², 2/le]
    # for DOFs [v₁ (disp), θ₁ (rot), v₂ (disp), θ₂ (rot)]
    scale = np.array([4. / le**2, 2. / le, 4. / le**2, 2. / le])

    for xi, w in zip(xi_g, w_g):
        d2H = np.array([3. * xi / 2.,
                        (3. * xi - 1.) / 2.,
                        -3. * xi / 2.,
                        (3. * xi + 1.) / 2.])
        d2N = scale * d2H  # curvature for DOFs [v₁, θ₁, v₂, θ₂]
        B_k = np.zeros(n_dof)
        B_k[[1, 2, 4, 5]] = d2N
        K += (w * jac * E * Iz) * np.outer(B_k, B_k)

    return K


def _local_element_stiffness_euler_exact_3d_single(beam_prop: dict, le: float) -> np.ndarray:
    """Euler-Bernoulli 3D element stiffness via Hermite curvature B-matrices (local frame).

    DOF order: [u₁, v₁, w₁, θx₁, θy₁, θz₁, u₂, v₂, w₂, θx₂, θy₂, θz₂]
    Convention: θz = +dv/dx,  θy = −dw/dx.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    le : float
        Element length.

    Returns
    -------
    np.ndarray, shape (12, 12)
        Local-frame stiffness matrix.
    """
    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2. * (1. + nu))
    Iy, Iz, J, A, _, _ = get_geometric_props(beam_prop)

    n_dof = 12
    K = np.zeros((n_dof, n_dof))

    # Axial (constant strain — analytical)
    ea = E * A / le
    K[0, 0] = K[6, 6] = ea
    K[0, 6] = K[6, 0] = -ea

    # Torsion (constant twist rate — analytical)
    gj = G * J / le
    K[3, 3] = K[9, 9] = gj
    K[3, 9] = K[9, 3] = -gj

    # Bending via 2-point Gauss integration (exact for cubic Hermite)
    xi_g, w_g = _gauss_legendre(2)
    jac = le / 2.
    scale = np.array([4. / le**2, 2. / le, 4. / le**2, 2. / le])

    for xi, w in zip(xi_g, w_g):
        d2H = np.array([3. * xi / 2.,
                        (3. * xi - 1.) / 2.,
                        -3. * xi / 2.,
                        (3. * xi + 1.) / 2.])
        d2N = scale * d2H

        # xy-plane bending (EIz): DOFs [v₁, θz₁, v₂, θz₂] → indices [1, 5, 7, 11]
        # θz = +dv/dx — no sign flip
        B_kxy = np.zeros(n_dof)
        B_kxy[[1, 5, 7, 11]] = d2N

        # xz-plane bending (EIy): DOFs [w₁, θy₁, w₂, θy₂] → indices [2, 4, 8, 10]
        # θy = -dw/dx — sign flip on rotation DOFs
        B_kxz = np.zeros(n_dof)
        B_kxz[[2, 8]] = d2N[[0, 2]]
        B_kxz[[4, 10]] = -d2N[[1, 3]]

        fac = w * jac
        K += (fac * E * Iz) * np.outer(B_kxy, B_kxy)
        K += (fac * E * Iy) * np.outer(B_kxz, B_kxz)

    return K


def _local_element_stiffness_euler_exact_2d_all(beam_prop: dict, le: np.ndarray) -> np.ndarray:
    """Vectorised 2D EB element stiffness (Hermite B-matrices).

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    le : np.ndarray, shape (n,)
        Element lengths.

    Returns
    -------
    np.ndarray, shape (n, 6, 6)
    """
    E = beam_prop['E']
    _, Iz, _, A, _, _ = get_geometric_props(beam_prop)

    n = le.shape[0]
    n_dof = 6
    K = np.zeros((n, n_dof, n_dof))

    # Axial
    ea = E * A / le  # (n,)
    K[:, 0, 0] += ea
    K[:, 3, 3] += ea
    K[:, 0, 3] -= ea
    K[:, 3, 0] -= ea

    # Bending
    xi_g, w_g = _gauss_legendre(2)
    jac = le / 2.  # (n,)
    # scale: (n, 4)
    scale = np.stack([4. / le**2, 2. / le, 4. / le**2, 2. / le], axis=-1)

    for xi, w in zip(xi_g, w_g):
        d2H = np.array([3. * xi / 2.,
                        (3. * xi - 1.) / 2.,
                        -3. * xi / 2.,
                        (3. * xi + 1.) / 2.])
        d2N = scale * d2H[None, :]  # (n, 4)

        B_k = np.zeros((n, n_dof))
        B_k[:, [1, 2, 4, 5]] = d2N

        fac = w * jac * E * Iz  # (n,)
        K += fac[:, None, None] * np.einsum('ni,nj->nij', B_k, B_k)

    return K


def _local_element_stiffness_euler_exact_3d_all(beam_prop: dict, le: np.ndarray) -> np.ndarray:
    """Vectorised 3D EB element stiffness (Hermite B-matrices).

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    le : np.ndarray, shape (n,)
        Element lengths.

    Returns
    -------
    np.ndarray, shape (n, 12, 12)
    """
    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2. * (1. + nu))
    Iy, Iz, J, A, _, _ = get_geometric_props(beam_prop)

    n = le.shape[0]
    n_dof = 12
    K = np.zeros((n, n_dof, n_dof))

    # Axial
    ea = E * A / le  # (n,)
    K[:, 0, 0] += ea
    K[:, 6, 6] += ea
    K[:, 0, 6] -= ea
    K[:, 6, 0] -= ea

    # Torsion
    gj = G * J / le  # (n,)
    K[:, 3, 3] += gj
    K[:, 9, 9] += gj
    K[:, 3, 9] -= gj
    K[:, 9, 3] -= gj

    # Bending
    xi_g, w_g = _gauss_legendre(2)
    jac = le / 2.  # (n,)
    scale = np.stack([4. / le**2, 2. / le, 4. / le**2, 2. / le], axis=-1)  # (n, 4)

    for xi, w in zip(xi_g, w_g):
        d2H = np.array([3. * xi / 2.,
                        (3. * xi - 1.) / 2.,
                        -3. * xi / 2.,
                        (3. * xi + 1.) / 2.])
        d2N = scale * d2H[None, :]  # (n, 4)

        B_kxy = np.zeros((n, n_dof))
        B_kxy[:, [1, 5, 7, 11]] = d2N

        B_kxz = np.zeros((n, n_dof))
        B_kxz[:, [2, 8]] = d2N[:, [0, 2]]
        B_kxz[:, [4, 10]] = -d2N[:, [1, 3]]

        fac = w * jac
        K += (fac * E * Iz)[:, None, None] * np.einsum('ni,nj->nij', B_kxy, B_kxy)
        K += (fac * E * Iy)[:, None, None] * np.einsum('ni,nj->nij', B_kxz, B_kxz)

    return K


def _local_element_stiffness_euler_condensed_single(
        beam_prop: dict, L: float,
        n_elem: int, ndim: int) -> np.ndarray:
    """Assemble and statically condense a chain of Hermite EB sub-elements.

    Each sub-element is a 2-node Hermite Euler-Bernoulli element.
    Interior DOFs (all intermediate nodes) are eliminated by Guyan condensation
    to yield a (2·dof × 2·dof) condensed stiffness identical in shape to a
    single exact element.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    L : float
        Total beam length.
    n_elem : int
        Number of Hermite sub-elements (>= 1).
    ndim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    np.ndarray
        Condensed stiffness matrix, shape (2*dof, 2*dof), local frame.
    """
    dof = 3 * (ndim - 1)
    le = L / n_elem
    n_total = (n_elem + 1) * dof

    K_loc = np.zeros((n_total, n_total))

    if ndim == 2:
        stiffness_fn = _local_element_stiffness_euler_exact_2d_single
    else:
        stiffness_fn = _local_element_stiffness_euler_exact_3d_single
    Ke = stiffness_fn(beam_prop, le)

    for i in range(n_elem):
        s = i * dof
        K_loc[s:s + 2 * dof, s:s + 2 * dof] += Ke

    b_dof = list(range(dof)) + list(range(n_total - dof, n_total))
    i_dof = list(range(dof, n_total - dof))

    if len(i_dof) == 0:
        return K_loc

    K_bb = K_loc[np.ix_(b_dof, b_dof)]
    K_bi = K_loc[np.ix_(b_dof, i_dof)]
    K_ib = K_loc[np.ix_(i_dof, b_dof)]
    K_ii = K_loc[np.ix_(i_dof, i_dof)]

    return K_bb - K_bi @ np.linalg.solve(K_ii, K_ib)


def _local_element_stiffness_euler_condensed_all(
        beam_prop: dict, L: np.ndarray,
        n_elem: int, ndim: int) -> np.ndarray:
    """Vectorised version of :func:`_local_element_stiffness_euler_condensed_single`.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    L : np.ndarray, shape (n,)
        Total beam lengths.
    n_elem : int
        Number of Hermite sub-elements per beam (uniform across this batch).
    ndim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    np.ndarray, shape (n, 2*dof, 2*dof)
    """
    n = L.shape[0]
    dof = 3 * (ndim - 1)
    le = L / n_elem                          # (n,)
    n_total = (n_elem + 1) * dof

    K_loc = np.zeros((n, n_total, n_total))

    stiffness_fn = (_local_element_stiffness_euler_exact_2d_all if ndim == 2
                    else _local_element_stiffness_euler_exact_3d_all)
    Ke = stiffness_fn(beam_prop, le)         # (n, 2*dof, 2*dof)

    for i in range(n_elem):
        s = i * dof
        K_loc[:, s:s + 2 * dof, s:s + 2 * dof] += Ke

    b_dof = list(range(dof)) + list(range(n_total - dof, n_total))
    i_dof = list(range(dof, n_total - dof))

    if len(i_dof) == 0:
        return K_loc

    K_bb = K_loc[:, np.ix_(b_dof, b_dof)[0], np.ix_(b_dof, b_dof)[1]]
    K_bi = K_loc[:, np.ix_(b_dof, i_dof)[0], np.ix_(b_dof, i_dof)[1]]
    K_ib = K_loc[:, np.ix_(i_dof, b_dof)[0], np.ix_(i_dof, b_dof)[1]]
    K_ii = K_loc[:, np.ix_(i_dof, i_dof)[0], np.ix_(i_dof, i_dof)[1]]

    return K_bb - K_bi @ np.linalg.solve(K_ii, K_ib)


def global_element_stiffness_euler_numeric_single(
        beam_prop: dict, d: np.ndarray,
        n_elem: int) -> np.ndarray:
    """Hermite EB element stiffness for a single beam in the global frame.

    Discretizes the beam into *n_elem* Hermite sub-elements, assembles their
    stiffnesses, eliminates interior DOFs by static condensation, and rotates
    the result into the global coordinate frame.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    d : np.ndarray, shape (dim,)
        Edge vector (tail → head).
    n_elem : int
        Number of Hermite sub-elements (>= 1).

    Returns
    -------
    np.ndarray, shape (num_elem_dof, num_elem_dof)
    """
    ndim = len(d)
    L = np.linalg.norm(d)
    K_cond = _local_element_stiffness_euler_condensed_single(beam_prop, L, n_elem, ndim)
    T = _transformation_matrix_single(d)
    return T.T @ K_cond @ T


def global_element_stiffness_euler_numeric_all(
        beam_prop: dict,
        d_vec: np.ndarray,
        n_elems: np.ndarray) -> np.ndarray:
    """Hermite EB element stiffness matrices in the global frame (vectorised).

    Beams are grouped by sub-element count so each group is processed in a
    single batched call.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    d_vec : np.ndarray, shape (num_edges, dim)
        Edge vectors.
    n_elems : np.ndarray, shape (num_edges,)
        Per-edge sub-element counts.

    Returns
    -------
    np.ndarray, shape (num_edges, num_elem_dof, num_elem_dof)
    """
    n_edges, ndim = d_vec.shape
    num_dof_e = 2 * 3 * (ndim - 1)
    L = np.linalg.norm(d_vec, axis=-1)

    K_cond = np.zeros((n_edges, num_dof_e, num_dof_e))
    for n_elem in np.unique(n_elems):
        mask = n_elems == n_elem
        K_cond[mask] = _local_element_stiffness_euler_condensed_all(
            beam_prop, L[mask], int(n_elem), ndim)

    T = _transformation_matrix_all(d_vec)
    return np.einsum('nki,nij,njl->nkl', T, K_cond, T)


def global_element_stiffness_euler_exact_single(beam_prop: dict, d: np.ndarray) -> np.ndarray:
    """EB element stiffness matrix in the global frame (single beam).

    Uses Hermite curvature B-matrices, which are equivalent to the
    Friedman-Kosmatka shape functions at Φ = 0 and to the analytical
    Timoshenko stiffness at Φ = 0.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    d : np.ndarray, shape (dim,)
        Edge vector (tail → head).

    Returns
    -------
    np.ndarray, shape (num_elem_dof, num_elem_dof)
    """
    L = np.linalg.norm(d)
    ndim = d.shape[0]
    if ndim == 2:
        K_loc = _local_element_stiffness_euler_exact_2d_single(beam_prop, L)
    else:
        K_loc = _local_element_stiffness_euler_exact_3d_single(beam_prop, L)
    T = _transformation_matrix_single(d)
    return T.T @ K_loc @ T


def global_element_stiffness_euler_exact_all(
        beam_prop: dict,
        d_vec: np.ndarray) -> np.ndarray:
    """EB element stiffness matrices in the global frame (vectorised).

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    d_vec : np.ndarray, shape (num_edges, dim)
        Edge vectors.

    Returns
    -------
    np.ndarray, shape (num_edges, num_elem_dof, num_elem_dof)
    """
    n_edges, ndim = d_vec.shape
    L = np.linalg.norm(d_vec, axis=-1)
    if ndim == 2:
        K_loc = _local_element_stiffness_euler_exact_2d_all(beam_prop, L)
    else:
        K_loc = _local_element_stiffness_euler_exact_3d_all(beam_prop, L)
    T = _transformation_matrix_all(d_vec)
    return np.einsum('nki,nij,njl->nkl', T, K_loc, T)


def global_element_stiffness_timoshenko_exact_single(
        beam_prop: dict, d: np.ndarray,
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
        data, rows, cols = _local_element_stiffness_timoshenko_exact_3d_single(beam_prop, L, derivative)
    else:
        data, rows, cols = _local_element_stiffness_timoshenko_exact_2d_single(beam_prop, L, derivative)

    K_elem = np.zeros(shape=(num_dof_elem, num_dof_elem))

    K_elem[rows, cols] = data

    T_elem = _transformation_matrix_single(d)

    # transform element stiffness in global frame
    Ke_global = T_elem.T.dot(K_elem.dot(T_elem))

    return Ke_global


def _transformation_matrix_single(d):
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


def local_element_stiffness_timoshenko_exact_all(
        beam_prop: dict, d: np.ndarray,
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
        :func:`global_element_stiffness_timoshenko_exact_single`). The default is None.

    Returns
    -------
    np.ndarray
        Array of local element stiffness matrices,
        shape (num_elements, num_elem_dof, num_elem_dof).
    """

    num_e = len(d)
    num_dof_e = 2 * 3 * (ndim - 1)

    if ndim == 2:
        data, rows, cols = _local_element_stiffness_timoshenko_exact_2d_single(beam_prop, d, derivative)
    else:
        data, rows, cols = _local_element_stiffness_timoshenko_exact_3d_single(beam_prop, d, derivative)

    data = np.array(data)

    K_elem = np.zeros((num_e, num_dof_e, num_dof_e))
    K_elem[:, rows, cols] = data.T

    return K_elem


def _transformation_matrix_all(d):
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


def global_element_stiffness_timoshenko_exact_all(
        beam_prop: dict, d_vec: np.ndarray,
        derivative: int | None = None) -> np.ndarray:
    """Element stiffness matrices for all beams in the global frame.

    Vectorised counterpart of :func:`global_element_stiffness_timoshenko_exact_single`.

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
        :func:`global_element_stiffness_timoshenko_exact_single`). The default is None.

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
        data, rows, cols = _local_element_stiffness_timoshenko_exact_2d_single(beam_prop, d, derivative)
    else:
        data, rows, cols = _local_element_stiffness_timoshenko_exact_3d_single(beam_prop, d, derivative)

    data = np.array(data)

    T = _transformation_matrix_all(d_vec)

    K_elem = np.zeros((nelem, num_dof_e, num_dof_e))
    K_elem[:, rows, cols] = data.T

    K = np.einsum('...ni,...ij,...jk->...nk', np.transpose(T, axes=(0, 2, 1)), K_elem, T)

    return K


def global_element_stiffness_truss_exact_single(
        beam_prop: dict, d: np.ndarray) -> np.ndarray:
    """Truss element stiffness matrix in the global frame (single element).

    Assumes pin-jointed bar: no bending, no rotational DOFs.  The element
    has ``2 * ndim`` DOFs ordered as ``[ux₁, uy₁, (uz₁,) ux₂, uy₂, (uz₂)]``.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.  Only ``'E'`` and the
        cross-sectional area (derived from the ``'name'``/size keys) are used.
    d : np.ndarray
        Vector from node 0 to node 1, shape (ndim,).  Its norm is the
        bar length.

    Returns
    -------
    np.ndarray
        Element stiffness matrix in the global frame,
        shape (2*ndim, 2*ndim).
    """
    E = beam_prop['E']
    _, _, _, A, _, _ = get_geometric_props(beam_prop)
    L = np.linalg.norm(d)
    gamma = E * A / L
    c = d / L
    c_ext = np.concatenate([c, -c])
    return gamma * np.outer(c_ext, c_ext)


def global_element_stiffness_truss_exact_all(
        beam_prop: dict, d_vec: np.ndarray) -> np.ndarray:
    """Truss element stiffness matrices in the global frame (all elements).

    Vectorised version of :func:`global_element_stiffness_truss_exact_single`.

    Parameters
    ----------
    beam_prop : dict
        Beam cross-section and elastic properties.
    d_vec : np.ndarray
        Edge vectors, shape (n_edges, ndim).

    Returns
    -------
    np.ndarray
        Element stiffness matrices, shape (n_edges, 2*ndim, 2*ndim).
    """
    E = beam_prop['E']
    _, _, _, A, _, _ = get_geometric_props(beam_prop)
    L = np.linalg.norm(d_vec, axis=-1)
    gamma = E * A / L
    c = d_vec / L[:, None]
    c_ext = np.concatenate([c, -c], axis=1)
    return gamma[:, None, None] * np.einsum('ni,nj->nij', c_ext, c_ext)
