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

from beam_networks.geo import get_geometric_props, get_geometric_props_derivative


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


def get_element_stiffness_global(beam_prop, d, derivative=None):
    """Single element stiffness matrix in global frame.

    Parameters
    ----------
    beam_prop : dict
        Beam properties (cross section and elastic constants)
    d : np.ndarray
        Vector pointing from one beam endpoint to the other
    sparse : bool, optional
        Return as sparse matrix if true (the default is True)
    derivative : None or int
        if not None returns the derivative with respect to a shape parameter.
        The shape parameter is determined by the value of derivative which
        depends on the shape of your beam.

    Returns
    -------
    np.ndarray or
        Element stiffness matrix in global frame
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


def get_element_stiffness_local_vec(beam_prop, d, ndim, derivative=None):
    """Get all element stiffness matrices in the local frame (w.o. transformation)

    Parameters
    ----------
    beam_prop : dict
        Beam properties (cross section and elastic constants)
    d : np.ndarray
        Edge lengths
    ndim : int
        Dimension of the problem
    derivative : None or int
        if not None returns the derivative with respect to a shape parameter.
        The shape parameter is determined by the value of derivative which
        depends on the shape of your beam.

    Returns
    -------
    np.ndarray
        Array with shape (num_elements, num_elem_dof, num_elem_dof)
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


def get_element_stiffness_global_vec(beam_prop, d_vec, derivative=None):
    """Get all element stiffness matrices in the global frame.

    Parameters
    ----------
    beam_prop : dict
        Beam properties (cross section and elastic constants)
    d : np.ndarray
        Vectors pointing from one beam endpoint to the other
    derivative : None or int
        if not None returns the derivative with respect to a shape parameter.
        The shape parameter is determined by the value of derivative which
        depends on the shape of your beam.
    Returns
    -------
    np.ndarray
        Array with shape (num_elements, num_elem_dof, num_elem_dof)
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
