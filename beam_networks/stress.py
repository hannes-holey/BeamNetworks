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
from beam_networks.stiffness import get_element_stiffness_local_vec
from beam_networks.geo import get_geometric_props


def vmises_stress(rhs, beam_prop, mode):
    """Wrapper for von Mises stress calculation

    Parameters
    ----------
    rhs : np.ndarray
        Element-wise right-hand side in local frame
    beam_prop : dict
        Beam properties
    mode : str
        Either 'max' or 'mean', i.e. calculate maximum or mean stress per beam

    Returns
    -------
    np.ndarray
        Von Mises stress, shape=(num_elements,)
    """

    if rhs.shape[1] == 6:
        return _vmises_stress_2d(rhs, beam_prop, mode)
    else:  # de.shape[1] == 12:
        return _vmises_stress_3d(rhs, beam_prop, mode)


def stretch_bend_ratio(rhs, beam_prop):

    if rhs.shape[1] == 6:
        return _stress_stretch_bend_ratio(rhs, beam_prop)
    else:
        raise RuntimeError("Not implemented for 3d")


def _stress_stretch_bend_ratio(rhs, beam_prop):

    Iy, Iz, _, A, kappa, ymax = get_geometric_props(beam_prop)

    F = rhs[:, 3]
    Q = rhs[:, 4]
    M0 = rhs[:, 2]
    M1 = rhs[:, 5]

    M = (np.abs(M0) + np.abs(M1)) / 2.

    s_bend = M * ymax / 2 / Iz
    s_stretch = np.abs(F) / A
    s_shear = np.abs(Q) / A

    return s_bend / (s_stretch + s_bend)


def _vmises_stress_2d(rhs, beam_prop, mode):
    """Von Mises stress calculation in local frame for 2D systems

    Parameters
    ----------
    rhs : np.ndarray
        Element-wise right-hand side in local frame
    beam_prop : dict
        Beam properties
    mode : str
        Either 'max' or 'mean', i.e. calculate maximum or mean stress per beam

    Returns
    -------
    np.ndarray
        Von Mises stress, shape=(num_elements,)
    """

    Iy, Iz, _, A, kappa, ymax = get_geometric_props(beam_prop)

    F = rhs[:, 3]
    Q = rhs[:, 4]
    M0 = rhs[:, 2]
    M1 = rhs[:, 5]

    if mode == 'max':
        M = np.maximum(np.abs(M0), np.abs(M1))
    else:  # mode == 'mean'
        M = (np.abs(M0) + np.abs(M1)) / 2.

    s = np.sqrt((F / A + M * ymax / Iz)**2 + 3. * (Q / A)**2)

    return s


def _vmises_stress_3d(rhs, beam_prop, mode):
    """Von Mises stress calculation in local frame for 3D systems

    Parameters
    ----------
    rhs : np.ndarray
        Element-wise right-hand side in local frame
    beam_prop : dict
        Beam properties
    mode : str
        Either 'max' or 'mean', i.e. calculate maximum or mean stress per beam

    Returns
    -------
    np.ndarray
        Von Mises stress, shape=(num_elements,)
    """
    Iy, Iz, Ip, A, kappa, ymax = get_geometric_props(beam_prop)

    # normal force
    F = rhs[:, 6]

    # shear force
    Q = np.sqrt(rhs[:, 7]**2 + rhs[:, 8]**2)

    # bending moment
    M0 = np.sqrt(rhs[:, 4]**2 + rhs[:, 5]**2)
    M1 = np.sqrt(rhs[:, 10]**2 + rhs[:, 11]**2)

    # twisting moment
    T0 = rhs[:, 3]
    T1 = rhs[:, 9]

    if mode == 'max':
        M = np.maximum(np.abs(M0), np.abs(M1))
        T = np.maximum(np.abs(M0), np.abs(M1))
    else:  # mode == 'mean'
        M = (np.abs(M0) + np.abs(M1)) / 2.
        T = (np.abs(T0) + np.abs(T1)) / 2.

    s = np.sqrt((F / A + M * ymax / Iz)**2 + 3. * (Q / A + T * ymax / Ip)**2)

    return s


def _get_element_dof_2d(coords, adj, edge_vec, sol_global):
    """Compute 2D element-wise solution vector in local frame.

    Parameters
    ----------
    coords : np.ndarray
        Nodal positions
    adj : np.ndarray
        Edges indices
    edge_vec : edge_vec
        Edge directors (normalized edge vectors)
    sol_global : np.ndarray
        Global solution vector

    Returns
    -------
    np.ndarray
        Array of local solutions
    """

    ux_global = sol_global[0::3]
    uy_global = sol_global[1::3]
    theta_global = sol_global[2::3]

    u_global = np.vstack([ux_global, uy_global]).T
    z_global = [0, 0, 1.]

    # tangential to beam axis
    beam_t = edge_vec
    beam_t_3D = np.hstack([beam_t, np.zeros((beam_t.shape[0], 1))])

    # orthogonal to beam axis
    beam_s = np.cross(z_global, beam_t_3D)[:, :2]

    dof_elem = np.array([np.einsum('...j,...j', u_global[adj[:, 0]], beam_t),
                         np.einsum('...j,...j', u_global[adj[:, 0]], beam_s),
                         theta_global[adj[:, 0]],
                         np.einsum('...j,...j', u_global[adj[:, 1]], beam_t),
                         np.einsum('...j,...j', u_global[adj[:, 1]], beam_s),
                         theta_global[adj[:, 1]],
                         ]).T

    return dof_elem


def _get_element_dof_3d(coords, adj, edge_vec, sol_global, rot=None):
    """Compute 3D element-wise solution vector in local frame.

    Parameters
    ----------
    coords : np.ndarray
        Nodal positions
    adj : np.ndarray
        Edges indices
    edge_vec : edge_vec
        Edge directors (normalized edge vectors)
    sol_global : np.ndarray
        Global solution vector
    rot : np.ndarray or None, optional
        If all elements have the same orientation, rot is the transformation matrix
        from the global to the local frame (the default is None).

    Returns
    -------
    np.ndarray
        Array of local solutions
    """

    ux_global = sol_global[0::6]
    uy_global = sol_global[1::6]
    uz_global = sol_global[2::6]
    tx_global = sol_global[3::6]
    ty_global = sol_global[4::6]
    tz_global = sol_global[5::6]

    if rot is not None:
        beam_l = rot.dot(np.array([1., 0., 0.])).reshape(3, 1).repeat(adj.shape[0], 1).T
        beam_m = rot.dot(np.array([0., 1., 0.])).reshape(3, 1).repeat(adj.shape[0], 1).T
        beam_n = rot.dot(np.array([0., 0., 1.])).reshape(3, 1).repeat(adj.shape[0], 1).T
    else:
        v_ref = np.random.rand(3)

        beam_l = edge_vec

        beam_m = np.cross(v_ref, beam_l)
        beam_m = beam_m / np.linalg.norm(beam_m, axis=-1)[:, None]

        beam_n = np.cross(beam_l, beam_m)
        beam_n = beam_n / np.linalg.norm(beam_n, axis=-1)[:, None]

    u_global = np.vstack([ux_global, uy_global, uz_global]).T
    t_global = np.vstack([tx_global, ty_global, tz_global]).T

    dof_elem = np.array([np.einsum('...j,...j', u_global[adj[:, 0]], beam_l),
                         np.einsum('...j,...j', u_global[adj[:, 0]], beam_m),
                         np.einsum('...j,...j', u_global[adj[:, 0]], beam_n),
                         np.einsum('...j,...j', t_global[adj[:, 0]], beam_l),
                         np.einsum('...j,...j', t_global[adj[:, 0]], beam_m),
                         np.einsum('...j,...j', t_global[adj[:, 0]], beam_n),
                         np.einsum('...j,...j', u_global[adj[:, 1]], beam_l),
                         np.einsum('...j,...j', u_global[adj[:, 1]], beam_m),
                         np.einsum('...j,...j', u_global[adj[:, 1]], beam_n),
                         np.einsum('...j,...j', t_global[adj[:, 1]], beam_l),
                         np.einsum('...j,...j', t_global[adj[:, 1]], beam_m),
                         np.einsum('...j,...j', t_global[adj[:, 1]], beam_n)
                         ]).T
    return dof_elem


def get_element_mises_stress(coords, adj, d_vec, sol, beam_prop, rot=None, mode='max', return_ratio=False):
    """Compute the von Mises stress for all beam elements

    Parameters
    ----------
    coords : np.ndarray
        Nodal positions
    adj : np.ndarray
        Edges indices
    d_vec : np.ndarray
        Edge vectors
    sol : np.ndarray
        Global solution vector
    beam_prop : dict
        Beam properties
    rot : np.ndarray or None, optional
        If all elements have the same orientation, rot is the transformation matrix
        from the global to the local frame (the default is None).
    mode : str, optional
        Either 'max' or 'mean', i.e. calculate maximum or mean stress per beam
        (the default is 'max').
    Returns
    -------
    np.ndarray
        Von Mises stress
    """
    _, ndim = coords.shape

    d = np.linalg.norm(d_vec, axis=-1)

    if ndim == 3:
        dof_elem = _get_element_dof_3d(coords, adj, d_vec / d[:, None], sol, rot)
    else:
        dof_elem = _get_element_dof_2d(coords, adj, d_vec / d[:, None], sol)

    K_elem = get_element_stiffness_local_vec(beam_prop, d, ndim)

    rhs = np.einsum('ijk,ik->ij', K_elem[:, :, :], dof_elem)
    svM = vmises_stress(rhs, beam_prop, mode)

    if return_ratio:
        ratio = stretch_bend_ratio(rhs, beam_prop)

        return svM, ratio

    else:
        return svM


def get_element_principal_stress(coords, adj, d_vec, sol, beam_prop, rot=None, mode='max'):
    """

    Parameters
    ----------
    coords : np.ndarray
        Nodal positions
    adj : np.ndarray
        Edges indices
    d_vec : np.ndarray
        Edge vectors
    sol : np.ndarray
        Global solution vector
    beam_prop : dict
        Beam properties
    rot : np.ndarray or None, optional
        If all elements have the same orientation, rot is the transformation matrix
        from the global to the local frame (the default is None).
    mode : str, optional
        Either 'max' or 'mean', i.e. calculate maximum or mean stress per beam
        (the default is 'max').
    Returns
    -------
    np.ndarray

    """
    _, ndim = coords.shape

    d = np.linalg.norm(d_vec, axis=-1)

    if ndim == 3:
        dof_elem = _get_element_dof_3d(coords, adj, d_vec / d[:, None], sol, rot)
    else:
        dof_elem = _get_element_dof_2d(coords, adj, d_vec / d[:, None], sol)

    K_elem = get_element_stiffness_local_vec(beam_prop, d, ndim)

    rhs = np.einsum('ijk,ik->ij', K_elem[:, :, :], dof_elem)

    return principal_stress(rhs, beam_prop, mode)


def principal_stress(rhs, beam_prop, mode):
    """Compute element-wise principal stress.

    Principal stress values are sorted in decreasing order along the second axis.

    Parameters
    ----------
    rhs : np.ndarray
        Element-wise right-hand side in local frame
    beam_prop : dict
        Beam properties
    mode : str
        Either 'max' or 'mean', i.e. calculate maximum or mean stress per beam

    Returns
    -------
    np.ndarry
        Principal stresses per beam (shape: (n_beams, 3))
    """

    sxx, syy, szz, syz, sxz, sxy = _stress(rhs, beam_prop, mode)
    stress = np.vstack([sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz]).T.reshape(-1, 3, 3)

    # not sorted!
    ev = np.linalg.eigvals(stress)

    return np.sort(ev, axis=1)[:, ::-1]


def _stress(rhs, beam_prop, mode):
    """Wrapper around stress calculation. Selects either 2D or 3D version.

    Parameters
    ----------
    rhs : np.ndarray
        Element-wise right-hand side in local frame
    beam_prop : dict
        Beam properties
    mode : str
        Either 'max' or 'mean', i.e. calculate maximum or mean stress per beam

    Returns
    -------
    np.ndarray
        Stress tensor elements, Voigt notation, shape=(6, num_elements,)
    """

    if rhs.shape[1] == 6:
        return _stress_2d(rhs, beam_prop, mode)
    else:  # de.shape[1] == 12:
        return _stress_3d(rhs, beam_prop, mode)


def _stress_2d(rhs, beam_prop, mode):
    """Stress in local frame for 2D systems

    Parameters
    ----------
    rhs : np.ndarray
        Element-wise right-hand side in local frame
    beam_prop : dict
        Beam properties
    mode : str
        Either 'max' or 'mean', i.e. calculate maximum or mean stress per beam

    Returns
    -------
    np.ndarray
        Stress tensor elements, Voigt notation, shape=(6, num_elements,)
    """

    Iy, Iz, _, A, kappa, ymax = get_geometric_props(beam_prop)

    F = rhs[:, 3]
    Q = rhs[:, 4]
    M0 = rhs[:, 2]
    M1 = rhs[:, 5]

    if mode == 'max':
        M = np.maximum(np.abs(M0), np.abs(M1))
    else:  # mode == 'mean'
        M = (np.abs(M0) + np.abs(M1)) / 2.

    sxx = F / A + M * ymax / Iz  # normal + bending
    sxy = Q / A  # shear

    _z = np.zeros_like(sxx)

    return sxx, _z, _z, _z, _z, sxy


def _stress_3d(rhs, beam_prop, mode):
    """Stress in local frame for 3D systems (only circular beam cross sections)

    Parameters
    ----------
    rhs : np.ndarray
        Element-wise right-hand side in local frame
    beam_prop : dict
        Beam properties
    mode : str
        Either 'max' or 'mean', i.e. calculate maximum or mean stress per beam

    Returns
    -------
    np.ndarray
        Stress tensor elements, Voigt notation, shape=(6, num_elements,)
    """
    Iy, Iz, Ip, A, kappa, ymax, zmax = get_geometric_props(beam_prop)

    # normal force
    F = rhs[:, 6]

    # TODO: don't average Q and M and return sxy and sxz
    # shear force
    Q = np.sqrt(rhs[:, 7]**2 + rhs[:, 8]**2)

    # bending moment
    M0 = np.sqrt(rhs[:, 4]**2 + rhs[:, 5]**2)
    M1 = np.sqrt(rhs[:, 10]**2 + rhs[:, 11]**2)

    # twisting moment
    T0 = rhs[:, 3]
    T1 = rhs[:, 9]

    if mode == 'max':
        M = np.maximum(np.abs(M0), np.abs(M1))
        T = np.maximum(np.abs(M0), np.abs(M1))
    else:  # mode == 'mean'
        M = (np.abs(M0) + np.abs(M1)) / 2.
        T = (np.abs(T0) + np.abs(T1)) / 2.

    sxx = F / A + M * ymax / Iz
    sxy = Q / A + T * ymax / Ip

    _z = np.zeros_like(sxx)

    return sxx, _z, _z, _z, _z, sxy


def principal_strain(coords, adj, d, sol, beam_prop, mode='mean'):

    exx, eyy, ezz, eyz, exz, exy = _strain(coords, adj, d, sol, beam_prop, mode=mode)

    strain = np.vstack([exx, exy, exz, exy, eyy, eyz, exz, eyz, ezz]).T.reshape(-1, 3, 3)

    ev = np.linalg.eigvals(strain)

    return np.sort(ev, axis=1)[:, ::-1]


# def _strain(coords, adj, d_vec, sol, beam_prop, mode='mean'):

#     d = np.linalg.norm(d_vec, axis=-1)

#     # beam geometric and elastic properties
#     E = beam_prop['E']
#     G = beam_prop['E']
#     nu = beam_prop['nu']
#     Iy, Iz, J, A, kappa, ymax, zmax = get_geometric_props(beam_prop)

#     PhiY = 12 * E * Iz / (kappa * G * A * d**2)
#     PhiZ = 12 * E * Iy / (kappa * G * A * d**2)

#     _, ndim = coords.shape
#     n = 3 * (ndim - 1) * 2

#     if mode == 'mean':
#         ymax = np.zeros((len(d), 4))
#         zmax = np.zeros((len(d), 4))
#     else:
#         ymax = np.array([ymax, -ymax, ymax, -ymax]).T
#         zmax = np.array([zmax, zmax, -zmax, -zmax]).T

#     if ndim == 3:  # n = 12
#         dof_elem = _get_element_dof_3d(coords, adj, d_vec, sol, rot=None)

#         # Node 1, displacement
#         ux_n1 = dof_elem[:, n // 2]
#         uy_n1 = dof_elem[:, n // 2 + 1]
#         uz_n1 = dof_elem[:, n // 2 + 2]

#         # Node 0, displacement
#         ux_n0 = dof_elem[:, 0]
#         uy_n0 = dof_elem[:, 1]
#         uz_n0 = dof_elem[:, 2]

#         # Node 1, rotations
#         tx_n1 = dof_elem[:, n // 2 + 3]
#         ty_n1 = dof_elem[:, n // 2 + 4]
#         tz_n1 = dof_elem[:, n // 2 + 5]

#         # Node 0, rotations
#         tx_n0 = dof_elem[:, 3]
#         ty_n0 = dof_elem[:, 5]
#         tz_n0 = dof_elem[:, 5]

#         # Delta, displ
#         du_x = ux_n1 - ux_n0
#         du_y = uy_n1 - uy_n0
#         du_z = uz_n1 - uz_n0

#         # Delta, rotations
#         dt_x = tx_n1 - tx_n0
#         dt_y = ty_n1 - ty_n0
#         dt_z = tz_n1 - tz_n0

#         # Avg. beam rotation around z, integrate shape function
#         Mt1 = shape_func_antideriv_FK(d, d, Phi=PhiZ)
#         Mt0 = shape_func_antideriv_FK(0., d, Phi=PhiZ)
#         wz = np.vstack([uy_n0, tz_n0, uy_n1, tz_n1])
#         tz = np.einsum('j..., j...', (Mt1 - Mt0) / d, wz)

#         # Avg. beam rotation around y, integrate shape function
#         Mt1 = shape_func_antideriv_FK(d, d, Phi=PhiY)
#         Mt0 = shape_func_antideriv_FK(0., d, Phi=PhiY)
#         wy = np.vstack([uz_n0, ty_n0, uz_n1, ty_n1])
#         ty = np.einsum('j..., j...', (Mt1 - Mt0) / d, wy)

#         # Strain tensor components
#         _exx = (du_x / d)[:, None] - (dt_z / d)[:, None] * ymax + (dt_y / d)[:, None] * zmax
#         exx = _exx[:, np.argmax(np.abs(_exx), axis=1)][:, 0]

#         _exz = ((du_z / d)[:, None] - (dt_x / d)[:, None] * ymax + ty[:, None]) / 2.
#         exz = _exz[:, np.argmax(np.abs(_exz), axis=0)][:, 0]

#         _exy = ((du_y / d)[:, None] - (dt_x / d)[:, None] * zmax - tz[:, None]) / 2.
#         exy = _exy[:, np.argmax(np.abs(_exy), axis=0)][:, 0]

#         eyy = -nu * du_x / d
#         ezz = -nu * du_x / d
#         eyz = np.zeros_like(d)

#     else:  # ndim == 2, n = 6
#         dof_elem = _get_element_dof_2d(coords, adj, d_vec, sol)

#         # node 1, displacement + rotation
#         ux_n1 = dof_elem[:, n // 2]
#         uy_n1 = dof_elem[:, n // 2 + 1]
#         tz_n1 = dof_elem[:, n // 2 + 2]

#         # node 2, displacement + rotation
#         ux_n0 = dof_elem[:, 0]
#         uy_n0 = dof_elem[:, 1]
#         tz_n0 = dof_elem[:, 2]

#         # deltas
#         du_x = ux_n1 - ux_n0
#         du_y = uy_n1 - uy_n0
#         dt_z = tz_n1 - tz_n0

#         # Avg. beam rotation, integrate shape function
#         Mt1 = shape_func_antideriv_FK(d, d, Phi=PhiZ)
#         Mt0 = shape_func_antideriv_FK(0., d, Phi=PhiZ)
#         wz = np.vstack([uy_n0, tz_n0, uy_n1, tz_n1])
#         tz = np.einsum('j..., j...', (Mt1 - Mt0) / d, wz)

#         # Strain tensor components
#         _exx = (du_x / d)[:, None] - (dt_z / d)[:, None] * ymax
#         exx = _exx[:, np.argmax(np.abs(_exx), axis=1)][:, 0]

#         eyy = -nu * du_x / d
#         ezz = np.zeros_like(d)
#         eyz = np.zeros_like(d)
#         exz = np.zeros_like(d)

#         exy = (du_y / d - tz) / 2.

#     return exx, eyy, ezz, eyz, exz, exy


# def shape_func_FK(x, L, Phi=1.):

#     x_n1 = x / L
#     x_n2 = (x / L)**2
#     x_n3 = (x / L)**3

#     nw0 = 1. / (1. + Phi) * (2. * x_n3 - 3. * x_n2 - Phi * x_n1 + (1. + Phi))
#     nw1 = L / (1. + Phi) * (x_n3 - (2. + Phi / 2.) * x_n2 + (1. + Phi / 2.) * x_n1)
#     nw2 = -1. / (1. + Phi) * (2. * x_n3 - 3. * x_n2 - Phi * x_n1)
#     nw3 = L / (1. + Phi) * (x_n3 - (1. - Phi / 2.) * x_n2 - Phi / 2. * x_n1)

#     nt0 = 6. / (1. + Phi) / L * (x_n2 - x_n1)
#     nt1 = 1. / (1. + Phi) * (3. * x_n2 - (4. + Phi) * x_n1 + (1. + Phi))
#     nt2 = -6. / (1. + Phi) / L * (x_n2 - x_n1)
#     nt3 = 1. / (1. + Phi) * (3. * x_n2 - (2. - Phi) * x_n1)

#     Nw = np.vstack([nw0, nw1, nw2, nw3])
#     Nt = np.vstack([nt0, nt1, nt2, nt3])

#     return Nw, Nt


# def shape_func_antideriv_FK(x, L, Phi=1.):

#     x_n1 = x / L
#     x_n2 = (x / L)**2
#     x_n3 = (x / L)**3

#     mt0 = 6. / (1. + Phi) * (x_n3 / 3. - x_n2 / 2.)
#     mt1 = L / (1. + Phi) * (x_n3 - (2. + Phi / 2.) * x_n2 + (1. + Phi) * x_n1)
#     mt2 = -6. / (1. + Phi) * (x_n3 / 3. - x_n2 / 2.)
#     mt3 = L / (1. + Phi) * (x_n3 - (1. - Phi / 2.) * x_n2)

#     Mt = np.vstack([mt0, mt1, mt2, mt3])

#     return Mt
