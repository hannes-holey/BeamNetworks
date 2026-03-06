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
from beam_networks.stiffness import local_element_stiffness_timoshenko_exact_vec
from beam_networks.geo import get_geometric_props


def vmises_stress(rhs: np.ndarray, beam_prop: dict, mode: str) -> np.ndarray:
    """Compute von Mises stress from element force/moment vectors.

    Dispatches to the 2D or 3D implementation based on the shape of *rhs*.

    Parameters
    ----------
    rhs : np.ndarray
        Element-wise internal force/moment vector in the local frame,
        shape (num_elements, 6) for 2D or (num_elements, 12) for 3D.
    beam_prop : dict
        Beam cross-section and elastic properties.
    mode : str
        Stress aggregation along each beam: ``'max'`` takes the end-point
        maximum, ``'mean'`` averages both ends.

    Returns
    -------
    np.ndarray
        Von Mises stress per element, shape (num_elements,).
    """

    if rhs.shape[1] == 6:
        return _vmises_stress_2d(rhs, beam_prop, mode)
    else:  # de.shape[1] == 12:
        return _vmises_stress_3d(rhs, beam_prop, mode)


def stretch_bend_ratio(rhs: np.ndarray, beam_prop: dict) -> np.ndarray:
    """Compute the bending-to-total stress ratio for each beam element.

    The ratio is defined as the bending stress divided by the sum of the
    bending and axial (stretching) stresses. A value of 1 indicates a
    purely bending-dominated beam; 0 indicates purely axial loading.

    Currently only implemented for 2D systems.

    Parameters
    ----------
    rhs : np.ndarray
        Element-wise internal force/moment vector in the local frame,
        shape (num_elements, 6).
    beam_prop : dict
        Beam cross-section and elastic properties.

    Returns
    -------
    np.ndarray
        Bending ratio per element, shape (num_elements,), values in [0, 1].

    Raises
    ------
    RuntimeError
        If called for a 3D system (rhs.shape[1] == 12).
    """
    if rhs.shape[1] == 6:
        return _stress_stretch_bend_ratio(rhs, beam_prop)
    else:
        raise RuntimeError("Not implemented for 3d")


def _stress_stretch_bend_ratio(rhs, beam_prop):

    Iy, Iz, _, A, kappa, ymax = get_geometric_props(beam_prop)

    F = rhs[:, 3]
    # Q = rhs[:, 4]
    M0 = rhs[:, 2]
    M1 = rhs[:, 5]

    M = (np.abs(M0) + np.abs(M1)) / 2.

    s_bend = M * ymax / 2 / Iz
    s_stretch = np.abs(F) / A
    # s_shear = np.abs(Q) / A

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


def get_element_mises_stress(coords: np.ndarray, adj: np.ndarray,
                             d_vec: np.ndarray, sol: np.ndarray,
                             beam_prop: dict, rot: np.ndarray | None = None,
                             mode: str = 'max',
                             return_ratio: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the von Mises stress for all beam elements.

    Transforms the global solution vector into element-local DOF vectors,
    computes internal forces and moments via the local stiffness matrix, and
    evaluates the von Mises stress.

    Parameters
    ----------
    coords : np.ndarray
        Nodal coordinates, shape (num_nodes, dim).
    adj : np.ndarray
        Edge connectivity (node index pairs), shape (num_edges, 2).
    d_vec : np.ndarray
        Edge vectors (tail → head), shape (num_edges, dim).
    sol : np.ndarray
        Global displacement solution vector, shape (num_nodes * dof_per_node,).
    beam_prop : dict
        Beam cross-section and elastic properties.
    rot : np.ndarray or None, optional
        If all elements share the same orientation, the rotation matrix from
        the global to the local frame, shape (3, 3). The default is None,
        which computes element-wise orientations from *d_vec*.
    mode : str, optional
        Stress aggregation: ``'max'`` takes the end-point maximum, ``'mean'``
        averages both ends. The default is ``'max'``.
    return_ratio : bool, optional
        If True, also return the bending-to-total stress ratio per element
        (2D only). The default is False.

    Returns
    -------
    svM : np.ndarray
        Von Mises stress per element, shape (num_edges,).
    ratio : np.ndarray
        Bending ratio per element, shape (num_edges,). Only returned when
        *return_ratio* is True.
    """
    _, ndim = coords.shape

    d = np.linalg.norm(d_vec, axis=-1)

    if ndim == 3:
        dof_elem = _get_element_dof_3d(coords, adj, d_vec / d[:, None], sol, rot)
    else:
        dof_elem = _get_element_dof_2d(coords, adj, d_vec / d[:, None], sol)

    K_elem = local_element_stiffness_timoshenko_exact_vec(beam_prop, d, ndim)

    rhs = np.einsum('ijk,ik->ij', K_elem[:, :, :], dof_elem)
    svM = vmises_stress(rhs, beam_prop, mode)

    if return_ratio:
        ratio = stretch_bend_ratio(rhs, beam_prop)

        return svM, ratio

    else:
        return svM


def get_element_principal_stress(coords: np.ndarray, adj: np.ndarray,
                                 d_vec: np.ndarray, sol: np.ndarray,
                                 beam_prop: dict,
                                 rot: np.ndarray | None = None,
                                 mode: str = 'max') -> np.ndarray:
    """Compute principal stresses for all beam elements.

    Parameters
    ----------
    coords : np.ndarray
        Nodal coordinates, shape (num_nodes, dim).
    adj : np.ndarray
        Edge connectivity (node index pairs), shape (num_edges, 2).
    d_vec : np.ndarray
        Edge vectors (tail → head), shape (num_edges, dim).
    sol : np.ndarray
        Global displacement solution vector, shape (num_nodes * dof_per_node,).
    beam_prop : dict
        Beam cross-section and elastic properties.
    rot : np.ndarray or None, optional
        Rotation matrix from global to local frame, shape (3, 3), shared by
        all elements. The default is None (element-wise orientations).
    mode : str, optional
        Stress aggregation: ``'max'`` takes the end-point maximum, ``'mean'``
        averages both ends. The default is ``'max'``.

    Returns
    -------
    np.ndarray
        Principal stresses per element sorted in descending order,
        shape (num_edges, 3).
    """
    _, ndim = coords.shape

    d = np.linalg.norm(d_vec, axis=-1)

    if ndim == 3:
        dof_elem = _get_element_dof_3d(coords, adj, d_vec / d[:, None], sol, rot)
    else:
        dof_elem = _get_element_dof_2d(coords, adj, d_vec / d[:, None], sol)

    K_elem = local_element_stiffness_timoshenko_exact_vec(beam_prop, d, ndim)

    rhs = np.einsum('ijk,ik->ij', K_elem[:, :, :], dof_elem)

    return principal_stress(rhs, beam_prop, mode)


def principal_stress(rhs: np.ndarray, beam_prop: dict, mode: str) -> np.ndarray:
    """Compute principal stresses from element force/moment vectors.

    Constructs the full stress tensor at the critical point of each beam
    cross-section and returns its eigenvalues in descending order.

    Parameters
    ----------
    rhs : np.ndarray
        Element-wise internal force/moment vector in the local frame,
        shape (num_elements, 6) for 2D or (num_elements, 12) for 3D.
    beam_prop : dict
        Beam cross-section and elastic properties.
    mode : str
        Stress aggregation: ``'max'`` takes the end-point maximum, ``'mean'``
        averages both ends.

    Returns
    -------
    np.ndarray
        Principal stresses per element sorted in descending order,
        shape (num_elements, 3).
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
    Iy, Iz, Ip, A, kappa, ymax = get_geometric_props(beam_prop)

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
