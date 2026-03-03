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
from beam_networks.utils import _dict_has_keys


def get_geometric_props(beam_prop: dict) -> tuple[float, float, float, float, float, float]:
    """Compute geometric properties of the beam cross-section.

    Parameters
    ----------
    beam_prop : dict
        Beam properties dict. Must contain ``'name'`` (``'circle'`` or
        ``'rectangle'``), ``'E'``, and ``'nu'``. Circle additionally requires
        ``'radius'``; rectangle requires ``'b'`` (width) and ``'h'`` (height).

    Returns
    -------
    Iy : float
        Second moment of area about the local y-axis.
    Iz : float
        Second moment of area about the local z-axis.
    Ip : float
        Second polar moment of area (Iy + Iz).
    A : float
        Cross-sectional area.
    kappa : float
        Timoshenko shear correction factor.
    ymax : float
        Maximum distance from the beam neutral axis to the outer surface
        (used in stress recovery).
    """

    required_keys = ['name', 'E', 'nu']
    assert _dict_has_keys(beam_prop, required_keys)
    assert beam_prop['name'] in ['circle', 'rectangle']

    nu = beam_prop['nu']

    if beam_prop['name'] == 'circle':
        required_keys_circle = ['radius']
        assert _dict_has_keys(beam_prop, required_keys_circle)

        radius = beam_prop['radius']
        Iz = np.pi * radius**4 / 4.
        Iy = np.pi * radius**4 / 4.
        Ip = Iy + Iz

        A = np.pi * radius**2
        ymax = radius
        kappa = 6. * (1. + nu) / (7. + 6. * nu)  # (Wikipedia)

        return Iy, Iz, Ip, A, kappa, ymax

    elif beam_prop['name'] == 'rectangle':
        # Only in 2D
        required_keys_rectangle = ['b', 'h']
        assert _dict_has_keys(beam_prop, required_keys_rectangle)

        b = beam_prop['b']
        h = beam_prop['h']
        Iz = h**3 * b / 12
        Iy = b**3 * h / 12
        Ip = Iy + Iz
        A = b * h
        kappa = 10. * (1. + nu) / (12. + 11. * nu)
        ymax = h

        return Iy, Iz, Ip, A, kappa, ymax


def get_geometric_props_derivative(beam_prop: dict,
                                   derivative: int | None) -> tuple[float, float, float, float, float, float]:
    """Derivatives of geometric cross-section properties w.r.t. a shape parameter.

    For circular cross-sections the derivative is always taken with respect to
    the radius (``derivative`` is ignored).
    For rectangular cross-sections, *derivative* selects the parameter:
    ``0`` → height *h*, ``1`` → width *b*.

    Parameters
    ----------
    beam_prop : dict
        Beam properties dict (see :func:`get_geometric_props`).
    derivative : int or None
        Shape-parameter selector. For circles: unused (pass any value or None).
        For rectangles: ``0`` for *h*, ``1`` for *b*.

    Returns
    -------
    dIy : float
        Derivative of Iy w.r.t. the selected shape parameter.
    dIz : float
        Derivative of Iz w.r.t. the selected shape parameter.
    dIp : float
        Derivative of Ip w.r.t. the selected shape parameter.
    dA : float
        Derivative of cross-sectional area.
    dkappa : float
        Derivative of the shear correction factor (zero for both shapes).
    dymax : float
        Derivative of the maximum surface distance.
    """

    required_keys = ['name', 'E', 'nu']
    assert _dict_has_keys(beam_prop, required_keys)
    assert beam_prop['name'] in ['circle', 'rectangle']

    if beam_prop['name'] == 'circle':
        required_keys_circle = ['radius']
        assert _dict_has_keys(beam_prop, required_keys_circle)

        radius = beam_prop['radius']

        dIz = np.pi * radius**3
        dIy = np.pi * radius**3
        dIp = dIy + dIz

        dA = 2*np.pi * radius

        dkappa = 0
        dymax = 1

    elif beam_prop['name'] == 'rectangle':
        # Only in 2D
        required_keys_rectangle = ['b', 'h']
        assert _dict_has_keys(beam_prop, required_keys_rectangle)

        b = beam_prop['b']
        h = beam_prop['h']

        # derivative with respect to h
        if derivative == 0:

            dIz = h**2 * b / 4
            dIy = b**3 / 12
            dIp = dIy + dIz
            dA = b
            dkappa = 0
            dymax = 1
        # derivative with respect to b
        elif derivative == 1:
            dIz = h**3 / 12
            dIy = b**2 * h / 4
            dIp = dIy + dIz
            dA = h
            dkappa = 0
            dymax = 0

    return dIy, dIz, dIp, dA, dkappa, dymax
