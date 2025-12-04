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

from beam_networks.geo import get_geometric_props


def cantilever_analytic(x, L, Fext, a, beam_prop):
    """

    Analytical solution for a cantilever beam with a point force and torque.

    (cf. Baier-Saip et al., Eur. J. Mech. / A Solids 79 (2020), Eq. 18)

    """

    N, P, M = Fext

    assert a <= 1.
    assert a > 0.

    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2 * (1. + nu))

    Iy, Iz, _, A, kappa, ymax = get_geometric_props(beam_prop)

    _x = x / L
    _xr = _x[_x > a]

    EA = E * A
    EI = E * Iz
    kAG = kappa * A * G

    ux = N * L / EA * _x
    ux[_x >= a] = N * L / EA * a

    uy = M * L**2 / (2 * EI) * _x**2 + P * L**3 / EI * (a / 2 * _x**2 - _x**3 / 6) + P * x / kAG
    uy[_x > a] = M * L**2 / EI * (a * _xr - a**2 / 2.) + P * L**3 / EI * (a**2 * _xr / 2. - a**3 / 6.) + P * L * a / kAG

    phi = M * L / EI * _x + P * L**2 / EI * (a * _x - _x**2 / 2)
    phi[_x > a] = M * L / EI * a + P * L**2 / EI * a**2 / 2

    # Gradients
    dux = N / EA * np.ones_like(_x)
    dux[_x > a] = 0.

    duy = M * L / EI * _x + P * L**3 / EI * (a * _x / L - _x**2 / L / 2) + P / kAG
    duy[_x > a] = M * L / EI * a + P * L**2 / EI * a**2 / 2

    dphi = M / EI + P * L**2 / EI * (a / L - _x / L)
    dphi[_x > a] = 0.

    exx = dux + np.sign(dphi) * ymax * dphi
    exy = duy - phi

    sxx = E * exx
    sxy = G * exy

    sVM = np.sqrt(sxx**2 + 3 * sxy**2)

    return ux, uy, phi, sVM
