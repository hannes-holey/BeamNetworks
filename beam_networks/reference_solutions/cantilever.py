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

from beam_networks.geometry.geo import get_geometric_props


def cantilever_analytic(x, L, Fext, a, beam_prop, euler_bernoulli=False):
    """

    Analytical solution for a cantilever beam with a point force and torque.

    (cf. Baier-Saip et al., Eur. J. Mech. / A Solids 79 (2020), Eq. 18)

    Parameters
    ----------
    x : array_like
        Positions along the beam axis.
    L : float
        Total beam length.
    Fext : array_like
        Applied loads [N, P, M] (axial force, transverse force, moment).
    a : float
        Normalised load position in (0, 1].
    beam_prop : dict
        Beam cross-section and elastic properties.
    euler_bernoulli : bool, optional
        If True, return the Euler-Bernoulli solution (shear-rigid limit
        kappa*G*A → ∞, i.e. the P/kAG shear terms are dropped).
        Default is False (full Timoshenko solution).
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

    shear_disp = 0. if euler_bernoulli else P * x / kAG
    shear_disp_r = 0. if euler_bernoulli else P * L * a / kAG
    shear_slope = 0. if euler_bernoulli else P / kAG

    ux = N * L / EA * _x
    ux[_x >= a] = N * L / EA * a

    uy = M * L**2 / (2 * EI) * _x**2 + P * L**3 / EI * (a / 2 * _x**2 - _x**3 / 6) + shear_disp
    uy[_x > a] = M * L**2 / EI * (a * _xr - a**2 / 2.) + P * L**3 / EI * (a**2 * _xr / 2. - a**3 / 6.) + shear_disp_r

    phi = M * L / EI * _x + P * L**2 / EI * (a * _x - _x**2 / 2)
    phi[_x > a] = M * L / EI * a + P * L**2 / EI * a**2 / 2

    # Gradients
    dux = N / EA * np.ones_like(_x)
    dux[_x > a] = 0.

    duy = M * L / EI * _x + P * L**3 / EI * (a * _x / L - _x**2 / L / 2) + shear_slope
    duy[_x > a] = M * L / EI * a + P * L**2 / EI * a**2 / 2

    dphi = M / EI + P * L**2 / EI * (a / L - _x / L)
    dphi[_x > a] = 0.

    exx = dux + np.sign(dphi) * ymax * dphi
    exy = duy - phi

    sxx = E * exx
    sxy = G * exy

    sVM = np.sqrt(sxx**2 + 3 * sxy**2)

    return ux, uy, phi, sVM
