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

from beam_networks.stiffness import get_element_stiffness_global


def test_K_derivatives_2d():

    # Example usage
    E = 1.
    nu = 0.3
    R = 0.05
    #
    d = np.array([1., 0.])
    # finite difference
    Ke = get_element_stiffness_global(beam_prop={'name': 'circle',
                                                 'radius': R,
                                                 'E': E,
                                                 'nu': nu},
                                      d=d,
                                      derivative=None)

    dr = 1e-10
    dKe_fdiff = ((get_element_stiffness_global(beam_prop={'name': 'circle',
                                                          'radius': R+dr,
                                                          'E': E,
                                                          'nu': nu},
                                               d=d,
                                               derivative=None)-Ke)/dr)

    dKe = get_element_stiffness_global(beam_prop={'name': 'circle',
                                                  'radius': R,
                                                  'E': E,
                                                  'nu': nu},
                                       d=d,
                                       derivative=0)

    np.testing.assert_almost_equal(dKe, dKe_fdiff)


def test_K_derivatives_3d():

    # Example usage
    E = 1.
    nu = 0.3
    R = 0.05
    #
    d = np.array([1., 0., 0.])
    # finite difference
    Ke = get_element_stiffness_global(beam_prop={'name': 'circle',
                                                 'radius': R,
                                                 'E': E,
                                                 'nu': nu},
                                      d=d,
                                      derivative=None)

    dr = 1e-11
    dKe_fdiff = ((get_element_stiffness_global(beam_prop={'name': 'circle',
                                                          'radius': R+dr,
                                                          'E': E,
                                                          'nu': nu},
                                               d=d,
                                               derivative=None)-Ke)/dr)

    dKe = get_element_stiffness_global(beam_prop={'name': 'circle',
                                                  'radius': R,
                                                  'E': E,
                                                  'nu': nu},
                                       d=d,
                                       derivative=0)

    np.testing.assert_almost_equal(dKe, dKe_fdiff)
