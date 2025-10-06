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
