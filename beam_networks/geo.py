import numpy as np
from beam_networks.utils import _dict_has_keys


def get_geometric_props(beam_prop):
    """Get geometric properties of the beam cross section.

    Parameters
    ----------
    beam_prop : dict
        Beam properties (cross section and elastic properties)

    Returns
    -------
    float
        Second moment of area Iy
    float
        Second moment of area Iz
    float
        Second polar moment of area Ip
    float
        Area
    float
        Shear correction factor
    float
        Maximum distance from beam neutral axis to the surface
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


def get_geometric_props_derivative(beam_prop, derivative):
    """Get derivative of geometric properties of the beam cross section with
    respect to radius

    Parameters
    ----------
    beam_prop : dict
        Beam properties (cross section and elastic properties)
    derivative : bool
        if not None returns the derivative with respect to a shape parameter.
        The shape parameter is determined by the value of derivative which
        depends on the shape of your beam.

    Returns
    -------
    float
        Derivative of second moment of area Iy
    float
        Derivative of second moment of area Iz
    float
        Derivative of second polar moment of area Ip
    float
        Derivative of area
    float
        Derivative of shear correction factor
    float
        Derivative of maximum distance from beam neutral axis to the surface
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
