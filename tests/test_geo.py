#
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

import pytest
import numpy as np
from beam_networks.geometry.geo import get_geometric_props, get_geometric_props_derivative


def circle_props(radius, nu=0.3):
    return {'name': 'circle', 'radius': radius, 'E': 1.0, 'nu': nu}


def rect_props(b, h, nu=0.3):
    return {'name': 'rectangle', 'b': b, 'h': h, 'E': 1.0, 'nu': nu}


# ---------------------------------------------------------------------------
# Rectangle cross-section
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('b, h', [(0.1, 0.2), (0.5, 0.5), (1.0, 0.3)])
def test_rectangle_props(b, h):
    Iy, Iz, Ip, A, kappa, ymax = get_geometric_props(rect_props(b, h))

    assert np.isclose(Iz, h**3 * b / 12.)
    assert np.isclose(Iy, b**3 * h / 12.)
    assert np.isclose(Ip, Iy + Iz)
    assert np.isclose(A, b * h)
    assert np.isclose(ymax, h)
    assert 0. < kappa < 1.


# ---------------------------------------------------------------------------
# Circle cross-section
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('R', [0.05, 0.1, 0.5])
def test_circle_props(R):
    Iy, Iz, Ip, A, kappa, ymax = get_geometric_props(circle_props(R))

    assert np.isclose(Iz, np.pi * R**4 / 4.)
    assert np.isclose(Iy, Iz)
    assert np.isclose(Ip, Iy + Iz)
    assert np.isclose(A, np.pi * R**2)
    assert np.isclose(ymax, R)
    assert 0. < kappa < 1.


# ---------------------------------------------------------------------------
# Rectangle derivatives vs finite difference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('b, h', [(0.1, 0.2), (0.5, 0.7), (0.3, 0.8)])
def test_rectangle_derivative_h(b, h):
    """Analytic d/dh should match numerical finite difference."""
    eps = 1e-6
    props = rect_props(b, h)
    props_plus = rect_props(b, h + eps)

    analytic = np.array(get_geometric_props_derivative(props, derivative=0))
    p0 = np.array(get_geometric_props(props))
    p1 = np.array(get_geometric_props(props_plus))
    numerical = (p1 - p0) / eps

    np.testing.assert_allclose(analytic, numerical, rtol=1e-4)


@pytest.mark.parametrize('b, h', [(0.1, 0.2), (0.5, 0.7), (0.3, 0.8)])
def test_rectangle_derivative_b(b, h):
    """Analytic d/db should match numerical finite difference."""
    eps = 1e-6
    props = rect_props(b, h)
    props_plus = rect_props(b + eps, h)

    analytic = np.array(get_geometric_props_derivative(props, derivative=1))
    p0 = np.array(get_geometric_props(props))
    p1 = np.array(get_geometric_props(props_plus))
    numerical = (p1 - p0) / eps

    np.testing.assert_allclose(analytic, numerical, rtol=1e-4)


@pytest.mark.parametrize('R', [0.05, 0.2])
def test_circle_derivative(R):
    """Analytic d/dR for circle should match numerical finite difference."""
    eps = 1e-6
    props = circle_props(R)
    props_plus = circle_props(R + eps)

    analytic = np.array(get_geometric_props_derivative(props, derivative=None))
    p0 = np.array(get_geometric_props(props))
    p1 = np.array(get_geometric_props(props_plus))
    numerical = (p1 - p0) / eps

    np.testing.assert_allclose(analytic, numerical, rtol=1e-4)
