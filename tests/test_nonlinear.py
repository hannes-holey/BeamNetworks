#
# Copyright 2026 Hannes Holey
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
"""Geometrically nonlinear solver tests.

Rolling-cantilever benchmark (Crisfield 1990): a cantilever of length L
loaded with a tip moment M = 2π EI / L bends into a complete circle, so
the free-end tip returns to the clamped-end position (origin).

The same geometry arises when a tip *rotation* of 2π is prescribed as a
Dirichlet BC instead of applying a moment, since a uniform arc is the
energy-minimising shape for prescribed end rotations with free translational
tip DOFs.  Both loading modes are tested here.
"""
import numpy as np
import pytest

from beam_networks.problem import ElasticNetwork
from beam_networks.geometry.geo import get_geometric_props


def _cantilever(ne, matrix):
    """Return a fresh cantilever ElasticNetwork with clamped root."""
    Lx = 10.
    beam_prop = {'b': 0.1, 'h': 0.1, 'E': 2.e11, 'nu': 0., 'name': 'rectangle'}
    x = np.linspace(0., Lx, ne + 1)
    nodes = np.column_stack([x, np.zeros_like(x)])
    edges = np.column_stack([np.arange(ne), np.arange(ne) + 1])
    net = ElasticNetwork(nodes, edges, beam_prop=beam_prop,
                         options={'vectorize': False, 'matrix': matrix,
                                  'verbose': False},
                         assemble_on_init=False)
    net.add_BC('clamp', 'D', 'node', [0], [0., 0., 0.])
    return net, Lx, beam_prop


@pytest.mark.parametrize('ne, matrix', [
    (20,  'dense'),
    (20,  'bsr'),
    (50,  'dense'),
    (50,  'bsr'),
])
def test_tip_moment_circle(ne, matrix):
    """Free tip must return to the origin after a full 2π tip moment."""
    net, Lx, beam_prop = _cantilever(ne, matrix)
    Iz = get_geometric_props(beam_prop)[1]
    Mref = 2. * np.pi * beam_prop['E'] * Iz / Lx

    net.add_BC('load', 'N', 'node', [ne], [None, None, Mref])
    net.solve_nonlinear(n_steps=100, tol=1e-9, verbose=False, matrix=matrix)

    np.testing.assert_allclose(net.displaced_nodes[-1], [0., 0.], atol=1e-4)


@pytest.mark.parametrize('ne, matrix', [
    (20, 'dense'),
    (20, 'bsr'),
    (50, 'bsr'),
])
def test_tip_rotation_circle(ne, matrix):
    """Prescribed tip rotation of 2π must produce the same full circle.

    Prescribing the rotational DOF at the tip to 2π (360°) with the tip
    translational DOFs left free gives a uniform-curvature arc — physically
    equivalent to a tip moment — so the displaced tip must also return to
    the clamped-end origin.
    """
    net, Lx, _ = _cantilever(ne, matrix)
    net.add_BC('tip_rot', 'D', 'node', [ne], [None, None, 2. * np.pi])
    net.solve_nonlinear(n_steps=100, tol=1e-9, verbose=False, matrix=matrix)

    np.testing.assert_allclose(net.displaced_nodes[-1], [0., 0.], atol=1e-4)


@pytest.mark.parametrize('ne', [20, 50])
def test_tip_rotation_semicircle(ne):
    """Prescribed tip rotation of π must produce a semicircle.

    For a uniform arc with total rotation θ the tip position is::

        x_tip = (L/θ) sin θ
        y_tip = (L/θ) (1 − cos θ)

    For θ = π: x_tip = 0, y_tip = 2L/π.
    """
    net, Lx, _ = _cantilever(ne, 'bsr')
    net.add_BC('tip_rot', 'D', 'node', [ne], [None, None, np.pi])
    net.solve_nonlinear(n_steps=100, tol=1e-9, verbose=False, matrix='bsr')

    tip = net.displaced_nodes[-1]
    x_exp = 0.
    y_exp = 2. * Lx / np.pi
    np.testing.assert_allclose(tip[0], x_exp, atol=1e-4)
    # y discretisation error ~ O((L/ne)^2); 1 % of y_exp covers ne=20
    np.testing.assert_allclose(tip[1], y_exp, rtol=1e-2)
