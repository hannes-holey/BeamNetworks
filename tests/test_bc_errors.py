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
from beam_networks.problem import BeamNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROPS = {'name': 'circle', 'radius': 0.05, 'E': 1.0, 'nu': 0.3}
_OPTS = {'matrix': 'bsr', 'vectorize': True, 'verbose': False}


def _three_node_line():
    """Minimal 2D line: nodes 0–1–2, two edges."""
    nodes = np.array([[0., 0.], [1., 0.], [2., 0.]])
    edges = np.array([[0, 1], [1, 2]])
    return BeamNetwork(nodes, edges, beam_prop=_PROPS, valid=True, options=_OPTS)


def _valid_problem():
    """Well-posed 2D cantilever ready to solve."""
    p = _three_node_line()
    p.add_BC('0', 'D', 'node', [0], [0., 0., 0.])
    p.add_BC('1', 'N', 'node', [2], [0., 1e-3, 0.])
    return p


# ---------------------------------------------------------------------------
# No boundary conditions
# ---------------------------------------------------------------------------

def test_no_bc_raises():
    """Solving without any BCs should raise RuntimeError."""
    p = _three_node_line()
    with pytest.raises(RuntimeError, match="No boundary conditions given"):
        p.solve()


# ---------------------------------------------------------------------------
# Invalid solver name
# ---------------------------------------------------------------------------

def test_invalid_solver_raises():
    """Unknown solver name should raise AssertionError (assert in solve.py)."""
    p = _valid_problem()
    with pytest.raises(AssertionError):
        p.solve(solver='mumps')


# ---------------------------------------------------------------------------
# Overlapping Neumann BCs
# ---------------------------------------------------------------------------

def test_overlapping_neumann_bcs():
    """Two Neumann BCs sharing a DOF should raise RuntimeError."""
    p = _three_node_line()
    p.add_BC('0', 'D', 'node', [0], [0., 0., 0.])
    # Two N BCs on the same node / same DOF component
    p.add_BC('n1', 'N', 'node', [2], [0., 1e-3, 0.])
    p.add_BC('n2', 'N', 'node', [2], [0., 2e-3, 0.])

    with pytest.raises(RuntimeError, match="Overlapping Neumann BCs"):
        p.solve()


# ---------------------------------------------------------------------------
# Overlapping Dirichlet BCs
# ---------------------------------------------------------------------------

def test_overlapping_dirichlet_bcs():
    """Two Dirichlet BCs sharing a DOF should raise RuntimeError."""
    p = _three_node_line()
    # Two D BCs on the same node (all DOF overlap)
    p.add_BC('d1', 'D', 'node', [0], [0., 0., 0.])
    p.add_BC('d2', 'D', 'node', [0], [0., 0., 0.])
    p.add_BC('load', 'N', 'node', [2], [0., 1e-3, 0.])

    with pytest.raises(RuntimeError, match="Overlapping Dirichlet BCs"):
        p.solve()


# ---------------------------------------------------------------------------
# Overlapping Dirichlet + Neumann BCs
# ---------------------------------------------------------------------------

def test_overlapping_dirichlet_neumann_bcs():
    """A DOF constrained by both D and N BCs should raise RuntimeError."""
    p = _three_node_line()
    # D at node 0 (ux=uy=θz=0) and N at node 0 (Fx applied — same ux DOF)
    p.add_BC('d0', 'D', 'node', [0], [0., 0., 0.])
    p.add_BC('n0', 'N', 'node', [0], [1e-3, None, None])  # Fx on same node

    with pytest.raises(RuntimeError, match="Overlapping Dirichlet and Neumann BCs"):
        p.solve()


# ---------------------------------------------------------------------------
# Successful solve (sanity check)
# ---------------------------------------------------------------------------

def test_valid_solve_succeeds():
    """A properly defined problem should solve without raising."""
    p = _valid_problem()
    p.solve()
    assert p.has_solution
