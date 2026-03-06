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

import warnings
import numpy as np
from beam_networks.geometry.selection import box_selection, point_selection
from beam_networks.io.validation import check_input_dict


# ---------------------------------------------------------------------------
# box_selection — relative fraction API
# ---------------------------------------------------------------------------

# 2D grid of 9 nodes in [0,1]×[0,1]
_NODES_2D = np.array([[x, y] for y in np.linspace(0, 1, 3) for x in np.linspace(0, 1, 3)])

# 3D grid of 8 nodes in [0,1]^3
_NODES_3D = np.array([[x, y, z]
                      for z in [0., 1.]
                      for y in [0., 1.]
                      for x in [0., 1.]])


def test_box_selection_left_half_2d():
    """lim=[0, 0.5, 0, 1] should select all nodes with x <= 0.5."""
    mask = box_selection(_NODES_2D, [0., 0.5, 0., 1.])
    selected = _NODES_2D[mask]
    assert np.all(selected[:, 0] <= 0.5 + 1e-12)


def test_box_selection_bottom_half_2d():
    """lim=[0, 1, 0, 0.5] should select all nodes with y <= 0.5."""
    mask = box_selection(_NODES_2D, [0., 1., 0., 0.5])
    selected = _NODES_2D[mask]
    assert np.all(selected[:, 1] <= 0.5 + 1e-12)


def test_box_selection_none_entries_2d():
    """None limits should fall back to domain min/max."""
    # [None, None, 0, 1] = full x range, full y range = all nodes
    mask = box_selection(_NODES_2D, [None, None, None, None])
    assert mask.sum() == len(_NODES_2D)


def test_box_selection_empty_2d():
    """An impossible region should yield no selected nodes."""
    # xlo > xhi effectively selects nothing
    mask = box_selection(_NODES_2D, [0.8, 0.2, 0., 1.])
    assert mask.sum() == 0


def test_box_selection_3d():
    """3D: select only the nodes in the lower-z half."""
    mask = box_selection(_NODES_3D, [0., 1., 0., 1., 0., 0.5])
    selected = _NODES_3D[mask]
    assert np.all(selected[:, 2] <= 0.5 + 1e-12)


def test_box_selection_returns_bool_mask():
    mask = box_selection(_NODES_2D, [0., 1., 0., 1.])
    assert mask.dtype == bool
    assert mask.shape == (len(_NODES_2D),)


# ---------------------------------------------------------------------------
# point_selection
# ---------------------------------------------------------------------------

def test_point_selection_nearest():
    """Should return the single closest node."""
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
    mask = point_selection(nodes, [0.1, 0.05])
    assert mask.sum() == 1
    # The nearest node to (0.1, 0.05) is (0, 0) = index 0
    assert mask[0]


def test_point_selection_multiple():
    """num=2 should return the 2 nearest nodes."""
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    mask = point_selection(nodes, [0., 0.], num=2)
    assert mask.sum() == 2
    # Two nearest to origin are (0,0) and one of (1,0)/(0,1)
    assert mask[0]  # (0,0) must be selected


def test_point_selection_exact_match():
    """If the point coincides with a node, that node is always selected."""
    nodes = np.array([[0., 0.], [2., 0.], [4., 0.]])
    mask = point_selection(nodes, [2., 0.])
    assert mask[1]


def test_point_selection_returns_bool_mask():
    nodes = np.array([[0., 0.], [1., 1.]])
    mask = point_selection(nodes, [0.5, 0.5])
    assert mask.dtype == bool
    assert mask.shape == (2,)


# ---------------------------------------------------------------------------
# check_input_dict
# ---------------------------------------------------------------------------

def test_check_input_dict_valid():
    """Valid entries should pass through unchanged."""
    d = {'matrix': 'bsr', 'vectorize': True, 'verbose': False}
    keys = ['matrix', 'vectorize', 'verbose']
    defaults = ['bsr', True, True]
    allowed = [['bsr', 'lil', 'dense'], None, None]

    result = check_input_dict(d, keys, defaults, allowed)
    assert result['matrix'] == 'bsr'
    assert result['vectorize'] is True
    assert result['verbose'] is False


def test_check_input_dict_invalid_falls_back():
    """Invalid entries should be replaced by their defaults (with a warning)."""
    d = {'matrix': 'unknown', 'vectorize': True, 'verbose': False}
    keys = ['matrix', 'vectorize', 'verbose']
    defaults = ['bsr', True, True]
    allowed = [['bsr', 'lil', 'dense'], None, None]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_input_dict(d, keys, defaults, allowed)

    assert result['matrix'] == 'bsr'  # fallback to default
    assert any("matrix" in str(warning.message) for warning in w)


def test_check_input_dict_missing_key_falls_back():
    """Missing keys should be filled with their defaults."""
    d = {'vectorize': True}  # 'matrix' missing
    keys = ['matrix', 'vectorize']
    defaults = ['bsr', True]
    allowed = [['bsr', 'lil', 'dense'], None]

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = check_input_dict(d, keys, defaults, allowed)

    assert result['matrix'] == 'bsr'


def test_check_input_dict_type_coercion():
    """Values of the correct type but wrong Python type should be coerced."""
    d = {'verbose': 'True'}  # string, not bool
    keys = ['verbose']
    defaults = [True]
    allowed = [None]

    result = check_input_dict(d, keys, defaults, allowed)
    # bool('True') = True (any non-empty string is truthy)
    assert isinstance(result['verbose'], bool)
