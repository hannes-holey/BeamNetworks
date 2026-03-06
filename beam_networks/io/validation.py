#
# Copyright 2025-2026 Hannes Holey
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
import warnings


def zero_pad_2d_array(arr):

    if arr.shape[1] == 2:
        return np.hstack([arr, np.zeros(arr.shape[0])[:, None]])
    else:
        return arr


def _dict_has_keys(d, required):
    """Check if dictionary has all required keys

    Parameters
    ----------
    d : dict
        Dictionary to test
    required : list
        Keys

    Returns
    -------
    bool
        True if all keys in required are in dictionary
    """

    return np.all([key in d.keys() for key in required])


def check_input_dict(container: dict, keys: list, defaults: list,
                     allowed: list) -> dict:
    """Validate and sanitise a dictionary of settings against expected keys.

    For each key in *keys*, the corresponding entry in *container* is checked
    against the type of the default value and, if *allowed* is not None, against
    the list of allowed values. Invalid or missing entries are replaced by the
    default with a warning.

    Parameters
    ----------
    container : dict
        Dictionary of settings to validate (modified in-place).
    keys : list of str
        Expected keys in *container*.
    defaults : list
        Default value for each key. The type of each default is used to coerce
        the stored value.
    allowed : list
        Allowed value list for each key, or None to accept any value of the
        correct type.

    Returns
    -------
    dict
        The validated (and possibly corrected) settings dictionary.
    """
    types = [type(d) for d in defaults]

    for k, t, d, a in zip(keys, types, defaults, allowed):
        if k in container.keys() and a is None:
            container[k] = t(container[k])
        elif k in container.keys() and container[k] in a:
            container[k] = t(container[k])
        else:
            warnings.warn(f"Invalid or missing option for '{k}'. Falling back to default ({d}).")
            container[k] = d

    return container
