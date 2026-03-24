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
"""Logging utilities for beam_networks.

The root ``beam_networks`` logger ships with a :class:`~logging.NullHandler`
so it is completely silent by default, as recommended for library code.

Users can enable output in two ways:

1. **Via the** ``verbose`` **flag** (any call-site that exposes one):
   ``verbose=True`` / ``verbose=1`` → INFO,  ``verbose=2`` → DEBUG.
   The flag internally calls :func:`set_level`, which also installs a
   :class:`~logging.StreamHandler` on the first call.

2. **Via the standard** :mod:`logging` **module** directly::

       import logging
       logging.getLogger("beam_networks").setLevel(logging.DEBUG)

"""
import logging
import sys

_root = logging.getLogger("beam_networks")
_root.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the child logger ``beam_networks.<name>``."""
    return logging.getLogger(f"beam_networks.{name}")


def verbose_to_level(verbose) -> int:
    """Map a *verbose* flag (bool or int) to a :mod:`logging` level integer.

    =============  ==============  ========================================
    *verbose*      level           description
    =============  ==============  ========================================
    ``False`` / 0  ``WARNING``     silent (no progress output)
    ``True``  / 1  ``INFO``        progress messages
    ≥ 2            ``DEBUG``       detailed diagnostics
    =============  ==============  ========================================
    """
    if isinstance(verbose, bool):
        return logging.INFO if verbose else logging.WARNING
    v = int(verbose)
    if v <= 0:
        return logging.WARNING
    if v == 1:
        return logging.INFO
    return logging.DEBUG


def set_level(level: int) -> None:
    """Set the log level for all ``beam_networks`` loggers.

    When *level* is at or below ``INFO``, a :class:`~logging.StreamHandler`
    writing to ``stdout`` is added automatically if no non-null handler is
    already attached to the root ``beam_networks`` logger.

    Parameters
    ----------
    level : int
        A :mod:`logging` level constant (e.g. ``logging.DEBUG``,
        ``logging.INFO``, ``logging.WARNING``).
    """
    _root.setLevel(level)
    if level <= logging.INFO:
        if not any(not isinstance(h, logging.NullHandler) for h in _root.handlers):
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            _root.addHandler(handler)
