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

from beam_networks.utils import box_selection, point_selection


def _get_bc_dof(nodes, select, selection, vector, num=1):
    """Transforms user input of selected nodes and displacement, load vectors
    to DOF.

    Parameters
    ----------
    nodes :  np.ndarray
        Nodal coordinates
    select : str
        How to select nodes, either 'box' or 'node'
    selection : array-like
        For box selection: lower and upper limits of bounding box in relative units,
        i.e. [xlo, xhi, ylo, yhi[, zlo, zhi]]. None entries cooresponds to the min/max limits, e.g.
        [None, None, 0.5, None, None, None] means upper half of the domain in the y-direction.
        For node selection: list of nodes.
    vector : array-like
        Vector of length 3 and 6 for 2D and 3D systems, respectively. None means the corresponding DOF is
        not affected.
        In 2D: [ux, uy, tz] (Dircihlet) or [Fx, Fy, Mz] (Neumann)
        In 3D: [ux, uy, uz, tx, ty, tz] (Dircihlet) or [Fx, Fy, Fz, Mx, My, Mz] (Neumann)
    num : int
            For point selection only, maximum number of closest nodes to select. The default is 1.

    Returns
    -------
    list
        List of key value pairs to be added to the boundary conditions

    Raises
    ------
    IOError
        Node selection method has not been specified
    """

    num_nodes, ndim = nodes.shape
    dof_per_node = 3 * (ndim - 1)

    if select == 'box':
        node_mask = box_selection(nodes, selection)
    elif select == 'node':
        node_mask = np.zeros(num_nodes, dtype=bool)
        node_mask[selection] = True
    elif select == 'point':
        node_mask = point_selection(nodes, selection, num)
    else:
        raise IOError("Either box or node selection must be given in BC dictionary")

    nodes = np.arange(num_nodes)[node_mask]
    all_dofs = np.arange(num_nodes * dof_per_node).reshape((num_nodes, dof_per_node))
    dof_mask = [i for i, k in enumerate(vector) if k is not None]
    vector = [float(v) for v in vector if v is not None]
    dofs = all_dofs[node_mask][:, dof_mask].flatten().tolist()
    dof_val = node_mask.sum() * vector
    dim = len(dof_mask)

    out = [('dof', dofs), ('dof_val', dof_val), ('dim', dim), ('nodes', nodes.tolist())]

    return out


def _assemble_BCs(bc_dict):
    """Assemble boundary conditions.

    Parameters
    ----------
    bc_dict : dict
        Boundary conditions dictionary

    Returns
    -------
    list
        Dirichlet DOFs
    list
        Dirichlet DOF values
    list
        Neumann DOFs
    list
        Neumann DOF values

    Raises
    ------
    RuntimeError
        For invalid BC definitons
    """

    _dof_D = [v for bc in bc_dict.values() if bc['type'] == 'D' and bc['active'] for v in bc['dof']]
    _val_D = [v for bc in bc_dict.values() if bc['type'] == 'D' and bc['active'] for v in bc['dof_val']]
    _dof_N = [v for bc in bc_dict.values() if bc['type'] == 'N' and bc['active'] for v in bc['dof']]
    _val_N = [v for bc in bc_dict.values() if bc['type'] == 'N' and bc['active'] for v in bc['dof_val']]

    # Check for double N BCs
    _, cNN = np.unique(_dof_N, return_counts=True)
    if np.any(cNN > 1):
        raise RuntimeError('Overlapping Neumann BCs')

    # Check for double D BCs
    _, cDD = np.unique(_dof_D, return_counts=True)
    if np.any(cDD > 1):
        raise RuntimeError('Overlapping Dirichlet BCs')

    # Check for overlapping D and N BCs
    _, cDN = np.unique(_dof_D + _dof_N, return_counts=True)
    if np.any(cDN > 1):
        raise RuntimeError('Overlapping Dirichlet and Neumann BCs')

    return _dof_D, _val_D, _dof_N, _val_N
