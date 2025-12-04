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
import os
import numpy as np
import tarfile
import shutil
import yaml
from meshio import Mesh
from meshio.xdmf import TimeSeriesWriter
import trimesh
import scipy.sparse as sp

from beam_networks.utils import zero_pad_2d_array
from beam_networks.stress import get_element_mises_stress


def _to_tar(filename, problem):
    """Create gzipped tar archive from an instance of the BeamNetwork class.

    Parameters
    ----------
    filename : str
        Name of the archive to write into
    problem : BeamNetwork
        Class instance
    """

    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    np.save(os.path.join(tmp_dir, 'nodes.npy'), problem._nodes)
    np.save(os.path.join(tmp_dir, 'edges.npy'), problem._edges)
    np.save(os.path.join(tmp_dir, 'active_edges.npy'), problem._active_edges)

    if problem._options['matrix'] == 'dense':
        np.save(os.path.join(tmp_dir, 'K.npy'), problem._K)
    else:
        np.save(os.path.join(tmp_dir, 'K_data.npy'), problem._K.data)
        np.save(os.path.join(tmp_dir, 'K_indices.npy'), problem._K.indices)
        np.save(os.path.join(tmp_dir, 'K_indptr.npy'), problem._K.indptr)

    with open(os.path.join(tmp_dir, 'bc.yaml'), 'w') as f:
        yaml.dump(problem._bc, f)

    with open(os.path.join(tmp_dir, 'prop.yaml'), 'w') as f:
        yaml.dump(problem._beam_prop, f)

    misc = problem._options
    misc['outdir'] = problem._outdir
    misc['boxsize'] = problem._boxsize.tolist()
    misc['periodic'] = problem._periodic.tolist()

    if problem.has_solution:
        np.save(os.path.join(tmp_dir, 'sol.npy'), problem.sol)
        np.save(os.path.join(tmp_dir, 'sVM.npy'), problem._sVM)
        misc['has_solution'] = True
    else:
        misc['has_solution'] = False

    with open(os.path.join(tmp_dir, 'misc.yaml'), 'w') as f:
        yaml.dump(misc, f)

    files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]

    with tarfile.open(filename, "w:gz") as tar:
        for file in files:
            tar.add(file, arcname=os.path.basename(file))

    shutil.rmtree(tmp_dir)


def _from_tar(filename):
    """Read tar archive to reconstruct the state of a previously saved object.

    Parameters
    ----------
    filename : str
        Name of the archive

    Returns
    -------
    np.ndarray
        Nodes

    np.ndarray
        Edges

    np.ndarray
        Active edges

    dict
        Beam properties

    scipy.sparse.bsr_array or np.ndarray
        Stiffness matrix

    dict
        Boundary conditions

    np.ndarray
        Nodal solution

    np.ndarray
        Elemental solution (von Mises stress)

    dict
        Miscellaneous (options, flags, etc...)
    """

    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=tmp_dir, filter='data')

    nodes = np.load(os.path.join(tmp_dir, 'nodes.npy'))
    edges = np.load(os.path.join(tmp_dir, 'edges.npy'))
    active_edges = np.load(os.path.join(tmp_dir, 'active_edges.npy'))

    with open(os.path.join(tmp_dir, 'bc.yaml'), 'r') as f:
        bc = yaml.safe_load(f)

    with open(os.path.join(tmp_dir, 'prop.yaml'), 'r') as f:
        beam_prop = yaml.safe_load(f)

    with open(os.path.join(tmp_dir, 'misc.yaml'), 'r') as f:
        misc = yaml.safe_load(f)

    if misc['has_solution']:
        sol = np.load(os.path.join(tmp_dir, 'sol.npy'))
        sVM = np.load(os.path.join(tmp_dir, 'sVM.npy'))
    else:
        sol = None
        sVM = None

    # Reconstruct stiffness matrix
    if misc['matrix'] == 'dense':
        K = np.load(os.path.join(tmp_dir, 'K.npy'))
    else:
        data = np.load(os.path.join(tmp_dir, 'K_data.npy'))
        indices = np.load(os.path.join(tmp_dir, 'K_indices.npy'))
        indptr = np.load(os.path.join(tmp_dir, 'K_indptr.npy'))

        num_nodes, dim = nodes.shape
        dof_per_node = 3 * (dim - 1)
        num_dof = num_nodes * dof_per_node

        K = sp.bsr_array((data, indices, indptr),
                         shape=(num_dof, num_dof),
                         blocksize=(dof_per_node, dof_per_node))

    shutil.rmtree(tmp_dir)

    return nodes, edges, active_edges, beam_prop, K, bc, sol, sVM, misc


def _write_vtk(file, nodes, edges, disp, rot, stress):
    """Write vtk file.

    Parameters
    ----------
    file : str or path-like
        filename
    nodes : numpy.ndarray
        Nodal coordinates
    edges : numpy.ndarray
        Edge indices
    disp : numpy.ndarray
        Nodal displacements
    rot : numpy.ndarray
        Nodal rotations
    stress : numpy.ndarray
        Element von Mises stress
    """

    cells = [
        ("line", edges),
    ]

    point_data = {"u": disp,
                  "theta": rot
                  }

    cell_data = {"svM": [stress]}

    mesh = Mesh(nodes,
                cells,
                point_data=point_data,
                cell_data=cell_data)

    mesh.write(file, file_format="vtk")


def _to_vtk_periodic(nodes,
                     pbc_nodes,
                     edges,
                     pbc_edges,
                     displacement,
                     rotation,
                     beam_prop,
                     file="out.vtk"):
    """Write vtk file for periodic structure.

    Create explicit copies of ghost nodes and.

    Parameters
    ----------
    nodes : numpy.ndarray
        Nodal coordinates
    pbc_nodes : numpy.ndarray
        Nodal coordinates of nodes involved in PBC-crossing edges
    edges : numpy.ndarray
        Edge indices
    pbc_edges : numpy.ndarray
        Edges crossing periodic boundaries
    displacement : numpy.ndarray
        Nodal displacements
    rotation : numpy.ndarray
        Nodal rotations
    beam_prop : dict
        Beam properties (elastic, geometry)
    file : str or path-like, optional
        Filename (the default is "out.vtk")
    """

    dim = nodes.shape[1]

    points = zero_pad_2d_array(nodes)
    disp = zero_pad_2d_array(displacement)
    rot = rotation
    lines = edges.copy()

    # Explicit copy of ghost nodes
    ghost_nodes = zero_pad_2d_array(pbc_nodes)
    points = np.vstack([points, ghost_nodes])
    n_pbc = ghost_nodes.shape[0] // 2

    # Update edge indices with those of ghost nodes
    offset = nodes.shape[0]
    e = np.vstack([np.arange(n_pbc), n_pbc + np.arange(n_pbc)]).T + offset
    lines[pbc_edges] = e

    # Append and update nodal and cell data
    _e = edges[pbc_edges].T.flatten()
    disp = np.vstack([disp, disp[_e]])
    rot = np.vstack([rot, rot[_e]])
    vectors = (points[lines[:, 1]] - points[lines[:, 0]])
    sol = np.hstack([disp[:, :dim], rot]).flatten()
    stress = get_element_mises_stress(points[:, :dim],
                                      lines,
                                      vectors[:, :dim],
                                      sol,
                                      beam_prop,
                                      rot=None,
                                      mode='mean')

    _write_vtk(file, points, lines, disp, rot, stress)


# def _to_vtk(nodes, edges, displacement, rotation, stress, file="foo.vtk"):
#     """Write structure and possibly solution to VTK file

#     Parameters
#     ----------

#     nodes : numpy.ndarray
#         Nodal coordinates
#     edges : numpy.ndarray
#         Edge indices
#     displacement : numpy.ndarray
#         Nodal displacements
#     rotation : numpy.ndarray
#         Nodal rotations
#     stress : numpy.ndarray
#         Element von Mises stress
#     file : str or path-like, optional
#         Filename (the default is "out.vtk")
#     """

#     points = zero_pad_2d_array(nodes)
#     disp = zero_pad_2d_array(displacement)

#     _write_vtk(file, points, edges, disp, rotation, stress)


def _to_vtk(file, coords, adj,
            r=None,
            u=None,
            f=None,
            stress=None):
    """Write structure and possibly solution to VTK file.

    Parameters
    ----------
    file : str, optional
        Output filename
    coords : np.ndarray
        Coordinates of node points with shape (n_nodes,ndim)
    adj : np.ndarray
        adjacency list of shape (n_beams,2)
    r : np.ndarray, optional
        radius of each beam shape (n_beams)
    u : np.ndarray, optional
        (generalized) displacements of each node shape (n_nodes,node_dof)
    f : np.ndarray, optional
        (generalized) forces of each node shape (n_nodes,node_dof)
    stress : np.ndarray, optional
        von Mises stress of each beam shape (n_beams)

    """

    #
    n_nodes, ndim = coords.shape
    n_dof = 3 * (ndim - 1)
    n_transl = ndim
    n_rot = n_dof - n_transl
    #
    cells = [("line", adj), ]
    # create mask for drawing degrees of freedom
    if u is not None or f is not None:
        # masks for drawing degrees of freedom
        if ndim == 2:
            mask = np.array([True, True, False])
        elif ndim == 3:
            mask = np.array([True, True, True, False, False, False])
        mask = np.tile(mask, n_nodes)
    # insert data for nodes
    point_data = {}
    if u is not None:
        point_data.update({"u": zero_pad_2d_array(u[mask].reshape(n_nodes, n_transl)),
                           "theta": u[~mask].reshape(n_nodes, n_rot)})
    if f is not None:
        point_data.update({"f": f[mask].reshape(n_nodes, n_transl),
                           "m": f[~mask].reshape(n_nodes, n_rot)})
    # insert data for elements
    cell_data = {}
    if r is not None:
        cell_data.update({"radius": [r]})
    if stress is not None:
        cell_data.update({"svM": [stress]})
    #
    mesh = Mesh(zero_pad_2d_array(coords),
                cells,
                point_data=point_data,
                cell_data=cell_data)

    mesh.write(file, file_format="vtk")
    return


def _to_xdmf(file,
             coords, adj,
             rs=None,
             us=None,
             fs=None,
             stresses=None):
    """Write structure and possibly solution to XDMF file for timeseries

    Parameters
    ----------
    file : str, optional
        Output filename
    coords : np.ndarray
        Coordinates of node points each with shape (n_nodes,ndim)
    adj : np.ndarray
        adjacency list of shape (n_beams,2)
    rs : list of np.ndarrays, optional
        list of length n_tsteps+1 with radii of each beam shape (n_beams)
    us : np.ndarray, optional
        list of length n_tsteps+1 with (generalized) displacements of each node shape (n_nodes,node_dof)
    fs : np.ndarray, optional
        list of length n_tsteps+1 with (generalized) forces of each node shape (n_nodes,node_dof)
    stresses : np.ndarray, optional
        list of length n_tsteps+1 with stresses of each beam shape (n_beams)

    """

    #
    n_nodes, ndim = coords.shape
    n_dof = 3 * (ndim - 1)
    n_transl = ndim
    n_rot = n_dof - n_transl
    #
    cells = [("line", adj), ]
    #
    if us is not None:
        # masks for drawing degrees of freedom
        if ndim == 2:
            mask = np.array([True, True, False])
        elif ndim == 3:
            mask = np.array([True, True, True, False, False, False])
        mask = np.tile(mask, n_nodes)
    #
    with TimeSeriesWriter(file+".xdmf") as writer:
        writer.write_points_cells(coords, cells)
        for i in np.arange(len(rs)-1):
            # insert data for nodes
            point_data = {}
            if us is not None:
                point_data.update({"u": us[i][mask].reshape(n_nodes, n_transl),
                                   "theta": us[i][~mask].reshape(n_nodes, n_rot), })
            if fs is not None:
                point_data.update({"f": fs[i][mask].reshape(n_nodes, n_transl),
                                   "m": fs[i][~mask].reshape(n_nodes, n_rot)})
            # insert data for elements
            cell_data = {}
            if rs is not None:
                cell_data.update({"radius": [rs[i], ]})
            if stresses is not None:
                cell_data.update({"svM": [stresses[i], ]})
            #
            writer.write_data(i,
                              point_data=point_data,
                              cell_data=cell_data)
    return


def _to_stl(file, coords, edges, props):

    meshes = []
    dim = coords.shape[1]
    coords = zero_pad_2d_array(coords)

    connectors_added = []

    for n0, n1 in edges:
        # for segment in segments:

        segment = np.hstack([coords[n0], coords[n1]])

        start = segment[:3]
        end = segment[3:]
        midpoint = (start + end) / 2
        vector = end - start
        length = np.linalg.norm(vector)
        z_axis = np.array([0, 0, 1])
        transformation_matrix = trimesh.geometry.align_vectors(z_axis, vector)
        transformation_matrix[:3, 3] = midpoint

        if props['name'] == 'circle':
            # capsule
            # cylinder = trimesh.creation.capsule(height=length,
            #                                     transform=transformation_matrix,
            #                                     radius=props['radius'],
            #                                     count=(18, 36))

            cylinder = trimesh.creation.cylinder(radius=props['radius'],
                                                 segment=segment.reshape(2, 3),
                                                 sections=36)

            meshes.append(cylinder)

            if n0 not in connectors_added:
                # rounding
                rc = trimesh.creation.icosphere(3, radius=props['radius'])
                rc.apply_translation(coords[n0])
                meshes.append(rc)
                connectors_added.append(n0)

            if n1 not in connectors_added:
                # rounding
                rc = trimesh.creation.icosphere(3, radius=props['radius'])
                rc.apply_translation(coords[n1])
                meshes.append(rc)
                connectors_added.append(n1)

        elif props['name'] == 'rectangle':
            # Create a box mesh for the beam with the given width and height
            box = box = trimesh.creation.box(extents=[props['b'], props['h'], length])
            box.apply_transform(transformation_matrix)
            meshes.append(box)

            if dim == 2:
                if n0 not in connectors_added:
                    seg = np.zeros(6).reshape(2, 3)
                    seg[0, :2] = coords[n0, :2]
                    seg[1, :2] = coords[n0, :2]
                    seg[0, 2] = -props['b'] / 2.
                    seg[1, 2] = props['b'] / 2.
                    rc = trimesh.creation.cylinder(radius=props['h'] / 2., segment=seg, sections=36)
                    meshes.append(rc)

                    connectors_added.append(n0)

                if n1 not in connectors_added:
                    seg = np.zeros(6).reshape(2, 3)
                    seg[0, :2] = coords[n1, :2]
                    seg[1, :2] = coords[n1, :2]
                    seg[0, 2] = -props['b'] / 2.
                    seg[1, 2] = props['b'] / 2.
                    rc = trimesh.creation.cylinder(radius=props['h'] / 2., segment=seg, sections=36)
                    meshes.append(rc)

                    connectors_added.append(n1)

    # mesh = trimesh.util.concatenate(meshes)
    # Create union of individual meshes, backend 'blender' not working in version 4.1, maybe try newer
    mesh = trimesh.boolean.union(meshes,
                                 engine='manifold'
                                 )

    print('Mesh is convex :', mesh.is_convex)
    print('Mesh is watertight :', mesh.is_watertight)
    print('Mesh is winding consistent :', mesh.is_winding_consistent)

    mesh.export(file)
