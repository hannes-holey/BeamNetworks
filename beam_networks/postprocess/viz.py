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
import meshio
import tqdm
import numpy as np
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib


def _plot_network(ax, nodes, edges, dr,
                  edge_data=None,
                  cax=None,
                  color='C0',
                  lw=2,
                  node_ids=False,
                  aspect=1.,
                  lim=None,
                  boxsize=None,
                  cbar_label=None,
                  cmap='plasma'):
    """Plot a 2D network

    Parameters
    ----------
    ax : matplotlib.pyplot.axis object
        Axis to plot into
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray
        Edge connectivity
    edge_data : np.ndarray, optional
        Scalar field to plot as color on edges (the default is None, which means no colormap)
    cax : matplotlib.pyplot.axis object or None, optional
        Axis to plot colorbar into if contour is not None (the default is None, which takes space from ax)
    color : str, optional
        If no contour is given, plot in this color (the default is 'C0')
    lw : float, optional
        Linewidth for edges (the default is 2)
    node_ids : bool, optional
            Print node numbers nect to undeformed structure (the default is False)
    aspect : float, optional
            Aspect ratio (the default is 1.)
    lim : tuple, optional
        Colorbar limits (the default is None, which takes the limits of contour)
    boxsize : list, optional
        Box size in the periodic directions. Non-periodic directions are False
        (the default is None, which does not plot the box).
    cbar_label: str
        Label for the colorbar (the default is None, which means no label)
    cmap: str
        Name of the matlpotlib colormap (default is 'plasma')
    """

    segments = []

    if boxsize is not None:
        Lx, Ly = boxsize

        # if  show_box:
        xlo, ylo = np.amin(nodes, axis=0)
        xhi, yhi = np.amax(nodes, axis=0)

        ax.plot([xhi - Lx, xhi], [yhi, yhi], color='0.5', zorder=-1)
        ax.plot([xhi - Lx, xhi], [yhi - Ly, yhi - Ly], color='0.5', zorder=-1)
        ax.plot([xhi, xhi], [yhi - Ly, yhi], color='0.5', zorder=-1)
        ax.plot([xhi - Lx, xhi - Lx], [yhi - Ly, yhi], color='0.5', zorder=-1)

    for n0, (dx, dy) in zip(edges[:, 0], dr[:, :2]):
        x0, y0 = nodes[n0, :2]
        line = (x0, y0), (x0 + dx, y0 + dy)

        segments.append(line)

    if edge_data is None:
        color = color
    else:
        sm, color = _array_to_colors(edge_data, lim=lim, cmap=cmap)
        if cax is not None:
            plt.colorbar(sm, ax=cax, orientation='horizontal',
                         label=cbar_label
                         )

    linecollection = LineCollection(segments=segments,
                                    linewidths=lw,
                                    colors=color)

    ax.add_collection(linecollection)

    if node_ids:
        for i, node in enumerate(nodes):
            x = node[0]
            y = node[1]

            ax.text(x, y, f'{i}', ha='left', va='top', fontsize=8,
                    bbox=dict(ec='none', fc='none'))

    xlo = min(np.amin(nodes[:, 0]), ax.get_xlim()[0])
    xhi = max(np.amax(nodes[:, 0]), ax.get_xlim()[1])
    ylo = min(np.amin(nodes[:, 1]), ax.get_ylim()[0])
    yhi = max(np.amax(nodes[:, 1]), ax.get_ylim()[1])

    ax.set_xlim(xlo - 0.01 * (xhi - xlo), xhi + 0.01 * (xhi - xlo))
    ax.set_ylim(ylo - 0.01 * (yhi - ylo), yhi + 0.01 * (yhi - ylo))

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.set_aspect(aspect)
    ax.set_box_aspect(aspect)

    return ax


def plot_solution_1D(ax, d, s, length=1.):
    """Plot a 1D solution along a 1D beam (composed of more than one element)

    Parameters
    ----------
    ax : matplotlib.pyplot.axis object
        Axis to plot into
    d : np.ndarray
        Solution (ux[0], uy[0], theta_z[0], ...)
    s : np.ndarray
        Von Mises stress per element
    length : float, optional
        Length of the beam (the default is 1.)
    """

    num_nodes = len(d) // 3

    xd = np.linspace(0., length, num_nodes)

    if num_nodes > 25:

        every = num_nodes // 25

        ax[0].plot(xd, d[0::3], '-')
        ax[1].plot(xd, d[1::3], '-')
        ax[2].plot(xd, d[2::3], '-')
        ax[3].plot((xd[1:] + xd[:-1]) / 2, s, '-')

        ax[0].plot(xd[::every], d[0::3][::every], 'X', color='C0')
        ax[1].plot(xd[::every], d[1::3][::every], 'X', color='C0')
        ax[2].plot(xd[::every], d[2::3][::every], 'X', color='C0')
        ax[3].plot(((xd[1:] + xd[:-1]) / 2)[::every], s[::every], 'X', color='C0')

    else:
        ax[0].plot(xd, d[0::3], '-X')
        ax[1].plot(xd, d[1::3], '-X')
        ax[2].plot(xd, d[2::3], '-X')
        ax[3].plot((xd[1:] + xd[:-1]) / 2, s, '-X')

    ax[0].legend()

    ax[0].set_ylabel(r'Displacement $u_\tilde{x}$')
    ax[1].set_ylabel(r'Displacement $u_\tilde{y}$')
    ax[2].set_ylabel(r'Rotation $\theta$')
    ax[3].set_ylabel(r'Equivalent stress $\sigma_\mathsf{vM}$')
    ax[3].set_xlabel(r'Beam coordinate $\tilde{x}$')


def _array_to_colors(arr, lim=None, cmap='plasma'):
    """Convert array into scalar mappable and colormap

    Parameters
    ----------
    arr : np.ndarray
        Array
    lim : tuple, optional
        Upper and lower limits (the default is None, which means array min and max are taken)

    Returns
    -------
    matplotlib.colormap.ScalarMappable
        scalar mappable
    np.ndarray
        colors
    """
    if lim is not None:
        vmin, vmax = lim
    else:
        vmin = arr.min()
        vmax = arr.max()
        if np.isclose(vmax - vmin, 0.):
            shift = max(0.1, np.abs(vmax))
            vmin -= shift
            vmax += shift

    if cmap not in matplotlib.colormaps.keys():
        cmap = 'plasma'

    cmap = matplotlib.colormaps[cmap]
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = cmap(norm(arr))

    return sm, colors


# Command-line scripts


def vtk2img(wdir=None, dpi=300, cbar_global_lim=False, ftype='png'):

    if wdir is None:
        parser = ArgumentParser()
        parser.add_argument('working_directory', default=os.getcwd())
        parser.add_argument('-d', '--dpi', default=300, type=int)
        parser.add_argument('-f', '--file_type', default='png', type=str)
        parser.add_argument('-g', '--cbar_global_limits', default=False, action='store_true')
        args = parser.parse_args()

        wdir = args.working_directory
        dpi = args.dpi
        cbar_global_lim = args.cbar_global_limits
        ftype = args.file_type

    all_vtk_files = sorted([os.path.join(wdir, f) for f in os.listdir(wdir) if f.endswith('.vtk')])

    if cbar_global_lim:
        lim = np.array([[meshio.read(file).cell_data['svM'][0].min(),
                         meshio.read(file).cell_data['svM'][0].max()] for file in all_vtk_files])

        vmin = np.amin(lim[:, 0])
        vmax = np.amax(lim[:, 1])
        lim = (vmin, vmax)

    else:
        lim = None

    for file in tqdm.tqdm(all_vtk_files):
        fig, ax = plt.subplots(1)

        mesh = meshio.read(file)

        nodes = mesh.points
        edges = mesh.cells_dict['line']

        svM = mesh.cell_data['svM'][0]

        _plot_network(ax, nodes, edges,
                      edge_data=svM,
                      cax=ax,
                      color='C0',
                      lw=2,
                      node_ids=False,
                      aspect=1.,
                      lim=lim)

        fname = file.rstrip('.vtk') + f'.{ftype}'

        fig.savefig(fname, dpi=dpi)
