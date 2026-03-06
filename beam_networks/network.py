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
import scipy.sparse as sp
from itertools import combinations
from scipy.spatial import ConvexHull

from beam_networks.utils import _remove_isolated_nodes_edges, _mic
from beam_networks.lattice import generate_cubic_lattice, generate_square_lattice, generate_bowtie_lattice
from typing import TYPE_CHECKING

from beam_networks.viz import _plot_network

if TYPE_CHECKING:
    import matplotlib


class Network:
    """Base class for network structure.

    Contains methods to generate lattice structures and some basic properties of the network.
    """

    def __init__(self,
                 nodes_positions,
                 edges_indices,
                 periodic=None,
                 boxsize=None,
                 valid=True):
        """

        Constructor.

        Parameters
        ----------
        nodes_positions : np.ndarray
            Coordinates of the nodes (2D or 3D)
        edges_indices : np.ndarray (of ints)
            Edge connectivity (indices of connected nodes)
        periodic : iterable, optional
            Use periodic boundary conditions (the default is None, which means no periodic BCs).
            Example: In a 3D system [True, False, False] means that only x is a periodic direction.
        boxsize : np.ndarray, optional
            Specify box dimensions (the default is None, which means that box dimensions are inferred from the nodes.)
        valid : bool, optional
            Skip initial sanity checks (the default is True, which assumes that the structure is valid.)
        """

        self.dim = nodes_positions.shape[1]

        if boxsize is None:
            self._boxsize = np.amax(nodes_positions, axis=0) - np.amin(nodes_positions, axis=0)
        else:
            self._boxsize = np.array(boxsize)

        if periodic is None:
            # set all directions to False
            self._periodic = np.zeros(self.dim, dtype=bool)
        else:
            self._periodic = np.array(periodic).astype(bool)

        e0, e1 = edges_indices.T
        r0 = nodes_positions[e0]
        r1 = nodes_positions[e1]
        dr = np.sqrt((r1 - r0)**2)

        if valid:
            self._nodes = nodes_positions
            self._edges = edges_indices
        else:
            # In nonperiodic directions, keep only edges that do not cross boundaries
            keep = np.all(dr[:, ~self._periodic] <= self._boxsize[~self._periodic] / 2., axis=-1).reshape(-1)
            edges_indices = edges_indices[keep]

            # Pre-process network // generate possible nodes and edges
            self._nodes, self._edges = self._preprocessing(nodes_positions, edges_indices)

        # Sort edges row-wise
        self._edges = np.sort(self._edges, axis=1)

        # Sort edges column-wise
        self._edges = self._edges[np.lexsort((self._edges[:, 1], self._edges[:, 0]))]

        # Masking arrays
        self._active_nodes = np.ones(len(self._nodes), dtype=bool)
        self._active_edges = np.ones(len(self._edges), dtype=bool)

    @property
    def nodes(self) -> np.ndarray:
        """Coordinates of active nodes.

        Returns
        -------
        np.ndarray
            Shape (num_nodes, dim).
        """
        return self._nodes[self._active_nodes]

    @property
    def num_nodes(self) -> int:
        """Number of active nodes.

        Returns
        -------
        int
        """
        return self.nodes.shape[0]

    @property
    def edges(self) -> np.ndarray:
        """Index pairs of active edges.

        Returns
        -------
        np.ndarray
            Integer array of shape (num_edges, 2).
        """
        return self._edges[self._active_edges]

    @property
    def num_edges(self) -> int:
        """Number of active edges.

        Returns
        -------
        int
        """
        return self.edges.shape[0]

    @property
    def Lx(self) -> float:
        """Box length in the x direction.

        Returns
        -------
        float
        """
        if self._boxsize[0] is None:
            return self.xhi - self.xlo
        else:
            return self._boxsize[0]

    @property
    def Ly(self) -> float:
        """Box length in the y direction.

        Returns
        -------
        float
        """
        if self._boxsize[1] is None:
            return self.yhi - self.ylo
        else:
            return self._boxsize[1]

    @property
    def Lz(self) -> float:
        """Box length in the z direction.

        Returns
        -------
        float
        """
        if self._boxsize[2] is None:
            return self.zhi - self.zlo
        else:
            return self._boxsize[2]

    @property
    def is_connected(self) -> bool:
        """Whether the active network forms a single connected component.

        Returns
        -------
        bool
        """
        graph = sp.csr_array((np.ones(self.num_edges, dtype=int), (self.edges[:, 0], self.edges[:, 1])),
                             shape=(self.num_nodes, self.num_nodes))

        n_comp, _ = sp.csgraph.connected_components(graph)

        return n_comp == 1

    @property
    def bounds(self) -> list:
        """Lower and upper bounds of the active nodal coordinates.

        Returns
        -------
        list
            ``[xlo, xhi, ylo, yhi]`` for 2D or
            ``[xlo, xhi, ylo, yhi, zlo, zhi]`` for 3D.
        """
        bounds = [self.xlo, self.xhi, self.ylo, self.yhi]
        if self.dim > 2:
            bounds.extend([self.zlo, self.zhi])

        return bounds

    @property
    def boxsize(self) -> tuple:
        """Box dimensions as a tuple ``(Lx, Ly)`` or ``(Lx, Ly, Lz)``."""
        if self.dim == 2:
            return self.Lx, self.Ly
        elif self.dim == 3:
            return self.Lx, self.Ly, self.Lz

    @property
    def xlo(self) -> float:
        """Minimum x-coordinate of all active nodes."""
        return np.amin(self._nodes[self._active_nodes, 0])

    @property
    def xhi(self) -> float:
        """Maximum x-coordinate of all active nodes."""
        return np.amax(self._nodes[self._active_nodes, 0])

    @property
    def ylo(self) -> float:
        """Minimum y-coordinate of all active nodes."""
        return np.amin(self._nodes[self._active_nodes, 1])

    @property
    def yhi(self) -> float:
        """Maximum y-coordinate of all active nodes."""
        return np.amax(self._nodes[self._active_nodes, 1])

    @property
    def zlo(self) -> float:
        """Minimum z-coordinate of all active nodes.

        .. note::
            Only valid for 3D networks. Raises ``IndexError`` for 2D networks.
        """
        return np.amin(self._nodes[self._active_nodes, 2])

    @property
    def zhi(self) -> float:
        """Maximum z-coordinate of all active nodes.

        .. note::
            Only valid for 3D networks. Raises ``IndexError`` for 2D networks.
        """
        return np.amax(self._nodes[self._active_nodes, 2])

    @property
    def bondlengths(self) -> np.ndarray:
        """Lengths of all active edges.

        Applies the minimum image convention for periodic boxes.

        Returns
        -------
        np.ndarray
            Array of edge lengths, shape (num_edges,).
        """
        return self._bondlengths[self._active_edges]

    @property
    def _bondlengths(self):
        """Length of the edges.

        Apply minimum image convention for periodic boxes.

        Returns
        -------
        np.ndarray
            Array of edge lengths
        """

        return np.linalg.norm(self._edge_vectors, axis=-1)

    @property
    def _edge_vectors(self):
        r1 = self._nodes[self._edges[:, 1]]
        r0 = self._nodes[self._edges[:, 0]]
        dr = _mic(r1 - r0, self._boxsize, self._periodic)

        return dr

    @property
    def edge_vectors(self) -> np.ndarray:
        """Vectors along all active edges (tail → head).

        Applies the minimum image convention for periodic boxes.

        Returns
        -------
        np.ndarray
            Shape (num_edges, dim).
        """
        return self._edge_vectors[self._active_edges]

    @property
    def pbc_edges(self) -> np.ndarray:
        """Boolean mask of active edges that cross periodic boundaries.

        Returns
        -------
        np.ndarray
            Boolean array of shape (num_edges,).
        """

        dr = self._nodes[self.edges[:, 1]] - self._nodes[self.edges[:, 0]]
        pbc_edges = np.any(np.abs(dr) > np.array(self.boxsize) / 2., axis=-1)

        return pbc_edges

    @property
    def volume(self) -> float:
        """Volume of the convex hull enclosing all active nodes.

        Returns
        -------
        float
        """

        return ConvexHull(self.nodes).volume

    @property
    def pbc_nodes(self) -> np.ndarray:
        """Ghost node coordinates for edges crossing periodic boundaries.

        For each boundary-crossing edge, one endpoint is translated into the
        neighbouring periodic image so the edge can be drawn without wrapping.
        Intended for visualisation (e.g. VTK output).

        Returns
        -------
        np.ndarray
            Stacked array of shifted node pairs, shape (2 * n_pbc_edges, dim).
        """

        r0 = self._nodes[self.edges[:, 0]].copy()
        r1 = self._nodes[self.edges[:, 1]].copy()
        box = np.array(self.boxsize)

        r0 = []
        r1 = []

        for e0, e1 in self.edges[self.pbc_edges]:

            l0 = self._nodes[e1] - self._nodes[e0] > box / 2.
            l1 = self._nodes[e0] - self._nodes[e1] > box / 2.

            if np.any(l0, axis=-1) and np.any(l1, axis=-1):
                _r0 = self._nodes[e0].copy()
                _r0[l0] += box[l0]
                r0.append(_r0)
                _r1 = self._nodes[e1].copy()
                _r1[l1] += box[l1]
                r1.append(_r1)
            else:
                if np.any(l0, axis=-1):
                    _r0 = self._nodes[e0].copy()
                    _r0[l0] += box[l0]
                    r0.append(_r0)
                    r1.append(self._nodes[e1])

                elif np.any(l1, axis=-1):
                    _r1 = self._nodes[e1].copy()
                    _r1[l1] += box[l1]
                    r1.append(_r1)
                    r0.append(self._nodes[e0])

        return np.vstack([r0, r1])

    @property
    def coordination(self) -> float:
        """Mean coordination number (average number of edges per node).

        Returns
        -------
        float
        """
        _, c = np.unique(self.edges, return_counts=True)
        return np.mean(c)

    def _get_angles(self):
        """Angles between pairs of edges.

        Returns only the sequence of nodes per angle, but not the actual angle.

        Returns
        -------
        np.ndarray

        """
        angles = []
        for n in range(len(self._nodes)):

            left_neighbors = self._edges[n == self._edges[:, 0]][:, 1]
            right_neighbors = self._edges[n == self._edges[:, 1]][:, 0]

            neighbors = np.hstack([left_neighbors, right_neighbors])

            if len(neighbors) > 1:
                for a, b in combinations(neighbors, 2):
                    angles.append([a, n, b])

        return np.array(angles)

    def _preprocessing(self, nodes, edges):
        """Sanity checks of input nodes and edges.

        Remove for isolated nodes and edges. Shift edge indices if those start with 1.


        Parameters
        ----------
        nodes : np.ndarray
            Node coordinate
        edges : np.ndarray
            Edge indices

        Returns
        -------
        np.ndarray
            Sanitized nodes
        np.ndarray
            Sanitized edges
        """

        assert np.amax(edges) < nodes.shape[0]
        assert np.all(edges >= 0)

        return _remove_isolated_nodes_edges(nodes, edges)

    @classmethod
    def generate_cubic_lattice(cls, a: float = 1.,
                               pbc: list | None = None,
                               bbox: list = [1., 1., 1.],
                               lattice_type: str = 'sc') -> "Network":
        """Generate a 3D cubic lattice.

        Parameters
        ----------
        a : float, optional
            Lattice constant (nearest-neighbour distance). The default is 1.
        pbc : iterable of bool, optional
            Periodic boundary condition flags per direction. The default is
            None (no periodic BCs). Example: ``[True, False, False]`` enables
            PBC in x only.
        bbox : array-like, optional
            Bounding box dimensions ``[Lx, Ly, Lz]``. The default is
            ``[1., 1., 1.]``.
        lattice_type : str, optional
            Name of the lattice type: ``'sc'``, ``'bcc'``, or ``'fcc'``.
            The default is ``'sc'``.

        Returns
        -------
        Network
            Instance with nodes and edges of the specified lattice.
        """

        lattice_coords, connections, pbc, bbox, valid = generate_cubic_lattice(a=a,
                                                                               pbc=pbc,
                                                                               bbox=bbox,
                                                                               lattice_type=lattice_type)

        return cls(lattice_coords, connections, valid=valid, periodic=pbc, boxsize=bbox)

    @classmethod
    def generate_square_lattice(cls, a: float = 1., bbox: list = [1., 1.],
                                lattice_type: str = 'sc') -> "Network":
        """Generate a 2D square (or triangular) lattice.

        Parameters
        ----------
        a : float, optional
            Lattice constant (nearest-neighbour distance). The default is 1.
        bbox : array-like, optional
            Bounding box dimensions ``[Lx, Ly]``. The default is ``[1., 1.]``.
        lattice_type : str, optional
            Name of the lattice type: ``'sc'`` (simple square) or ``'fcc'``
            (face-centred, equivalent to a triangular lattice).
            The default is ``'sc'``.

        Returns
        -------
        Network
            Instance with nodes and edges of the specified lattice.
        """

        lattice_coords, connections = generate_square_lattice(a=a,
                                                              bbox=bbox,
                                                              lattice_type=lattice_type)

        return cls(lattice_coords, connections)

    @classmethod
    def generate_bowtie_lattice(cls, a: float = 1., w: float = 0.1,
                                bbox: list = [1., 1.]) -> "Network":
        """Generate a 2D bowtie lattice.

        Parameters
        ----------
        a : float, optional
            Lattice constant (unit cell size). The default is 1.
        w : float, optional
            Offset parameter controlling node positions within a unit cell.
            Larger values move nodes further from the cell edges.
            The default is 0.1.
        bbox : array-like, optional
            Bounding box dimensions ``[Lx, Ly]``. The default is ``[1., 1.]``.

        Returns
        -------
        Network
            Instance with nodes and edges of the bowtie lattice.
        """

        lattice_coords, connections = generate_bowtie_lattice(a=a,
                                                              w=w,
                                                              bbox=bbox)

        return cls(lattice_coords, connections)

    def plot(self, ax, node_ids: bool = False, cax=None, aspect: float = 1.,
             lim: tuple | None = None, lw: float = 2.,
             scale: float = 1.) -> "matplotlib.axes.Axes":
        """Generate a 2D plot of the undeformed network.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to draw into.
        node_ids : bool, optional
            If True, print node indices next to each node. The default is False.
        cax : matplotlib.axes.Axes or None, optional
            Axes for the colourbar (unused here; retained for API consistency
            with :meth:`~beam_networks.problem.BeamNetwork.plot`).
            The default is None.
        aspect : float, optional
            Aspect ratio of the axes. The default is 1.
        lim : tuple or None, optional
            Colourbar limits (unused here). The default is None.
        lw : float, optional
            Line width for the beam edges. The default is 2.
        scale : float, optional
            Scale factor applied to node positions (for visualisation only).
            The default is 1.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the network drawn into it.
        """

        # undeformed
        ax = _plot_network(ax, self.nodes, self.edges, self.edge_vectors, color='0.7',
                           node_ids=node_ids, lw=lw)

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

        ax.set_aspect(aspect)

        return ax
