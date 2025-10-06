import numpy as np
import scipy.sparse as sp
from itertools import combinations
from scipy.spatial import cKDTree, ConvexHull

from beam_networks.utils import _remove_isolated_nodes_edges, _mic


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

        # In nonperiodic directions, keep only edges that do not cross boundaries
        keep = np.all(dr[:, ~self._periodic] <= self._boxsize[~self._periodic] / 2., axis=-1).reshape(-1)
        edges_indices = edges_indices[keep]

        if valid:
            self._nodes = nodes_positions
            self._edges = edges_indices
        else:
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
    def nodes(self):
        """Exposes only activated nodes.

        Returns
        -------
        np.ndarray

        """
        return self._nodes[self._active_nodes]

    @property
    def num_nodes(self):
        """Number of active nodes

        Returns
        -------
        int

        """
        return self.nodes.shape[0]

    @property
    def edges(self):
        """Exposes only active edges.

        Returns
        -------
        np.ndarray

        """
        return self._edges[self._active_edges]

    @property
    def num_edges(self):
        """Number of active edges

        Returns
        -------
        int

        """
        return self.edges.shape[0]

    @property
    def Lx(self):
        """Boxlength (x)

        Returns
        -------
        float

        """
        if self._boxsize[0] is None:
            return self.xhi - self.xlo
        else:
            return self._boxsize[0]

    @property
    def Ly(self):
        """Boxlength (y)

        Returns
        -------
        float

        """
        if self._boxsize[1] is None:
            return self.yhi - self.ylo
        else:
            return self._boxsize[1]

    @property
    def Lz(self):
        """Boxlength (z)

        Returns
        -------
        float

        """
        if self._boxsize[2] is None:
            return self.zhi - self.zlo
        else:
            return self._boxsize[2]

    @property
    def is_connected(self):
        """Check if structure is fully connected

        Returns
        -------
        bool

        """
        graph = sp.csr_array((np.ones(self.num_edges, dtype=int), (self.edges[:, 0], self.edges[:, 1])),
                             shape=(self.num_nodes, self.num_nodes))

        n_comp, _ = sp.csgraph.connected_components(graph)

        return n_comp == 1

    @property
    def bounds(self):
        """Lower and upper bounds of the nodal coordinates

        Returns
        -------
        list
           xlo, xhi, ylo, yhi[, zlo, zhi]
        """
        bounds = [self.xlo, self.xhi, self.ylo, self.yhi]
        if self.dim > 2:
            bounds.extend([self.zlo, self.zhi])

        return bounds

    @property
    def boxsize(self):
        if self.dim == 2:
            return self.Lx, self.Ly
        elif self.dim == 3:
            return self.Lx, self.Ly, self.Lz

    @property
    def xlo(self):
        return np.amin(self._nodes[self._active_nodes, 0])

    @property
    def xhi(self):
        return np.amax(self._nodes[self._active_nodes, 0])

    @property
    def ylo(self):
        return np.amin(self._nodes[self._active_nodes, 1])

    @property
    def yhi(self):
        return np.amax(self._nodes[self._active_nodes, 1])

    @property
    def zlo(self):
        return np.amin(self._nodes[self._active_nodes, 2])

    @property
    def zhi(self):
        return np.amax(self._nodes[self._active_nodes, 2])

    @property
    def bondlengths(self):
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
    def edge_vectors(self):
        return self._edge_vectors[self._active_edges]

    @property
    def pbc_edges(self):
        """Mask for edges that cross periodic boundary conditions.

        Returns
        -------
        np.ndarray
        """

        dr = self._nodes[self.edges[:, 1]] - self._nodes[self.edges[:, 0]]
        pbc_edges = np.any(np.abs(dr) > np.array(self.boxsize) / 2., axis=-1)

        return pbc_edges

    @property
    def volume(self):
        """Compute volume of network as convex hull.

        Returns
        -------
        float

        """

        return ConvexHull(self.nodes).volume

    @property
    def pbc_nodes(self):
        """Nodal positions for edges that cross periodic boundary conditions.
        Nodes are shifted to the neighboring periodic images for visualization purposes.

        Returns
        -------
        np.ndarray
            Nodal coordinates
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
    def coordination(self):
        """Mean coordination number.

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
    def generate_cubic_lattice(cls, a=1.,
                               pbc=None,
                               bbox=[1., 1., 1.], lattice_type='sc'):
        """

        Generates a cubic lattice.

        Parameters
        ----------
        a : float, optional
            Lattice constant (distance between nearest neighbors)
        pbc: iterable, optional
            Periodic boundary conditions (the default is None, which means no periodic BCs).
        bbox : array-like
            Size of the bounding box, filled with repeated unit cells
            (the default is [1., 1., 1.])
        lattice_type : str, optional
            Name of the lattice type ['sc', 'bcc', 'fcc'] (the default is 'sc')

        Returns
        -------
        beam_networks.network.Network
            Class instance with nodes and edges given by the prescribed lattice
        """

        if pbc is None:
            # set all directions to False
            pbc = np.zeros(3, dtype=bool)
        else:
            pbc = np.array(pbc).astype(bool)

        a0 = a

        if lattice_type == 'sc':
            basis = np.array([
                [0, 0, 0],            # Corner of the cube
            ]) * a
        elif lattice_type == 'fcc':
            basis = np.array([
                [0, 0, 0],            # Corner of the cube
                [0.5, 0.5, 0],        # Face centers
                [0.5, 0, 0.5],
                [0, 0.5, 0.5]
            ]) * a
            a0 = a * np.sqrt(2) / 2
        elif lattice_type == 'bcc':
            basis = np.array([
                [0, 0, 0],       # Corner of the cube
                [0.5, 0.5, 0.5]  # Center of the cube
            ]) * a
            a0 = a * np.sqrt(3) / 2
        else:
            raise RuntimeError("Lattice must be one of ['sc', 'fcc', 'bcc']")

        nx, ny, nz = (np.array(bbox) / np.maximum(np.ones(3) * a,
                                                  np.amax(basis, axis=0))).astype(int) + 1

        # Generate the grid of unit cells
        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
        )

        # Stack the grid into vectors of unit cell origins and scale by a
        grid_coords = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3) * a

        # Add the basis atoms to the grid
        lattice_coords = grid_coords[:, None, :] + basis
        lattice_coords = lattice_coords.reshape(-1, 3)  # Reshape into list of 3D coordinates

        # Select only nodes within the box
        mask = np.logical_and(np.all(lattice_coords <= bbox, axis=-1),
                              np.all(lattice_coords >= 0., axis=-1))
        lattice_coords = lattice_coords[mask]

        # Now we find connections between neighboring atoms using a KD-Tree
        kdtree = cKDTree(lattice_coords)
        connections = kdtree.query_pairs(r=a0 + 1e-5, output_type='ndarray')  # Neighbors within a0

        for d in np.arange(3)[pbc]:

            pbcx_mask_left = np.isclose(lattice_coords[:, d], 0.)
            pbcx_mask_right = np.isclose(lattice_coords[:, d], bbox[d])

            pbcx_nodes_left = np.arange(lattice_coords.shape[0])[pbcx_mask_left]
            pbcx_nodes_right = np.arange(lattice_coords.shape[0])[pbcx_mask_right]

            new_connections = np.copy(connections)

            for pl, pr in zip(pbcx_nodes_left, pbcx_nodes_right):

                new_connections[connections[:, 0] == pl, 0] = pr
                new_connections[connections[:, 1] == pl, 1] = pr

            connections = new_connections

        connections = np.unique(connections, axis=0)

        # Need to delete obsolete nodes for pbc, thus set valid to false
        if np.any(pbc):
            valid = False
        else:
            valid = True

        return cls(lattice_coords, connections, valid=valid, periodic=pbc, boxsize=bbox)

    @classmethod
    def generate_square_lattice(cls, a=1., bbox=[1., 1.], lattice_type='sc'):
        """

        Generates a cubic lattice.

        Parameters
        ----------
        a : float, optional
            Lattice constant (distance between nearest neighbors)
        bbox : array-like
            Size of the bounding box, filled with repeated unit cells
            (the default is [1., 1., 1.])
        lattice_type : str, optional
            Name of the lattice type ['sc', 'bcc', 'fcc'] (the default is 'sc')

        Returns
        -------
        beam_networks.network.Network
            Class instance with nodes and edges given by the prescribed lattice
        """

        a0 = a

        if lattice_type == 'sc':
            basis = np.array([
                [0, 0],            # Corner of the cube
            ]) * a
        elif lattice_type == 'fcc':
            basis = np.array([
                [0, 0],            # Corner of the cube
                [0.5, 0],             # Face centers
                [0, 0.5]
            ]) * a
            a0 = a * np.sqrt(2) / 2
        elif lattice_type == 'bcc':
            basis = np.array([
                [0, 0],       # Corner of the cube
                [0.5, 0.5]  # Center of the cube
            ]) * a
            a0 = a * np.sqrt(3) / 2
        else:
            raise RuntimeError("Lattice must be one of ['sc', 'fcc', 'bcc']")

        nx, ny = (np.array(bbox) / np.maximum(np.ones(2) * a,
                                              np.amax(basis, axis=0))).astype(int) + 1

        # Generate the grid of unit cells
        grid_x, grid_y = np.meshgrid(
            np.arange(nx), np.arange(ny), indexing='ij')

        # Stack the grid into vectors of unit cell origins and scale by a
        grid_coords = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2) * a

        # Add the basis atoms to the grid
        lattice_coords = grid_coords[:, None, :] + basis
        lattice_coords = lattice_coords.reshape(-1, 2)  # Reshape into list of 3D coordinates

        # Select only nodes within the box
        mask = np.logical_and(np.all(lattice_coords <= bbox, axis=-1),
                              np.all(lattice_coords >= 0., axis=-1))
        lattice_coords = lattice_coords[mask]

        # Now we find connections between neighboring atoms using a KD-Tree
        kdtree = cKDTree(lattice_coords)
        connections = kdtree.query_pairs(r=a0 + 1e-5, output_type='ndarray')  # Neighbors within a0

        return cls(lattice_coords, connections)

    @classmethod
    def generate_bowtie_lattice(cls, a=1., w=0.1, bbox=[1., 1.]):
        """

        Generates a bowtie lattice.

        Parameters
        ----------
        a : float, optional
            Lattice constant (distance between nearest neighbors)
        bbox : array-like
            Size of the bounding box, filled with repeated unit cells
            (the default is [1., 1., 1.])

        Returns
        -------
        beam_networks.network.Network
            Class instance with nodes and edges given by the prescribed lattice
        """

        a0 = a

        basis = np.array([
            [w, 0.0],
            [1. - w, 0.0],
            [0.5 - w, 0.5],
            [0.5 + w, 0.5]
        ]) * a

        a0 = a * np.sqrt(2) / 2

        nx, ny = (np.array(bbox) / np.maximum(np.ones(2) * a,
                                              np.amax(basis, axis=0))).astype(int) + 1

        # Generate the grid of unit cells
        grid_x, grid_y = np.meshgrid(
            np.arange(nx), np.arange(ny), indexing='ij')

        # Stack the grid into vectors of unit cell origins and scale by a
        grid_coords = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2) * a

        # Add the basis atoms to the grid
        lattice_coords = grid_coords[:, None, :] + basis
        lattice_coords = lattice_coords.reshape(-1, 2)  # Reshape into list of 3D coordinates

        # Select only nodes within the box
        mask = np.logical_and(np.all(lattice_coords <= bbox, axis=-1),
                              np.all(lattice_coords >= 0., axis=-1))
        lattice_coords = lattice_coords[mask]

        kdtree = cKDTree(lattice_coords)
        _connections = kdtree.query_pairs(r=1.0, output_type='ndarray')  # Neighbors within a0
        d = np.linalg.norm(lattice_coords[_connections[:, 0]] - lattice_coords[_connections[:, 1]], axis=-1)

        # Now we find connections between neighboring atoms using a KD-Tree
        d0 = np.linalg.norm(basis[1] - basis[0])
        d1 = np.linalg.norm(basis[2] - basis[0])

        connections_1 = _get_connections(lattice_coords * np.array([1., 1.]), d0)
        connections_2 = _get_connections(lattice_coords * np.array([1., 1.]), d1)

        connections = np.vstack([
            connections_1,
            connections_2,
        ])

        return cls(lattice_coords, connections)


def _get_connections(coords, d):

    hi = d + 1e-8
    lo = d - 1e-8

    kdtree = cKDTree(coords)
    connections_hi = kdtree.query_pairs(r=hi, output_type='ndarray')  # Neighbors within a0

    connections = []
    for e0, e1 in connections_hi:
        d = np.linalg.norm(coords[e1] - coords[e0])
        if d > lo:
            connections.append([e0, e1])

    return np.array(connections)
