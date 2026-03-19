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
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

from beam_networks.network import Network
from beam_networks.solvers.linear import solve
from beam_networks.fem.assembly import assemble_global_system
from beam_networks.postprocess.stress import get_element_mises_stress, get_element_principal_stress
from beam_networks.fem.bc import _get_bc_dof, _assemble_BCs
from beam_networks.io.validation import check_input_dict
from beam_networks.geometry.selection import _remove_isolated_nodes_edges, _mic
from beam_networks.postprocess.viz import _plot_network, _plot_network_3d
from beam_networks.io.formats import _to_vtk, _to_vtk_periodic, _to_stl, _from_tar, _to_tar
from beam_networks.solvers.nonlinear import solve_nonlinear as _solve_nonlinear
from beam_networks.fem.corotational import (
    element_mises_stress_2d as _corot_mises_2d,
    element_mises_stress_3d as _corot_mises_3d,
)

if TYPE_CHECKING:
    import matplotlib


def _default_ref_vectors(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Auto-generate one reference vector per 3D element.

    For each element, picks the global axis ([0,1,0], [0,0,1], or [1,0,0])
    that is least parallel to the chord, guaranteeing a well-conditioned
    local frame for any chord orientation.
    """
    e0i, e1i = edges[:, 0], edges[:, 1]
    chord = nodes[e1i] - nodes[e0i]                          # (M, 3)
    chord_hat = chord / np.linalg.norm(chord, axis=1, keepdims=True)

    candidates = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    dots = np.abs(chord_hat @ candidates.T)                  # (M, 3)
    best = np.argmin(dots, axis=1)                           # (M,)
    return candidates[best]


class ElasticNetwork(Network):
    """Elastic network of 1D structural elements. Derives from Network.

    Adding cross-section and elastic properties to the edges of a network
    turns it into an elastic network. Supports Timoshenko and Euler-Bernoulli
    beam elements as well as pin-jointed truss (bar) elements. Solves the
    linear elastic problem given appropriate boundary conditions.
    """

    def __init__(self,
                 nodes: np.ndarray,
                 edges: np.ndarray,
                 beam_prop: dict = {'name': 'circle', 'radius': 1.,
                                    'E': 1., 'nu': 0.3},
                 periodic: list | None = None,
                 boxsize: np.ndarray | None = None,
                 valid: bool = True,
                 options: dict = {'vectorize': True, 'matrix': 'bsr', 'verbose': True},
                 outdir: str = '.',
                 assemble_on_init: bool = True) -> None:
        """Constructor.

        Initialize the network and assemble its global stiffness matrix.

        Parameters
        ----------
        nodes : np.ndarray
            Nodal coordinates, shape (N, 2) for 2D or (N, 3) for 3D.
        edges : np.ndarray
            Edge connectivity as integer node index pairs, shape (M, 2).
        beam_prop : dict, optional
            Beam cross-section and elastic properties. Must contain 'name'
            ('circle' or 'rectangle'), 'E' (Young's modulus), and 'nu'
            (Poisson's ratio). 'circle' additionally requires 'radius';
            'rectangle' requires 'b' (width) and 'h' (height).
            Default: ``{'name': 'circle', 'radius': 1., 'E': 1., 'nu': 0.3}``.
        periodic : list of bool, optional
            Periodic boundary condition flags per spatial direction, e.g.
            ``[True, False, False]`` enables PBC in x only.
            The default is None (no periodic BCs).
        boxsize : np.ndarray, optional
            Explicit box dimensions. The default is None, in which case
            dimensions are inferred from the nodal coordinates.
        valid : bool, optional
            If True, skip sanity checks on the input topology (isolated node
            removal, etc.). Set to False when loading untrusted external data.
            The default is True.
        options : dict, optional
            Assembly options:

            ``'vectorize'`` (bool)
                Pre-compute all element stiffness matrices at once before
                assembly. Faster for large networks but uses more memory.
                Default True.
            ``'matrix'`` (``'bsr'`` | ``'lil'`` | ``'dense'``)
                Sparse format for the global stiffness matrix. Default ``'bsr'``.
            ``'verbose'`` (bool)
                Show a progress bar during assembly. Default True.
            ``'euler_bernoulli'`` (bool)
                Use Euler-Bernoulli beam theory (ignore shear deformation).
                Default False (Timoshenko).
            ``'n_elem_per_length'`` (float or None)
                Number of FEM sub-elements per unit length. When set, each
                beam is discretised into sub-elements and interior DOFs are
                eliminated by static condensation. The element type is chosen
                automatically: Lagrange (Timoshenko) or Hermite
                (Euler-Bernoulli), both with reduced integration. Default None
                (exact analytical stiffness).
        outdir : str, optional
            Directory for output files. Created if it does not exist.
            The default is the current working directory.
        assemble_on_init : bool, optional
            If True, assemble the global stiffness matrix immediately.
            Set to False to defer assembly (e.g. when restoring from a file).
            The default is True.
        """

        super().__init__(nodes, edges,
                         periodic=periodic,
                         boxsize=boxsize,
                         valid=valid)

        self._options = check_input_dict(options,
                                         ['verbose', 'vectorize', 'matrix'],
                                         [True, True, 'bsr'],
                                         [None, None, ['bsr', 'lil', 'dense']])
        self._options['n_elem_per_length'] = options.get('n_elem_per_length', None)
        self._options['min_element_length'] = options.get('min_element_length', None)
        self._options['euler_bernoulli'] = bool(options.get('euler_bernoulli', False))
        # Advanced FEM options (not part of the public API).
        # Shape function type is selected automatically (Lagrange for Timoshenko,
        # Hermite for Euler-Bernoulli); reduced integration is always used.
        self._options['fem_poly_order'] = options.get('fem_poly_order', 3)
        self._options['fem_n_gauss'] = options.get('fem_n_gauss', None)

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self._beam_prop = beam_prop
        self._outdir = outdir
        self._verbose = options['verbose']

        if assemble_on_init:
            self._assemble_global_system()
        else:
            self._K = None

        self._bc = {}
        self._bc_changed = True
        self.has_solution = False

    def save(self, filename: str) -> None:
        """Save the current state of the solver as a gzipped tar archive.

        Parameters
        ----------
        filename : str
            Path to the output archive file.
        """
        _to_tar(filename, self)

    @classmethod
    def load(cls, filename: str, recompute: bool = True) -> "ElasticNetwork":
        """Create a class instance from a tar archive.

        Parameters
        ----------
        filename : str
            Path to the archive previously created with :meth:`save`.
        recompute : bool, optional
            If True, re-solve the linear system after loading. If False,
            restore the archived solution (if present) without recomputing.
            The default is True.

        Returns
        -------
        ElasticNetwork
            A new instance restored from the archive.
        """

        nodes, edges, active_edges, beam_prop, K, bc, sol, sVM, misc = _from_tar(filename)

        new = cls(nodes, edges,
                  beam_prop=beam_prop,
                  options=misc,
                  periodic=misc['periodic'],
                  boxsize=misc['boxsize'],
                  outdir=misc.get('outdir', '.'),
                  valid=True,
                  assemble_on_init=False)

        new._active_edges = active_edges
        new._K = K
        new._bc_changed = True
        new._bc = bc

        if recompute:
            new.solve()
        else:
            if misc['has_solution']:
                new.sol = sol
                new._sVM = sVM
                new.has_solution = True
            else:
                new.has_solution = False

        return new

    @classmethod
    def from_network(cls,
                     network: Network,
                     beam_prop: dict = {'name': 'circle', 'radius': 1.,
                                        'E': 1., 'nu': 0.3},
                     options: dict = {'vectorize': True, 'matrix': 'bsr', 'verbose': True},
                     outdir: str = '.',
                     assemble_on_init: bool = True) -> "ElasticNetwork":
        """Create an elastic network from an existing :class:`~beam_networks.network.Network` instance.

        Parameters
        ----------
        network : beam_networks.network.Network
            Underlying network providing nodes and edges.
        beam_prop : dict, optional
            Beam cross-section and elastic properties (see :meth:`__init__`).
        options : dict, optional
            Assembly options (see :meth:`__init__`).
        outdir : str, optional
            Directory for output files. The default is the current working directory.
        assemble_on_init : bool, optional
            If True, assemble the global stiffness matrix immediately.
            The default is True.

        Returns
        -------
        ElasticNetwork
            A new instance sharing the topology of *network*.
        """

        return cls(network._nodes,
                   network._edges,
                   beam_prop=beam_prop,
                   periodic=network._periodic,
                   boxsize=network.boxsize,
                   valid=True,
                   options=options,
                   outdir=outdir,
                   assemble_on_init=assemble_on_init)

    @classmethod
    def from_cubic_lattice(cls,
                           a: float = 1.,
                           pbc: list | None = None,
                           bbox: tuple = (1., 1., 1.),
                           lattice_type: str = 'sc',
                           beam_prop: dict = {'name': 'circle', 'radius': 1.,
                                              'E': 1., 'nu': 0.3},
                           options: dict = {'vectorize': True, 'matrix': 'bsr', 'verbose': True},
                           outdir: str = '.',
                           assemble_on_init: bool = True) -> "ElasticNetwork":
        """Create an elastic network from a 3D cubic lattice.

        Thin wrapper around :meth:`~beam_networks.network.Network.generate_cubic_lattice`
        that directly returns a :class:`ElasticNetwork` instance.

        Parameters
        ----------
        a : float, optional
            Lattice constant (nearest-neighbour distance). The default is 1.
        pbc : list of bool, optional
            Periodic boundary condition flags per direction. The default is None
            (no periodic BCs).
        bbox : tuple of float, optional
            Bounding box dimensions ``(Lx, Ly, Lz)``. The default is ``(1., 1., 1.)``.
        lattice_type : str, optional
            One of ``'sc'`` (simple cubic), ``'bcc'``, or ``'fcc'``.
            The default is ``'sc'``.
        beam_prop : dict, optional
            Beam cross-section and elastic properties (see :meth:`__init__`).
        options : dict, optional
            Assembly options (see :meth:`__init__`).
        outdir : str, optional
            Directory for output files. The default is the current working directory.
        assemble_on_init : bool, optional
            If True, assemble the global stiffness matrix immediately.
            The default is True.

        Returns
        -------
        ElasticNetwork
            A new instance built on the specified cubic lattice.
        """

        lattice = Network.generate_cubic_lattice(a=a,
                                                 pbc=pbc,
                                                 bbox=bbox,
                                                 lattice_type=lattice_type)
        return cls.from_network(lattice,
                                beam_prop=beam_prop,
                                options=options,
                                outdir=outdir,
                                assemble_on_init=assemble_on_init)

    @classmethod
    def from_square_lattice(cls,
                            a: float = 1.,
                            bbox: tuple = (1., 1.),
                            lattice_type: str = 'sc',
                            beam_prop: dict = {'name': 'circle', 'radius': 1.,
                                               'E': 1., 'nu': 0.3},
                            options: dict = {'vectorize': True, 'matrix': 'bsr', 'verbose': True},
                            outdir: str = '.',
                            assemble_on_init: bool = True) -> "ElasticNetwork":
        """Create an elastic network from a 2D square lattice.

        Parameters
        ----------
        a : float, optional
            Lattice constant (nearest-neighbour distance). The default is 1.
        bbox : tuple of float, optional
            Bounding box dimensions ``(Lx, Ly)``. The default is ``(1., 1.)``.
        lattice_type : str, optional
            One of ``'sc'`` (simple square) or ``'fcc'`` (face-centred, i.e.
            triangular). The default is ``'sc'``.
        beam_prop : dict, optional
            Beam cross-section and elastic properties (see :meth:`__init__`).
        options : dict, optional
            Assembly options (see :meth:`__init__`).
        outdir : str, optional
            Directory for output files. The default is the current working directory.
        assemble_on_init : bool, optional
            If True, assemble the global stiffness matrix immediately.
            The default is True.

        Returns
        -------
        ElasticNetwork
            A new instance built on the specified square lattice.
        """

        lattice = Network.generate_square_lattice(a=a,
                                                  bbox=bbox,
                                                  lattice_type=lattice_type)
        return cls.from_network(lattice,
                                beam_prop=beam_prop,
                                options=options,
                                outdir=outdir,
                                assemble_on_init=assemble_on_init)

    @classmethod
    def from_bowtie_lattice(cls,
                            a: float = 1.,
                            w: float = 0.1,
                            bbox: tuple = (1., 1.),
                            beam_prop: dict = {'name': 'circle', 'radius': 1.,
                                               'E': 1., 'nu': 0.3},
                            options: dict = {'vectorize': True, 'matrix': 'bsr', 'verbose': True},
                            outdir: str = '.',
                            assemble_on_init: bool = True) -> "ElasticNetwork":
        """Create an elastic network from a 2D bowtie lattice.

        Parameters
        ----------
        a : float, optional
            Lattice constant (unit cell size). The default is 1.
        w : float, optional
            Offset parameter controlling the bowtie node positions within a
            unit cell. The default is 0.1.
        bbox : tuple of float, optional
            Bounding box dimensions ``(Lx, Ly)``. The default is ``(1., 1.)``.
        beam_prop : dict, optional
            Beam cross-section and elastic properties (see :meth:`__init__`).
        options : dict, optional
            Assembly options (see :meth:`__init__`).
        outdir : str, optional
            Directory for output files. The default is the current working directory.
        assemble_on_init : bool, optional
            If True, assemble the global stiffness matrix immediately.
            The default is True.

        Returns
        -------
        ElasticNetwork
            A new instance built on the bowtie lattice.
        """

        lattice = Network.generate_bowtie_lattice(a=a,
                                                  w=w,
                                                  bbox=bbox)
        return cls.from_network(lattice,
                                beam_prop=beam_prop,
                                options=options,
                                outdir=outdir,
                                assemble_on_init=assemble_on_init)

    @property
    def has_bc(self) -> bool:
        """Whether at least one boundary condition has been added."""
        return len(self._bc) > 0

    @property
    def dof_per_node(self) -> int:
        """Number of degrees of freedom per node.

        For beams: 3 for 2D (ux, uy, θz) and 6 for 3D (ux, uy, uz, θx, θy, θz).
        For trusses: 2 for 2D (ux, uy) and 3 for 3D (ux, uy, uz).
        """
        if self._beam_prop.get('truss', False):
            return self.dim
        return 3 * (self.dim - 1)

    @property
    def num_dof(self) -> int:
        """Number of unconstrained (free) degrees of freedom.

        Equal to ``num_nodes * dof_per_node`` minus the number of constrained
        DOFs imposed by Dirichlet and Neumann boundary conditions.
        """
        n_dof = self.num_nodes * self.dof_per_node

        if self.has_bc:
            n_dof -= len(self._dof_D)
            n_dof -= len(self._dof_N)

        return n_dof

    @property
    def displaced_nodes(self) -> np.ndarray:
        """Nodal coordinates in the deformed configuration.

        Returns
        -------
        np.ndarray
            Shape (num_nodes, dim). Returns undeformed coordinates if no
            solution is available.
        """
        return self.nodes + self.displacement

    @property
    def displaced_edge_vectors(self) -> np.ndarray:
        """Edge vectors in the deformed configuration.

        Applies the minimum image convention for periodic systems.

        Returns
        -------
        np.ndarray
            Shape (num_edges, dim).
        """
        r1 = self.displaced_nodes[self.edges[:, 1]]
        r0 = self.displaced_nodes[self.edges[:, 0]]
        dr = _mic(r1 - r0, self._boxsize, self._periodic)

        return dr

    @property
    def displacement(self) -> np.ndarray:
        """Nodal displacements extracted from the global solution vector.

        Returns
        -------
        numpy.ndarray
            Shape (num_nodes, dim). Zero array if no solution is available.
        """
        if self.has_solution:
            u = self.sol.reshape(-1, self.dof_per_node)
            return u[:, :self.dim]
        else:
            return np.zeros((self.num_nodes, self.dim))

    @property
    def rotation(self) -> np.ndarray:
        """Nodal rotations extracted from the global solution vector.

        Returns
        -------
        numpy.ndarray
            Shape (num_nodes, 1) for 2D beams (θz), (num_nodes, 3) for 3D
            beams (θx, θy, θz), or (num_nodes, 0) for trusses (no rotations).
            Zero array if no solution is available.
        """
        n_rot = self.dof_per_node - self.dim
        if self.has_solution:
            u = self.sol.reshape(-1, self.dof_per_node)
            return u[:, self.dim:]
        else:
            return np.zeros((self.num_nodes, n_rot))

    @property
    def stiffness(self) -> np.ndarray:
        """Global stiffness matrix.

        Returns
        -------
        numpy.ndarray or scipy.sparse.bsr_array
            The assembled stiffness matrix, or None if not yet assembled.
        """
        return self._K

    @property
    def has_stiffness(self) -> bool:
        """Whether the global stiffness matrix has been assembled."""
        return self._K is not None

    def _compute_edge_discretization(self) -> np.ndarray | None:
        """Return per-edge element count array, or None for exact mode.

        Uses ``n_elem_per_length`` from options to compute the number of
        FEM sub-elements per edge.  Returns None when ``n_elem_per_length``
        is None, which activates the exact Timoshenko assembly path.
        """
        density = self._options['n_elem_per_length']
        if density is None:
            return None
        min_len = self._options['min_element_length']
        L = self.bondlengths
        n = np.maximum(1, np.ceil(L * density).astype(int))
        if min_len is not None:
            # Reduce n where L/n < min_len, but never below 1
            n_max = np.maximum(1, np.floor(L / min_len).astype(int))
            n = np.minimum(n, n_max)
        return n

    def _assemble_global_system(self):
        """
        Assemble the stiffness matrix in the global coordinate system using Timoshenko beam theory.
        """

        if self._verbose:
            print(f"Assemble elastic network with {self.num_nodes} nodes and {self.num_edges} edges")

        n_elems = self._compute_edge_discretization()

        beam_prop = dict(self._beam_prop)
        beam_prop['euler_bernoulli'] = self._options['euler_bernoulli']

        self._K = assemble_global_system(self._nodes,
                                         self._edges,
                                         self._edge_vectors,
                                         beam_prop,
                                         vectorize=self._options['vectorize'],
                                         matrix=self._options['matrix'],
                                         n_elems=n_elems,
                                         fem_poly_order=self._options['fem_poly_order'],
                                         fem_n_gauss=self._options['fem_n_gauss'],
                                         )

    def add_BC(self, name: str, type: str, select: str, selection,
               vector, active: bool = True, num_per_point: int = 1) -> None:
        """Add a boundary condition.

        Parameters
        ----------
        name : str
            Unique identifier for this boundary condition. If a BC with this
            name already exists it will be overridden with a warning.
        type : str
            BC type: ``'D'`` for Dirichlet (prescribed displacement/rotation)
            or ``'N'`` for Neumann (prescribed force/moment).
        select : str
            Node selection method: ``'box'``, ``'node'``, or ``'point'``.
        selection : array-like
            For ``'box'``: relative limits ``[xlo, xhi, ylo, yhi[, zlo, zhi]]``
            where each value is a fraction of the domain extent (0–1). ``None``
            entries fall back to the domain minimum or maximum, e.g.
            ``[None, None, 0.5, None]`` selects the upper half in y.
            For ``'node'``: explicit list of node indices.
            For ``'point'``: coordinates of the target point (see *num_per_point*).
        vector : array-like
            DOF values for the selected nodes. Length 3 for 2D, 6 for 3D.
            ``None`` entries leave the corresponding DOF unaffected.

            * 2D Dirichlet: ``[ux, uy, θz]``
            * 2D Neumann:   ``[Fx, Fy, Mz]``
            * 3D Dirichlet: ``[ux, uy, uz, θx, θy, θz]``
            * 3D Neumann:   ``[Fx, Fy, Fz, Mx, My, Mz]``
        active : bool, optional
            Whether this BC participates in the next solve. The default is True.
        num_per_point : int, optional
            For ``'point'`` selection only: number of closest nodes to include.
            The default is 1.
        """

        assert select in ['box', 'node', 'point']
        assert type in ['D', 'N']

        if name not in self._bc.keys():
            self._bc[name] = {}
        else:
            warnings.warn(f"Boundary condition '{name} has been overridden'")

        self._bc[name].update([('type', type), ('active', active)])
        self._bc[name].update(
            _get_bc_dof(self.nodes, select, selection, vector,
                        num_per_point, dof_per_node=self.dof_per_node))
        self._bc_changed = True

    def delete_BC(self, name: str) -> None:
        """Delete a boundary condition.

        Parameters
        ----------
        name : str
            Name of the BC to remove. No-op if the name does not exist.
        """
        self._bc.pop(name, None)
        self._bc_changed = True

    def scale_BC(self, type: str, factor: float) -> None:
        """Scale all boundary condition values of a given type by a scalar factor.

        Parameters
        ----------
        type : str
            BC type to scale: ``'D'`` for Dirichlet or ``'N'`` for Neumann.
        factor : float
            Multiplicative scaling factor applied to all DOF values of the
            selected BC type.
        """

        if type == 'D':
            self._val_D = list(np.array(self._val_D) * factor)
        elif type == 'N':
            self._val_N = list(np.array(self._val_N) * factor)

    def modify_BC(self, name: str, vector) -> None:
        """Modify the values of an existing boundary condition.

        Changes the prescribed values without altering which DOFs are
        constrained. ``None`` entries in *vector* must appear in the same
        positions as in the original call to :meth:`add_BC`. Useful for
        incrementally stepping a load or displacement.

        Parameters
        ----------
        name : str
            Name of the BC to modify.
        vector : array-like
            New DOF values. Length 3 for 2D, 6 for 3D. ``None`` entries
            leave the corresponding DOF constraint unchanged (position must
            match the original ``None`` pattern).

            * 2D Dirichlet: ``[ux, uy, θz]``
            * 2D Neumann:   ``[Fx, Fy, Mz]``
            * 3D Dirichlet: ``[ux, uy, uz, θx, θy, θz]``
            * 3D Neumann:   ``[Fx, Fy, Fz, Mx, My, Mz]``
        """

        assert name in self._bc.keys()

        nodes = self._bc[name]['nodes']
        dof_mask = [i for i, k in enumerate(vector) if k is not None]
        assert len(dof_mask) == self._bc[name]['dim']
        vector = [float(v) for v in vector if v is not None]
        dof_val = len(nodes) * vector

        self._bc[name].update([('dof_val', dof_val)])
        self._bc_changed = True

    def assemble_BCs(self) -> None:
        """Assemble boundary conditions into DOF index lists.

        Collects all active boundary conditions into the internal Dirichlet
        (``_dof_D``, ``_val_D``) and Neumann (``_dof_N``, ``_val_N``) lists
        consumed by :meth:`solve`. Called automatically by :meth:`solve` when
        the BC state has changed since the last call.
        """

        self._dof_D, self._val_D, self._dof_N, self._val_N = _assemble_BCs(self._bc)
        _excluded_nodes = [v for k, bc in self._bc.items() if not k.startswith('sym') for v in bc['nodes']]
        self._possible_edges = np.unique([j for j, (l, r) in enumerate(self._edges)
                                          if l not in _excluded_nodes and r not in _excluded_nodes])

        self._bc_changed = False

    def solve(self, solver: str = 'cg', stress_mode: str = 'mean',
              verbosity: int = 0, tol: float = 1e-10,
              scale: bool = True) -> None:
        """Solve the linear elastic system.

        Assembles boundary conditions if they have changed, solves the
        reduced linear system, and stores the global displacement vector
        (``sol``), reaction forces (``Freact``), and von Mises stress
        (``_sVM``). Sets ``has_solution = True`` on success.

        Parameters
        ----------
        solver : str, optional
            Linear solver.  Available options:

            * ``'cg'``       — unpreconditioned conjugate gradient
            * ``'direct'``   — sparse LU via ``spsolve``
            * ``'cholesky'`` — sparse Cholesky via CHOLMOD
              (requires ``scikit-sparse``)
            * ``'ilu'``      — CG + incomplete LU (ILU)
            * ``'ssor'``     — CG + SSOR (ω = 1)
            * ``'amg'``      — CG + smoothed-aggregation AMG
              (requires ``pyamg``); optimal for homogeneous lattices
            * ``'amg_rs'``   — CG + Ruge–Stüben AMG
              (requires ``pyamg``); better for heterogeneous / diluted networks

            The default is ``'cg'``.
        stress_mode : str, optional
            How to aggregate von Mises stress along each beam: ``'max'``
            takes the end-point maximum, ``'mean'`` averages both ends.
            The default is ``'mean'``.
        verbosity : int, optional
            Diagnostic output level (active for sparse solvers only):

            * 0 — silent (default)
            * 25–49 — print displacement and reaction force statistics
            * 50–99 — additionally print stiffness matrix statistics
            * ≥100 — additionally print the condition number

        tol : float, optional
            Relative convergence tolerance for iterative solvers.
            Convergence is declared when ``‖r‖ / ‖b‖ < tol`` in the
            Jacobi-scaled system.  The default is 1e-10.
        scale : bool, optional
            Apply symmetric Jacobi scaling to the reduced system before
            solving.  Set to ``False`` to work with the raw (unscaled) system.
            The default is ``True``.

        Raises
        ------
        RuntimeError
            If no boundary conditions have been defined.
        """

        if not self.has_stiffness:
            self._assemble_global_system()

        if self._bc_changed:
            self.assemble_BCs()

        if not self.has_bc:
            raise RuntimeError(
                'No boundary conditions given. Nothing to solve here.')

        sol, F, info = solve(self._K,
                             self._dof_D,
                             self._val_D,
                             self._dof_N,
                             self._val_N,
                             solver=solver,
                             verbosity=verbosity,
                             tol=tol,
                             scale=scale)

        self.has_solution = info == 0

        if self.has_solution:
            self.sol = sol
            self.Freact = self._get_reaction_forces(F)
            self.compute_equivalent_stress(mode=stress_mode)

    def solve_nonlinear(self, n_steps: int = 100, max_iter: int = 10000,
                        tol: float = 1e-9, verbose: bool = True,
                        callback=None, matrix: str = 'dense',
                        ref_vectors=None) -> None:
        """Solve the geometrically nonlinear elastic system.

        Uses a load-stepped Newton–Raphson scheme with the Crisfield (1990)
        co-rotational formulation. Large rigid-body rotations are handled
        exactly; strains in the local element frame remain small. An Updated
        Lagrangian approach is used: the reference configuration is advanced
        to the deformed state after each converged load step.

        Supports both 2D networks (3 DOFs per node: ux, uy, θz) and 3D
        networks (6 DOFs per node: ux, uy, uz, θx, θy, θz).

        Only Neumann (force/moment) boundary conditions are load-stepped.
        Dirichlet (prescribed displacement) BCs are ramped linearly if
        non-zero values were provided via :meth:`add_BC`, otherwise enforced
        as fixed supports throughout.

        Parameters
        ----------
        n_steps : int, optional
            Number of load increments. The total Neumann load is divided
            equally across all steps. The default is 100.
        max_iter : int, optional
            Maximum Newton–Raphson iterations per load step. The default is
            10 000.
        tol : float, optional
            Convergence tolerance on the incremental displacement norm.
            The default is 1e-9.
        verbose : bool, optional
            Print per-step convergence information. The default is True.
        callback : callable or None, optional
            If provided, called at the end of each converged load step as
            ``callback(step, nodes_current, sol_total)`` where *nodes_current*
            is the reference nodal array after committing the step and
            *sol_total* is the accumulated displacement from the original
            nodes. Useful for plotting intermediate deformed shapes. The
            default is None.
        matrix : {'dense', 'bsr'}, optional
            Storage format for the element-level tangent stiffness assembly.
            ``'dense'`` uses plain NumPy arrays and ``numpy.linalg.solve``
            for the Newton–Raphson linear step; suitable for small networks.
            ``'bsr'`` assembles a ``scipy.sparse.bsr_array`` and uses
            ``scipy.sparse.linalg.spsolve`` for the linear step; recommended
            for large networks. The default is ``'dense'``.
        ref_vectors : np.ndarray, shape (M, 3), or None, optional
            *3D only.* One reference vector per element that defines the local
            e2 (in-plane) axis. Must not be parallel to any element chord.
            When ``None`` (default), a per-element default is chosen
            automatically: ``[0, 1, 0]`` for elements not aligned with the
            y-axis, ``[0, 0, 1]`` otherwise.

        Raises
        ------
        RuntimeError
            If no boundary conditions have been defined.
        """
        if self._bc_changed:
            self.assemble_BCs()

        if not self.has_bc:
            raise RuntimeError(
                "No boundary conditions given. Nothing to solve here.")

        rv = None
        if self.dim == 3:
            if ref_vectors is not None:
                rv = np.asarray(ref_vectors, dtype=float)
            else:
                rv = _default_ref_vectors(self._nodes, self._edges)

        self.sol = _solve_nonlinear(
            self._nodes,
            self._edges,
            self._beam_prop,
            dof_D=self._dof_D,
            dof_N=self._dof_N,
            val_N=self._val_N,
            val_D=self._val_D,
            n_steps=n_steps,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            callback=callback,
            matrix=matrix,
            ndim=self.dim,
            ref_vectors=rv,
        )
        self.has_solution = True

        if self.dim == 2:
            self._sVM = _corot_mises_2d(self._nodes, self._edges, self.sol, self._beam_prop)
        else:
            self._sVM = _corot_mises_3d(self._nodes, self._edges, self.sol, self._beam_prop, rv)

    def _get_reaction_forces(self, F):
        """Extract reaction forces from global force vector

        Parameters
        ----------
        F : np.ndarray
            Global force vector returned by solve method

        Returns
        -------
        dict
            Dictionary with force vectors per Dirichlet BC.
        """

        out = {}

        for name, bc in self._bc.items():
            if bc['type'] == 'D':
                dim = bc['dim']
                dof = self._bc[name]['dof']
                out[name] = F[dof].reshape(dim, -1).T

        return out

    def _get_group_disp(self, d):
        """Extract displacements/rotations from global displacement vector

        Parameters
        ----------
        d : np.ndarray
            Global solution vector returned by solve method

        Returns
        -------
        dict
            Dictionary with displacements/rotations per Neumann BC.
        """
        out = {}

        for name, bc in self._bc.items():
            if bc['type'] == 'N':
                dim = bc['dim']
                dof = self._bc[name]['dof']
                out[name] = d[dof].reshape(dim, -1).T

        return out

    def compute_equivalent_stress(self, mode: str = 'mean') -> None:
        """Compute the von Mises stress for all beams and store it in ``_sVM``.

        Called automatically by :meth:`solve`. Use this method directly to
        recompute with a different aggregation mode after solving.

        Parameters
        ----------
        mode : str, optional
            Stress aggregation along each beam: ``'max'`` takes the
            end-point maximum, ``'mean'`` averages both ends.
            The default is ``'mean'``.
        """
        self._sVM = get_element_mises_stress(self.nodes,
                                             self.edges,
                                             self.edge_vectors,
                                             self.sol,
                                             self._beam_prop,
                                             rot=None, mode=mode)

    def compute_principal_stresses(self, mode: str = 'max') -> np.ndarray:
        """Compute the principal stresses for all beam elements.

        Parameters
        ----------
        mode : str, optional
            Stress aggregation along each beam: ``'max'`` takes the
            end-point maximum, ``'mean'`` averages both ends.
            The default is ``'max'``.

        Returns
        -------
        np.ndarray
            Principal stresses sorted in descending order,
            shape (num_edges, 3).
        """
        pS = get_element_principal_stress(self.nodes,
                                          self.edges,
                                          self.edge_vectors,
                                          self.sol,
                                          self._beam_prop,
                                          rot=None, mode=mode)

        return pS

    def compute_ratio(self) -> np.ndarray:
        """Compute the bending-to-total stress ratio for all beam elements.

        The ratio is defined as the bending stress divided by the sum of
        bending and axial stresses, giving a value in [0, 1] where 1 means
        purely bending-dominated and 0 means purely axially dominated.

        Returns
        -------
        np.ndarray
            Bending ratio per beam, shape (num_edges,).
        """
        _, ratio = get_element_mises_stress(self.nodes,
                                            self.edges,
                                            self.edge_vectors,
                                            self.sol,
                                            self._beam_prop,
                                            rot=None, return_ratio=True)

        return ratio

    def scale_solution(self, scale_factor: float) -> None:
        """Scale the global displacement solution vector by a scalar factor.

        Useful for superimposing solutions or converting between unit systems.

        Parameters
        ----------
        scale_factor : float
            Multiplicative factor applied to all displacement and rotation DOFs.
        """
        self.sol *= scale_factor

    def _resolve_contour(self, contour):
        """Resolve a contour specification to a ``(array, label)`` pair.

        Parameters
        ----------
        contour : np.ndarray, str, or None
            If an ndarray it is returned with an empty label.
            Supported string keys:

            Per-edge quantities
              * ``'stress'`` / ``'svm'``  — von Mises equivalent stress
              * ``'ratio'``               — bending-to-total stress ratio (2D only)
              * ``'p1'``, ``'p2'``, ``'p3'`` — principal stresses (descending)

            Per-node quantities (averaged over the two endpoint nodes)
              * ``'u'``                        — displacement magnitude
              * ``'ux'``, ``'uy'``, ``'uz'``  — displacement components
              * ``'r'``                        — rotation magnitude
              * ``'rz'``                       — rotation θz (2D and 3D)
              * ``'rx'``, ``'ry'``             — rotations θx, θy (3D only)

        Returns
        -------
        data : np.ndarray or None
        label : str
            Human-readable name suitable for a colourbar title.
        """
        if contour is None:
            return None, ''

        if isinstance(contour, np.ndarray):
            return contour, ''

        if not isinstance(contour, str):
            raise TypeError(f"contour must be an ndarray, str, or None; got {type(contour)}")

        key = contour.lower().strip()

        # --- per-edge quantities -----------------------------------------
        if key in ('stress', 'svm'):
            if self._sVM is None:
                raise RuntimeError("No stress available; call solve() or solve_nonlinear() first.")
            return self._sVM, 'von Mises stress'

        if key == 'ratio':
            return self.compute_ratio(), 'bending ratio'

        if key in ('p1', 'p2', 'p3'):
            idx = {'p1': 0, 'p2': 1, 'p3': 2}[key]
            labels = {'p1': 'principal stress 1',
                      'p2': 'principal stress 2',
                      'p3': 'principal stress 3'}
            return self.compute_principal_stresses()[:, idx], labels[key]

        # --- per-node quantities averaged to edges -----------------------
        disp = self.displacement  # (N, dim)

        if key == 'u':
            node_data, label = np.linalg.norm(disp, axis=1), 'U (magnitude)'
        elif key == 'ux':
            node_data, label = disp[:, 0], 'UX'
        elif key == 'uy':
            node_data, label = disp[:, 1], 'UY'
        elif key == 'uz':
            if self.dim < 3:
                raise ValueError("'uz' is only available for 3D networks.")
            node_data, label = disp[:, 2], 'UZ'
        elif key in ('r', 'rx', 'ry', 'rz'):
            rot = self.rotation  # (N, n_rot): 2D→(N,1), 3D→(N,3)
            if key == 'r':
                node_data, label = np.linalg.norm(rot, axis=1), 'R (magnitude)'
            elif key == 'rz':
                # rz is the only rotation in 2D (index 0); last column in 3D
                node_data, label = rot[:, -1], 'RZ'
            else:
                if self.dim < 3:
                    raise ValueError(f"'{key}' is only available for 3D networks.")
                idx = {'rx': 0, 'ry': 1}[key]
                node_data, label = rot[:, idx], key.upper()
        else:
            valid = ("'stress'/'svm', 'ratio', 'p1', 'p2', 'p3', "
                     "'u', 'ux', 'uy', 'uz', 'r', 'rx', 'ry', 'rz'")
            raise ValueError(f"Unknown contour key '{contour}'. Valid options: {valid}")

        # Average the per-node scalar to per-edge
        return 0.5 * (node_data[self.edges[:, 0]] + node_data[self.edges[:, 1]]), label

    def plot(self,
             ax=None,
             contour: np.ndarray | str | None = None,
             lim: tuple | None = None,
             scale: float = 1.,
             show_undeformed: bool = False,
             lw: float = 3.,
             cmap: str = 'plasma',
             # 2-D only
             node_ids: bool = False,
             cax=None,
             aspect: float = 1.,
             # 3-D only
             plotter=None,
             scalar_bar_args: dict | None = None):
        """Plot the network, dispatching to 2-D (matplotlib) or 3-D (PyVista).

        The backend is chosen automatically from the network dimension
        (``self.dim``).  Common parameters work identically in both backends;
        backend-specific parameters are silently ignored when they do not apply.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            *(2-D only)* Axes to draw into.  A new figure with one axes is
            created when None.
        contour : numpy.ndarray, str, or None, optional
            Per-edge scalar used to colour the deformed network.  Accepts a
            raw array or any string key recognised by :meth:`_resolve_contour`
            (``'stress'``, ``'u'``, ``'ux'``, ``'uy'``, ``'uz'``,
            ``'r'``, ``'rx'``, ``'ry'``, ``'rz'``,
            ``'ratio'``, ``'p1'``, ``'p2'``, ``'p3'``).
            Default is None (uniform colour).
        lim : tuple or None, optional
            Colour-bar limits ``(vmin, vmax)``.  None uses the data range.
        scale : float, optional
            Displacement magnification factor.  Default is 1.
        show_undeformed : bool, optional
            Overlay the undeformed network in grey.  Default is False.
        lw : float, optional
            Line width (points for 2-D, screen pixels for 3-D).  Default is 3.
        cmap : str, optional
            Colormap name.  Default is ``'plasma'``.
        node_ids : bool, optional
            *(2-D only)* Annotate nodes with their index.  Default is False.
        cax : matplotlib.axes.Axes or None, optional
            *(2-D only)* Axes for the colour bar.  None steals space from *ax*.
        aspect : float, optional
            *(2-D only)* Axes aspect ratio.  Default is 1.
        plotter : pyvista.Plotter or None, optional
            *(3-D only)* Existing plotter to draw into.  A new one is created
            when None.
        scalar_bar_args : dict or None, optional
            *(3-D only)* Extra kwargs forwarded to the PyVista scalar bar.

        Returns
        -------
        matplotlib.axes.Axes  (2-D networks)
            Call ``plt.show()`` or ``fig.savefig()`` as usual.
        pyvista.Plotter  (3-D networks)
            Call ``.show()`` to open the interactive window.
        """
        contour, label = self._resolve_contour(contour)
        disp_nodes = self.nodes + scale * self.displacement

        if self.dim == 2:
            import matplotlib.pyplot as plt
            if ax is None:
                _, ax = plt.subplots()

            if show_undeformed:
                ax = _plot_network(ax, self.nodes, self.edges, self.edge_vectors,
                                   color='0.7', node_ids=node_ids, lw=lw)

            disp_dr = _mic(disp_nodes[self.edges[:, 1]] - disp_nodes[self.edges[:, 0]],
                           self._boxsize, self._periodic)
            ax = _plot_network(ax, disp_nodes, self.edges, disp_dr, contour,
                               cax=cax, lim=lim, lw=lw, cmap=cmap)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_aspect(aspect)
            return ax

        else:
            try:
                import pyvista as pv
            except ImportError as exc:
                raise ImportError(
                    "3-D plotting requires pyvista: pip install beam_networks[viz]"
                ) from exc

            if plotter is None:
                plotter = pv.Plotter()

            if show_undeformed:
                _plot_network_3d(plotter, self.nodes, self.edges, line_width=lw)

            sba = dict(scalar_bar_args) if scalar_bar_args else {}
            if label and 'title' not in sba:
                sba['title'] = label

            _plot_network_3d(plotter, disp_nodes, self.edges,
                             edge_data=contour, cmap=cmap, lim=lim,
                             line_width=lw, color='steelblue',
                             scalar_bar_args=sba)
            plotter.set_background('white')
            plotter.add_axes()
            return plotter

    def to_vtk(self, file: str = "foo.vtk") -> None:
        """Write the network structure and solution to a VTK file.

        When a solution is available, nodal displacements, rotations, and
        per-element von Mises stress are included. For periodic structures,
        ghost copies of boundary-crossing edges are written for visualisation.
        Output is placed in ``outdir`` (set at construction time).

        Parameters
        ----------
        file : str, optional
            Output filename relative to ``outdir``. The default is
            ``"foo.vtk"``.
        """

        if np.any(self._periodic):
            _to_vtk_periodic(self.nodes,
                             self.pbc_nodes,
                             self.edges,
                             self.pbc_edges,
                             self.displacement,
                             self.rotation,
                             self._beam_prop,
                             os.path.join(self._outdir, file))
        else:
            if not self.has_solution:
                _to_vtk(os.path.join(self._outdir, file),
                        coords=self.nodes,
                        adj=self.edges,
                        r=None,
                        u=None,
                        f=None,
                        stress=None)
            else:
                _to_vtk(os.path.join(self._outdir, file),
                        coords=self.nodes,
                        adj=self.edges,
                        r=None,
                        u=np.hstack((self.displacement, self.rotation)).flatten(),
                        f=None,
                        stress=self._sVM,
                        dof_per_node=self.dof_per_node)

    def to_stl(self, file: str, clean: bool = False, tol: float = 1e-6) -> None:
        """Write the elastic network geometry to an STL file.

        Each beam is tessellated as a cylindrical or rectangular solid
        (depending on the cross-section type in ``beam_prop``).

        Parameters
        ----------
        file : str
            Output filename relative to ``outdir``.
        clean : bool, optional
            If True, remove isolated nodes and dangling edges before export.
            The default is False.
        tol : float, optional
            Geometric tolerance used during mesh cleaning. The default is 1e-6.
        """

        if clean:
            # stress_tolerance = tol * self._beam_prop['E']
            # mask = self._sVM > stress_tolerance
            nodes, edges = _remove_isolated_nodes_edges(self.nodes, self.edges)
        else:
            nodes = self._nodes
            edges = self.edges

        _to_stl(os.path.join(self._outdir, file), nodes, edges, self._beam_prop)
