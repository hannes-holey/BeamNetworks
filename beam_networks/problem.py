import os
import numpy as np
import warnings

from beam_networks.network import Network
from beam_networks.solve import solve
from beam_networks.assembly import assemble_global_system
from beam_networks.stress import get_element_mises_stress, get_element_principal_stress
from beam_networks.bc import _get_bc_dof, _assemble_BCs
from beam_networks.utils import check_input_dict, _remove_isolated_nodes_edges, _mic
from beam_networks.viz import _plot_network, _plot_crack_graph
from beam_networks.io import _to_vtk, _to_vtk_periodic, _to_stl, _from_tar, _to_tar


class BeamNetwork(Network):
    """Beam Network structure. Derives from Network.

    Adding a cross section and elastic properties to the edges of a network
    makes it a beam network. Solves the elastic problem given appropriate
    boundary conditions.
    """

    def __init__(self,
                 nodes,
                 edges,
                 beam_prop={'name': 'circle', 'radius': 1.,
                            'E': 1., 'nu': 0.3},
                 periodic=None,
                 boxsize=None,
                 valid=True,
                 options={'vectorize': True, 'matrix': 'bsr', 'verbose': True},
                 outdir='.',
                 assemble_on_init=True):
        """Constructor.

        Initialize the network and assemble its global stiffness matrix.

        Parameters
        ----------
        nodes : np.ndarray
            Nodal coordinates
        edges : np.ndarray (of ints)
            Edge indices
        beam_prop : dict, optional
            Beam properties (cross section and elastic properties)
            (the default is {'name': 'circle', 'radius', 'E', 'nu'})
        periodic : bool, optional
            Use periodic boundary conditions (the default is False)
        boxsize : np.ndarray, optional
            Specify box dimensions (the default is None, which means that box dimensions are inferred from the nodes.)
        valid : bool, optional
            Skip initial sanity checks (the default is True, which assumes that the structure is valid.)
        options : dict, optional
            Solver options
        outdir : str, optional
            Directory where output is written into (the default is the current working directory)
        assemble_on_init : bool, optional
            Flag to activate the assembly of the stiffness matrix at initialization (the default is True)
        """

        super().__init__(nodes, edges,
                         periodic=periodic,
                         boxsize=boxsize,
                         valid=valid)

        self._options = check_input_dict(options,
                                         ['verbose', 'vectorize', 'matrix'],
                                         [True, True, 'bsr'],
                                         [None, None, ['bsr', 'lil', 'dense']])

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

    def save(self, filename):
        """Save the current state of the solver as gzipped tar archive

        Parameters
        ----------
        filename : str
            Name of the archive
        """
        _to_tar(filename, self)

    @classmethod
    def load(cls, filename, recompute=True):
        """Create a class instance from a tar archive.

        Parameters
        ----------
        filename : str
            Name of the archive

        Returns
        -------
        BeamNetwork
            A new class instance
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

    @property
    def has_bc(self):
        return len(self._bc) > 0

    @property
    def dof_per_node(self):
        return 3 * (self.dim - 1)

    @property
    def num_dof(self):

        n_dof = self.num_nodes * self.dof_per_node

        if self.has_bc:
            n_dof -= len(self._dof_D)
            n_dof -= len(self._dof_N)

        return n_dof

    @property
    def displaced_nodes(self):
        return self.nodes + self.displacement

    @property
    def displaced_edge_vectors(self):
        r1 = self.displaced_nodes[self.edges[:, 1]]
        r0 = self.displaced_nodes[self.edges[:, 0]]
        dr = _mic(r1 - r0, self._boxsize, self._periodic)

        return dr

    @property
    def displacement(self):
        """Nodal displacements

        Returns
        -------
        numpy.ndarray
            Shape (num_nodes, dim)
        """
        if self.has_solution:
            u = self.sol.reshape(
                len(self.sol) // self.dof_per_node, self.dof_per_node)
            return u[:, :self.dof_per_node // self.dim + 1]
        else:
            return np.zeros((self.num_nodes, self.dof_per_node // self.dim + 1))

    @property
    def rotation(self):
        """Nodal rotations

        Returns
        -------
        numpy.ndarray
            Shape (num_nodes, 3) for 3D; (num_nodes, 1) for 2D
        """
        if self.has_solution:
            u = self.sol.reshape(
                len(self.sol) // self.dof_per_node, self.dof_per_node)
            return u[:, self.dof_per_node // self.dim + 1:]
        else:
            return np.zeros((self.num_nodes, self.dof_per_node // self.dim + 1))

    @property
    def stiffness(self):
        """Global stiffness matrix

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_array
        """
        return self._K

    @property
    def has_stiffness(self):
        return self._K is not None

    def _assemble_global_system(self):
        """
        Assemble the stiffness matrix in the global coordinate system using Timoshenko beam theory.
        """

        if self._verbose:
            print(f"Assemble beam network with {self.num_nodes} nodes and {self.num_edges} edges")

        self._K = assemble_global_system(self._nodes,
                                         self._edges,
                                         self._edge_vectors,
                                         self._beam_prop,
                                         vectorize=self._options['vectorize'],
                                         matrix=self._options['matrix'],
                                         verbose=self._verbose
                                         )

    def add_BC(self, name, type, select, selection, vector, active=True, num_per_point=1):
        """Add boundary condition.

        Parameters
        ----------
        name : str
            Name/Identifier of the BC
        type : str
            BC type, either 'D' for Dirichlet or 'N' for Neumann
        select : str
            How to select nodes, either 'box', 'node', or 'point'
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
        num_per_point : int
            For point selection only, maximum number of closest nodes to select. The default is 1.
        """

        assert select in ['box', 'node', 'point']
        assert type in ['D', 'N']

        if name not in self._bc.keys():
            self._bc[name] = {}
        else:
            warnings.warn(f"Boundary condition '{name} has been overridden'")

        self._bc[name].update([('type', type), ('active', active)])
        self._bc[name].update(_get_bc_dof(self.nodes, select, selection, vector, num_per_point))
        self._bc_changed = True

    def delete_BC(self, name):
        """Delete boundary condition.

        Parameters
        ----------
        name : str
            Name of the BC to be removed
        """
        self._bc.pop(name, None)
        self._bc_changed = True

    def scale_BC(self, type, factor):
        """Scale all BCs of a certain type by a given factor

        Parameters
        ----------
        type : str
            BC type (either 'D' or 'N')
        factor : float
            factor
        """

        if type == 'D':
            self._val_D = list(np.array(self._val_D) * factor)
        elif type == 'N':
            self._val_N = list(np.array(self._val_N) * factor)

    def modify_BC(self, name, vector):
        """Modify an existing BC by changing its vector.
        This does not change the constraint DOFs, i.e. vector needs to have 'None' in the same place as before.
        Useful for stepwise change of a load/displacement

        Parameters
        ----------
        name : str
            Name of the BC to be removed
        vector : array-like
            Vector of length 3 and 6 for 2D and 3D systems, respectively. None means the corresponding DOF is
            not affected.
            In 2D: [ux, uy, tz] (Dircihlet) or [Fx, Fy, Mz] (Neumann)
            In 3D: [ux, uy, uz, tx, ty, tz] (Dircihlet) or [Fx, Fy, Fz, Mx, My, Mz] (Neumann)
        """

        assert name in self._bc.keys()

        nodes = self._bc[name]['nodes']
        dof_mask = [i for i, k in enumerate(vector) if k is not None]
        assert len(dof_mask) == self._bc[name]['dim']
        vector = [float(v) for v in vector if v is not None]
        dof_val = len(nodes) * vector

        self._bc[name].update([('dof_val', dof_val)])
        self._bc_changed = True

    def assemble_BCs(self):

        self._dof_D, self._val_D, self._dof_N, self._val_N = _assemble_BCs(self._bc)
        _excluded_nodes = [v for k, bc in self._bc.items() if not k.startswith('sym') for v in bc['nodes']]
        self._possible_edges = np.unique([j for j, (l, r) in enumerate(self._edges)
                                          if l not in _excluded_nodes and r not in _excluded_nodes])

        self._bc_changed = False

    def solve(self, solver=None, stress_mode='mean', preconditioner=None,
              verbosity=0):
        """Solve the linear system.

        Stores the solution for all DOFs and computes the reaction forces at constraint DOFs

        Parameters
        ----------
        solver : str, optional
            Type of solver to use, currently 'direct' and 'cg' available (the default is 'direct')
        stress_mode : str
            Either 'max' or 'mean', i.e. calculate maximum or mean von Mises 
            stress per beam
        preconditioner : str, optional
            Type of preconditioner, 'diagonal' or None (the default is None)
        verbosity : int, optional
            level of verbosity, if between 25/50 print information about 
            displacements and reaction forces, if between 50/100 print 
            information about stiffness matrix, if greater equal 100 print 
            condition number (default is 0). Only active for sparse matrices.

        Raises
        ------
        RuntimeError
            System can only be solved if boundary conditions are given.
        """

        if not self.has_stiffness:
            self._assemble_global_system()

        if self._bc_changed:
            self.assemble_BCs()

        if solver is None:
            if self.num_dof < 20_000:
                solver = 'direct'
            else:
                solver = 'cg'

        if not self.has_bc:
            raise RuntimeError(
                'No boundary conditions given. Nothing to solve here.')

        sol, F, info = solve(self._K,
                             self._dof_D,
                             self._val_D,
                             self._dof_N,
                             self._val_N,
                             solver=solver,
                             preconditioner=preconditioner,
                             verbosity=verbosity)

        self.has_solution = info == 0

        if self.has_solution:
            self.sol = sol
            self.Freact = self._get_reaction_forces(F)
            self.compute_equivalent_stress(mode=stress_mode)

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

    def compute_equivalent_stress(self, mode='mean'):
        """Calculate von Mises stress and store it in self._sVM.

        Parameters
        ----------
        mode : str
            Either 'max' or 'mean', i.e. calculate maximum or mean stress per beam

        Returns
        -------
        None
        """
        self._sVM = get_element_mises_stress(self.nodes,
                                             self.edges,
                                             self.edge_vectors,
                                             self.sol,
                                             self._beam_prop,
                                             rot=None, mode=mode)

    def compute_principal_stresses(self, mode='max'):
        pS = get_element_principal_stress(self.nodes,
                                          self.edges,
                                          self.edge_vectors,
                                          self.sol,
                                          self._beam_prop,
                                          rot=None, mode=mode)

        return pS

    def compute_ratio(self):

        _, ratio = get_element_mises_stress(self.nodes,
                                            self.edges,
                                            self.edge_vectors,
                                            self.sol,
                                            self._beam_prop,
                                            rot=None, return_ratio=True)

        return ratio

    def scale_solution(self, scale_factor):
        self.sol *= scale_factor

    def plot(self, ax, removed_edges=None, node_ids=False, contour=None, cax=None, aspect=1., lim=None, lw=2.,
             scale=1.):
        """Generate 2D plot of the deformed network structure.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis object
            Axis to plot into
        removed_edges : iterable, optional
            List of edge indices which have been removed from the original structure (the default is None)
        node_ids : bool, optional
            Print node numbers nect to undeformed structure (the default is False)
        contour : numpy.ndarray, optional
            Scalar field to plot as color on edges (the default is None, which means no coloring)
        cax : matplotlib.pyplot.axis object or None, optional
            Axis to plot colorbar into if contour is not None (the default is None, which takes space from ax)
        aspect : float, optional
            Aspect ratio (the default is 1.)
        lim : tuple, optional
            Colorbar limits (the default is None, which takes the limits of contour)

        Returns
        -------
        matplotlib.pyplot.axis object
            The plotted axis
        """

        # periodic_box = [L if p else p for (
        #     L, p) in zip(self._boxsize, self._periodic)]

        # undeformed
        ax = _plot_network(ax, self.nodes, self.edges, self.edge_vectors, color='0.7',
                           node_ids=node_ids, lw=lw)

        # deformed
        ax = _plot_network(ax, self.displaced_nodes, self.edges, self.displaced_edge_vectors, contour,
                           cax=cax, lim=lim, lw=lw)

        if removed_edges is not None:
            _plot_crack_graph(ax, self.nodes, self._edges, removed_edges)

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

        ax.set_aspect(aspect)

        return ax

    def to_vtk(self, file="foo.vtk"):
        """Write structure and possibly solution to VTK file

        Parameters
        ----------
        file : str, optional
            Output filename (the default is "foo.vtk")
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
                        stress=self._sVM)

    def to_stl(self, file, clean=False, tol=1e-6):

        if clean:
            # stress_tolerance = tol * self._beam_prop['E']
            # mask = self._sVM > stress_tolerance
            nodes, edges = _remove_isolated_nodes_edges(self.nodes, self.edges)
        else:
            nodes = self._nodes
            edges = self.edges

        _to_stl(os.path.join(self._outdir, file), nodes, edges, self._beam_prop)
