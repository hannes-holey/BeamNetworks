API Reference
=============

Main classes
------------

Three classes form the public API of **beam_networks**.  They live at the
top level of the package and are the primary entry points for users:

:class:`~beam_networks.network.Network`
   Manages the network topology — nodes, edges, and periodic boundary
   conditions.  Provides geometric properties (edge lengths, edge vectors,
   coordination number) and lattice-generation factory methods for common 2D
   and 3D lattice types.

:class:`~beam_networks.problem.BeamNetwork`
   Extends :class:`~beam_networks.network.Network` with the full FEM problem
   setup: cross-section and material parameters, Dirichlet and Neumann boundary
   conditions, stiffness assembly, linear solve, stress recovery, and VTK/STL
   file output.

:class:`~beam_networks.fracture.FractureProblem`
   Extends :class:`~beam_networks.problem.BeamNetwork` with progressive element
   removal to simulate fracture.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: class.rst

   beam_networks.network.Network
   beam_networks.problem.BeamNetwork
   beam_networks.fracture.FractureProblem


``beam_networks.fem`` — Finite element machinery
-------------------------------------------------

All components of the FEM pipeline live here: shape functions, element
stiffness matrices (both analytical and numerically integrated), global
assembly, and boundary condition handling.

Assembly
~~~~~~~~

Builds the sparse global stiffness matrix :math:`K` by summing element-level
contributions, each rotated from the local beam frame into the global
coordinate frame.  Supports exact analytical stiffness matrices as well as
FEM assembly with Lagrange (Timoshenko) or Hermite (Euler-Bernoulli) shape
functions and static condensation of interior DOFs.

.. autosummary::
   :nosignatures:

   beam_networks.fem.assembly.assemble_global_system

.. automodule:: beam_networks.fem.assembly
   :members:
   :undoc-members:
   :show-inheritance:

Element stiffness
~~~~~~~~~~~~~~~~~

Per-element stiffness matrices in the global frame.  Four families are
provided — exact analytical or numerically integrated, for Timoshenko or
Euler-Bernoulli beam theory — each in a single-element (loop) and a
vectorised (all-elements-at-once) form.

.. autosummary::
   :nosignatures:

   beam_networks.fem.stiffness.global_element_stiffness_timoshenko_exact_loop
   beam_networks.fem.stiffness.global_element_stiffness_timoshenko_exact_vec
   beam_networks.fem.stiffness.local_element_stiffness_timoshenko_exact_vec
   beam_networks.fem.stiffness.global_element_stiffness_timoshenko_numeric_loop
   beam_networks.fem.stiffness.global_element_stiffness_timoshenko_numeric_vec
   beam_networks.fem.stiffness.global_element_stiffness_euler_exact_loop
   beam_networks.fem.stiffness.global_element_stiffness_euler_exact_vec
   beam_networks.fem.stiffness.global_element_stiffness_euler_numeric_loop
   beam_networks.fem.stiffness.global_element_stiffness_euler_numeric_vec

.. automodule:: beam_networks.fem.stiffness
   :members:
   :undoc-members:
   :show-inheritance:

Shape functions and quadrature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gauss–Legendre quadrature points and weights, and the shape-function families
used by the FEM element stiffness routines: Lagrange (for Timoshenko beams),
cubic Hermite (for Euler-Bernoulli beams), and the exact Friedman–Kosmatka
(FK) Timoshenko shape functions.

.. automodule:: beam_networks.fem.basis
   :members:
   :undoc-members:
   :show-inheritance:

Boundary conditions
~~~~~~~~~~~~~~~~~~~

Translates user-facing boundary condition specifications (box/point/node
selection, displacement and force vectors) into DOF index lists consumed by
the solver.

.. automodule:: beam_networks.fem.bc
   :members:
   :undoc-members:
   :show-inheritance:


``beam_networks.geometry`` — Geometry and mesh
----------------------------------------------

Cross-section properties, lattice generators, periodic-cell tessellation,
and spatial selection utilities.

Cross-section geometry
~~~~~~~~~~~~~~~~~~~~~~

Computes cross-section properties — area, second moments of area, and the
Timoshenko shear-correction factor — for circular and rectangular sections.
Analytic first-order derivatives with respect to the shape parameters are
available to support gradient-based sensitivity analyses.

.. autosummary::
   :nosignatures:

   beam_networks.geometry.geo.get_geometric_props
   beam_networks.geometry.geo.get_geometric_props_derivative

.. automodule:: beam_networks.geometry.geo
   :members:
   :undoc-members:
   :show-inheritance:

Lattice generation
~~~~~~~~~~~~~~~~~~

Factory functions for common 2D and 3D lattice topologies (square, triangular,
cubic, BCC, FCC, bowtie).

.. automodule:: beam_networks.geometry.lattice
   :members:
   :undoc-members:
   :show-inheritance:

Tessellation
~~~~~~~~~~~~

Tiles a unit cell across periodic boundaries to build large network structures.

.. automodule:: beam_networks.geometry.tesselate
   :members:
   :undoc-members:
   :show-inheritance:

Spatial selection
~~~~~~~~~~~~~~~~~

Node-selection helpers (bounding-box and nearest-point queries), edge
filtering by length, network cleaning (isolated-node and dangling-bond
removal), spatial reflection, and construction of a contact-edge graph from
a disk-packing file.

.. autosummary::
   :nosignatures:

   beam_networks.geometry.selection.box_selection
   beam_networks.geometry.selection.point_selection
   beam_networks.geometry.selection.get_edges_from_disks

.. automodule:: beam_networks.geometry.selection
   :members:
   :undoc-members:
   :show-inheritance:


``beam_networks.io`` — Input / Output
--------------------------------------

File I/O (VTK, XDMF, STL, tar archives) and input-dictionary validation.

File formats
~~~~~~~~~~~~

Read and write network data and simulation results in various formats.

.. automodule:: beam_networks.io.formats
   :members:
   :undoc-members:
   :show-inheritance:

Input validation
~~~~~~~~~~~~~~~~

Validates and sanitises user-provided option dictionaries.

.. automodule:: beam_networks.io.validation
   :members:
   :undoc-members:
   :show-inheritance:


``beam_networks.postprocess`` — Post-processing
------------------------------------------------

Post-processes nodal displacements into element-level stresses and
produces plots of the network.

Stress recovery
~~~~~~~~~~~~~~~

Computes von Mises equivalent stress, principal stresses, and the
bending-to-total stress ratio (useful for distinguishing stretch- from
bending-dominated deformation).

.. autosummary::
   :nosignatures:

   beam_networks.postprocess.stress.get_element_mises_stress
   beam_networks.postprocess.stress.get_element_principal_stress
   beam_networks.postprocess.stress.vmises_stress
   beam_networks.postprocess.stress.principal_stress
   beam_networks.postprocess.stress.stretch_bend_ratio

.. automodule:: beam_networks.postprocess.stress
   :members:
   :undoc-members:
   :show-inheritance:

Visualisation
~~~~~~~~~~~~~

2D network plots (undeformed and deformed, with optional per-edge colouring)
and VTK-to-image rendering utilities.

.. automodule:: beam_networks.postprocess.viz
   :members:
   :undoc-members:
   :show-inheritance:


``beam_networks.solvers`` — Linear solvers
------------------------------------------

Solves the constrained linear system :math:`K\,u = f` after applying
Dirichlet boundary conditions.  Supports a direct sparse factorisation
(default) and a preconditioned conjugate-gradient method.

.. autosummary::
   :nosignatures:

   beam_networks.solvers.linear.solve

.. automodule:: beam_networks.solvers.linear
   :members:
   :undoc-members:
   :show-inheritance:


``beam_networks.reference_solutions`` — Analytic solutions
----------------------------------------------------------

Closed-form reference solutions used for verification and convergence studies.

.. automodule:: beam_networks.reference_solutions.cantilever
   :members:
   :undoc-members:
   :show-inheritance:
