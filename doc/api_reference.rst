API Reference
=============

Main classes
------------

Two classes form the public API of **beam_networks**:

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

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: class.rst

   beam_networks.network.Network
   beam_networks.problem.BeamNetwork
   beam_networks.fracture.FractureProblem

Supporting modules
------------------

The main classes delegate to the following modules for specific computations.

Assembly
~~~~~~~~

Builds the sparse global stiffness matrix :math:`K` by summing element-level
contributions, each rotated from the local beam frame into the global coordinate
frame.

.. autosummary::
   :nosignatures:

   beam_networks.assembly.assemble_global_system

.. automodule:: beam_networks.assembly
   :members:
   :undoc-members:
   :show-inheritance:

Solver
~~~~~~

Solves the constrained linear system :math:`K\,u = f` after applying Dirichlet
boundary conditions.  Supports a direct sparse factorisation (default) and a
preconditioned conjugate-gradient method.

.. autosummary::
   :nosignatures:

   beam_networks.solve.solve

.. automodule:: beam_networks.solve
   :members:
   :undoc-members:
   :show-inheritance:

Element stiffness
~~~~~~~~~~~~~~~~~

Constructs the per-element stiffness matrix (12 × 12 in 3D, 6 × 6 in 2D) for
Timoshenko beams following the standard FEM derivation.  Vectorised forms
operate on all elements simultaneously for efficient assembly.

.. autosummary::
   :nosignatures:

   beam_networks.stiffness.get_element_stiffness_global
   beam_networks.stiffness.get_element_stiffness_local_vec
   beam_networks.stiffness.get_element_stiffness_global_vec

.. automodule:: beam_networks.stiffness
   :members:
   :undoc-members:
   :show-inheritance:

Cross-section geometry
~~~~~~~~~~~~~~~~~~~~~~

Computes cross-section properties — area, second moments of area, and the
Timoshenko shear-correction factor — for circular and rectangular sections.
Analytic first-order derivatives with respect to the shape parameters are
available to support gradient-based sensitivity analyses.

.. autosummary::
   :nosignatures:

   beam_networks.geo.get_geometric_props
   beam_networks.geo.get_geometric_props_derivative

.. automodule:: beam_networks.geo
   :members:
   :undoc-members:
   :show-inheritance:

Stress recovery
~~~~~~~~~~~~~~~

Post-processes nodal displacements into element-level stresses: von Mises
equivalent stress, principal stresses, and the bending-to-total stress ratio
(useful for distinguishing stretch- from bending-dominated deformation).

.. autosummary::
   :nosignatures:

   beam_networks.stress.get_element_mises_stress
   beam_networks.stress.get_element_principal_stress
   beam_networks.stress.principal_stress
   beam_networks.stress.vmises_stress
   beam_networks.stress.stretch_bend_ratio

.. automodule:: beam_networks.stress
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
~~~~~~~~~

Spatial selection helpers (box and sphere regions), input-dictionary
validation, and construction of a contact-edge network from a set of
overlapping disks.

.. autosummary::
   :nosignatures:

   beam_networks.utils.box_selection
   beam_networks.utils.point_selection
   beam_networks.utils.check_input_dict
   beam_networks.utils.get_edges_from_disks

.. automodule:: beam_networks.utils
   :members:
   :undoc-members:
   :show-inheritance:

Additional modules
~~~~~~~~~~~~~~~~~~

.. automodule:: beam_networks.bc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: beam_networks.lattice
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: beam_networks.io
   :members:
   :undoc-members:
   :show-inheritance:
