#
# Copyright 2026 Hannes Holey
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


def _gauss_legendre(n_gauss):
    """Gauss-Legendre quadrature points and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n_gauss)


def _lagrange_basis(n_nodes, xi):
    """Evaluate Lagrange shape functions and their xi-derivatives at a point.

    Nodes are equally spaced in [-1, 1]:  xi_k = -1 + 2*k/(n_nodes-1).

    Parameters
    ----------
    n_nodes : int
        Number of nodes (polynomial degree = n_nodes - 1).
    xi : float
        Evaluation point in [-1, 1].

    Returns
    -------
    N : np.ndarray, shape (n_nodes,)
        Shape function values.
    dN : np.ndarray, shape (n_nodes,)
        Shape function derivatives with respect to xi.
    """
    nodes = np.linspace(-1., 1., n_nodes)

    N = np.ones(n_nodes)
    dN = np.zeros(n_nodes)

    for k in range(n_nodes):
        for j in range(n_nodes):
            if j != k:
                N[k] *= (xi - nodes[j]) / (nodes[k] - nodes[j])

    for k in range(n_nodes):
        for j in range(n_nodes):
            if j != k:
                prod = 1. / (nodes[k] - nodes[j])
                for m in range(n_nodes):
                    if m != k and m != j:
                        prod *= (xi - nodes[m]) / (nodes[k] - nodes[m])
                dN[k] += prod

    return N, dN


def _hermite_basis(xi):
    """Cubic Hermite shape functions on the reference element ξ ∈ [−1, 1].

    Returns the four C¹-Hermite shape functions and their ξ-derivatives at the
    scalar evaluation point *xi*.  The four DOFs are ordered as

        [f(−1),  (df/dξ)(−1),  f(+1),  (df/dξ)(+1)]

    i.e. the *derivative* DOF is with respect to the **reference** coordinate ξ,
    not the physical coordinate x.

    Explicit expressions::

        H₁(ξ) = (2 − 3ξ + ξ³) / 4
        H₂(ξ) = (1 − ξ − ξ² + ξ³) / 4
        H₃(ξ) = (2 + 3ξ − ξ³) / 4
        H₄(ξ) = (−1 − ξ + ξ² + ξ³) / 4

    Verification of boundary conditions (rows = functions, cols = ξ = ±1)::

        H₁: f(−1) = 1,  H₁'(−1) = 0,  H₁(+1) = 0,  H₁'(+1) = 0
        H₂: f(−1) = 0,  H₂'(−1) = 1,  H₂(+1) = 0,  H₂'(+1) = 0
        H₃: f(−1) = 0,  H₃'(−1) = 0,  H₃(+1) = 1,  H₃'(+1) = 0
        H₄: f(−1) = 0,  H₄'(−1) = 0,  H₄(+1) = 0,  H₄'(+1) = 1

    H₁ + H₃ = 1 (partition of unity for the displacement part).

    Conversion to physical DOFs
    ---------------------------
    The physical element runs from x = 0 to x = l with Jacobian dx/dξ = l/2.
    When the desired nodal quantities are **physical-space** derivatives
    (e.g. the beam rotation θ = dv/dx), note::

        df/dx = (df/dξ) · (2/l)   →   (df/dξ) = θ · (l/2)

    so the shape functions for physical DOFs
    [f(0), (df/dx)(0), f(l), (df/dx)(l)] are::

        Ñ₁ = H₁,         Ñ₂ = H₂ · (l/2),
        Ñ₃ = H₃,         Ñ₄ = H₄ · (l/2)

    and their x-derivatives are::

        dÑ₁/dx = (dH₁/dξ)·(2/l),   dÑ₂/dx = dH₂/dξ,
        dÑ₃/dx = (dH₃/dξ)·(2/l),   dÑ₄/dx = dH₄/dξ

    Timoshenko beam note
    --------------------
    In Timoshenko theory the transverse displacement v and the rotation θ are
    **independent** fields, so full Hermite interpolation for v (with θ = dv/dx
    enforced as a DOF) is only appropriate for Euler–Bernoulli beams.

    For Timoshenko beams a locking-free approach is *linked interpolation*:
    interpolate v with the four Hermite DOFs [v₁, θ₁, v₂, θ₂] and derive
    the shear-consistent rotation as θ̃ = dv/dx + γ̄ where γ̄ is a constant
    shear strain augmentation.  Alternatively, use separate Lagrange
    interpolation for both v and θ with reduced integration (see
    :func:`_lagrange_basis` and :func:`_gauss_legendre`).

    Parameters
    ----------
    xi : float
        Reference coordinate in [−1, 1].

    Returns
    -------
    N : np.ndarray, shape (4,)
        Shape function values [H₁, H₂, H₃, H₄].
    dN : np.ndarray, shape (4,)
        Shape function derivatives dHₖ/dξ.
    """
    N = np.empty(4)
    N[0] = (2. - 3.*xi + xi**3) / 4.
    N[1] = (1. - xi - xi**2 + xi**3) / 4.
    N[2] = (2. + 3.*xi - xi**3) / 4.
    N[3] = (-1. - xi + xi**2 + xi**3) / 4.

    dN = np.empty(4)
    dN[0] = (-3. + 3.*xi**2) / 4.
    dN[1] = (-1. - 2.*xi + 3.*xi**2) / 4.
    dN[2] = (3. - 3.*xi**2) / 4.
    dN[3] = (-1. + 2.*xi + 3.*xi**2) / 4.

    return N, dN


def _timoshenko_basis_FK(xi, L, Phi=0.):
    """Exact Timoshenko beam shape functions after Friedman & Kosmatka (1993).

    Returns the displacement (Nw) and rotation (Nt) shape-function matrices for
    a two-node Timoshenko beam element of length *L*, evaluated at one or more
    reference coordinates *xi* ∈ [−1, 1].

    The mapping to physical coordinates is  x = L/2 · (ξ + 1), so ξ = −1
    corresponds to node 1 (x = 0) and ξ = +1 to node 2 (x = L).

    The four DOFs are ordered as  [w₁, θ₁, w₂, θ₂]  (values at ξ = ±1).

    Shear-influence parameter
    -------------------------
    Φ = 12 EI / (κ G A L²)

    where κ is the shear correction factor.  Φ → 0 recovers the
    Euler–Bernoulli cubic-Hermite shape functions.  A typical value for a
    moderately thick beam is Φ ~ 1.

    Shape functions (expressed via s = (ξ+1)/2 ∈ [0, 1])
    ------------------------------------------------------
    Transverse displacement  w = Nw · d::

        Nw₁ =  1/(1+Φ) · [2s³ − 3s² − Φs + (1+Φ)]
        Nw₂ =  L/(1+Φ) · [s³ − (2+Φ/2)s² + (1+Φ/2)s]
        Nw₃ = −1/(1+Φ) · [2s³ − 3s² − Φs]
        Nw₄ =  L/(1+Φ) · [s³ − (1−Φ/2)s² − (Φ/2)s]

    Section rotation  θ = Nt · d::

        Nt₁ =  6/(1+Φ)/L · [s² − s]
        Nt₂ =  1/(1+Φ)   · [3s² − (4+Φ)s + (1+Φ)]
        Nt₃ = −6/(1+Φ)/L · [s² − s]
        Nt₄ =  1/(1+Φ)   · [3s² − (2−Φ)s]

    The rotation field θ is the *section* rotation (independent of dw/dx),
    consistent with Timoshenko kinematics.  The shear strain is
    γ = dw/dx − θ.

    Parameters
    ----------
    xi : float or array_like
        Reference coordinate(s) in [−1, 1].
    L : float
        Element length.
    Phi : float, optional
        Shear-influence parameter Φ = 12EI/(κGAL²).  Default 0 (EB limit).

    Returns
    -------
    Nw : np.ndarray, shape (4, n)
        Displacement shape functions; n = number of evaluation points.
    Nt : np.ndarray, shape (4, n)
        Rotation shape functions.

    References
    ----------
    Z. Friedman, J. B. Kosmatka, "An improved two-node Timoshenko beam
    finite element", Computers & Structures 47(3), 473–481, 1993.
    """
    xi = np.asarray(xi, dtype=float)
    c = 1. / (1. + Phi)
    s = (xi + 1.) / 2.       # s = x/L ∈ [0, 1]
    s2 = s ** 2
    s3 = s ** 3

    nw0 = c * (2.*s3 - 3.*s2 - Phi*s + (1. + Phi))
    nw1 = c * L * (s3 - (2. + Phi/2.)*s2 + (1. + Phi/2.)*s)
    nw2 = c * (-(2.*s3 - 3.*s2 - Phi*s))
    nw3 = c * L * (s3 - (1. - Phi/2.)*s2 - (Phi/2.)*s)

    nt0 = c * 6./L * (s2 - s)
    nt1 = c * (3.*s2 - (4. + Phi)*s + (1. + Phi))
    nt2 = c * (-6./L) * (s2 - s)
    nt3 = c * (3.*s2 - (2. - Phi)*s)

    Nw = np.vstack([nw0, nw1, nw2, nw3])
    Nt = np.vstack([nt0, nt1, nt2, nt3])

    return Nw, Nt
