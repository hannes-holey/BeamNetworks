import numpy as np

from beam_networks.stiffness import get_element_stiffness_global
from beam_networks.geo import get_geometric_props


def get_Ke(r0, r1, d0, d1, beam_prop):

    E = beam_prop['E']
    nu = beam_prop['nu']
    G = E / (2. * (1 + nu))
    _, Iz, _, A, kappa, _ = get_geometric_props(beam_prop)

    EA = E * A
    GA = kappa * G * A
    EI = E * Iz

    # Get rigid body rotation (alpha)
    # disp
    u0 = d0[:2]
    u1 = d1[:2]

    # rot
    t0 = d0[2]
    t1 = d1[2]

    l0 = np.sqrt(np.sum((r1 - r0)**2))
    ln = np.sqrt(np.sum((r1 + u1 - r0 - u0)**2))

    c0 = (r1[0] - r0[0]) / l0
    s0 = (r1[1] - r0[1]) / l0
    c = (r1[0] + u1[0] - r0[0] - u0[0]) / ln
    s = (r1[1] + u1[1] - r0[1] - u0[1]) / ln

    sin_a = c0 * s - s0 * c
    cos_a = c0 * c + s0 * s

    if sin_a >= 0. and cos_a >= 0.:
        alpha = np.arcsin(sin_a)
    elif sin_a >= 0. and cos_a >= 0.:
        alpha = np.arccos(cos_a)
    elif sin_a < 0. and cos_a >= 0.:
        alpha = np.arcsin(sin_a)
    else:
        alpha = -np.arccos(cos_a)

    assert np.abs(alpha) < np.pi

    # Compute local stiffness and local force vector
    ul = np.zeros(3)
    Kl = np.zeros((3, 3))
    Kl[0, 0] = EA / l0
    Kl[1, 1] = EI / l0
    Kl[2, 2] = GA / l0

    ul[0] = ln - l0
    ul[1] = t0 - alpha
    ul[2] = t1 - alpha

    fl = Kl @ ul

    # Local to global transformation
    B = np.array([
        [-c, -s, 0., c, s, 0.],
        [-s / ln, c / ln, 1, s / ln, -c / ln, 0.],
        [-s / ln, c / ln, 0., s / ln, -c / ln, 1.],
    ])

    # Forces: local to global
    fg = B.T @ fl

    # Tangent stiffness: K_local + K_geometric
    r = np.array([-c, -s, 0, c, s, 0])
    z = np.array([s, -c, 0, -s, c, 0])
    zz = z[:, None] @ z[None, :]
    rz = r[:, None] @ z[None, :]
    Kg = B.T @ Kl @ B + zz * fl[0] / ln + (rz + rz.T) / ln**2 * (fl[1] + fl[2])

    return Kg, fg


def assemble_K(nodes, edges, beam_prop, sol=None):
    """Assemble global stiffness matrix as dense array.

    Vectorized version.

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray (of ints)
        Edge connectivity
    beam_prop : dict
        Beam properties (cross section and elastic properties)

    Returns
    -------
    np.ndrarray
        The global stiffness matrix
    """

    num_nodes, ndim = nodes.shape
    num_dof_per_node = 3 * (ndim - 1)
    num_dof = num_nodes * num_dof_per_node

    K_global = np.zeros((num_dof, num_dof))
    F_global = np.zeros(num_dof)

    for i, element in enumerate(edges):
        e0, e1 = element
        s1 = slice(e0 * num_dof_per_node, (e0 + 1) * num_dof_per_node)
        s2 = slice(e1 * num_dof_per_node, (e1 + 1) * num_dof_per_node)

        r0_e = nodes[e0]
        r1_e = nodes[e1]

        if sol is not None:
            u0_e = sol[s1]
            u1_e = sol[s2]
            Ke, fe = get_Ke(r0_e, r1_e, u0_e, u1_e, beam_prop)
            F_global[s1] += fe[:num_dof_per_node]
            F_global[s2] += fe[num_dof_per_node:]
        else:
            Ke = get_element_stiffness_global(beam_prop, r1_e - r0_e)

        K_global[s1, s1] += Ke[:num_dof_per_node, :num_dof_per_node]
        K_global[s1, s2] += Ke[:num_dof_per_node, num_dof_per_node:]
        K_global[s2, s1] += Ke[num_dof_per_node:, :num_dof_per_node]
        K_global[s2, s2] += Ke[num_dof_per_node:, num_dof_per_node:]

    return K_global, F_global


def solve_lin(nodes, edges, f_ext, bc_D):

    # geometric linear
    Klin, _ = assemble_K(nodes, edges)
    Klin[:, bc_D] = 0.
    Klin[bc_D, :] = 0.
    Klin[bc_D, bc_D] = 1.
    ulin = np.linalg.solve(Klin, f_ext)
    ulin[bc_D] = 0.

    print('L: ', np.linalg.norm(Klin @ ulin - f_ext))

    return ulin


def solve_nonlin(nodes, edges, sol, f_ext, prop, bc_D):

    it = 10_000
    for i in range(it):
        # geometric nonlinear
        K, F = assemble_K(nodes, edges, prop, sol)
        res = F - f_ext

        K[:, bc_D] = 0.
        K[bc_D, :] = 0.
        K[bc_D, bc_D] = 1.
        F[bc_D] = 0.

        # TODO: reduced system
        du = -np.linalg.solve(K, res)
        du[bc_D] = 0.

        sol += du

        norm = np.linalg.norm(du)
        if norm < 1e-9:
            print('N: ', norm, f'({i})')
            break

    return sol
