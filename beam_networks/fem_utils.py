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
