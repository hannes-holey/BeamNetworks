import numpy as np
import scipy.sparse as sp

from beam_networks.utils import _intersect


def _get_graph_removed_edges(nodes, edges, removed_edges, rc=4.5):
    """Construct a new graph connecting midpoints of a subset of edges
    (here: removed edges)

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray
        Edge connectivity
    removed_edges : array-like
        Edge indices which form the nodes of the new graph
    rc : float, optional
        Cutoff radius specifying how far we search for neighboring nodes
        (the default is 4.5)

    Returns
    -------
    np.ndarray
        Nodes of the new graph (edge midpoints)
    scipy.sparse.csr_matrix
       Adjacency matrix of the new graph
    """

    _crack_edges = []
    _edge_len = []

    rm_edges_mp = (nodes[edges[removed_edges][:, 0], :] + nodes[edges[removed_edges][:, 1], :]) / 2.
    active_mask = np.ones(edges.shape[0], dtype=bool)
    active_mask[removed_edges] = False

    for i, (xi, yi) in enumerate(rm_edges_mp):

        for j, (xj, yj) in enumerate(rm_edges_mp[i+1:]):

            d = (xj - xi)**2 + (yj - yi)**2

            if d < rc**2:

                A = [xi, yi]
                B = [xj, yj]

                C = nodes[edges[active_mask][:, 0]].T
                D = nodes[edges[active_mask][:, 1]].T

                intersec = _intersect(A, B, C, D)

                if not np.any(intersec):
                    _crack_edges.append([i, i + j + 1])
                    _crack_edges.append([i + j + 1, i])
                    _edge_len.append(np.sqrt(d))
                    _edge_len.append(np.sqrt(d))

    _crack_edges = np.array(_crack_edges, dtype=int)

    if _crack_edges.ndim == 1:
        return None, None
    else:
        graph = sp.csr_matrix((_edge_len, [_crack_edges[:, 0], _crack_edges[:, 1]]), shape=(removed_edges.shape[0],
                                                                                            removed_edges.shape[0]))

        return rm_edges_mp, graph


def _get_crack_path(graph_nodes, graph, nbound=5):
    """Reconstruct crack path in a 2D cracked network structure.

    Parameters
    ----------
    graph_nodes : np.ndarray
        Nodal coordinates
    graph : scipy.sparse.csr_matrix
        Adjacency matrix of the graph
    nbound : int, optional
        Number of nodes a the side considered as possible starting points of the crack path
        (the default is 5).

    Returns
    -------
    np.ndarray
        Crack path edge indices
    float
        Length of the crack path
    """

    x_sorted = np.argsort(graph_nodes[:, 0])
    nbound = min(nbound, len(x_sorted) - 1)

    start = np.arange(nbound)
    end = -(np.arange(nbound) + 1)

    lengths = []
    preds = []

    for s in start:
        for e in end:
            source = x_sorted[s]
            sink = x_sorted[e]

            length, predecessor = sp.csgraph.yen(csgraph=graph,
                                                 source=source,
                                                 sink=sink,
                                                 K=1,
                                                 directed=True,
                                                 unweighted=False,
                                                 return_predecessors=True)

            if len(length) == 1:
                lengths.append(length)
                preds.append(predecessor)

    if len(lengths) > 0:
        predecessor = preds[np.argmax(lengths)]

        path = sp.csgraph.reconstruct_path(csgraph=graph, predecessors=predecessor[0], directed=True)
        path_edges = np.vstack(np.nonzero(path)).T

        return path_edges, np.amax(lengths)
    else:
        return None, None


def get_crack(nodes, edges, removed_edges, nbound=5):
    """Reconstruct crack path in a 2D cracked network structure.

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray
        Edge indices
    removed_edges : array-like
        Edge indices which have been removed from the original structure
    nbound : int, optional
        Number of nodes a the side considered as possible starting points of the crack path
        (the default is 5).

    Returns
    -------
    np.ndarray
        Crack path nodes
    np.ndarray
        Crack path edge indices
    float
        Length of the crack path
    """

    # get auxilliary graph
    graph_nodes, graph = _get_graph_removed_edges(nodes, edges, removed_edges)

    # Find longest crack within the auxiliary graphs
    if graph_nodes is not None:
        path_edges, length = _get_crack_path(graph_nodes, graph, nbound=nbound)
        return graph_nodes, path_edges, length
    else:
        return None, None, None
