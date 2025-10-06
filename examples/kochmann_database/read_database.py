import numpy as np
from scipy.sparse import load_npz, triu


def load_database(n_samples=None, start_index=None, split=False):
    """Load dataset from
    https://doi.org/10.1038/s41467-023-42068-x downloaded from
    https://www.research-collection.ethz.ch/handle/20.500.11850/618078.

    Parameters
    ----------
    n_samples : int
        number of samples to be drawn.
    start_index : int
        index of first sample to be drawn.
    split : bool
        if True return the adj. matrices as a list where every entry is
        the adj. matrix of an individual cell.

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (n_nodes_tesselate,ndim).
    adj : np.ndarray
        adjacency list of shape (n_bonds_tesselate,2).
    cell : np.ndarray
        cell lengths of shape (ndim).

    """
    if start_index is None:
        start_index = 0
    if n_samples is None:
        n_samples = 965685
    # always 27 nodes so if we want to draw individual samples, just chunk
    # each array in to pieces of 27 rows (apart from the stiffness vector)
    start_index = start_index * 27
    stop_index = start_index + n_samples * 27

    coords = load_npz("node-position.npz").toarray()[start_index:stop_index]

    offset = load_npz("node-offset.npz").toarray()[start_index:stop_index]

    adj = load_npz("adjacency-matrix.npz")[start_index:stop_index, :]
    # adj = triu(load_npz("data/adjacency-matrix.npz")[start_index:stop_index])
    # print(adj.shape)
    # adj = np.column_stack(adj.nonzero())
    if split:
        adj = [np.column_stack(triu(adj[27*i:27*(i+1)]).nonzero())
               for i in np.arange(n_samples)]

    stiffness_vec = np.loadtxt("stiffness-vec.csv",
                               delimiter=",")[start_index:stop_index]

    return coords, offset, adj, stiffness_vec


def check_identity(start_ind=0, nsamples=15):
    """Take first sample and compare it with the subsequent ones in terms of
    coordinate positions, adjacency, stiffness etc. to determine what changes.
    The information is printed out to the screen.

    Parameters
    ----------
    start_ind : int
        index of first sample to be drawn.
    n_samples : int
        number of samples to be drawn.

    Returns
    -------

    """

    #
    coords, offset, adj, stiffness_vec = load_database(n_samples=nsamples,
                                                       start_index=start_ind,
                                                       split=True)
    coords1, offset1, adj1, stiffness_vec1 = coords[:27, :], offset[:27, :], adj[0], stiffness_vec[:27, :]
    #
    for i in range(1, 15):
        start_index = i * 27
        stop_index = (i + 1) * 27
        coords2 = coords[start_index:stop_index]
        offset2 = offset[start_index:stop_index]
        adj2 = adj[i]
        stiffness_vec2 = stiffness_vec[start_index:stop_index]

        print("Sample Nr. ", i)
        print("Coordinates identical",
              np.all(np.isclose(coords1, coords2)))
        print("Offset identical",
              np.all(np.isclose(offset1, offset2)))
        print("Adjacency identical",
              np.all(np.isclose(adj1, adj2)))
        print("Stiffness identical",
              np.all(np.isclose(stiffness_vec1, stiffness_vec2)))
        print()
    return
