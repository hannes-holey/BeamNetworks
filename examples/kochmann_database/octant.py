import numpy as np


def build_full_cell(octant=None, adj=None):
    """Build full cell from octant as provided by the dataset from
    https://doi.org/10.1038/s41467-023-42068-x downloaded from
    https://www.research-collection.ethz.ch/handle/20.500.11850/618078


    Reference: https://en.wikipedia.org/wiki/Octant_(solid_geometry)

    Parameters
    ----------
    octant : np.ndarray
        coordinates of shape (n_nodes,ndim).
    adj : np.ndarray
        adjacency list of shape (n_bonds,2).

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (n_nodes_new,ndim).
    adj : np.ndarray
        adjacency list of shape (n_bonds_new,2).

    """

    n_nodes = octant.shape[0]

    # final results
    full_cell = np.empty((0, 3))
    A = []
    octant_idx = 0

    # convert to numpy array
    adj = np.array(adj)  # convert to numpy array

    # Get the 8 octants and concatenate them to build the full cell
    # octants are labeled with the corresponding sign combination,
    # where p <=> + and m <=> -

    # (+,+,+) octant
    ppp = octant.copy()
    full_cell = np.vstack((full_cell, ppp))
    for b in adj:
        A.append([b[0]+n_nodes*octant_idx, b[1]+n_nodes*octant_idx])
    octant_idx += 1

    # (-,+,+) octant
    mpp = octant.copy()
    mpp[:, 0] = -1.*mpp[:, 0]
    full_cell = np.vstack((full_cell, mpp))
    for b in adj:
        A.append([b[0]+n_nodes*octant_idx, b[1]+n_nodes*octant_idx])
    octant_idx += 1

    # (+,-,+) octant
    pmp = octant.copy()
    pmp[:, 1] = -1.*pmp[:, 1]
    full_cell = np.vstack((full_cell, pmp))
    for b in adj:
        A.append([b[0]+n_nodes*octant_idx, b[1]+n_nodes*octant_idx])
    octant_idx += 1

    # (-,-,+) octant
    mmp = octant.copy()
    mmp[:, 0] = -1.*mmp[:, 0]
    mmp[:, 1] = -1.*mmp[:, 1]
    full_cell = np.vstack((full_cell, mmp))
    for b in adj:
        A.append([b[0]+n_nodes*octant_idx, b[1]+n_nodes*octant_idx])
    octant_idx += 1

    # (+,+,-) octant
    ppm = octant.copy()
    ppm[:, 2] = -1.*ppm[:, 2]
    full_cell = np.vstack((full_cell, ppm))
    for b in adj:
        A.append([b[0]+n_nodes*octant_idx, b[1]+n_nodes*octant_idx])
    octant_idx += 1

    # (-,+,-) octant
    mpm = octant.copy()
    mpm[:, 0] = -1.*mpm[:, 0]
    mpm[:, 2] = -1.*mpm[:, 2]
    full_cell = np.vstack((full_cell, mpm))
    for b in adj:
        A.append([b[0]+n_nodes*octant_idx, b[1]+n_nodes*octant_idx])
    octant_idx += 1

    # (+,-,-) octant
    pmm = octant.copy()
    pmm[:, 1] = -1.*pmm[:, 1]
    pmm[:, 2] = -1.*pmm[:, 2]
    full_cell = np.vstack((full_cell, pmm))
    for b in adj:
        A.append([b[0]+n_nodes*octant_idx, b[1]+n_nodes*octant_idx])
    octant_idx += 1

    # (-,-,-) octant
    mmm = -1*octant.copy()
    full_cell = np.vstack((full_cell, mmm))
    for b in adj:
        A.append([b[0]+n_nodes*octant_idx, b[1]+n_nodes*octant_idx])

    full_cell[full_cell == -0.] = 0.  # fix -zeros to 'standard' zeros

    return full_cell, A


def trim_cell(coords=None, A=None):
    """When building the cell by mirroring the first octant, some nodes get
    doubled. With this function we want to get rid of these nodes and remap the
    coordinates to a smaller dataset of non-redundant nodes.

    Parameters
    ----------
    coords : np.ndarray
        coordinates of shape (n_nodes,ndim).
    A : np.ndarray
        adjacency list of shape (n_bonds,2).

    Returns
    -------
    coords : np.ndarray
        coordinates of shape (n_nodes_new,ndim).
    adj : np.ndarray
        adjacency list of shape (n_bonds_new,2).

    """

    # Remove duplicate nodes
    unique_coords = np.unique(coords, axis=0)
    unique_A = None

    # Mapping dictionaries to associate old node labels with the new ones
    # idx -> (x,y,z)_old
    dic_uniq_idx_old_nodes = {}
    for i, node in enumerate(coords):
        dic_uniq_idx_old_nodes[i] = node

    # (x,y,z)_unique -> idx
    dic_uniq_node_idx = {}
    for i, node in enumerate(unique_coords):
        dic_uniq_node_idx[str(node)] = i  # XXX

    # Get new adjacency matrix
    unique_A = [[dic_uniq_node_idx[str(dic_uniq_idx_old_nodes[b[0]])],
                 dic_uniq_node_idx[str(dic_uniq_idx_old_nodes[b[1]])]] for b in A]
    unique_A = np.array(unique_A)
    unique_A = np.unique(unique_A, axis=0)

    # Non-connected nodes: remove nodes if not in unique adjacency matrix
    conn_coords = np.array([uc for uc in unique_coords if dic_uniq_node_idx[str(uc)] in unique_A])

    # (x,y,z)_conn -> idx
    dic_conn_nodes_idx = {}
    for i, node in enumerate(conn_coords):
        dic_conn_nodes_idx[str(node)] = i  # XXX

    # remap adjacency matrix to the new indexes
    conn_A = [[dic_conn_nodes_idx[str(unique_coords[b[0]])],
               dic_conn_nodes_idx[str(unique_coords[b[1]])]] for b in unique_A]
    conn_A = np.array(conn_A)

    return conn_coords, conn_A
