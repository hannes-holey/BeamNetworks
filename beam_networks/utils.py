import numpy as np
import warnings


def _mic(dr, boxsize, periodic):
    """Apply minimum image convention

    Parameters
    ----------
    dr : numpy.ndarray
        Edge vectors (n_elem, dim)
    boxsize : numpy.ndarray
        Box dimensions (dim,)
    periodic : numpy.ndarray
        PBC flags (bool) (dim,)

    Returns
    -------
    numpy.ndarray
        Corrected edge vectors
    """

    for i, (L, p) in enumerate(zip(boxsize, periodic)):
        if p:
            m = np.abs(dr[:, i]) > L / 2.
            dr[m, i] -= np.sign(dr[m, i]) * L

    return dr


def zero_pad_2d_array(arr):

    if arr.shape[1] == 2:
        return np.hstack([arr, np.zeros(arr.shape[0])[:, None]])
    else:
        return arr


def _dict_has_keys(d, required):
    """Check if dictionary has all required keys

    Parameters
    ----------
    d : dict
        Dictionary to test
    required : list
        Keys

    Returns
    -------
    bool
        True if all keys in required are in dictionary
    """

    return np.all([key in d.keys() for key in required])


def coupon_collector_bound(num_different_coupons: int, sureness: float = 1e-3) -> int:
    r"""A lower bound to how many coupons you need to buy to
    to be sure you will see all the coupon collection.

    bound derived from
    https://en.wikipedia.org/wiki/Coupon_collector%27s_problem

    $$
    P\left [ T > \beta n \log n \right ] = P \left [ \bigcup_i {Z}_i^{\beta n \log n} \right ]
    \le n \cdot P [ {Z}_1^{\beta n \log n} ] \le n^{-\beta + 1}
    $$

    Parameters
    ----------
    num_different_coupons : int
        Number of different coupon types
    sureness : float, optional
        Sureness (the default is 1e-3, which means 99.9% probability of seeing all coupons)

    Returns
    -------
    int
        Value of the lower bound
    """
    float_bound = num_different_coupons * \
        (np.log(num_different_coupons) - np.log(sureness))
    int_bound = int(float_bound) + 1
    return int_bound


def box_selection(nodes, lim):
    """Select nodes which are within a cuboid.

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    lim : iterable
        Lower and upper bounds of the cuboid

    Returns
    -------
    np.ndarray
        Masking array for nodes
    """

    if nodes.shape[1] == 2:
        _xlo, _ylo = np.amin(nodes, axis=0)
        _xhi, _yhi = np.amax(nodes, axis=0)

        xlo = _xlo if lim[0] is None else _xlo + lim[0] * (_xhi - _xlo)
        xhi = _xhi if lim[1] is None else _xlo + lim[1] * (_xhi - _xlo)
        ylo = _ylo if lim[2] is None else _ylo + lim[2] * (_yhi - _ylo)
        yhi = _yhi if lim[3] is None else _ylo + lim[3] * (_yhi - _ylo)

        mask_x = np.logical_and(nodes[:, 0] >= xlo, nodes[:, 0] <= xhi)
        mask_y = np.logical_and(nodes[:, 1] >= ylo, nodes[:, 1] <= yhi)

        mask_nodes = np.logical_and(mask_x, mask_y)

    else:
        _xlo, _ylo, _zlo = np.amin(nodes, axis=0)
        _xhi, _yhi, _zhi = np.amax(nodes, axis=0)

        xlo = _xlo if lim[0] is None else _xlo + lim[0] * (_xhi - _xlo)
        xhi = _xhi if lim[1] is None else _xlo + lim[1] * (_xhi - _xlo)
        ylo = _ylo if lim[2] is None else _ylo + lim[2] * (_yhi - _ylo)
        yhi = _yhi if lim[3] is None else _ylo + lim[3] * (_yhi - _ylo)
        zlo = _zlo if lim[4] is None else _zlo + lim[4] * (_zhi - _zlo)
        zhi = _zhi if lim[5] is None else _zlo + lim[5] * (_zhi - _zlo)

        mask_x = np.logical_and(nodes[:, 0] >= xlo, nodes[:, 0] <= xhi)
        mask_y = np.logical_and(nodes[:, 1] >= ylo, nodes[:, 1] <= yhi)
        mask_z = np.logical_and(nodes[:, 2] >= zlo, nodes[:, 2] <= zhi)

        mask_nodes = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask_nodes


def point_selection(nodes, point, num=1):
    """Select nodes which are closest to a given point.

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    point : array-like
        Coordinates of the point
    num : int
        Number of points. The default is 1.

    Returns
    -------
    np.ndarray
        Masking array for nodes
    """

    point = np.array(point)[None, :]
    distance = np.sqrt(np.sum((nodes - point)**2, axis=-1))

    node_mask = np.zeros(nodes.shape[0], dtype=bool)
    node_mask[np.argsort(distance)[:num]] = True

    return node_mask


def _ccw(A, B, C):
    """Check if points A, B, and C are counter-clockwise (2D only)

    Parameters
    ----------
    A : array-like
        Coordinates of point A
    B : array-like
        Coordinates of point B
    C : array-like
        Coordinates of point C

    Returns
    -------
    bool
        True if counter-clockwise
    """

    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C

    return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)


def _intersect(A, B, C, D):
    """Check if segments AB and CD intersect (in 2D)

    Parameters
    ----------
    A : array-like
        Coordinates of point A
    B : array-like
        Coordinates of point B
    C : array-like
        Coordinates of point C
    D : array-like
        Coordinates of point D

    Returns
    -------
    bool
        True if segments intersect
    """
    m_1 = _ccw(A, C, D) != _ccw(B, C, D)
    m_2 = _ccw(A, B, C) != _ccw(A, B, D)

    return np.logical_and(m_1, m_2)


def _threshold_bondlength(nodes, edges, threshold=None):
    """Select bondlengths smaller than a certain threshold


    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray
        Edge connectivity
    threshold : float, optional
        Bondlength thresokd (the default is None, which falls back to 50% of the maximum box dimension)

    Returns
    -------
    np.ndarray
        Masking array
    """

    if threshold is None:
        lengths = np.amax(nodes, axis=0) - np.amin(nodes, axis=0)
        threshold = 0.5 * np.amax(lengths)

    e0, e1 = edges
    bondlengths = np.sqrt(np.sum((nodes[e1, :] - nodes[e0, :])**2, axis=-1))

    return bondlengths < threshold


def _remove_isolated_nodes_edges(nodes, edges, max_depth=None):
    """Remove isolated nodes and dangling edges from a network.

    Parameters
    ----------
    nodes : np.ndarray
        Nodal coordinates
    edges : np.ndarray
        Edge connectivity
    max_depth : int, optional
        Maximum depth until which dangling bonds are removed (the default is None)

    Returns
    -------
    np.ndarray
        nodes
    np.ndarray
        edges
    """

    initial_nodes_size = nodes.shape[0]
    initial_edges_size = edges.shape[0]

    if max_depth is None:
        max_depth = initial_edges_size

    print(f"Pre-processing: reading input with {initial_nodes_size} nodes and {initial_edges_size} edges.")

    i = 0

    # Remove dangling bonds
    while i < max_depth:
        _, inv, counts = np.unique(edges, return_counts=True, return_inverse=True)
        tmp = (counts == 1)[inv].reshape(-1, 2)
        mask = np.any(tmp, axis=-1)
        edges = edges[~mask]

        if mask.sum() == 0:
            break
        i += 1

    # Remove isolated nodes and adjust edge indices
    unique_nodes_ids = np.unique(edges)
    isolated_nodes = np.setdiff1d(np.arange(nodes.shape[0]), unique_nodes_ids)
    new_nodes = nodes[unique_nodes_ids]

    new_edges = edges.copy()
    for node in isolated_nodes:
        new_edges[edges > node] -= 1

    nodes_diff = initial_nodes_size - new_nodes.shape[0]
    edges_diff = initial_edges_size - new_edges.shape[0]

    print(f"Pre-processing: removing {nodes_diff} isloated nodes and {edges_diff} dangling bonds.")

    return new_nodes, new_edges


def check_input_dict(container, keys, defaults, allowed):
    """Sanitize output settings


    Parameters
    ----------
    options : dict
        Output settings

    Returns
    -------
    dict
        Sanitized output settings
    """
    types = [type(d) for d in defaults]

    for k, t, d, a in zip(keys, types, defaults, allowed):
        if k in container.keys() and a is None:
            container[k] = t(container[k])
        elif k in container.keys() and container[k] in a:
            container[k] = t(container[k])
        else:
            warnings.warn(f"Invalid or missing option for '{k}'. Falling back to default ({d}).")
            container[k] = d

    return container


def get_edges_from_disks(file, cutoff=3., atol=1e-5):
    """Get adjacency matrix from a collextion of hard disks in 2D

    The input file has the format:

    L
    n_1, x_1, y_1, r_1
    n_2, x_2, y_2, r_2
    ...
    n_n, x_n, y_n, r_n

    where n_* denotes the disk index (1 based, will be shifted to zero),
    x_*, y_* are the coordinates,
    and r_* the radius of the disk

    Parameters
    ----------
    file : str
        Filename
    cutoff : float, optional
        Distance cutoff to check for possible neighbors (the default is 3.)
    atol : float, optional
        Absoulte tolerance for comparison between distance and
        composite radius of possibly neighboring disks (the default is 1e-5)

    Returns
    -------
    np.ndarray
        Nodes
    np.ndarray
        Edges
    float
        Lx
    float
        Ly
    """

    nodes = np.loadtxt(file, skiprows=1)
    edges = []

    with open(file) as f:
        Lx = float(f.readline())
    Ly = Lx

    # _, Lx, Ly, _ = np.amax(nodes, axis=0) - np.amin(nodes, axis=0)

    for i, node in enumerate(nodes):

        n0, x0, y0, r0 = node

        n_tmp, x_tmp, y_tmp, r_tmp = nodes[i+1:, :].T

        dx = (x_tmp - x0)
        dy = (y_tmp - y0)

        dx[dx > Lx / 2.] -= Lx
        dx[dx <= -Lx / 2.] += Lx

        dy[dy > Ly / 2.] -= Ly
        dy[dy <= -Ly / 2.] += Ly

        d = np.sqrt(dx**2 + dy**2)

        neigh_mask = d <= cutoff

        for n, x, y, r in zip(n_tmp[neigh_mask],
                              x_tmp[neigh_mask],
                              y_tmp[neigh_mask],
                              r_tmp[neigh_mask]):

            dx = (x - x0)
            dy = (y - y0)

            if dx > Lx / 2.:
                dx -= Lx
            if dx <= -Lx / 2.:
                dx += Lx

            if dy > Ly / 2.:
                dy -= Ly
            if dy <= -Ly / 2.:
                dy += Ly

            d = np.sqrt(dx**2 + dy**2)

            if np.isclose(d - (r + r0), 0., atol=atol, rtol=0.):
                edges.append([n0, n])

    edges = np.array(edges).astype(int)
    nodes = nodes[:, 1:3]

    edges -= 1

    return nodes, edges, Lx, Ly


def _reflect(nodes, edges, plane='x_min'):
    """Apply symmetry BCs and return nodes and edges of the full unfolded structure.

    Parameters
    ----------
    nodes : numpy.ndarray
        Nodal coordinates
    edges : numpy.ndarray
        Edges indices
    plane : str, optional
        String representation of the mirror plane, e.g. 'x_min', 'y_max' (the default is 'x_min',
        which reflects in negative x direction at the minimum coordinate)

    Returns
    -------
    np.ndarray
        Nodes
    np.ndarray
        Edges
    """
    _axis, _loc = plane.split('_')
    axis = {'x': 0, 'y': 1, 'z': 2}[_axis]

    if _loc == 'min':
        loc = np.min(nodes[:, axis])
    elif _loc == 'max':
        loc = np.max(nodes[:, axis])
    else:
        loc = 0.

    # copy nodes which are not on symmetrie plane
    to_reflect = np.invert(np.isclose(nodes[:, axis], loc))
    to_reflect_ids = np.arange(nodes.shape[0])[to_reflect]
    to_reflect_ids_ctd = np.arange(to_reflect.sum())

    new_nodes = nodes[to_reflect].copy()
    new_nodes[:, axis] -= loc
    new_nodes[:, axis] *= -1
    new_nodes[:, axis] += loc

    nodes = np.vstack([nodes, new_nodes])

    new_edges = edges[:].copy()

    shift = len(nodes) - len(new_nodes)

    new_edges = []

    for n0, i0 in zip(to_reflect_ids, np.arange(len(to_reflect_ids))):
        # reflected node is left node
        for n1 in edges[edges[:, 0] == n0, 1]:
            # right node is also reflected
            if n1 in to_reflect_ids:
                new_edges.append([i0 + shift, to_reflect_ids_ctd[to_reflect_ids == n1][0] + shift])
            # right node is not reflected
            else:
                new_edges.append([i0 + shift, n1])

        # reflected node is right node
        for n1 in edges[edges[:, 1] == n0, 0]:
            # left node is not reflected
            if n1 not in to_reflect_ids:
                new_edges.append([n1, i0 + shift])

    new_edges = np.array(new_edges)
    new_edges[new_edges[:, 0] > new_edges[:, 1]] = new_edges[new_edges[:, 0] > new_edges[:, 1], ::-1]

    edges = np.vstack([edges, new_edges])

    return nodes, edges
