import numpy as np
from beam_networks.utils import _remove_isolated_nodes_edges


def test_linear_chain():

    _x = np.linspace(0., 1., 5)
    nodes = np.vstack([_x, np.zeros_like(_x)]).T
    _xid = np.arange(4)
    _yid = np.arange(1, 5)
    edges = np.vstack([_xid, _yid]).T

    new_nodes, new_edges = _remove_isolated_nodes_edges(nodes, edges, max_depth=1)

    np.testing.assert_array_equal(new_nodes, nodes[1:-1])
    np.testing.assert_array_equal(new_edges, edges[:2])

    new_nodes, new_edges = _remove_isolated_nodes_edges(nodes, edges, max_depth=2)

    np.testing.assert_array_equal(new_nodes, np.empty(shape=(0, 2)))
    np.testing.assert_array_equal(new_edges, np.empty(shape=(0, 2)))


def test_cube():
    """

    +
    |
    +     +
    |    /
    +---+           +---+
    |   |   ==>     |   |
    +---+           +---+


    """

    nodes = np.array([[0., 0.],  # 0
                      [1., 0.],  # 1
                      [1., 1.],  # 2
                      [0., 1.],  # 3
                      [1., 1.],  # 4
                      [2., 2.],  # 5
                      [0., 2.],  # 6
                      [0., 3.],  # 7
                      ])

    edges = np.array([[0, 1],
                      [0, 3],
                      [1, 2],
                      [2, 3],
                      [2, 5],
                      [3, 6],
                      [3, 7],
                      ])

    new_nodes, new_edges = _remove_isolated_nodes_edges(nodes, edges)

    np.testing.assert_array_equal(new_nodes, nodes[:4])
    np.testing.assert_array_equal(new_edges, edges[:4])
