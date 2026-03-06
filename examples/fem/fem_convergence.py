#
# Copyright 2025-2026 Hannes Holey
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

import os
import numpy as np
import matplotlib.pyplot as plt

from beam_networks.problem import BeamNetwork
from beam_networks.geometry.geo import get_geometric_props
from beam_networks.reference_solutions.cantilever import cantilever_analytic


PROPS = {'name': 'circle', 'radius': 0.05, 'E': 1., 'nu': 0.3}


def _cantilever_2d(num_nodes, length, options):
    x = np.linspace(0., length, num_nodes)
    nodes = np.column_stack([x, np.zeros(num_nodes)])
    edges = np.column_stack([np.arange(num_nodes - 1), np.arange(1, num_nodes)])
    return BeamNetwork(nodes, edges, beam_prop=PROPS, valid=True, options=options)


def _disp_fem(length, num_nodes, density, fem_poly_order, fem_n_gauss):
    """Two-node cantilever with transverse tip load; return uy at tip."""
    _, Iz, _, A, kappa, _ = get_geometric_props(PROPS)
    P = 0.05 * PROPS['E'] * Iz / length**3

    opts = {'vectorize': True,
            'matrix': 'dense',
            'verbose': False,
            'n_elem_per_length': density,
            'fem_poly_order': fem_poly_order,
            'fem_n_gauss': fem_n_gauss,
            }

    net = _cantilever_2d(num_nodes, length, opts)
    net.add_BC('fix', 'D', 'point', [0., 0.], [0., 0., 0.])
    net.add_BC('load', 'N', 'point', [length, 0.], [0., P, 0.])
    net.solve()

    return net.nodes[:, 0], net.displacement, net.rotation


def plot_p_convergence():

    # Analytic result
    length = 50.
    _, Iz, _, _, _, _ = get_geometric_props(PROPS)
    P = 0.05 * PROPS['E'] * Iz / length**3
    x = np.linspace(0., length, 100)
    _, uy_ref, phi_ref, _ = cantilever_analytic(x, length, [0., P, 0.], 1.0, PROPS)

    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex='col')

    ax[0, 0].plot(x, uy_ref, '--', color='0.0')
    ax[1, 0].plot(x, phi_ref, '--', color='0.0')

    num_nodes = 2
    n_sub = 1

    for i, red in enumerate([True, False]):

        errors_u = []
        errors_phi = []

        density = n_sub / length
        ps = [1, 2, 3]
        for p in ps:

            # reduced_integration
            n = p if red else p + 1

            x_fem, u_fem, phi_fem = _disp_fem(length, num_nodes, density, p, n)
            uy_fem = u_fem[:, 1]

            if red:
                ax[0, 0].plot(x_fem, uy_fem)
                ax[1, 0].plot(x_fem, phi_fem)

            errors_u.append(np.abs(uy_ref[-1] - uy_fem[-1]))
            errors_phi.append(np.abs(phi_ref[-1] - phi_fem[-1]))

        ax[0, i+1].plot(ps, errors_u)
        ax[1, i+1].plot(ps, errors_phi)

    fig.suptitle('p refinement')

    ax[0, 0].set_title('Cantilver')
    ax[0, 0].set_ylabel(r'$u$')
    ax[1, 0].set_ylabel(r'$\theta$')
    ax[1, 0].set_xlabel(r'$x$')

    ax[0, 1].set_title('Reduced integration (p)')
    ax[0, 1].set_ylabel(r'$|\Delta u|$')
    ax[1, 1].set_ylabel(r'$|\Delta \theta|$')
    ax[0, 1].set_yscale('log')
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_xlabel('p')

    ax[0, 2].set_title('Exact integration (p+1)')
    ax[0, 2].set_ylabel(r'$|\Delta u|$')
    ax[1, 2].set_ylabel(r'$|\Delta \theta|$')
    ax[0, 2].set_yscale('log')
    ax[1, 2].set_yscale('log')
    ax[1, 2].set_xlabel('p')


def plot_h_convergence():

    # Analytic result
    length = 50.
    _, Iz, _, _, _, _ = get_geometric_props(PROPS)
    P = 0.05 * PROPS['E'] * Iz / length**3
    x = np.linspace(0., length, 100)
    _, uy_ref, phi_ref, _ = cantilever_analytic(x, length, [0., P, 0.], 1.0, PROPS)

    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex='col')

    ax[0, 0].plot(x, uy_ref, '--', color='0.0')
    ax[1, 0].plot(x, phi_ref, '--', color='0.0')

    num_nodes = 3
    p = 1

    for i, red in enumerate([True, False]):

        # reduced_integration
        n = p if red else p + 1

        errors_u = []
        errors_phi = []

        n_subs = [1, 5, 10, 100]
        for n_sub in n_subs:
            density = n_sub / length

            x_fem, u_fem, phi_fem = _disp_fem(length, num_nodes, density, p, n)
            uy_fem = u_fem[:, 1]

            if red:
                ax[0, 0].plot(x_fem, uy_fem, '-x')
                ax[1, 0].plot(x_fem, phi_fem, '-x')

            errors_u.append(np.abs(uy_ref[-1] - uy_fem[-1]))
            errors_phi.append(np.abs(phi_ref[-1] - phi_fem[-1]))

        ax[0, i+1].plot(n_subs, errors_u)
        ax[1, i+1].plot(n_subs, errors_phi)

    fig.suptitle(f'h refinement (p={p})')

    ax[0, 0].set_title('Cantilver')
    ax[0, 0].set_ylabel(r'$u$')
    ax[1, 0].set_ylabel(r'$\theta$')
    ax[1, 0].set_xlabel(r'$x$')

    ax[0, 1].set_title('Reduced integration (p)')
    ax[0, 1].set_ylabel(r'$|\Delta u|$')
    ax[1, 1].set_ylabel(r'$|\Delta \theta|$')
    ax[0, 1].set_yscale('log')
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_xlabel('n_sub')

    ax[0, 2].set_title('Exact integration (p+1)')
    ax[0, 2].set_ylabel(r'$|\Delta u|$')
    ax[1, 2].set_ylabel(r'$|\Delta \theta|$')
    ax[0, 2].set_yscale('log')
    ax[1, 2].set_yscale('log')
    ax[1, 2].set_xlabel('n_sub')


if __name__ == "__main__":

    plt.style.use(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'beams.mplstyle'))
    plot_h_convergence()
    plot_p_convergence()

    plt.show()
