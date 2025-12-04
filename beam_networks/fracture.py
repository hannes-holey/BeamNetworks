#
# Copyright 2025 Hannes Holey
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
import pandas as pd

from .problem import BeamNetwork
from .assembly import assemble_global_system


class FractureProblem(BeamNetwork):

    def __init__(self, *args, save_trajectory=False, no_output=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_trajectory = save_trajectory

        keys = ['fracture_energy',
                'stress_strain',
                'avalanche_size',
                'removed_edges',
                'cracked_edges']

        self._initialize_output_buffer(keys)

    @property
    def removed_edges(self):
        """List of inactive edges.

        Returns
        -------
        list
            Inactive edges indices
        """
        return np.arange(len(self._active_edges))[np.invert(self._active_edges)]

    def _initialize_output_buffer(self, keys, values=None):
        """Initialize dictionary to store and later save output.

        Parameters
        ----------
        keys : list of str
            Names of output variables
        values : list, optional
            Values to fill at initialization (the default is None, which initializes empty lists)
        """
        self._output = {}

        if values is None:
            values = [[] for _ in range(len(keys))]

        for key, value in zip(keys, values):
            self._output[key] = value

    def failure_criterion(self, name='vM'):

        assert name in ['vM', 'pS', 'pE']

        pS = self.compute_principal_stresses()

        if name == 'vM':
            f = np.sqrt(0.5 * ((pS[:, 0] - pS[:, 1])**2 + (pS[:, 1] - pS[:, 2])**2 + (pS[:, 2] - pS[:, 0])**2))
        elif name == 'pS':
            f = pS[:, 0]
        elif name == 'pE':
            f = pS[:, 0] - self._beam_prop['nu'] * (pS[:, 1] + pS[:, 2])

        return f

    def run(self,
            mode='cascade',
            sign=-1,
            dist=None,
            fail_crit='vM',
            no_output=False):

        i = 0

        # Generic filename
        def fname(i):
            return os.path.join(f'fracture-{i:04d}.vtk')

        # Store backup
        K_before = self._K.copy()
        edges_before = self._active_edges.copy()

        # Initialize
        s = 0
        buffer = [[0., 0.]]
        avalanche_size = []
        cracked_edges = []

        # Get first stress distribution
        u = sign * 1. * self.Ly
        self.modify_BC('1', [None, u, None])
        self.solve(stress_mode='max')

        E = self._beam_prop.get('E')
        fail = self._beam_prop.get('strength', 0.1 * E)

        if dist is None:
            failure_threshold = fail * np.ones_like(self._active_edges)
        else:
            failure_threshold = fail * dist

        overstress = self.failure_criterion(name=fail_crit) / failure_threshold[self._active_edges]
        rescale = True

        while self.is_connected:
            if rescale:  # BC
                # Rescale displacement according to current max. overstressed beam
                factor = 1. / overstress.max()
                u *= factor
                self.scale_BC('D', factor)

            self.solve(stress_mode='max')

            # Reaction forces
            Fy1 = self.Freact['1']

            overstress = self.failure_criterion(name=fail_crit) / failure_threshold[self._active_edges]
            active_edge_ids = np.arange(self._edges.shape[0])[self._active_edges]
            index_max_stressed_edge = active_edge_ids[np.argmax(overstress)]

            if self.save_trajectory or i == 0:
                self.to_vtk(fname(i))
                i += 1

            # Save stress and strain data
            Fy1 = self.Freact['1']
            # Save stress and strain data
            buffer.append([sign * u / self.Ly, sign * Fy1.sum() / self.Lx])

            if mode == "adiabatic":
                if np.isclose(overstress.max(), 1.):
                    # break bond
                    self._crack_edge(index_max_stressed_edge)
                    cracked_edges.append(index_max_stressed_edge)
                    rescale = True
                else:
                    rescale = True

            else:
                if overstress.max() - 1. > -1e-12:
                    # break bonds as long as no bond exceeds the critical Mises stress
                    self._crack_edge(index_max_stressed_edge)
                    cracked_edges.append(index_max_stressed_edge)
                    s += 1
                    rescale = False
                else:
                    # Save and reset avalanche size
                    if s > 0:
                        avalanche_size.append(s)
                    s = 0

                    rescale = True

        self._sVM = np.zeros(self.edges.shape[0])

        self.to_vtk(fname(i))

        buffer = np.array(buffer)
        self._output['removed_edges'].append(self.removed_edges)
        self._output['stress_strain'].append(buffer)
        self._output['avalanche_size'].append(np.array(avalanche_size))
        self._output['cracked_edges'].append(np.array(cracked_edges))

        if self.has_solution:
            fracture_energy = self._get_energy(buffer, p=0.5)
            self._output['fracture_energy'].append(fracture_energy)
        else:
            self._output['fracture_energy'].append(np.nan)

        # Restore previous network
        self._K = K_before
        self._active_edges = edges_before

    def write(self):
        """Wrapper for different output writers. Currently only calls HDF5 output."""
        self._write_hdf5()

    def _get_energy(self, data, p=0.0):

        strain, stress = data.T

        if len(strain) < 2:
            return 0.
        elif len(strain) == 2:
            return 0.5 * strain[1] * stress[1]
        else:
            init_stiffness = stress[1] / strain[1]
            stress_p_init = p * init_stiffness * strain

            if p > 0.05:
                xarg = np.argmin(np.abs(stress[1:] - stress_p_init[1:]))
                xarg += 1
            else:
                xarg = -1

            E_frac_p = np.trapezoid(stress[:xarg], x=strain[:xarg])

            return E_frac_p

    def _crack_edge(self, index):
        """Flip edge without rescaling beam cross sections

        This just adds or subtracts parts of the global stiffness matrix.

        Parameters
        ----------
        index : int, or list of ints
            Indices to flip
        """

        if isinstance(index, np.int64) or isinstance(index, int):
            index = [index, ]

        for i in index:
            if self._active_edges[i]:
                # Remove bond
                self._active_edges[i] = False
                self._K -= assemble_global_system(self._nodes,
                                                  self._edges[i][None, :],
                                                  self._edge_vectors[i][None, :],
                                                  self._beam_prop,
                                                  vectorize=self._options['vectorize'],
                                                  matrix=self._options['matrix'],
                                                  verbose=False)

    def _write_hdf5(self):
        """Write output buffer to HDF5 file.
        """

        fname = os.path.join(self._outdir, 'data.h5')
        if os.path.exists(fname):
            os.rename(fname, fname + '.bak')

        df = pd.DataFrame(data=self._output)
        df.to_hdf(fname, key='data')

    def _pop_output_buffer(self, index=-1):
        """Pop index from output buffer.

        Parameters
        ----------
        index : int, optional
            Index to pop (the default is -1, which removes the last added item)
        """

        for key in self._output.keys():
            self._output[key].pop(index)
