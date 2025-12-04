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
import pytest
import subprocess


@pytest.mark.parametrize('example_directory', [
    'first_steps',
    'geometry',
])
def test_examples_in_subdir(tmp_path, example_directory):

    test_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path. join(test_path, '..', 'examples', example_directory)
    examples = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.py')])

    cmds = [['python', file, '-o', tmp_path] for file in examples]

    for cmd in cmds:
        subprocess.run(cmd, shell=False, check=True)
