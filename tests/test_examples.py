import os
import pytest
import subprocess


# @pytest.mark.slow
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
