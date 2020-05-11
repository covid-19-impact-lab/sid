import os
from pathlib import Path

import pytest


@pytest.mark.optional
def test_notebooks():
    """Run the simulation notebook.

    source: https://nbconvert.readthedocs.io/en/latest/execute_api.html

    """

    repo_path = Path(__file__).resolve().parent.parent
    tutorials_path = repo_path / "docs" / "source" / "tutorials"
    os.chdir(tutorials_path)

    ipynbs = list(tutorials_path.glob("*.ipynb"))
    # Remove checkpoints.
    ipynbs = [ipynb for ipynb in ipynbs if ".ipynb_checkpoints" not in ipynb.as_posix()]

    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    for nb_name in ipynbs:
        with open(nb_name) as f:
            notebook = nbformat.read(f, as_version=4)
        ExecutePreprocessor(timeout=600).preprocess(notebook)
