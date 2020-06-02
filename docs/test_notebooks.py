import os
from pathlib import Path

import pytest


def find_notebooks():
    repo_path = Path(__file__).resolve().parent.parent
    src_path = repo_path / "docs" / "source"

    ipynbs = list(src_path.glob("*/*.ipynb"))
    # Remove checkpoints.
    ipynbs = [ipynb for ipynb in ipynbs if ".ipynb_checkpoints" not in ipynb.as_posix()]
    return ipynbs


NOTEBOOKS = find_notebooks()


@pytest.mark.optional
@pytest.mark.parametrize("nb_path", NOTEBOOKS)
def test_notebooks(nb_path):
    """Run the simulation notebook.

    source: https://nbconvert.readthedocs.io/en/latest/execute_api.html

    """
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    os.chdir(nb_path.parent)
    with open(nb_path) as f:
        notebook = nbformat.read(f, as_version=4)
    ExecutePreprocessor(timeout=1200).preprocess(notebook)
