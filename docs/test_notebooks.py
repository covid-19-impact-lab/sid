import os
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.mark.optional
def test_notebooks():
    """Run the simulation notebook.

   source: https://nbconvert.readthedocs.io/en/latest/execute_api.html

   """
    repo_path = Path(__file__).resolve().parent.parent
    tutorials_path = repo_path / "docs" / "source" / "tutorials"
    ipynbs = list(tutorials_path.glob("*.ipynb"))
    for nb_name in ipynbs:
        with open(tutorials_path / nb_name) as f:
            notebook = nbformat.read(f, as_version=4)
    os.chdir(tutorials_path)
    ExecutePreprocessor(timeout=600).preprocess(notebook)
