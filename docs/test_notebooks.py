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
    assert True is False, "test_notebooks ran!"
