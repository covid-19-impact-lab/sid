from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sid.config import INDEX_NAMES


@pytest.fixture(autouse=True)
def _patch_doctest_namespace(doctest_namespace):
    """Patch the namespace for doctests.

    This function adds some packages to namespace of every doctest.

    """
    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd


@pytest.fixture
def initial_states():
    return pd.read_csv(Path(__file__).resolve().parent / "test_states.csv").astype(
        "category"
    )


@pytest.fixture
def params():
    return pd.read_csv(
        Path(__file__).resolve().parent / "test_params.csv",
        index_col=INDEX_NAMES,
    )
