from pathlib import Path

import pandas as pd
import pytest
from sid.config import INDEX_NAMES


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
