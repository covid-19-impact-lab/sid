from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def initial_states():
    return pd.read_csv(Path(__file__).resolve().parent / "test_states.csv")
