import numpy as np
import pandas as pd

from sid.simulate import _process_initial_states
from sid.simulate import _process_simulation_results


def test_process_data_w_index():
    df = pd.DataFrame(
        data=np.arange(9).reshape(3, 3), columns=["index", "a", "b"]
    ).set_index("index")

    states, index_names = _process_initial_states(df, [])

    assert isinstance(states.index, pd.RangeIndex)
    assert "index" in states

    to_concat = []
    for period in range(2):
        to_concat.append(states.copy().assign(period=period))

    simulation_results = _process_simulation_results(
        to_concat, index_names, {"column_name": "period"}
    )

    assert simulation_results.index.names == ["period", "index"]


def test_process_data_w_multiindex():
    df = pd.DataFrame(
        data=np.arange(9).reshape(3, 3), columns=["index_a", "index_b", "a"]
    ).set_index(["index_a", "index_b"])

    states, index_names = _process_initial_states(df, [])

    assert isinstance(states.index, pd.RangeIndex)
    assert all(col in states.columns for col in ["index_a", "index_b", "a"])

    to_concat = []
    for period in range(2):
        to_concat.append(states.copy().assign(period=period))

    simulation_results = _process_simulation_results(
        to_concat, index_names, {"column_name": "period"}
    )

    assert simulation_results.index.names == ["period", "index_a", "index_b"]
