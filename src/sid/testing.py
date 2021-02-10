"""This module holds the interface for the testing models."""
import itertools
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from sid.testing_allocation import allocate_tests
from sid.testing_allocation import update_pending_tests
from sid.testing_demand import calculate_demand_for_tests
from sid.testing_processing import process_tests


def perform_testing(
    date: pd.Timestamp,
    states: pd.DataFrame,
    params: pd.DataFrame,
    testing_demand_models: Dict[str, Dict[str, Any]],
    testing_allocation_models: Dict[str, Dict[str, Any]],
    testing_processing_models: Dict[str, Dict[str, Any]],
    seed: itertools.count,
    columns_to_keep: Optional[List[str]] = None,
):
    """Perform testing."""
    if columns_to_keep is None:
        columns_to_keep = []

    if testing_demand_models:
        demands_test, channel_demands_test = calculate_demand_for_tests(
            states,
            testing_demand_models,
            params,
            date,
            columns_to_keep,
            seed,
        )
        allocated_tests = allocate_tests(
            states, testing_allocation_models, demands_test, params, date, seed
        )

        states = update_pending_tests(states, allocated_tests)

        to_be_processed_tests = process_tests(
            states, testing_processing_models, params, date, seed
        )
    else:
        channel_demands_test = None
        to_be_processed_tests = None

    return states, channel_demands_test, to_be_processed_tests
