"""This module contains routines to validate inputs to functions."""
import numpy as np
import pandas as pd
from sid.config import BOOLEAN_STATE_COLUMNS
from sid.config import INDEX_NAMES
from sid.countdowns import COUNTDOWNS


def validate_params(params):
    """Validate the parameter DataFrame."""
    if not isinstance(params, pd.DataFrame):
        raise ValueError("params must be a DataFrame.")

    params = params.copy()
    if not (
        isinstance(params.index, pd.MultiIndex) and params.index.names == INDEX_NAMES
    ):
        raise ValueError(
            "params must have the index levels 'category', 'subcategory' and 'name'."
        )

    if np.any(params.index.to_frame().isna()):
        raise ValueError(
            "No NaNs allowed in the params index. Repeat the previous index level "
            "instead."
        )

    if params.index.duplicated().any():
        raise ValueError("No duplicates in the params index allowed.")

    if params["value"].isna().any():
        raise ValueError("The 'value' column of params must not contain NaNs.")

    cd_names = sorted(COUNTDOWNS)
    gb = params.loc[cd_names].groupby(INDEX_NAMES[:2])
    prob_sums = gb["value"].sum()
    problematic = prob_sums[~prob_sums.between(1 - 1e-08, 1 + 1e-08)].index.tolist()
    assert (
        len(problematic) == 0
    ), f"The following countdown probabilities don't add up to 1: {problematic}"

    first_levels = params.index.get_level_values("category")
    assort_prob_matrices = [
        x for x in first_levels if x.startswith("assortative_matching_")
    ]
    for name in assort_prob_matrices:
        meeting_prob = params.loc[name]["value"].unstack()
        assert len(meeting_prob.index) == len(
            meeting_prob.columns
        ), f"assortative probability matrices must be square but isn't for {name}."
        assert (
            meeting_prob.index == meeting_prob.columns
        ).all(), (
            f"assortative probability matrices must be square but isn't for {name}."
        )
        assert (meeting_prob.sum(axis=1) > 0.9999).all() & (
            meeting_prob.sum(axis=1) < 1.00001
        ).all(), (
            f"the meeting probabilities of {name} do not add up to one in every row."
        )


def validate_initial_states_and_infections(initial_states, initial_infections):
    if not isinstance(initial_states, pd.DataFrame):
        raise ValueError("initial_states must be a DataFrame.")

    if not isinstance(initial_infections, pd.Series):
        raise ValueError("initial_infections must be a pandas Series.")

    if not initial_infections.index.equals(initial_states.index):
        raise ValueError("initial_states and initial_infections must have same index.")


def validate_prepared_initial_states(states):
    if np.any(states.drop(columns="pending_test_date", errors="ignore").isna()):
        raise ValueError("'initial_states' are not allowed to contain NaNs.")

    for column in BOOLEAN_STATE_COLUMNS:
        if states[column].dtype != "bool":
            raise ValueError(f"Column '{column}' must be a boolean.")


def validate_models(
    contact_models,
    contact_policies,
    testing_demand_models,
    testing_allocation_models,
    testing_processing_models,
):
    """Check the user inputs."""
    if not isinstance(contact_models, dict):
        raise ValueError("contact_models must be a dictionary.")

    for cm_name, cm in contact_models.items():
        if not isinstance(cm, dict):
            raise ValueError(f"Each contact model must be a dictionary: {cm_name}.")

    if not isinstance(contact_policies, dict):
        raise ValueError("policies must be a dictionary.")

    for name, pol in contact_policies.items():
        if not isinstance(pol, dict):
            raise ValueError(f"Each policy must be a dictionary: {name}.")
        if "affected_contact_model" not in pol:
            raise KeyError(
                f"contact_policy {name} must have a 'affected_contact_model' specified."
            )
        model = pol["affected_contact_model"]
        if model not in contact_models:
            raise ValueError(f"Unknown affected_contact_model for {name}.")
        if "policy" not in pol:
            raise KeyError(f"contact_policy {name} must have a 'policy' specified.")

        # the policy must either be a callable or a number between 0 and 1.
        if not callable(pol["policy"]):
            if not isinstance(pol["policy"], (float, int)):
                raise ValueError(
                    f"The 'policy' entry of {name} must be callable or a number."
                )
            elif (pol["policy"] > 1.0) or (pol["policy"] < 0.0):
                raise ValueError(
                    f"If 'policy' is a number it must lie between 0 and 1. "
                    f"For {name} it is {pol['policy']}."
                )
            else:
                recurrent = contact_models[model]["is_recurrent"]
                assert not recurrent or pol["policy"] == 0.0, (
                    f"Specifying multipliers for recurrent models such as {name} for "
                    f"{pol['affected_contact_model']} will not change the contacts "
                    "of anyone because for recurrent models it is only checked"
                    "where the number of contacts is larger than 0. "
                    "This is unaffected by any multiplier other than 0"
                )

    for testing_model in [
        testing_demand_models,
        testing_allocation_models,
        testing_processing_models,
    ]:
        for name in testing_model:
            if not isinstance(testing_model[name], dict):
                raise ValueError(f"Each testing model must be a dictionary: {name}.")

            if "model" not in testing_model[name]:
                raise ValueError(
                    f"Each testing model must have a 'model' entry: {name}."
                )