"""This module contains routines to validate inputs to functions."""
import inspect
import warnings
from typing import Callable
from typing import List

import numpy as np
import pandas as pd
from sid.config import BOOLEAN_STATE_COLUMNS
from sid.config import INDEX_NAMES
from sid.countdowns import COUNTDOWNS_WITH_DRAWS
from sid.time import get_date


COMMON_ARGS = ("states", "params", "seed")

NECESSARY_CONTACT_MODEL_KEYS = ("is_recurrent", "model", "assort_by")
NECESSARY_CONTACT_POLICY_KEYS = ("policy",)


def validate_params(params: pd.DataFrame) -> None:
    """Validate the parameter DataFrame."""
    if not isinstance(params, pd.DataFrame):
        raise ValueError("params must be a DataFrame.")

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

    cd_names = sorted(COUNTDOWNS_WITH_DRAWS)
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

    try:
        relative_limit = params.loc[
            ("health_system", "icu_limit_relative", "icu_limit_relative"), "value"
        ]
    except KeyError:
        warnings.warn(
            "A limit of ICU beds is not specified in 'params'. Individuals who need "
            "intensive care will decease immediately.\n\n"
            "Set ('health_system', 'icu_limit_relative', 'icu_limit_relative') in "
            "'params' to beds per 100,000 individuals to silence the warning."
        )
    else:
        if relative_limit < 1:
            warnings.warn("The limit for ICU beds per 100,000 individuals is below 1.")


def validate_initial_states(initial_states):
    if not isinstance(initial_states, pd.DataFrame):
        raise ValueError("initial_states must be a DataFrame.")

    if np.any(initial_states.isna()):
        raise ValueError("'initial_states' are not allowed to contain NaNs.")


def validate_prepared_initial_states(states, duration):
    columns_with_nans = ["pending_test_date", "virus_strain"]
    if np.any(states.drop(columns=columns_with_nans, errors="ignore").isna()):
        raise ValueError("'initial_states' are not allowed to contain NaNs.")

    for column in BOOLEAN_STATE_COLUMNS:
        if states[column].dtype != "bool":
            raise ValueError(f"Column '{column}' must be a boolean.")

    end_previous_simulation = get_date(states)
    new_start_simulation = end_previous_simulation + pd.Timedelta(1, unit="day")
    if not new_start_simulation == duration["start"]:
        raise ValueError(
            "The resumed simulation does not start where the former ended. The former "
            f"ended on {end_previous_simulation.date()} and should be continued on "
            f"{new_start_simulation.date()}, but the specified 'duration' starts "
            f"{duration['start'].date()}."
        )


def validate_contact_models(contact_models):
    """Validate contact models and policies."""
    if not isinstance(contact_models, dict):
        raise ValueError("contact_models must be a dictionary.")

    for name, model in contact_models.items():
        if not isinstance(model, dict):
            raise ValueError(f"Each contact model must be a dictionary: {name}.")

        missing_keys = set(NECESSARY_CONTACT_MODEL_KEYS) - set(model)
        if missing_keys:
            raise ValueError(
                f"Contact model {name} is missing the following keys: {missing_keys}"
            )

        if model["is_recurrent"] and "assort_by" not in model:
            raise ValueError(
                f"{name} is a recurrent contact model without an assort_by."
            )

        _validate_model_function(
            name, "contact_models", model.get("model"), COMMON_ARGS
        )


def validate_contact_policies(contact_policies, contact_models):
    if not isinstance(contact_policies, dict):
        raise ValueError("'contact_policies' must be a dictionary.")

    for name, policy in contact_policies.items():
        if not isinstance(policy, dict):
            raise ValueError(f"Contact policy '{name}' is not a dictionary.")

        missing_keys = set(NECESSARY_CONTACT_POLICY_KEYS) - set(policy)
        if missing_keys:
            raise ValueError(
                f"The contact policy '{name}' is missing the following keys: "
                f"{missing_keys}."
            )

        affected_model = policy.get("affected_contact_model")
        if affected_model is not None and affected_model not in contact_models:
            raise ValueError(
                f"The contact policy '{name}' affects the contact model "
                f"'{affected_model}' which is unknown."
            )

        if callable(policy["policy"]):
            _validate_model_function(
                name, "contact_policies", policy["policy"], COMMON_ARGS + ("contacts",)
            )
        elif isinstance(policy["policy"], (float, int)):
            if contact_models[affected_model]["is_recurrent"]:
                if policy["policy"] != 0:
                    raise ValueError(
                        f"Specifying multipliers for recurrent models such as {name} "
                        f"for {policy['affected_contact_model']} will not change the "
                        "contacts of anyone because for recurrent models it is only "
                        "checked where the number of contacts is larger than 0. "
                        "This is unaffected by any multiplier other than 0"
                    )
            else:
                if not 0 <= policy["policy"] <= 1:
                    raise ValueError(
                        f"The policy of contact policy '{name}' is not a number in "
                        "[0, 1]."
                    )
        else:
            raise ValueError(
                f"The 'policy' entry of contact policy '{name}' must be callable or "
                "a number in [0, 1]."
            )


def validate_testing_models(
    testing_demand_models, testing_allocation_models, testing_processing_models
):
    """Validate models for testing."""
    for group_name, testing_model, args in [
        ("testing_demand_models", testing_demand_models, COMMON_ARGS),
        (
            "testing_allocation_models",
            testing_allocation_models,
            ("n_allocated_tests", "demands_test") + COMMON_ARGS,
        ),
        (
            "testing_processing_models",
            testing_processing_models,
            ("n_to_be_processed_tests",) + COMMON_ARGS,
        ),
    ]:
        if not isinstance(testing_model, dict):
            raise ValueError(f"'{group_name}' must be a dictionary.")

        for name, model in testing_model.items():
            if not isinstance(model, dict):
                raise ValueError(
                    f"Each model of '{group_name}' must be a dictionary: {name}."
                )

            _validate_model_function(name, group_name, model.get("model"), args)


def validate_vaccination_models(vaccination_models):
    """Validate vaccination models."""
    if not isinstance(vaccination_models, dict):
        raise ValueError("'vaccination_models' must be a dictionary.")

    for name, model in vaccination_models.items():
        if not isinstance(model, dict):
            raise ValueError(f"Vaccination model '{name}' is not a dictionary.")

        _validate_model_function(
            name,
            "vaccination_models",
            model.get("model"),
            ("receives_vaccine",) + COMMON_ARGS,
        )


def validate_return_is_series_or_ndarray(x, model_name, model_group, index):
    if isinstance(x, (pd.Series, np.ndarray)):
        return pd.Series(data=x, index=index)
    else:
        raise ValueError(
            f"The model '{model_name}' of '{model_group}' does not return a "
            f"pandas.Series or a numpy.ndarray, but {x} has type {type(x)}."
        )


def _validate_model_function(
    model_name: str, model_group: str, model: Callable, args: List[str]
) -> None:
    if not callable(model):
        raise TypeError(
            f"The model '{model_name}' of '{model_group}' is not a callable, but "
            f"{model}."
        )

    signature = inspect.signature(model)
    missing_arguments = set(args) - set(signature.parameters)

    if missing_arguments:
        raise ValueError(
            f"The model '{model_name}' of '{model_group}' is missing some mandatory "
            f"arguments: {missing_arguments}."
        )
