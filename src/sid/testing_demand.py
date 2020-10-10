"""Contains the code for calculating the demand for tests."""
import numpy as np
import pandas as pd
from sid.shared import random_choice


def calculate_demand_for_tests(states, testing_demand_models, params, seed):
    """Calculate the demand for tests.

    The following is a three-staged process:

    1. Compute the probability for each demand model that an individual demands a test.
    2. Sample which individuals demand a test.
    3. For those demanding a test, sample why they want a test.

    Sampling whether an individual requests any test at all and, then, sampling the
    reason for the test is computationally beneficial in contract to sampling with the
    probability of each demand model the demand and the reason. The first approach
    always involves two steps whereas the complexity of the latter increases with the
    number of demand models.

    Args:
        states (pandas.DataFrame): The states of all individuals.
        testing_demand_models (dict): A dictionary containing the demand models for
            testing.
        params (pandas.DataFrame): The parameter DataFrame.
        seed (itertools.count): The seed counter.

    Returns:
        (tuple): Tuple containing.

            - demands_test (pandas.Series): A boolean series indicating which person
              demands a test.
            - demands_test_reason (pandas.Series): A series indicating the demand model
              which made the individual ask for a test.

    """
    demand_probabilities = _calculate_demand_probabilities(
        states, testing_demand_models, params
    )

    demands_test = _sample_which_individuals_demand_a_test(demand_probabilities, seed)
    demands_test_reason = _sample_reason_for_demanding_a_test(
        demand_probabilities, demands_test, seed
    )

    return demands_test, demands_test_reason


def _calculate_demand_probabilities(states, testing_demand_models, params):
    """Calculate the demand probabilities for each test demand model.

    Args:
        states (pandas.DataFrame): The states of all individuals.
        testing_demand_models (dict): A dictionary containing the demand models for
            testing.
        params (pandas.DataFrame): The parameter DataFrame.

    Returns:
        demand_probabilities (pandas.DataFrame): Contains for each individual and every
            demand model the probability that the individual will request a test.

    """
    demand_probabilities = pd.DataFrame(index=states.index)
    for name, model in testing_demand_models.items():
        loc = model.get("loc", params.index)
        func = model["model"]

        demand_probabilities[name] = func(states, params.loc[loc])

    return demand_probabilities


def _sample_which_individuals_demand_a_test(demand_probabilities, seed):
    """Sample which individuals demand a test.

    At first, compute the probabilities that each individual will demand no test at all
    and the corresponding probability that an individual demands at least one test.

    Then, sample individuals which demand a test.

    Args:
        demand_probabilities (pandas.DataFrame): Contains for each individual and every
            demand model the probability that the individual will request a test.
        seed (itertools.count): The seed counter.

    Returns:
        demands_test (pandas.Series): A boolean series indicating individuals who demand
            a test.

    """
    np.random.seed(next(seed))

    probability_demands_no_test = (1 - demand_probabilities).prod(axis=1)
    probability_demands_any_test = 1 - probability_demands_no_test
    probabilities = np.column_stack(
        (probability_demands_any_test, probability_demands_no_test)
    )

    demands_test = random_choice([True, False], probabilities)
    demands_test = pd.Series(index=demand_probabilities.index, data=demands_test)

    return demands_test


def _sample_reason_for_demanding_a_test(demand_probabilities, demands_test, seed):
    """Sample reason for demanding a test.

    Args:
        demand_probabilities (pandas.DataFrame): Contains for each individual and every
            demand model the probability that the individual will request a test.
        demands_test (pandas.Series): A boolean series indicating individuals who demand
            a test.
        seed (itertools.count): The seed counter.

    Returns:
        demands_test_reason (pandas.Series): Shows which demand model caused the bid.

    """
    np.random.seed(next(seed))

    normalized_probabilities = _normalize_probabilities(
        demand_probabilities[demands_test]
    )
    sampled_reason = random_choice(
        demand_probabilities.columns.tolist(), normalized_probabilities
    )

    demands_test_reason = pd.Series(index=demands_test.index, data="")
    demands_test_reason.loc[demands_test] = sampled_reason

    return demands_test_reason


def _normalize_probabilities(probabilities):
    """Normalize probabilities such that they sum to one.

    Args:
        probabilities (pandas.DataFrame): Contains probabilities for each individual and
            demand model to request a test.

    Returns:
        (pandas.Series): A series with normalized probabilities summing up to one.

    """
    return probabilities.divide(probabilities.sum(axis=1), axis=0)
