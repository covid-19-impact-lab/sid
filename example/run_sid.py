import numpy as np
import pandas as pd

from sid import simulate

if __name__ == "__main__":
    params = pd.read_csv("../sid/params.csv").set_index(["category", "name"])
    initial_states = pd.read_csv("initial_states.csv")
    initial_infections = pd.Series([True] * 5 + [False] * 10)
    cm = {
        "contact_type": "standard",
        "loc": params.index,
        "model": "meet_two_people",
    }
    contact_models = {"work_contacts": cm}
    policies = {}
    n_periods = 5
    assort_by = ["age_group", "region"]

    np.random.seed(123)
    res = simulate(
        params=params,
        initial_states=initial_states,
        initial_infections=initial_infections,
        contact_models=contact_models,
        policies=policies,
        n_periods=n_periods,
        assort_by=assort_by,
    )

    pd.to_pickle(res, "res.pickle")
