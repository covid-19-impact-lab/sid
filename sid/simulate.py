def simulate(
    params, initial_states, contact_models, policies, n_periods,
):
    """Simulate the spread of covid-19.

    Args:
        params (pd.DataFrame): DataFrame with parameters that influence the number of
            contacts, contagiousness and dangerousness of the disease, ...
        initial_states (pd.DataFrame): See :ref:`states`
        contact_models (list): List of dictionaries where each dictionary describes a
            channel by which contacts can be formed. See :ref:`contact_models`.
        policies (list): List of dictionaries with contact and testing policies. See
            :ref:`policies`
        n_periods (int): Number of periods to simulate.

    """
    pass


def simulate_one_period(
    params, states, contact_models, testing_model, policies, period, indexer,
):
    pass
