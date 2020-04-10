def get_dangerous_contacts(states, contacts, params, assort_by=None):
    """Match individuals that have contact.

    Matching is by default assortative based on region and age_group. Other variables
    can be provided as well. The

    Args:
        states (pd.DataFrame): Everything that influences matching probabilities.
        contacts (pd.DataFrame): Total number of contacts.
        params (pd.DataFrame):
        assort_by (list): List of state variables that influence matching probabilities


    Returns:
        dangerous_contacts (pd.DataFrame): Number of contacts of each type with infected
        people. One column per contact type.

    """
    assort_by = ["region", "age_group"] if assort_by is None else assort_by
    assort_by = [var for var in assort_by if var in states.columns]

    pass
