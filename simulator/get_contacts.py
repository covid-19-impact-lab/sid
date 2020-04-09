def get_contacts(model_spec, states, params):
    """Calculate number of contacts of different types.

    Args:
        model_spec (list): A list of dictionaries. Dictionary entries are:
            - type (str): Different contact types can have different contagion
                rates. contact_types will also be used as name of resulting columns.
                If several dictionaries refer to the same contact type, they are added.
                This can for example be used if we do not want to distinguish types of
                contacts, but fit  separate contact models for work, free-time and
                errands to our data.
            - loc: Expression to select Parameters via params.loc
            - contact_model (str or callable): A function to calculate contacts of a
              given type from states.
        states (pd.DataFrame):
        params (pd.DataFrame):

    Returns:
        contacts (pd.DataFrame): DataFrame with one column per contact type.


    """
    pass
