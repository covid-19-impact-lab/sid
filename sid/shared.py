import numpy as np
import pandas as pd

from sid.config import DTYPE_GROUP_CODE


def factorize_assortative_variables(states, assort_by):
    """Factorize assortative variables.

    This function forms unique values by combining the different values of assortative
    variables. If there are no assortative variables, a single group is assigned to all
    states.

    The group codes are converted to a lower dtype to save memory.

    Args:
        states (pandas.DataFrame): The user-defined initial states.
        assort_by (list, optional): List of variable names. Contacts are assortative by
            these variables.

    Returns:
        group_codes (numpy.ndarray): Array containing the code for each states.
        group_codes_values (np.ndarray): One-dimensional array where positions
            correspond the values of assortative variables to form the group.

    """
    if assort_by:
        assort_by_series = [states[col].to_numpy() for col in assort_by]
        group_codes, group_codes_values = pd.factorize(
            pd._libs.lib.fast_zip(assort_by_series), sort=True
        )
        group_codes = group_codes.astype(DTYPE_GROUP_CODE)

    else:
        group_codes = np.zeros(len(states), dtype=np.uint8)
        group_codes_values = [(0,)]

    return group_codes, group_codes_values
