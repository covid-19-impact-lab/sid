"""This module contains everything related to the individual immunity level."""
from typing import Union

import numba as nb
import numpy as np


@nb.njit  # pragma: no cover
def get_immunity_level_after_infection(virus_strain_i):  # noqa: U100
    """This needs to be taken from the medical literature."""
    return 0.95


def combine_first_factorized_immunity(
    immune_recurrent: Union[np.ndarray, None],
    immune_random: Union[np.ndarray, None],
) -> np.ndarray:
    """Combine immunity level from recurrent and random meetings."""
    if immune_recurrent is None:
        combined = immune_random
    elif immune_random is None:
        combined = immune_recurrent
    else:
        combined = np.maximum(immune_recurrent, immune_random)

    return combined
