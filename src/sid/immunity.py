"""This module contains everything related to the individual immunity level."""
import numpy as np
from sid.config import DTYPE_IMMUNITY


def get_immunity_level_after_infection(virus_strain_i):  # noqa: U100
    """This needs to be taken from the medical literature."""
    return 0.95


def combine_first_factorized_immunity(
    immune_recurrent: DTYPE_IMMUNITY, immune_random: DTYPE_IMMUNITY
) -> DTYPE_IMMUNITY:
    """Combine immunity level from recurrent and random meetings."""
    return np.maximum(immune_recurrent, immune_random)
