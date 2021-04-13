"""This module contains the code to for applying contact policies."""
import itertools
from typing import Any
from typing import Dict

import pandas as pd
from sid.validation import validate_return_is_series_or_ndarray


def apply_contact_policies(
    contact_policies: Dict[str, Dict[str, Any]],
    contacts: pd.DataFrame,
    states: pd.DataFrame,
    date: pd.Timestamp,
    seed: itertools.count,
) -> pd.DataFrame:
    """Apply policies to contacts."""
    for name, policy in contact_policies.items():

        affected_cm = policy["affected_contact_model"]
        if affected_cm in contacts.columns:
            if policy["start"] <= date <= policy["end"]:
                func = policy["policy"]
                model_specific_contacts = contacts[affected_cm]

                if isinstance(policy["policy"], (float, int)):
                    model_specific_contacts = model_specific_contacts * policy["policy"]
                else:
                    model_specific_contacts = func(
                        states=states,
                        contacts=model_specific_contacts,
                        seed=next(seed),
                    )

                model_specific_contacts = validate_return_is_series_or_ndarray(
                    model_specific_contacts, name, "contact_policies", states.index
                )

                contacts[affected_cm] = model_specific_contacts

    return contacts
