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
        if policy["start"] <= date <= policy["end"]:
            func = policy["policy"]

            affected_cm = policy.get("affected_contact_model")
            affected_contacts = (
                contacts if affected_cm is None else contacts[affected_cm]
            )

            if isinstance(policy["policy"], (float, int)):
                affected_contacts = affected_contacts * policy["policy"]
            else:
                affected_contacts = func(
                    states=states,
                    contacts=affected_contacts,
                    seed=next(seed),
                )

            if affected_cm is None:
                contacts = affected_contacts
            else:
                affected_contacts = validate_return_is_series_or_ndarray(
                    affected_contacts, name, "contact_policies", states.index
                )
                contacts[affected_cm] = affected_contacts

    return contacts
