"""This module contains the code to for applying contact policies."""
import itertools
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import pandas as pd
from sid.time import timestamp_to_sid_period
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


def compute_pseudo_effect_sizes_of_policies(
    policies: Dict[str, Dict[str, Any]], states: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, Union[float, pd.Series]]]:
    """Compute pseudo effect sizes of policies.

    This function returns either the multiplicand of a policy or computes a pseudo
    effect size by passing a :class:`pandas.Series` with booleans set to ``True`` to the
    policy function. The effect size is the average of the returned
    :class:`pandas.Series`.

    How is the effect size interpreted.

    1. The effect is an average treatment effect on the whole population since this
       simple approach cannot infer which share of the population is affected by the
       policy.

    2. The pseudo effect size can be interpreted as an effect on the average of
       contacts for this contact model.

    Parameters
    ----------
    policies : Dict[str, Dict[str, Any]]
        A policy dictionary.
    states : pd.DataFrame, optional
        The states.

    Returns
    -------
    effects : Dict[str, Dict[str, Union[float, pd.Series]]]
        A dictionary containing for each contact model the mean effect size and a raw
        effect size if a pseudo effect was computed.

    """
    states = states.copy(deep=True) if isinstance(states, pd.DataFrame) else states

    effects = {}
    for name, policy in policies.items():
        date = _find_date_where_policy_is_active(policy)
        if states is not None:
            states["date"] = date
            states["period"] = timestamp_to_sid_period(date)

        index = range(1_000) if states is None else states.index
        if policy.get("affected_contact_model") is None:
            contacts = pd.Series(index=index, data=True).to_frame()
        else:
            contacts = pd.Series(
                index=index, data=True, name=policy["affected_contact_model"]
            )

        if callable(policy["policy"]):
            result = policy["policy"](contacts=contacts, states=states, seed=0)
            result = result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
            result = result.astype("float") if result.dtype == "object" else result
            effects[name] = {"raw_effect": result, "mean": result.mean()}
        else:
            effects[name] = {"mean": policy["policy"]}

    return effects


def _find_date_where_policy_is_active(policy):
    return policy.get("start", policy.get("end", pd.Timestamp("2020-03-09")))
