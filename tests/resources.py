import pandas as pd


def meet_two(states, params):  # noqa: U100
    return pd.Series(index=states.index, data=2)


CONTACT_MODELS = {
    "standard": {
        "model": meet_two,
        "assort_by": ["age_group", "region"],
        "is_recurrent": False,
    }
}
