import pandas as pd


def meet_two_people(states, params, period):
    return pd.DataFrame(2, index=states.index, columns=["normal_contacts"])
