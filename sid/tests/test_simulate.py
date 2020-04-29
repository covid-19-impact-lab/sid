import pandas as pd

from sid.simulate import simulate


def test_simple_run(params, initial_states, tmpdir):
    initial_infections = pd.Series(index=initial_states.index, data=False)
    initial_infections.iloc[0] = True

    def meet_two(states, params, period):
        return pd.Series(index=states.index, data=2)

    contact_models = {"standard": {"model": meet_two, "contact_type": "standard"}}

    df = simulate(
        params,
        initial_states,
        initial_infections,
        contact_models,
        assort_by=["age_group", "region"],
        path=tmpdir,
    )
    df = df.compute()

    assert isinstance(df, pd.DataFrame)
