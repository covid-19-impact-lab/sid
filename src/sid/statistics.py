def calculate_r_effective(df, window_length=7):
    """Calculate the effective reproduction number.

    More information can be found here: https://bit.ly/2VZOR5a.

    Args:
        df (pandas.DataFrame): states DataFrame for which to calculate R_e, usually
            the states of one day.
        window_length (int): how many days to use to identify the previously infectious
            people. The lower, the more changes in behavior can be seen, but the smaller
            the number of people on which to calculate R_e.

    Returns:
        r_effective (float): mean number of people infected by someone whose infectious
            spell ended in the last *window_length* days.

    """
    prev_infected = df[df["cd_infectious_false"].between(-window_length, 0)]
    # the infection counter is only reset to zero once a person becomes infected again
    # so abstracting from very fast reinfections its mean among those that
    # ceased to be infectious in the last window_length is R_e.
    r_effective = prev_infected["n_has_infected"].mean()
    return r_effective


def calculate_r_zero(df, window_length=7):
    """Calculate the basic replication number R_0.

    This is done by dividing the effective reproduction number by the share of
    susceptible people in the DataFrame. Using R_e and the share of the susceptible
    people from the very last period of the time means that heterogeneous matching and
    changes in the rate of immunity are neglected.

    More explanation can be found here: https://bit.ly/2VZOR5a.

    Args:
        df (pandas.DataFrame): states DataFrame for which to calculate R_0, usually the
            states of one period.
        window_length (int): how many days to use to identify the previously infectious
            people. The lower, the more changes in behavior can be seen, but the smaller
            the number of people on which to calculate R_0.

    Returns:
        r_zero (float): mean number of people that would have been infected by someone
            whose infectious spell ended in the last *window_length* days if everyone
            had been susceptible, neglecting heterogeneous matching and changes in the
            rate of immunity.

    """
    r_effective = calculate_r_effective(df=df, window_length=window_length)
    pct_susceptible = 1 - df["immune"].mean()
    r_zero = r_effective / pct_susceptible
    return r_zero
