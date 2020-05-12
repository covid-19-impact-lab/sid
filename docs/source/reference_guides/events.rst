======
Events
======

Motivation
----------

Events allow to model infections which have exogenous causes and occur besides
contacts to infectious people. Examples of such events are

- Imported cases which caused the center of infections in Munich when ski-tourists
  returned.

- A carnival session which sparked the outbreak in `Gangelt
  <https://www.land.nrw/sites/default/files/asset/document/
  zwischenergebnis_covid19_case_study_gangelt_0.pdf>`_.

- The RKI models a steady inflow of one infection per day.


Specifying an event
-------------------

Multiple events are stored in a dictionary. Each event has a name in this dictionary
like ``"gangelt"``. The values of the dictionary are again dictionaries which specify
the event and could have the following keys.

.. code-block:: python

    {
        "gangelt": {
            # "loc": slice(None),  # Returns all parameters which is the default.
            "model": carnival_session,
        }
    }


``"loc"``
^^^^^^^^^

Expression to select a subset of ``params``. This is mostly relevant if pre-implemented
events are used (e.g. ``import_cases``) and the params can be used to
select covariates from ``states``. The same contact model could be used with a different
parameterization. This key is optional.


``"model"``
^^^^^^^^^^^

A model is a function which receives ``states`` and ``params`` and returns a boolean
series where new infections are marked with ``True``. The new infections per day from
contacts and multiple events are merged with a logical OR.

.. code-block:: python

    from sid import get_date


    def carnival_session(states, params):
        date = get_date(states)  # Helper to get the current date from states.
        date_of_carnival_session = pd.to_datetime("2020-02-15")

        newly_infected = pd.Series(data=False, index=states.index)

        if date == date_of_carnival_session:

            adults_in_heinsberg = states.query(
                "county == 'Kreis Heinsberg' and age_group >= '10 - 19'"
            ).index

            infection_prob = params.loc[("infection_prob", "gangelt", None), "value"]
            infected_indices = np.random.choice(
                adults_in_heinsberg, size=300 * infection_prob, replace=False
            )

            newly_infected.loc[infected_indices] = True

        return newly_infected
