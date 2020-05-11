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
  <https://www.land.nrw/sites/default/files/asset/document/zwischenergebnis_covid19_case_study_gangelt_0.pdf>`_.

- The RKI models a steady inflow of one infection per day.


Specifying an event
-------------------

Multiple events are stored in a dictionary. Each event has a name in this dictionary
like ``"gangelt"``. The values of the dictionary are again dictionaries which specify the event and could have the following keys.

.. code-block:: python

    {
        "gangelt": {
            # "loc": slice(None),
            "model": carnival_session,
        }
    }


``"loc"``
^^^^^^^^^

Expression to select a subset of ``params``. This is mostly relevant if pre-implemented
contact models are used (e.g. ``linear_contact_model``) and the params can be used to
select covariates from ``states``. The same contact model could be used with a different
parameterization. This key is optional.


``"model"``
^^^^^^^^^^^

A model is a function which receives ``newly_infected``, ``states``, ``params`` and
``date``

.. code-block:: python

    def carnival_session(newly_infected, states, params, date):
        date_of_carnival_session = pd.to_datetime("2020-02-15")

        if date == date_of_carnival_session:

            adults_in_heinsberg = states.query(
                "county == 'Kreis Heinsberg' and age_group >= '10 - 19'"
            ).index

            infection_rate = params.loc[("infection_prob", "gangelt", None), "value"]
            infected = np.random.choice(adults_in_heinsberg, size=300 * infection_prob)

            newly_infected.loc[infected] = True

        return newly_infected
