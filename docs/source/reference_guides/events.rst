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

An event is a dictionary with the following keys.

.. code-block:: python

    {
        "start": "2020-02-15",
        "end": "2020-02-15",
        "model": carnival_session,
    }


``"start"`` and ``"end"``
^^^^^^^^^^^^^^^^^^^^^^^^^

The start and end keys allow to perform quick checks for whether an event is active or
not. The event can only occur in this time interval.


``"model"``
^^^^^^^^^^^

A model is a function which receives ``newly_infected``, ``states``, ``params`` and
``date``

.. code-block:: python

    def carnival_session(newly_infected, states, params, data):
        adults_in_heinsberg = states.query(
            "county == 'Kreis Heinsberg' and age_group >= '10 - 19'"
        ).index

        infection_rate = params.loc[("infection_prob", "gangelt", None), "value"]
        infected = np.random.choice(adults_in_heinsberg, size=300 * infection_prob)

        newly_infected.loc[infected] = True

        return newly_infected
