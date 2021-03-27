Rapid Tests
===========

In contrast to the other :doc:`testing mechanism <testing>` which represents PCR tests,
rapid tests are less reliable, but yield a test result on the same day they were
administered.

That means that people can take rapid tests and, after that, decide to have contacts or
not depending on the outcome. This way one is able to able implement policies like
mass-testing for schools and work places.

To implement the latter scenarios, two components need to be prepared:

- ``rapid_test_models``: These models allow to assign rapid tests to individuals. The
  test outcome for each individual is then sampled from the sensitivity and specificity
  of the test.

- ``rapid_test_reaction_models``: These models have access to the results of the rapid
  tests and allow to adjust the number of contacts a person has depending on the
  outcome.


``rapid_test_models``
---------------------

``rapid_test_models`` is a dictionary of dictionaries. Here is an example:

.. code-block:: python

    {
        "rapid_tests_for_schools": {
            "model": distribute_rapid_tests_at_schools,
            # "loc": ...,
            # "start": ...,
            # "end": ...,
        }
    }


The function of the model ``"rapid_tests_for_schools"`` has the following signature:

.. code-block:: python

    def distribute_rapid_tests_at_schools(
        receives_rapid_test, states, params, contacts, seed
    ):
        pass

The function receives a boolean series, ``receives_rapid_test`` which indicates people
who already receive rapid tests so that test are not assigned twice.

``contacts`` are a :class:`pandas.DataFrame` with the number of contacts for each
contact model. This allows to distribute tests, for example, to schools if contact
models for schools are active and there are no policies which have shut down schools.

``states`` contains a column called ``cd_received_rapid_test`` which gives you the
number of days since the last rapid test for this person. A person has a countdown value
of 0 if she received the rapid test on the same day. Then, value decreases each period
by -1 such that -2 indicates that the last test happened two days ago.

The return of the function is a boolean :class:`pandas.Series` which indicates
additional individuals who receive rapid tests. The indicator will be automatically
combined with ``receives_rapid_test``.


``rapid_test_reaction_models``
------------------------------

``rapid_test_reaction_models`` allow to change the number of contacts in response to the
outcome of the rapid test.

The models are also passed to the simulation function as a dictionary of dictionaries.

.. code-block:: python

    {
        "react_to_positive_rapid_test": {
            "model": react_to_positive_rapid_test,
            # "loc": ...,
            # "start": ...,
            # "end": ...,
        }
    }

The signature of the function looks like this:

.. code-block:: python

    def react_to_positive_rapid_test(contacts, states, params, seed):
        pass

``contacts`` is a :class:`pandas.DataFrame` which includes the planned contacts of
individuals which can be deactivated.

``states`` includes a column called ``is_tested_positive_by_rapid_test`` which indicates
individuals who are tested positive by a rapid test in the same period.
