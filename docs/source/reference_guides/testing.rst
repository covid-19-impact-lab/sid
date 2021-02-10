.. _testing:

Testing
=======

For an introduction to specify testing, have a look at the `testing tutorial
<../tutorials/how_to_test.ipynb>`_

In the following, the three different testing models are described and how to pass them
to :func:`sid.get_simulate_func`.

There are three categories of testing models:

- ``testing_demand_models`` are used to compute the demand for tests. Each model returns
  a :class:`pandas.Series` that contains for each individual the probability of
  demanding a test.

- ``testing_allocation_models`` are used to assign who will receive a test.

- ``testing_processing_models`` are used to start the execution of tests.


General
-------

Regardless of the category of testing models, each argument receives a dictionary where
keys represent the names of the models. The values are dictionaries with the
following keys:

- ``"model"`` accepts a function. The three different categories of functions are
  explained below.
- ``"loc"`` allows to pass only a subset of parameters to the functions.
- ``"start"`` and ``"end"`` accept dates to activate and deactivate models based on the
  date.

.. code-block:: python

    testing_models = {
        "name": {
            "model": function,
            # "loc": ...,
            # "start": "2019-02-29",
            # "end": pd.Timestamp("2019-07-30"),
        }
    }


.. _testing_demand_models:

Demand models
-------------

Demand models for tests allow to compute the demand for tests. Each demand model
represents a channel for why the individual requests a test, e.g., symptomatic or
positive cases among household members, mandatory testing at the workplace, response to
mass testing with monetary incentives. The function returns a :class:`pandas.Series`
where each entry specifies the probability that the individual demands a test for this
reason.

The probabilities of all demand models are combined to the probability that an
individual will at least demand one test from any model. Then, the individuals who
demand a test are sampled with this probability.

As an optional feature, you can add ``"channel_demands_test"`` to ``saved_columns``
which allows you to store which demand model was responsible. It also costs a little bit
of runtime and memory which is why it is deactivated by default.

A demand model accepts only the ``states``, ``params`` and a ``seed`` as arguments.

.. code-block:: python

    def demand_tests(states, params, seed):
        pass


.. _testing_allocation_models:

Allocation models
-----------------

Models for the allocation of tests allow to distribute tests to individuals who demand a
test. Models may reflect different priorities given by policies.

For simulations with a simple testing scheme, you can insert a constant limit of tests
in the parameters under

.. code-block:: python

    params.loc[("testing", "allocation", "rel_available_tests"), "value"]

which is the number of available tests per 100,000 people. If the limit is exceeded, a
warning will be raised. If the number of available tests is a daily changing resource,
you can disable all warnings by setting the parameter to
:class:`np.inf`.

An allocation model for tests has the following interface:

.. code-block:: python

    def allocate_tests(n_allocated_tests, demands_test, states, params, seed):
        pass

- ``n_allocated_tests`` returns the number of available tests minus the already
  allocated tests.

- ``demands_test`` is a :class:`pandas.Series` with boolean values for individuals
  demanding a test which is also updated between each allocation model.


.. _testing_processing_models:

Processing models
-----------------

Models for processing tests allow to start the processing of administered tests. They
allow to implement different processing schemes which are valuable if laboratories are
at their limits. For example, a switch from first-in first-out to last-in first-out may
have positive effects.

As before, a constant limit of tests per 100,000 individuals which can be processed
daily can be set in

.. code-block:: python

    params.loc[("testing", "processing", "rel_available_capacity"), "value"]

Processing models have the following interface:

.. code-block:: python

    def process_tests(n_to_be_processed_tests, states, params, seed):
        pass

- ``n_to_be_processed_tests`` yields the number of remaining tests in this period which
  can be distributed.

- The tests who are still waiting to be processed can be located in ``states`` in the
  column ``pending_test`` and ``pending_test_date`` yields the date when the test was
  administered.
