.. _contact_models:

================
`contact_models`
================


Motivation
==========


One of the main design goals of SID was to achieve an interpretable model of contacts between individuals that can be informed by economic theory and calibrated with data. The contact models should allow for heterogeneity in contact rates based on age, regions and other observable characteristics. Moreover, different types of contacts (e.g. distant and close contacts) should be possible.


Specifying One Contact Model
============================

One contact model is a dictionary with the following entries:

`"contact_type"`
----------------

A string with the name of the contact type. Examples could be "close" or "distant". It is possible to have any number of contact types. If several dictionaries have the same type, they get added. Different types have different probabilities of contagion.


`"id"`
------

A string that gives a name to the contact model. An example could be `"work_contacts"`. This is mainly used to match policies to contact models. See :ref:`policies`. The id has to be unique.


`"loc"`
-------

Expression to select a subset of `params`. This is mostly relevant if pre-implemented
contact models are used (e.g. `linear_contact_model`) and the params can be used to select covariates from `states`.

`"model"`
---------

Either the name of a function defined in `sid.contact_models` or a function that takes states, params and period as arguments and returns a series of contacts that has the same index as states. An example is:

.. code-block:: python

    def some_contact_model(states, params, period):
        # calculate a pd.Series with contacts for each row in the states DataFrame
        return series

The function can depend on period in order to implement policies.


Combining Contact Models
========================

Contact models are combined automatically into a DataFrame with one column per contact type. The user can just provide a list of contact models.
