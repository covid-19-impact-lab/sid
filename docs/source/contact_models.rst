.. _contact_models:

================
Contact Models
================

Motivation
----------

One of the main design goals of SID was to achieve an interpretable model of contacts
between individuals that can be informed by economic theory and calibrated with data.
The contact models should allow for heterogeneity in contact rates based on age, regions
and other observable characteristics. Moreover, different types of contacts (e.g.
distant and close contacts) should be possible.

``contact_models`` are specified as a dictionary of dictionaries. The keys in the outer
dictionary are the name of the contact model which is also used to map policies to
contact models. An example of a name could be "work_close" for a contact model that
describes the number of close contacts at work. The values are dictionaries that
describe one contact model.

Below we first describe how one contact model is described and then how they are
combined.


Specifying One Contact Model
----------------------------

One contact model is a dictionary with the following entries:

``"contact_type"``
^^^^^^^^^^^^^^^^^^

A string with the name of the contact type. Examples could be "close" or "distant". It
is possible to have any number of contact types. If several contact models have the same
type, they get added. Different contact_types have different probabilities of contagion.


``"loc"``
^^^^^^^^^

Expression to select a subset of ``params``. This is mostly relevant if pre-implemented
contact models are used (e.g. ``linear_contact_model``) and the params can be used to
select covariates from ``states``. Optional.

``"model"``
^^^^^^^^^^^

Either the name of a function defined in ``sid.contact_models`` or a function that takes
states, params and period as arguments and returns a series of contacts that has the
same index as states. An example is:

.. code-block:: python

    def some_contact_model(states, params, period):
        # calculate a pd.Series with contacts for each row in the states DataFrame
        return series

The function can depend on period in order to implement policies.


Combining Contact Models
------------------------

The ``simulate`` function takes a dictionary of contact models, where the values are
dictionaries as described above and the keys are the name of the contact model.

The results of the contact models are combined automatically into a DataFrame with one
column per contact type.
