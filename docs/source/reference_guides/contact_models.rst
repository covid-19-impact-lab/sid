.. _contact_models:

==============
Contact Models
==============

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


``"loc"``
^^^^^^^^^

Expression to select a subset of ``params``. This is mostly relevant if pre-implemented
contact models are used (e.g. ``linear_contact_model``) and the params can be used to
select covariates from ``states``. Optional.

``"model"``
^^^^^^^^^^^

Either ``"meet_group"`` or a function that takes states, params and date as arguments
and returns a series that has the same index as states. The values of the Series are
numbers of contacts. An example is:

.. code-block:: python

    def meet_two_people(states, params, date):
        return pd.Series(index=states.index, data=2)

The function can depend on the date in order to implement policies. The returned values
can be floating point numbers. In that case they will be automatically rounded to
integers in a way that preserves the total number of contacts.

If the model is ``"meet_group"``, the matching is fully assortative and exhaustive. I.e.
each individual meets all others who have the exact same value in all ``assort_by``
variables. This can be used to model recurrent contacts inside a household, a school
class or at the workplace.


.. _assort_by:

``"assort_by"``
^^^^^^^^^^^^^^^

A single variable or list of variables according to which the matching is assortative.
All ``assort_by`` variables must be categorical. Individuals who have the same value in
all ``assort_by`` variables belong to one group. The ``params`` DataFrame contains
entries that govern the probability of meeting people from the own group. The index
entry of that parameter values is
``("assortative_matching", "name_of_contact_model", variable_name)``.

The remaining probability mass is spread on all other groups, adjusting for group sizes
and number of planned contacts in each group.

If the model is ``"meet_group"`` there must be exactly one ``assort_by`` variable. If a
person has zero contacts in this contact model, it must have a unique value in the
``assort_by`` variable. Example: an individual who does not go to school needs a unique
value in the variable that indicates school classes.


Combining Contact Models
------------------------

The ``simulate`` function takes a dictionary of contact models, where the values are
dictionaries as described above and the keys are the name of the contact model.

The results of the contact models are combined automatically into a DataFrame with one
column per contact model.
