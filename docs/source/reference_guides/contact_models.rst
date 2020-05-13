.. _contact_models:

==============
Contact Models
==============

Motivation
----------

One of the main design goals of SID is to achieve an interpretable model of contacts
between individuals that can be informed by economic theory and calibrated with data.
The contact models should allow for heterogeneity in contact rates based on age, regions
and other observable characteristics. Moreover, different types of contacts (e.g.
distant and close contacts) should be possible.

``contact_models`` are specified as a dictionary of dictionaries. The keys in the outer
dictionary are the name of the contact model. This key is also used to map policies to
contact models. An example of a name could be "work_close" for a contact model that
describes the number of close contacts at work. The values are dictionaries that
describe one contact model.

Below we first describe how one contact model is described and then how they are
combined.


Specifying One Contact Model
----------------------------

One contact model is a dictionary with the following entries:

.. _is_recurrent:

``"is_recurrent"``
^^^^^^^^^^^^^^^^^^

Boolean flag to mark models that describe recurrent contacts such as families, school
classes and the workplace.


``"loc"``
^^^^^^^^^

Expression to select a subset of ``params``. This is mostly relevant if pre-implemented
contact models are used (e.g. ``linear_contact_model``) and the params can be used to
select covariates from ``states``. The same contact model could be used with a different
parameterization. This key is optional.


``"model"``
^^^^^^^^^^^

A function that takes states and params as arguments and returns a Series that has
the same index as states. The values of the Series are the numbers of contacts for each
person. An example is:

.. code-block:: python

    from sid import get_date


    def meet_two_people(states, params):
        # date = get_date(states)  # Get date from states for conditional contacts.

        return pd.Series(index=states.index, data=2)

The returned values can be floating point numbers. In that case they will be
automatically rounded to integers in a way that preserves the total number of contacts.

For recurrent contact models the matching is fully assortative and exhaustive, i.e. each
individual meets all others who have the exact same value in all ``assort_by``
variables. In that case the values of the Series are not interpreted quantitatively and
we just check if they are zero (in which case an individual will not have contacts) or
larger than zero (in which case she will meet all people in her group). These model
functions can be used to turn recurrent contact models on and off or to implement that
school classes only meet on weekdays, make sick individuals stay home, etc..

.. _assort_by:

``"assort_by"``
^^^^^^^^^^^^^^^

A single variable or list of variables according to which the matching is assortative.
All ``assort_by`` variables must be categorical. Individuals who have the same value in
all ``assort_by`` variables belong to one group. The ``params`` DataFrame contains
entries that govern the probability of meeting people from the own group. The index
entry of that parameter values is ``("assortative_matching", name_of_contact_model,
variable_name)``.

The remaining probability mass is spread on all other groups, adjusting for group sizes
and number of planned contacts in each group.

There are two ways to implement that a person has zero contacts in a recurrent contact
model: The preferred is to return a zero in the "model" function for these individual.
Alternatively, people without contacts in a recurrent contact model can have unique
values in the assort_by variables such that their group only contains them alone.
Example: an individual who does not go to school needs a unique value in the variable
that indicates school classes.


Combining Contact Models
------------------------

The ``simulate`` function takes a dictionary of contact models, where the values are
dictionaries as described above and the keys are the name of the contact model.

The results of the contact models are combined automatically into a DataFrame with one
column per contact model.
