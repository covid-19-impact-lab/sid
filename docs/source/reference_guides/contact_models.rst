.. _contact_models:

==============
Contact Models
==============

Motivation
----------

One of the main design goals of SID is to achieve an interpretable model of contacts
between individuals that can be calibrated with data and subject to policies.
The contact models should allow for heterogeneity in contact rates based on observable
characteristics. Moreover, different types of contacts (e.g. work and leisure contacts)
should be possible.

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

Boolean flag to mark models that describe recurrent contacts such as families and school
classes.


``"loc"``
^^^^^^^^^

Expression to select a subset of ``params``. This is mostly relevant if pre-implemented
contact models are used (e.g. ``linear_contact_model``) and the params can be used to
select covariates from ``states``. The same contact model could be used with a different
parameterization. This key is optional.


``"model"``
^^^^^^^^^^^

A function that takes states and params as arguments and returns a Series that has the
same index as states. The values of the Series are the numbers of contacts for each
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
larger than zero (in which case she will meet all people in her group who also
participate in that contact model in that period). These model functions can be used to
turn recurrent contact models on and off or to implement that school classes only meet
on weekdays, make sick individuals stay home, etc..

Note that sid automatically implements that dead people and people who need intensive
care do not have contacts anymore.

.. _assort_by:

``"assort_by"``
^^^^^^^^^^^^^^^

A single variable or list of variables according to which the matching is assortative.
All ``assort_by`` variables must be categorical. Individuals who have the same value in
all ``assort_by`` variables belong to one group.

The matching probabilities for the assortative matching process are collected in
probability matrices (also called transition matrices). A probability matrix is a square
matrix in which rows sum to one. The element in row i and column j is the
probability that someone from group i meets a person from group j.

Often the total number of groups (n_groups) is very high and the full group probability
matrix has n_groups * n_groups entries. Thus, it is not possible to estimate this matrix
precisely, without imposing some further structure.

There are three ways of specifying the group probability matrix. In all cases, we first
create one probability matrix per assort_by variable and combine them to the full group
probability matrix, assuming independence.

Here are the three ways of describing the per-variable probability matrices (starting
with the most parsimonious one):

1. Specifying one probability per assort_by variable. This is the probability of meeting
   someone with the same value of the assort_by variable. The remaining probability mass
   is spread uniformly over all other values. The params index of this probability is
   ``("assortative_matching", name_of_contact_model, name_of_variable)``
2. Specifying one probability per value of the assort_by variable. This is interpreted
   as the diagonal of the per-variable probability matrix. The remaining probability
   mass is spread uniformly on the off diagonal elements. In this case, the params index
   is as follows: The first level is
   ``f"assortative_matching_{model_name}_{variable}"``. The second and third level are
   the values of the variable, i.e. the second and third level have to be identical.
3. Specify the full per-variable probability matrix. In this case the first index level
   is the same as in case 2. The second index level indicates the row label of the
   probability matrix. The third index level indicates the column of the probability
   matrix.

There are two ways to implement that a person has zero contacts in a recurrent contact
model: The preferred is to return a zero in the "model" function for this individual.
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
