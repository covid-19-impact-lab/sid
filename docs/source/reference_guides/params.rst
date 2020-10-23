.. _params:

``params``
==========

``params`` is a DataFrame that contains all parameters that quantify how the disease
spreads in the `"value"` column. It can also contain parameters for the contact models
such as the degree of assortativeness by a certain variable. Putting those parameters
into a DataFrame, allows us to optimize over them using `estimagic
<https://estimagic.readthedocs.io/en/latest/>`_. Make sure to read about the basic
structure of `params DataFrames
<https://estimagic.readthedocs.io/en/latest/optimization/params.html>`_ in estimagic,
before you continue.

``params`` has a three level index. The first level is "category", the second is the
"subcategory", the third is called "name". The values are stored in the "value" column.

We provide epidemiological estimates for many of these variables in the
``covid_epi_params.csv`` with explanatory notes and links to their sources.

Currently, we have the following categories:


Assortative Matching (``assortative_matching``)
-----------------------------------------------

As the assortative matching parameters depend on the contact models, we don't provide
any defaults. They must be added by the user.

We suggest to implement assortative matching by ``age_group`` and ``region``. However,
you are free to implement assortative matching by any variable in your ``states``
dataset. Having assortative matching not only adds realism to your model but also
reduces running time.

For more information on assortative matching see :ref:`assort_by`.

.. raw:: html

    <div class="d-flex flex-row gs-torefguide">
        <span class="badge badge-info">To tutorials and explanations</span>

Have a look at the `simulation tutorial <../tutorials/how_to_simulate.ipynb>`_ and
the `assortative matching notebook <../explanations/assortative_matching.ipynb>`_
to see some example contact models and assortative matching parameters.

.. raw:: html

    </div>


Health System (``health_system``)
---------------------------------

The default parameters in this category only include the number of free beds in
intensive care units which determine how many individuals with serious infection cases
survive.


Infection Probabilities (``infection_prob``)
--------------------------------------------

As the infection probabilities depend on the contact models, wo don't provide any
defaults. They must be added by the user.

.. raw:: html

    <div class="d-flex flex-row gs-torefguide">
        <span class="badge badge-info">To tutorials and explanations</span>

Have a look at the `simulation tutorial <../tutorials/how_to_simulate.ipynb>`_ and the
`assortative matching notebook <../explanations/assortative_matching.ipynb>`_ to see
some example contact models.

.. raw:: html

    </div>


Countdowns
----------

Every countdown described in :ref:`countdowns` has its own category, describing its
distribution.

If the distribution does not depend on the age group, the subcategory is "all". If the
distribution depends on the age group then the subcategory takes the values of the age
groups. The states DataFrame then must contain a column called "age_group" with the age
groups and their values must match the ones in the subcategory column. In each case the
"name" column contains the possible realizations and the "value" column contains the
probability. Probabilities for each group must add up to one.

Here is an example with hypothetical numbers:

.. csv-table:: Hypotetical Parameter Values
    :header: category, subcategory, name, value

    cd_symptoms_true  , all              , -1 (= never)      , 0.25
    cd_symptoms_true  , all              , 3                 , 0.75
    ...               , ...              , ...               , ...
    cd_infectious_true, 0-9 (age group), 3 (possible value), 0.6 (probability)
    cd_infectious_true, 0-9 (age group), 5 (possible value), 0.3 (probability)
    cd_infectious_true, 0-9 (age group), 7 (possible value), 0.1 (probability)
    cd_infectious_true, 10-20          , 3 (possible value), 0.6 (probability)
    ...               , ...              , ...               , ...

The following section describes the epidemiological parameters we provide for Covid-19
and their sources.
