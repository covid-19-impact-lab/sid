.. _states:

The ``states`` DataFrame
========================


Introduction
------------

The ``states`` DataFrame is the core of the COVID-19 simulator. It contains all
characteristics of all individuals in the population. This includes all variables that
influence the number of contacts, the dangerousness of the disease as well as the health
status.

All variables in ``states`` should be categorical with meaningful categories that can be
directly used for plotting. Internally, we will work with the numeric codes.

No NaNs are allowed in ``states``. If there are NaNs in the dataset with background
characteristics, the user has to impute values or drop those observations.


Health States
-------------

Our model combines an infection, a contact and an economic model.

In many ways our model is similar to a Susceptible-Exposed-Infected-Recovered (SEIR)
model. However, we represent the health state as booleans as opposed to one categorical
variable.

- ``ever_infected``: Set to True when an infection takes place, stays True forever.
- ``ever_infected``: Set to True when an infection takes place, stays True forever.
- ``immune``: Set to True when an infection takes place, gets a countdown.
- ``immune``: Set to True when an infection takes place, gets a countdown.
- ``infectious``: After an infection, ``infectious_countdown`` is triggered with a
  random time.
- ``infectious``: After an infection, ``infectious_countdown`` is triggered with a
  random time.
- ``received_test_result``: Whether an individual just received a test result. Exists as
  ``True`` only internally in :func:`updates_states` at the end of simulating a period
  when a countdown for a receiving a test result has inspired. Affects ``knows_immune``
  and ``knows_infectious`` and their related countdowns.
- ``knows_immune`` is an indicator for an individual which received a positive test
  result and is still immune.
- ``knows_infectious`` is an indicator for an individual which has been tested an is
  still infectious.


Infection Counter
-----------------

There is an integer column called ``infection_counter`` that counts how many people each
person infected during there current or most recent infection with the virus. This is
set to zero in the moment an infection takes place. It will mainly be used to calculate
the `basic and effective replication number
<https://en.wikipedia.org/wiki/Basic_reproduction_number>`_.

The effective replication number over the past k days can be calculated by averaging
over the infection counter of all individuals who became non-infectious in the past k
days, i.e. those with ``cd_infectious_false`` between 0 and -k.

People can become non-infectious for the following reasons:

- recovery from normal symptoms.
- recovery from intensive care.
- death.

In all cases ``cd_infectious_false`` is set to zero when infectiousness stop, even if
the end of infectiousness was not triggered by that countdown.


.. _countdowns:

Evolution of States
-------------------

Some of the states have stochastic transitions that can be triggered by events that
happened several days earlier. For example, an infection on day t leads to symptoms only
with a probability smaller than one and these symptoms set in after a stochastic
incubation time that can last several days.

Such transitions are handled by countdowns. Countdowns are integer variables. They are
negative if no change was triggered and positive otherwise. Each day they get reduced by
one. If they hit zero, the state variables they are associated with is changed.

We have the following countdowns:

- ``cd_infectious_true``: Time from point of infection until an infected person becomes
  infectious. This period is often called the latency period.
- ``cd_infectious_false``: How long infectiousness lasts.
- ``cd_immune_false``: How long immunity lasts after an infection.
- ``cd_symptoms_true``: Time between the onset of infectiousness and the appearance of
  symptoms. -1 if a person does not develop symptoms in response to an infection.
  Symptoms can only start after a person has become infectious.
- ``cd_symptoms_false``: Duration of symptoms if a person does not require intensive
  care.
- ``cd_needs_icu_true``: Time between the onset of symptoms until a person needs
  intensive care. -1 if a person does not require intensive care.
- ``cd_needs_icu_false``: How long a person needs intensive care before they recover if
  a person does not die beforehand.
- ``cd_dead_true``: Time until a person dies, from start of intensive care. -1 if a
  person will survive intensive care of any length.
- ``cd_ever_infected``: Time passed since the last infection. The countdown starts at
  zero and counts towards negative infinity. Use in combination with ``ever_infected``
  to have the subset of people who were ever infected.

The following countdowns are related to testing.

- ``cd_received_test_result_true``: Time until a test is processed.
- ``cd_knows_immune_false``: Time between receiving a positive test result and losing
  immunity.
- ``cd_knows_infectious_false``: Time between receiving a positive test result and being
  infectious to stop spreading the disease.


Background Characteristics
--------------------------

Background characteristics do not change over time. Their distribution should be taken
from representative datasets for each country or simulated from a distribution that is
calibrated from such a dataset. They can influence matching probabilities (e.g.
assortative meetings by region and age group) and the course of the disease (e.g. higher
fatality rate for risk groups). Moreover, they can be useful for visualization purposes.

There are no mandatory background characteristics if there is only one distribution for
every countdown. Since age is the most important predictor of disease progression, we
assume that if countdown distributions for different subcategories are supplied that the
``states`` DataFrame contains a column named ``age_group``. Have a look at the
``params`` table to see how countdowns are specified, how our age groups look like and
where we take the estimates from.

Other background characteristics you may want to include are:

- variables governing the assortativeness of the matching of individuals, such as region
  of residence
- individual characteristics that influence how many contacts a person has, such as
  gender or occupation.
- identifiers for recurrent contact models such as households or school classes.
