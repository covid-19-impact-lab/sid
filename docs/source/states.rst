.. _states:

======================
The `states` DataFrame
======================


Introduction
============

The `states` DataFrame is the core of the covid simulator. It contains all characteristics of all individuals in the population. This includes all variables that influence the number of contacts, the dangerousness of the disease as well as the health status.

Al variables in `states` should be categorical with meaningful categories that can be directly used for plotting. Internally, we will work with the codes.

No missings are allowed in `states`. If there are missings in the dataset with background characteristics, the user has to impute values or drop those observations.


Health States
=============

Our Model combines

is similar to a Susceptible-Exposed-Infected-Recovered (SEIR) model. However, we represent the health states as booleans as opposed to one categorical variable.

- **ever_infected**: Set to True when an infection takes place, stays True forever.
- **immune**: Set to True when an infection takes place, gets a countdown.
- **infectious**: After an infection, `infectious_countdown` is triggered with a random time.
- **knows**: Whether an individual knows if he is infectious or not.
- **symptoms**: Whether an individual has symptoms. Only infectious people can have symptoms
- **icu**: Whether an individual needs intensive care. Only possible for people with symptoms.
- **dead**: Whether an individual is dead. People die if they need icu but do not get it or despite icu with a certain probability.

Infection Counter
=================

There is an integer column called ``infection_counter`` that counts how many people each person infected during there current or most recent infection with the virus. This is set to zero in the moment an infection takes place. It will mainly be used to calculate ``r_zero``.

``r_zero`` can be counted by taking all individuals who became non-infectious in the past k days, i.e. those with ``cd_infectious_false`` between 0 and -k.

People can become non-infectious for the following reasons:
- recovery from normal symptoms
- recovery from intensive care
- death
In all cases ``cd_infectious_false`` is set to zero when infectiousness stop, even if the end of infectiousness was not triggered by that countdown.

Evolution of States
===================

Some of the states have stochastic transition that can be triggered by events that happened several periods earlier. For example, an infection in period t leads to symptoms only with a probability smaller than one and these symptoms set in after a stochastic incubation time that can last several days.

Such transitions are handled by countdowns. Countdowns are integer variables. They are negative if no change was triggered and positive otherwise. Each period they get reduced by one. If they hit zero, the state variables they are associated with is changed. Since all variables are booleans, the end of the countdown means just that the current value is negated. Thus the same countdown can be used for switches from True to False and from False to True.

We have the following countdowns:

- **cd_infectious_true**: Time until an infected person becomes infectious
- **cd_infectious_false**: How long infectiousness lasts
- **cd_immune_false**: How long immunity lasts after infection
- **cd_symptoms_true**: Time until symptoms start.
- **cd_symptoms_false**: How long symptoms last
- **cd_needs_icu_true**: Time until a person needs intensive care, from start of symptoms
- **cd_dead**: Time until a person dies, from start of intensive care
- **cd_needs_icu_false**: How long a person needs intensive care
- **cd_knows_true **: Time from test until tested person knows the result


Background Characteristics
==========================

Background characteristics do not change over time. Their distribution should be taken from representative datasets for each country or simulated from a distribution that is calibrated from such a dataset. The can influence matching probabilities (e.g. assortative meetings by region and age group) and the course of the disease (e.g. higher fatality rate for risk groups). Moreover, they can be useful for visualization purposes.


Mandatory Background Characteristics
------------------------------------

- `age_group`: There will be assortative matching in the sense that people have more contact with people from their own age group. I suggest to have age groups in ten-year bins, i.e. 0-9, 10-19, ...; We can take the pathogenesis data from `here <https://towardsdatascience.com/agent-based-simulation-of-covid-19-health-and-economical-effects-6aa4ae0ff397>`_ or `the paper they cite <https://spiral.imperial.ac.uk:8443/bitstream/10044/1/77482/8/2020-03-16-COVID19-Report-9.pdf>`_.
- `region`: Region is only used for assortative matching and possibly to draw a map of the simulation output. In German Data we could go down to Landkreis in the time series. Bundesländer would also be ok.

Optional Background Characteristics
-----------------------------------
