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

is similar to a Susceptible-Exposed-Infected-Recovered (SEIR) model. However, we represent the health states as booleans as opposed to one categorical variable. This makes it easier to add different states later. Some of the states have stochastic transition that can be triggered by events that happened several periods earlier. For example, an infection in period t leads to symptoms only with a probability smaller than one and these symptoms set in after a stochastic incubation time that can last several days. Such transitions are handled by countdowns. Countdowns are integer variables. They are negative if no change was triggered and positive otherwise. Each period they get reduced by one. If they hit zero, the state variables they are associated with is changed.

- **ever_infected**: Set to True when an infection takes place, stays True forever. This is mainly to calculate cumulative infections, not used anywhere in the simulation
- **immune**: Set to True when as soon as an infection takes place, gets a countdown.

- **infectious**: After an infection, `infectious_countdown` is triggered with a random time span.
- **knows_infectious**:
- **symptoms**:
- **needs_icu**
- **in_icu**
- **dead**



Evolution States
================





Background Characteristics
==========================

Background characteristics do not change over time. Their distribution should be taken from representative datasets for each country or simulated from a distribution that is calibrated from such a dataset. The can influence matching probabilities (e.g. assortative meetings by region and age group) and the course of the disease (e.g. higher fatality rate for risk groups). Moreover, they can be useful for visualization purposes. We want at least:

- `age_group`: There will be assortative matching in the sense that people have more contact with people from their own age group. I suggest to have age groups in ten-year bins, i.e. 0-9, 10-19, ...; We can take the pathogenesis data from `here <https://towardsdatascience.com/agent-based-simulation-of-covid-19-health-and-economical-effects-6aa4ae0ff397>`_ or `the paper they cite <https://spiral.imperial.ac.uk:8443/bitstream/10044/1/77482/8/2020-03-16-COVID19-Report-9.pdf>`_.
- `region`: Region is only used for assortative matching and possibly to draw a map of the simulation output. In German Data we could go down to Landkreis in the time series. Bundesländer would also be ok.
- `pre_health`: Pre-existing conditions could influence the pathogenesis. However, it is difficult to get parameters to quantify this influence. This is where we would need help from doctors.
