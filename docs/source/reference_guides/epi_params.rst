.. _epi-params:

================================
Epidemiological Parameters
================================

This section presents the epidemiological parameters for CoViD-19.
Their values are stored in covid-epi-params.csv.

Health System
===============

This category only contains the icu_limit_relative.
This entry gives the number of ICU beds per individual.

Sources:
    - `Deutsche Krankenhausgesellschaft <https://www.dkgev.de/dkg/coronavirus-fakten-und-infos/>`_
    - `Intensivregister <https://www.intensivregister.de/#/intensivregister>`_

Countdowns
===============

--------------------------------------------------------------------------
Countdown from Infection to the Beginn of Infectiousness
--------------------------------------------------------------------------

The period between infection and onset of infectiousness is called latent or latency
period.

However, the latency period is rarely given in epidemiological reports on CoViD-19.
Instead, scientists and agencies usually report the incubation period,
the period from infection to the onset of symptoms.
A few studies used measurements of virus shedding to estimate infectiousness during
the course of the disease.
When measurements started before the onset of symptoms the development of the viral
load before symptoms gives us an indication of number of days between the onset of
infectiousness and symptoms.

The `ECDC (2020-06-24, question 5) <https://www.ecdc.europa.eu/en/covid-19/questions-answers>`_
gives a period of 1-2 days between the onset of infectiousness and the onset of symptoms.
This is in line with a `study published in Nature <https://doi.org/10.1038/s41591-020-0869-5>`_
who estimate the onset of infectiousness at 2.3 days (CI: 0.8–3.0 days) before symptoms.
This also aligns with a
`study published in the Lancet <https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30361-3/fulltext>`_.

For our estimates of the latency period we assume a latency period equal to the
incubation period minus 2 days.

Estimates of the incubation period usually give a range from 2 to 12 days.
We follow the distribution reported by
`Lauer et al. (2020) <https://www.acpjournals.org/doi/full/10.7326/M20-0504>`_.
They report the following percentiles for the incubation period:

.. csv-table::
    :header: "percentile", "incubation period"

        02.5%, 2.2
        25.0%, 4
        50.0%, 5.2
        75.0%, 6.8
        97.5%, 11.5

We discretize this to the following distribution:

.. csv-table::
    :header: "probability", "incubation period"

    10.0%, 3
    25.0%, 4
    35.0%, 5
    15.0%, 6
    10.0%, 8
    05.0%, 10

We do not separate between age groups as
`He et al. (2020-04-15) <https://doi.org/10.1038/s41591-020-0869-5>`_
do not report differences in viral loads across age groups and disease severity.

Thus, our latency periods are

.. csv-table::
    :header: "probability", "latency period"

    10.0%, 1
    25.0%, 2
    35.0%, 3
    15.0%, 4
    10.0%, 6
    05.0%, 8

These numbers also agree with estimates by
`Linton et al. (2020) <https://www.mdpi.com/2077-0383/9/2/538/htm>`_ and
`He et al. (2020-05-29) <https://onlinelibrary.wiley.com/doi/full/10.1002/jmv.26041>`_.

This countdown is called `cd_infectious_true`.

However, calculating back from the symptomatic cases leaves the case of asymptomatic
cases unclear.
To our knowledge no estimates for the latency period of asymptomatic cases of
CoViD-19 exist.
We assume it to be the same for symptomatic and asymptomatic cases.

--------------------------------------------------------------------------------
Countdown From Infectiousness to Symptoms (Length of the Presymptomatic Stage)
--------------------------------------------------------------------------------

`cd_symptoms_true` is the time between the onset of infectiousness and the onset of
symptoms.

As we used the incubation time (the time from infection to symptoms) to calculate the
latency period, the length of `cd_symptoms_true` follows mechanically from the
estimated number of days by which infectiousness precedes symptoms.
In the case of CoViD-19 we assume that the countdown is either 2 or 3 for symptomatic
courses of the disease.

However, a significant share of infected and infectious individuals never develop
symptoms.

A big problem with estimating the share of asymptomatic individuals is that they can
be difficult to find.
In addition tests have been a precious resource in the fight against CoViD-19 -
usually reserved for those with symptoms and their contacts.
Korea has had a stellar performance in testing a large fraction of its population.
We therefore rely on the
`Korean CDC reported 33% of asymptomatic cases <https://www.ijidonline.com/article/S1201-9712(20)30344-1/abstract>`_.

Other sources with more or less similar estimates of asymptomatic cases include:
- 13% of Chinese children (<15 years) (`Dong et al. (2020) <https://pediatrics.aappublications.org/content/145/6/e20200702>`_)

- 15-20% on the Diamond Princess (`Mizumoto et al. (2020) <https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2020.25.10.2000180/#html_fulltext>`_)

- 30.8% (CI: 7.7–53.8%) from Japanese evacuees (`Nishiura and Kobayashi <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7270890/>`_)

- 46% (CI: 18-74%) from a meta study by (`He et al. (2020-05-29) <https://onlinelibrary.wiley.com/doi/full/10.1002/jmv.26041>`_)


-------------------------------------------------------
Duration of Infectiousness in the Absence of Symptoms
-------------------------------------------------------

There is evidence that there is no difference in the transmission rates of
coronavirus between symptomatic and asymptomatic patients
(https://pubmed.ncbi.nlm.nih.gov/32442131/)
and that the viral load between symptomatic and asymptomatic individuals are similar
(https://www.nejm.org/doi/10.1056/NEJMc2001737).

[To be written]
