.. _epi-params:

Epidemiological Parameters
==========================

This section presents the epidemiological parameters for COVID-19. Their values are
stored in `covid_epi_params.csv
<https://github.com/COVID-19-impact-lab/sid/blob/main/src/sid/covid_epi_params.csv>`_.

Health System
-------------

This category only contains the icu_limit_relative. This entry gives the number of ICU
beds per individual.

Sources:
    - `Deutsche Krankenhausgesellschaft
      <https://www.dkgev.de/dkg/coronavirus-fakten-und-infos/>`_
    - `Intensivregister <https://www.intensivregister.de/#/intensivregister>`_


Immunity Countdown
------------------

Due to the novelty of COVID-19, no reliable information on the duration of immunity
exists yet. However, according to `the German RKI <https://www.rki.de/DE/Content/InfAZ/
N/Neuartiges_Coronavirus/Steckbrief.html#doc13776792bodyText14>`_ people have been
reported to develop specific anti bodies within two weeks of infection and evidence from
the related SARS and MERS viruses suggest immunity of up to 3 years.
Note, however, that there have also been
`case reports of reinfection <https://doi.org/10.1136/bmj.m3340>`_
SID supports both heterogeneous, stochastic and no lasting immunity so we can check
how different immunity patterns affect our results.
As baseline we set the length of immunity to 1000 days (2.7 years) for all individuals.


Countdown from Infection to the Begin of Infectiousness
-------------------------------------------------------

The period between infection and onset of infectiousness is called latent or latency
period. The corresponding countdown is called `cd_infectious_true`.

However, the latency period is rarely given in epidemiological reports on COVID-19.
Instead, scientists and agencies usually report the incubation period, the period from
infection to the onset of symptoms. A few studies used measurements of virus shedding to
estimate infectiousness during the course of the disease. When measurements started
before the onset of symptoms the development of the viral load before symptoms gives us
an indication of number of days between the onset of infectiousness and symptoms.

The `ECDC (2020-06-24, question 5)
<https://www.ecdc.europa.eu/en/COVID-19/questions-answers>`_ gives a period of 1-2 days
between the onset of infectiousness and the onset of symptoms. This is in line with a
`study published in Nature <https://doi.org/10.1038/s41591-020-0869-5>`_ who estimate
the onset of infectiousness at 2.3 days (CI: 0.8–3.0 days) before symptoms. This also
aligns with a `study published in the Lancet
<https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30361-3/fulltext>`_.

Estimates of the incubation period usually give a range from 2 to 12 days.

A `meta-analysis by McAloon et al. (2020-07-23)
<https://bmjopen.bmj.com/content/bmjopen/10/8/e039652.full.pdf>`_ comes to the
conclusion that "The incubation period distribution may be modelled with a lognormal
distribution with pooled mu and sigma parameters (95% CIs) of 1.63 (95% CI 1.51 to 1.75)
and 0.50 (95% CI 0.46 to 0.55), respectively."

We discretize the incubation period as follows:

.. image:: ../_static/images/incubation_period.png

To arrive at the latency period we abstract 1 from each of these, except for the one
that is one. This is because countdowns are reduced by one before they cause changes,
i.e. if a countdown "starts" at zero it will not have an effect while a countdown of
one will take effect at the end of the current period (i.e. after individuals meet).

Thus, we arrive at the following lengths for the latency period:

.. csv-table::
    :header: "length of latency", "probability"

        1, 0.39
        2, 0.35
        3, 0.22
        5, 0.04

We do not separate between age groups as
`He et al. (2020-04-15) <https://doi.org/10.1038/s41591-020-0869-5>`_
do not report differences in viral loads across age groups and disease severity.

.. We follow the
.. distribution reported by `Lauer et al. (2020)
.. <https://www.acpjournals.org/doi/full/10.7326/M20-0504>`_. They report the following
.. percentiles for the incubation period:
..
.. .. csv-table::
..     :header: "percentile", "incubation period"
..
..         2.5%, 2.2
..         25.0%, 4
..         50.0%, 5.2
..         75.0%, 6.8
..         97.5%, 11.5
..
.. We interpolate these percentiles to create an empiric cdf of the incubation period:
..
.. .. image:: ../_static/images/cd_infectious_true_cdf.png
..
.. With the resulting distribution:
..
.. .. image:: ../_static/images/cd_infectious_true_full.png
..
.. For our estimates of the latency period we assume a latency period equal to the
.. incubation period minus 2 days.
..
.. These numbers also agree with estimates by
.. `Linton et al. (2020) <https://www.mdpi.com/2077-0383/9/2/538/htm>`_ and
.. `He et al. (2020-05-29) <https://onlinelibrary.wiley.com/doi/full/10.1002/jmv.26041>`_.

However, calculating back from the symptomatic cases leaves the case of asymptomatic
cases unclear. To our knowledge no estimates for the latency period of asymptomatic
cases of COVID-19 exist. We assume it to be the same for symptomatic and asymptomatic
cases.


Length of the Presymptomatic Stage
----------------------------------

The presymptomatic stage is the time between the onset of infectiousness and the
onset of symptoms. The corresponding countdown is called `cd_symptoms_true`.

As we used the incubation time (the time from infection to symptoms) to calculate the
latency period, the length of `cd_symptoms_true` follows mechanically from the estimated
number of days by which infectiousness precedes symptoms. In the case of COVID-19 we
assume that the countdown is either 2 or 3 for symptomatic courses of the disease.

However, a significant share of infected and infectious individuals never develop
symptoms.

A big problem with estimating the share of asymptomatic individuals is that they can be
difficult to find. In addition, in the beginning, tests have been a precious resource
in the fight against COVID-19 - usually reserved for those with symptoms and their
contacts.
Korea has had a stellar performance in testing a large fraction of its population.
We therefore rely on the `Korean CDC reported 33% of asymptomatic cases
<https://www.ijidonline.com/article/S1201-9712(20)30344-1/abstract>`_.

Other sources with more or less similar estimates of asymptomatic cases include:
    - 13% of Chinese children (<15 years) (`Dong et al. (2020)
      <https://pediatrics.aappublications.org/content/145/6/e20200702>`_)
    - 15-20% on the Diamond Princess (`Mizumoto et al. (2020)
      <https://www.eurosurveillance.org/content/10.2807/
      1560-7917.ES.2020.25.10.2000180/#html_fulltext>`_)
    - 30.8% (CI: 7.7–53.8%) from Japanese evacuees (`Nishiura and Kobhttps://bmjopen.bmj.com/content/bmjopen/10/8/e039652.full.pdfayashi
      <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7270890/>`_)
    - 46% (CI: 18-74%) from a meta study by (`He et al. (2020-05-29)
      <https://onlinelibrary.wiley.com/doi/full/10.1002/jmv.26041>`_)

We split the remaining probability mass evenly between 2 and 3 days for the
presymptomatic stage.

Duration of Symptoms (and Infectiousness)
-----------------------------------------

SID assumes that symptomatic individuals stop being infectious when their symptoms cease.
This is governed by the `cd_symptoms_false` countdown.
Asymptomatic individuals have a separate counter `cd_infectious_false`.

.. symptomatic

We use the duration reported by `Bi et al. (2020-03-19, Figure
S3, panel 2)
<https://www.medrxiv.org/content/10.1101/2020.03.03.20028423v3.article-info>`_ to
recovery of mild and moderate cases that we assume to not require intensive care as
estimates for the duration of symptoms and infectiousness for asymptomatic and non-ICU
cases.

.. image:: ../_static/images/time_to_recovery.png

We collapse the data to the following distribution:

.. csv-table::
    :header: "probability", "days until recovery"

    10%, 15
    30%, 18
    30%, 22
    30%, 27

For the asymptomatic cases we assume this to be the distribution of duration of
infectiousness (plus 2-3 days that usually lie between onset of infectiousness and
symptoms for symptomatic individuals) as `evidence suggests little differences
<https://pubmed.ncbi.nlm.nih.gov/32442131/>`_ in the transmission rates of corona virus
between symptomatic and asymptomatic patients and that `the viral load
<https://www.nejm.org/doi/10.1056/NEJMc2001737>`_ between symptomatic and asymptomatic
individuals are similar. These are the values of `cd_infectious_false`.

.. warning::

    However, `this meta-analysis <https://doi.org/10.1101/2020.04.25.20079889>`_ reports
    an estimated mean time from symptom onset to end of infectiousness of 13.4 days
    (95%CI: 10.9-15.8) with shorter estimates for children and less severe cases.
    However, they do not provide information on dispersion parameters. Note that these
    numbers are not as important for our estimates on the spread of the disease as
    agents in sid (can) reduce their contacts (often drastically) once they have
    symptoms.

For symptomatic cases we need to rescale as a proportion of the symptomatic individuals
will require ICU and they get the counter for `cd_symptoms_false` set to -1 as their
symptoms will not end until they exit ICU or die.

The data on how many percent of symptomatic patients will require ICU is pretty thin. We
rely on data by `the US CDC
<https://www.cdc.gov/mmwr/volumes/69/wr/mm6924e2.htm?s_cid=mm6924e2_w#T3_down>`_.

.. warning::

    The CDC's reported age gradient is very small. Only 3.6% of individuals over 80
    years old require intensive care. While the death rate is 28.7%. This seems to stem
    from the ICU share assuming no ICU for those where ICU information is missing. We
    therefore use the maximum of the death and ICU rate.

Other sources often only report the proportion of hospitalized cases admitted to ICU.
According to the collection of the `MIDAS network <https://midasnetwork.us/COVID-19/>`_
the proportion of hospitalized cases to ICU reported were: 0.06, 0.11, 0.26, 0.167
According to the information provided by the `RKI <https://www.rki.de/DE/Content/InfAZ
/N/Neuartiges_Coronavirus/Steckbrief.html#doc13776792bodyText19>`_ the proportion of
hospitalized cases in Germany was around 20%. `In Shanghai the rate is reported to be
8.8%. <https://doi.org/10.1016/j.jinf.2020.03.004>`_


Time from Symptom Onset to Admission to ICU
-------------------------------------------

`Chen et al. (2020-03-02) <https://doi.org/10.1016/j.jinf.2020.03.004>`_ estimate the
time from symptom onset to ICU admission as 8.5 +/- 4 days.

This aligns well with numbers reported for the time from first symptoms to
hospitalization: `The Imperial College reports a mean of 5.76 with a standard deviation
of 4. <https://spiral.imperial.ac.uk/bitstream/10044/1/77344/
12/2020-03-11-COVID19-Report-8.pdf>`_ This is also in line with the `durations collected
by the RKI <https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/
Steckbrief.html#doc13776792bodyText16>`_. We assume that the time between symptom onset
and ICU takes 4, 6, 8 or 10 days with equal probabilities.

These times mostly matter for the ICU capacities rather than the spread of the disease
as symptomatic individuals reduce their social contacts in our model.


Death and Recovery from ICU
---------------------------

`The RKI <https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/
Steckbrief.html#doc13776792bodyText23>`_ cites that a share of 40% of patients admitted
to the ICU died. In Italy `Grasselli et al. (2020-04-06)
<https://jamanetwork.com/journals/jama/fullarticle/2764365>`_ report that 26% of ICU
patients died. We take the midpoint of 33%.

.. warning::

    There exist studies where the share of people who died is much larger than the share
    of patients admitted to ICU. For example `Richardson et al.
    <https://jamanetwork.com/journals/jama/article-abstract/2765184>`_ report 14% ICU
    and 21% death rate. In sid only individuals admitted to intensive care can die.

We assume that patiens in ICU that die do so after 3 weeks. This follows the `3 to 6
weeks of hospital duration reported by the RKI <https://www.rki.de/DE/Content/InfAZ/N/
Neuartiges_Coronavirus/Steckbrief.html#doc13776792bodyText18>`_.

This also aligns with `Chen et al. (2020-04-02)
<https://doi.org/10.1016/j.jinf.2020.03.004>`_ where over 50% of ICU patients still had
fever after 20 days at the hospital.

We use a smaller time until ICU exit for those surviving, assuming they "only" require 2
weeks of ICU care.

As with admission we do not distinguish between hospital and ICU exit.
