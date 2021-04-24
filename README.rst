sid
===

.. start-badges

.. image:: https://img.shields.io/pypi/v/sid-dev?color=blue
    :alt: PyPI
    :target: https://pypi.org/project/sid-dev

.. image:: https://img.shields.io/pypi/pyversions/sid-dev
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/sid-dev

.. image:: https://img.shields.io/conda/vn/conda-forge/sid-dev.svg
    :target: https://anaconda.org/conda-forge/sid-dev

.. image:: https://img.shields.io/conda/pn/conda-forge/sid-dev.svg
    :target: https://anaconda.org/conda-forge/sid-dev

.. image:: https://img.shields.io/pypi/l/sid-dev
    :alt: PyPI - License
    :target: https://pypi.org/project/sid-dev

.. image:: https://readthedocs.org/projects/sid-dev/badge/?version=latest
    :target: https://sid-dev.readthedocs.io/en/latest

.. image:: https://img.shields.io/github/workflow/status/covid-19-impact-lab/sid/Continuous%20Integration%20Workflow/main
   :target: https://github.com/covid-19-impact-lab/sid/actions?query=branch%3Amain

.. image:: https://codecov.io/gh/covid-19-impact-lab/sid/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/covid-19-impact-lab/sid

.. image:: https://results.pre-commit.ci/badge/github/covid-19-impact-lab/sid/main.svg
    :target: https://results.pre-commit.ci/latest/github/covid-19-impact-lab/sid/main
    :alt: pre-commit.ci status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. end-badges


Features
--------

sid is an agent-based simulation model for infectious diseases like COVID-19. It scales
from simple examples to complex models which makes it an ideal tool for prototyping,
educational purposes, and research.

sid's focus is on predicting the effects of non-pharmaceutical interventions on the
spread of an infectious disease. To accomplish this task it is built to capture
important aspects of contacts between people. In particular, sid has the following
features:

1. At the core of the model, people meet people based on a matching algorithm. It is
   possible to represent a variety of contact types like households, leisure activities,
   school classes and nurseries with teachers and several types of contacts at the
   workplace. Contact types can be random or recurrent and vary in frequency.

2. Policies allow to shut down contact types entirely or partially. The reduction of
   contacts can be random or systematic, e.g., to allow for essential workers.

3. Infection probabilities vary across contact types and depending on the age of the
   susceptible individual, but are invariant to policies which reduce contacts. The
   invariance is an essential property for predicting the effects of policies for which
   empirical data does not exist.

4. The model achieves a good fit on German infection and fatality rate data even if only
   the infection probabilities are fit to the data and the remaining parameters are
   calibrated from the medical literature and data on contact frequencies.

5. The model allows for two testing mechanisms, representing PCR and rapid tests. PCR
   tests always reveal the true health status of the tested individual after some days
   which can be used for testing policies or to differentiate between known and unknown
   infections.

   In contrast, rapid tests immediately return the test outcome and identify infected
   people based on the sensitivity and specificity of the test. It is possible to
   implement reactions to the outcome of the test enabling individuals to plan meetings,
   test with a rapid test, and to refrain from meeting if the test is positive.

6. Mutations may lead to multiple, prevalent virus strains with different
   characteristics. For now, sid is able to model an unlimited amount of virus strains
   with different degrees of infectiousness.

7. It is possible to implement models for vaccinating people who, then, gain perfect
   immunity from the disease.

More information can be found in the `discussion paper
<https://www.iza.org/publications/dp/13899>`_ or in the `documentation
<https://sid-dev.readthedocs.io/en/latest/>`_.


.. start-installation

Installation
------------

sid is available on `PyPI <https://pypi.org/project/sid-dev>`_ and on `Anaconda.org
<https://anaconda.org/conda-forge/sid-dev>`_ and can be installed with

.. code-block:: bash

    $ pip install sid-dev

    # or

    $ conda install -c conda-forge sid-dev

.. end-installation


Publications
------------

sid has been featured in some publications which are listed here:

- Gabler, J., Raabe, T., & Röhrl, K. (2020). `People Meet People: A Microlevel Approach
  to Predicting the Effect of Policies on the Spread of COVID-19
  <http://ftp.iza.org/dp13899.pdf>`_.

- Dorn, F., Gabler, J., von Gaudecker, H. M., Peichl, A., Raabe, T., & Röhrl, K. (2020).
  `Wenn Menschen (keine) Menschen treffen: Simulation der Auswirkungen von
  Politikmaßnahmen zur Eindämmung der zweiten Covid-19-Welle
  <https://www.ifo.de/DocDL/sd-2020-digital-15-dorn-etal-politikmassnahmen-covid-19-
  zweite-welle.pdf>`_. ifo Schnelldienst Digital, 1(15).

- Gabler, J., Raabe, T., Röhrl, K., & Gaudecker, H. M. V. (2020). `Die Bedeutung
  individuellen Verhaltens über den Jahreswechsel für die Weiterentwicklung der
  Covid-19-Pandemie in Deutschland <http://ftp.iza.org/sp99.pdf>`_ (No. 99). Institute
  of Labor Economics (IZA).

- Gabler, J., Raabe, T., Röhrl, K., & Gaudecker, H. M. V. (2021). `Der Effekt von
  Heimarbeit auf die Entwicklung der Covid-19-Pandemie in Deutschland
  <http://ftp.iza.org/sp100.pdf>`_ (No. 100). Institute of Labor Economics (IZA).


Citation
--------

If you rely on sid for your own research, please cite it with

.. code-block::

    @article{Gabler2020,
      Title = {
        People Meet People: A Microlevel Approach to Predicting the Effect of Policies
        on the Spread of COVID-19
      },
      Author = {Gabler, Janos and Raabe, Tobias and R{\"o}hrl, Klara},
      Year = {2020},
      Publisher = {IZA Discussion Paper}
    }
