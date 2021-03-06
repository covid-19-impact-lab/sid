sid
===

.. start-badges

.. image:: https://anaconda.org/covid-19-impact-lab/sid/badges/version.svg
    :target: https://anaconda.org/covid-19-impact-lab/sid

.. image:: https://anaconda.org/covid-19-impact-lab/sid/badges/platforms.svg
    :target: https://anaconda.org/covid-19-impact-lab/sid

.. image:: https://readthedocs.org/projects/sid-dev/badge/?version=latest
    :target: https://sid-dev.readthedocs.io/en/latest

.. image:: https://codecov.io/gh/covid-19-impact-lab/sid/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/covid-19-impact-lab/sid

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://results.pre-commit.ci/badge/github/covid-19-impact-lab/sid/main.svg
    :target: https://results.pre-commit.ci/latest/github/covid-19-impact-lab/sid/main
    :alt: pre-commit.ci status

.. end-badges


Features
--------

sid is a simulation-based model for infectious diseases like COVID-19. It combines
features of a prototypical SEIR (Susceptible-Exposed-Infected-Recovered) model and an
agent-based simulation model.

sid's focus is on predicting the effects of policies on the spread of an infectious
disease. To accomplish this task it is built to capture important aspects of human
contacts. In particular, sid has the following features:

1. At the core of the model, people meet people based on a matching algorithm. We
   distinguish various types of contacts. Currently, these are households, leisure
   activities, schools, nurseries and several types of contacts at the workplace.
   Contact types can be random or recurrent and vary in frequency.

2. Policies can be implemented as shutting down contact types entirely or partially. The
   reduction of contacts can be random or systematic, e.g., to allow for essential
   workers.

3. Infection probabilities vary across contact types, but are invariant to policies
   which reduce contacts.

4. The model achieves a good fit on German infection and fatality rate data even if only
   the infection probabilities are fit to the data and the remaining parameters are
   calibrated from the medical literature and datasets on contact frequencies.

More information can be found in the `discussion paper
<https://www.iza.org/publications/dp/13899>`_ or in the `documentation
<https://sid-dev.readthedocs.io/en/latest/>`_.


.. start-installation

Installation
------------

sid will soon become available as a conda package and will be installed with

.. code-block:: bash

    $ conda install -c covid-19-impact-lab sid

For now, clone the repository and install the package with pip or conda.

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
