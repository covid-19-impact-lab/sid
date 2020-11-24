sid
===

.. start-badges

.. .. image:: https://anaconda.org/covid-19-impact-lab/sid/badges/version.svg
..     :target: https://anaconda.org/covid-19-impact-lab/sid

.. .. image:: https://anaconda.org/covid-19-impact-lab/sid/badges/platforms.svg
..     :target: https://anaconda.org/covid-19-impact-lab/sid

.. image:: https://readthedocs.org/projects/sid-dev/badge/?version=latest
    :target: https://sid-dev.readthedocs.io/en/latest

.. image:: https://codecov.io/gh/covid-19-impact-lab/sid/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/covid-19-impact-lab/sid

.. image:: https://img.shields.io/badge/License-none-yellow.svg
    :target: https://opensource.org/licenses/none

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. end-badges


Features
--------

sid is a simulation-based model for infectious diseases like COVID-19. It combines
features of a prototypical SEIR (Susceptible-Exposed-Infected-Recovered) model and an
agent-based simulation model.

sid's focus is on predicting the effects of policies on the spread of an infectious
disease. To accomplish this task it is built to capture important aspects of human
contacts. In particular, sid has the following features:

1. At the core of the model, people meet people based on a matching algorithm. One can
   distinguish various types of contacts, for example, households, leisure activities,
   schools, nurseries, and contacts at the workplace. Contact types can be random or
   recurrent and vary in frequency.

2. Policies can be implemented as shutting down contact types entirely or partially. The
   reduction of contacts can be random or systematic, e.g., to allow for essential
   workers.

3. Infection probabilities vary across contact types and matching patterns and disease
   progressions according to individual characteristics such as age and gender.


.. start-installation

Installation
------------

sid will soon become available as a conda package. To install sid, type

.. code-block:: bash

    $ conda install -c covid-19-impact-lab sid


.. end-installation
