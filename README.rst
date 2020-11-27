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
