Changes
=======

This is a record of all past sid releases and what went into them in reverse
chronological order. Releases follow `semantic versioning <https://semver.org/>`_ and
all releases are available on `Anaconda.org
<https://anaconda.org/covid-19-impact-lab/sid>`_.


0.0.5 - 2021-xx-xx
------------------

- :gh:`112` implements a gantt chart to visualize policies.


0.0.4 - 2021-04-24
------------------

- :gh:`92` adds an interface for rapid tests.
- :gh:`93` enhances the validation mechanism.
- :gh:`94` scales a single vaccination model to multiple vaccination models.
- :gh:`95` enhances the documentation and fixes resuming simulations.
- :gh:`96` changes the initialization of countdowns and removes draws created for
  countdowns without randomness.
- :gh:`97` improves the test coverage.
- :gh:`98` fixes typo.
- :gh:`99` and :gh:`103` simplify ``factorize_assortative_variables``.
- :gh:`101` removes ``"is_active"`` from policies.
- :gh:`102` separates the calculation of contacts from applying policies.
- :gh:`104` implements a seasonality factor which scales infection probabilities.
- :gh:`106` allows policies to affect all contacts and not a single contact model.
- :gh:`107` allows compute derived state variables which can be used across model
  features to save some computations.
- :gh:`108` enhances dtype conversion of random contact models.
- :gh:`110` fixes a ``SettingWithCopy`` warning in ``contacts.py``.
- :gh:`111` leads the migration from ``versioneer`` to ``setuptools_scm``.


0.0.3 - 2021-03-23
------------------

- :gh:`88` adds models to vaccinate individuals.
- :gh:`91` adds realistic parameters for when vaccines become effective.


0.0.2 - 2021-03-23
------------------

- :gh:`59` removes the ``optional_state_columns`` which is now controlled by
  ``saved_columns`` as well.
- :gh:`60` adds many more tests to push coverage beyond 70% and enriches the
  documentation.
- :gh:`67` allows to indicate already factorized ``assort_by`` variables to reduce
  memory consumption.
- :gh:`70` follows :gh:`67` and ensures that the unique values of ``assort_by``
  variables are always sorted to maintain a stable ordering. The PR also reworks the
  factorization such that it is only done once.
- :gh:`71` separates recurrent from random contacts and how infections are calculated
  for each type of contact.
- :gh:`72` allows sid to be packaged on PyPI and adds versioneer.
- :gh:`75` passes sid's seed to the testing models.
- :gh:`76` removes ``share_known_cases`` which should now be implemented with testing
  models.
- :gh:`79` implements a multiplier for infection probabilities.
- :gh:`81` sets the default start date for testing models to the first burn-in period of
  the initial conditions.
- :gh:`83` adds an interface to have multiple virus strains with different
  infectiousness.
- :gh:`84` does some clean up in the matching algorithm.
- :gh:`85` adds seeds to events.
- :gh:`86` renames the package such that it is published on PyPI and Anaconda as
  sid-dev.


0.0.1 - 2021-01-05
------------------

- The PRs ranging from :gh:`1` to :gh:`64` form the first release of sid. It is also the
  basis of the report `Die Bedeutung individuellen Verhaltens über den Jahreswechsel für
  die Weiterentwicklung der Covid-19-Pandemie in Deutschland
  <http://ftp.iza.org/sp99.pdf>`_.
