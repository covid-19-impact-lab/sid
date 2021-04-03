Changes
=======

This is a record of all past sid releases and what went into them in reverse
chronological order. Releases follow `semantic versioning <https://semver.org/>`_ and
all releases are available on `Anaconda.org
<https://anaconda.org/covid-19-impact-lab/sid>`_.


0.0.4 - 2021-xx-xx
------------------

- :gh:`92` adds an interface for rapid tests.
- :gh:`93` enhances the validation mechanism.
- :gh:`94` scales a single vaccination model to multiple vaccination models.
- :gh:`96` changes the initialization of countdowns and removes draws created for
  countdowns without randomness.


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
