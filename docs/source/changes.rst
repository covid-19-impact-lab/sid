Changes
=======

This is a record of all past sid releases and what went into them in reverse
chronological order. Releases follow `semantic versioning <https://semver.org/>`_ and
all releases are available on `Anaconda.org
<https://anaconda.org/covid-19-impact-lab/sid>`_.


0.0.2 - 2021-xx-xx
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
- :gh:`75` passes sid's seed to the testing models.


0.0.1 - 2021-01-05
------------------

- The PRs ranging from :gh:`1` to :gh:`64` form the first release of sid. It is also the
  basis of the report `Die Bedeutung individuellen Verhaltens über den Jahreswechsel für
  die Weiterentwicklung der Covid-19-Pandemie in Deutschland
  <http://ftp.iza.org/sp99.pdf>`_.
