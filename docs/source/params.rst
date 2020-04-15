.. _params:

========
`params`
========


`params` is a DataFrame that contains all parameters that quantify how the disease spreads in the `"value"` column. Putting those parameters into a DataFrame, allows us to optimize over them using `estimagic <https://estimagic.readthedocs.io/en/latest/>`_. Make sure to read about the basic structure of `params DataFrames <https://estimagic.readthedocs.io/en/latest/optimization/params.html>`_ in estimagic, before you continue.


`params` has a two level index. The first level is a category, the second is a name. Currently, we have the following categories:


-
