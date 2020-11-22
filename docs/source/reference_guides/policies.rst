.. _policies:

========
Policies
========

In SID we can implement nearly any type of policy as a modification of the
:ref:`contact_models`. However, to keep things separable and modular, policies can also
specified outside the contact models in a separate, specialized `contact_policies`
dictionary.


``contact_policies``
--------------------

The contact policies are a nested dictionary, mapping the policy's name to its
specification. You can choose any name you wish.

The specification must contain an `affected_contact_model` entry, a `policy` entry and
provide when the policy is active. The `affected_contact_model` gives the name of the
contact model whose output, the `contacts` (a pandas.Series with the same index as the
`states` DataFrame), will be modified while the policy is active. The `policy` entry is
either a float or a function. If it is a float, the contacts are simply multiplied with
the this number. For non-recurrent contact models we will round the results for you to
have the wanted reduction on average. If it is a function, it should take the `states`,
`contacts` and `params` as inputs and return a modified `contacts` Series.

To specify when the policy is active, you have three options:

- You provide a `start` and an `end` date. For example, you could specify school
  closures during the first lockdown which started on the 22nd of March and ended on the
  20th of April as following

  .. code-block:: python

    {
        "1st_lockdown_school": {
            "affected_contact_model": "school",
            "policy": 0,
            "start": "2020-03-22",
            "end": "2020-04-20",
        },
    }

- You provide an `is_active` function that maps the `states` to either `True` or
  `False`. Then the policy will be called whenever the `is_active` function returns
  True.

- You provide both and the policy is active on those periods between `start` and `end`
  where the `is_active` policy returns to `True`.

For an example on how to specify contact policies, look at the `contact policies
tutorial <../tutorials/how_to_specify_policies.ipynb>`_
