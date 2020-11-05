.. _policies:

========
Policies
========

The most general way of implementing policies is to modify :ref:`contact_models`.
However, this can be rather cumbersome. Therefore, we offer a quick way of implementing
policies that modify contact models or implement testing strategies.


``contact_policies``
--------------------

Contact policies are multipliers on a contact model that are active on certain dates or
when ``states`` fulfills some condition. Here is an example:

.. code-block:: python

    def activate_contact_policy(states):
        """Activate policy if 20% of the population are infectious."""
        return states["infectious"].mean() > 0.2


    policies = {
        "work_close": {
            "start": "2020-02-01",
            "end": "2020-02-15",
            "multiplier": 0.5,
            "is_active": activate_contact_policy,
        }
    }

``"work_close"`` is the name of the contact model the policies refers to. ``"start"``
and ``"end"`` define the time period in dates when the policy is active. They are
optional. ``"multiplier"`` will be multiplied with the number of contacts. It is no
problem if the multiplication leads to non-integer number of contacts. We will
automatically round them in a way that preserves the total number of contacts as well as
possible. ``"is_active"`` is a function that returns a bool. This is also optional. A
policy is only active if ``is_active & pol["start"] <= date <= pol["end"]``.
