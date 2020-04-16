.. _policies:

========
Policies
========


The most general way of implementing policies is to modify :ref:`contact_models`. However, this can be rather cumbersome. Therefore we offer a quick way of implementing policies that modify contact models or implement testing strategies.


``contact_policies``
====================

Contact policies are multipliers on a contact model that are active in certain periods or when the ``states`` fulfills some condition. Here is an example:

.. code-block:: python

    def contact_policy_is_active(states):
        return states["infectious"].mean() > 0.2

    policies = {
        "work_close": {
            "start": 10,
            "end": 15,
            "multiplier": 0.5,
            "is_active": contact_policy_is_active,
        }


``"work_close"`` is the name of the contact model the policies refers to. ``"start"`` and ``"end"`` define the periods in which the policy is active. They are optional. ``"multiplier"`` will be multiplied with the number of contact. It is no problem if the multiplication leads to non-integer number of contacts. We will automatically round them in a way that preserves the total number of contacts. as good as possible. ``"is_active"`` is a function that returns a bool. This is also optional.


``testing_policies``
====================

Testing policies determine which individuals receive a test. Testing policies are not implemented yet, but will be a dict of dicts of the following form:

.. code-block:: python

    def testing_policy_is_active(states):
        return states["immune"].mean() < 0.5

    {
        "test_old_people": {
            "start": 20,
            "end": 100,
            "query": "age_group == '> 60' & symptoms",
            "probability": 0.7,
            "is_active": testing_policy_is_active,
        }
    }


``"test_old_people"`` is the name of the testing policy. ``"start"`` and ``"end"`` define the periods in which the policy is active. They are optional. ``"query"`` selects the people who get the test. We will handle the case where individuals are selected by multiple testing policies automatically and test them just once. ``"probability"`` can be used to incorporate randomness in the testing policy. It is optional. ``"is_active"`` is a function that returns a bool. This is also optional.
