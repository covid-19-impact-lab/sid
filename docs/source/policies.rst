.. _policies:

==========
`policies`
==========


The most general way of implementing policies is to modify :ref:`contact_models`. However, this can be rather cumbersome. Therefore we offer a quick way of implementing policies that act as multipliers on contacts of a certain type. A policy is a dictionary. Here is an example:


.. code-block:: python

    def contact_policy_is_active(states, rates):
        return states["infectious"].mean() > 0.2


    def testing_policy_is_active(states, rates):
        return states["immune"].mean() < 0.5

    policies = [
    {
        "policy_type": "contact"
        "start": 10,
        "end": 15,
        "contact_id":
        "multiplier": 0.5
        "is_active": contact_policy_is_active
    }


    {
        "policy_type": "testing",
        "start": 20,
        "end": 100,
        "query": "age_group == '> 60' & symptoms",
        "probability": 0.7,
        "is_active": testing_policy_is_active,
    }
