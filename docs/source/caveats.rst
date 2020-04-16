.. _caveats:

=======
Caveats
=======


Interpretation of contact channels
==================================

In the current implementation we cannot interpret easily which contact types lead to most infection because we first do all matchings for the first contact type, then for the second and so on. This means that the first contact type has higher chances to infect people because for the second contact type there are already more immune people. The bias should however be small. If we get interesting results that some contact type is responsible for a large share of infections, we can move it to the end and find out if the results persist.


Problems in the matching process
================================

It is relatively hard to generate a feasible matching from a list of contact numbers. For computational reasons, we implement this as a multi-stage sampling, where we first sample from which group (defined by age, region or other variables) the second person is and then sample the person. The second stage group probabilities are contacts each individual is scheduled to have divided by the sum of contacts from that group. The following problematic cases can arise:
1. The first and second person are the same
2. The sampled group does not contain individuals with scheduled contacts
In both cases we will simply not have a contact. More sophisticated solutions are possible, but since the number of contacts has a large random component anyways, they are probably not necessary.
