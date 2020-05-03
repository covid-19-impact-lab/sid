.. _caveats:

=======
Caveats
=======

Problems in the matching process
--------------------------------

It is relatively hard to generate a feasible matching from a list of contact numbers.
For computational reasons, we implement this as a multi-stage sampling, where we first
sample from which group (defined by age, region or other variables) the second person is
and then sample the person. The second stage group probabilities are contacts each
individual is scheduled to have divided by the sum of contacts from that group. The
following problematic cases can arise:

1. The first and second person are the same

2. The sampled group does not contain individuals with scheduled contacts

In both cases we will simply not have a contact. More sophisticated solutions are
possible, but since the number of contacts has a large random component anyways, they
are probably not necessary.
