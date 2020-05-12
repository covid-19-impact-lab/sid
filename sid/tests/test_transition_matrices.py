import pandas as pd
from pandas.testing import assert_frame_equal

from sid.transition_matrices import create_transition_matrix_from_own_prob
from sid.transition_matrices import join_transition_matrices


def test_join_transition_matrices():
    t1 = create_transition_matrix_from_own_prob(0.6, ["bla", "blubb"])
    t2 = create_transition_matrix_from_own_prob(pd.Series([0.6, 0.7], index=["a", "b"]))
    exp_data = [
        [0.36, 0.24, 0.24, 0.16],
        [0.18, 0.42, 0.12, 0.28],
        [0.24, 0.16, 0.36, 0.24],
        [0.12, 0.28, 0.18, 0.42],
    ]
    ind_tups = [("bla", "a"), ("bla", "b"), ("blubb", "a"), ("blubb", "b")]

    ind = pd.MultiIndex.from_tuples(ind_tups)
    expected = pd.DataFrame(exp_data, columns=ind, index=ind)

    calculated = join_transition_matrices([t1, t2])

    assert_frame_equal(calculated, expected)
