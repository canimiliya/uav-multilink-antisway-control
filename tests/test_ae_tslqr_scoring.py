def ae_score(position, orientation, acquisition, rate):
    return position + 0.25 * orientation + 0.50 * acquisition + 0.05 * rate


def test_score_is_monotone_in_each_penalty():
    base = ae_score(1, 1, 1, 1)
    assert ae_score(1.1, 1, 1, 1) > base
    assert ae_score(1, 1.1, 1, 1) > base
    assert ae_score(1, 1, 1.1, 1) > base
    assert ae_score(1, 1, 1, 1.1) > base

