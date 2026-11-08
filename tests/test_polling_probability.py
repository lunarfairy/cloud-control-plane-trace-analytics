"""Periodic sampling of fragmented violation intervals under a uniform phase."""
import pytest

from cptrace.study2 import polling_recall


@pytest.mark.parametrize('segments,expected', [
    ([(0, 5), (60, 65)], 5 / 60),  # same phases in two periods
    ([(0, 5), (30, 35)], 10 / 60),
    ([(55, 65)], 10 / 60),  # projection crosses the period boundary
    ([(0, 60)], 1.0),
    ([(0, 5), (3, 8)], 8 / 60),
    ([], 0.0),
])
def test_prediction_is_fraction_of_phases_that_detect_episode(monkeypatch, segments, expected):
    monkeypatch.setattr('cptrace.study2.A.oracle_timeline', lambda r: ([0.0, 120.0], []))
    record = {'sid': 'S0000'}
    episode = dict(code='F1', wall=120.0, cfg=sum(b-a for a,b in segments),
                   segs=segments, censored=False)
    result = polling_recall([record], {'S0000': [episode, dict(episode, code='S1')]}, deltas=(1,), boot=2)
    assert result['1']['pred_F'] == pytest.approx(expected)
    assert result['1']['pred_S'] == pytest.approx(expected)
