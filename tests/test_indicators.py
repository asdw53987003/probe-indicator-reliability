import numpy as np
import pandas as pd

from probe_reliability import indicators_table, sums_from_points, thin_records
from probe_reliability.indicators import gap_limit, hdc


def _pass():
    # three successive steps of -2.5 m/s^2 form one episode; the 70 s gap is not differenced
    return pd.DataFrame({'vehicle': 'X', 'unit': 'u', 'pass': 'p', 't': [0, 1, 2, 3, 4, 5, 75, 76],
                         'speed': [25, 25, 22.5, 20, 17.5, 17.5, 30, 30]})


def test_episode_counted_once_and_gap_not_bridged():
    s = sums_from_points(_pass(), tau=1.0)
    assert int(s['hd_events'].iloc[0]) == 1
    km = (25 + 23.75 + 21.25 + 18.75 + 17.5 + 30) / 1000.0
    assert abs(s['km'].iloc[0] - km) < 1e-12


def test_gap_limit():
    assert gap_limit(1) == 60.0
    assert gap_limit(45) == 90.0


def test_separate_runs_count_twice():
    p = pd.DataFrame({'vehicle': 'X', 'unit': 'u', 'pass': 'p', 't': np.arange(8),
                      'speed': [25, 22, 22, 22, 19, 19, 19, 19]})
    assert int(sums_from_points(p)['hd_events'].iloc[0]) == 2


def test_thin_records_keeps_every_tau_th_record_per_vehicle():
    p = pd.DataFrame({'vehicle': ['A'] * 10 + ['B'] * 10, 't': list(range(10)) * 2, 'unit': 'u', 'pass': 'p',
                      'speed': 20.0})
    th = thin_records(p, 3, seed=1)
    for v, g in th.groupby('vehicle'):
        assert len(set(np.asarray(g['t']) % 3)) == 1
        assert len(g) in (3, 4)
    assert len(thin_records(p, 1)) == len(p)


def test_indicator_definitions():
    rec = pd.DataFrame({'vehicle': ['a', 'b'], 'unit': ['u', 'u'], 'km': [1.0, 3.0], 'kmv': [20.0, 3 * 24.0],
                        'kmv2': [400.0, 3 * 576.0], 'kma2': [0.04, 3 * 0.09], 'km_acc': [1.0, 3.0],
                        'hd_events': [1, 2]})
    units = pd.DataFrame({'length_km': [2.0]}, index=pd.Index(['u'], name='unit'))
    t = indicators_table(rec, units)
    assert abs(t.loc['u', 'MSP'] - 23.0) < 1e-12
    assert abs(t.loc['u', 'VSD'] - np.sqrt((400 + 3 * 576) / 4 - 23 ** 2)) < 1e-12
    assert abs(t.loc['u', 'ASD'] - np.sqrt((0.04 + 0.27) / 4)) < 1e-12
    assert abs(t.loc['u', 'HDR'] - 75.0) < 1e-12
    assert abs(t.loc['u', 'veq'] - 2.0) < 1e-12
    assert abs(hdc(75.0, 2.0, 7.0)[()] - 75.0 * 2.0 / 100 / 7) < 1e-12
