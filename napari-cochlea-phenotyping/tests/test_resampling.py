import numpy as np
import pandas as pd
import pytest

from napari_cochlea_phenotyping._resampling import (cluster_bootstrap, compare_groups_by_mouse, distinct_resamples,
                                                    field_summary, small_sample_factor)


def test_distinct_resamples():
    assert [distinct_resamples(n) for n in (1, 2, 3, 8)] == [1, 3, 10, 6435]


def test_small_sample_factor():
    assert small_sample_factor(3) == pytest.approx(2.69, abs=0.01)
    assert small_sample_factor(12) == pytest.approx(1.17, abs=0.01)


def test_identical_mice_give_no_width():
    v = np.tile([1.0, 2.0, 3.0], 4); c = np.repeat(["a", "b", "c", "d"], 3)
    r = cluster_bootstrap(v, c, n_boot=500)
    assert r["low"] == pytest.approx(2.0) and r["high"] == pytest.approx(2.0) and r["n_clusters"] == 4


def test_mouse_interval_wider_than_cell_interval():
    rng = np.random.default_rng(0)
    mice = np.repeat(np.arange(6), 20); v = rng.normal(0, 1, 6)[mice] + rng.normal(0, 0.3, mice.size)
    r = cluster_bootstrap(v, mice, n_boot=2000)
    r_cells = cluster_bootstrap(v, np.arange(v.size), n_boot=2000)
    assert (r["high"] - r["low"]) > 2 * (r_cells["high"] - r_cells["low"])


def test_field_summary():
    r = field_summary([1.0, 2.0, 3.0, np.nan, 4.0])
    assert r["cells"] == 4 and r["median"] == pytest.approx(2.5) and r["q25"] == pytest.approx(1.75)


def test_compare_groups_by_mouse():
    d = pd.DataFrame(dict(group=np.repeat(["control"] * 3 + ["mutant"] * 3, 5),
                          mouse=np.repeat(list("abcdef"), 5),
                          h=np.repeat([4.0, 4.1, 4.2, 3.0, 3.1, 3.2], 5)))
    r = compare_groups_by_mouse(d, "h", "group", "mouse", "control", "mutant")
    assert r["difference"] == pytest.approx(-1.0)
    assert r["relabelings"] == 20 and r["p_permutation"] == pytest.approx(0.1) and r["smallest_possible_p"] == pytest.approx(0.1)
    assert len(r["leave_one_mouse_out"]) == 6 and r["low"] < -1.0 < r["high"]


def test_mouse_in_both_groups_rejected():
    d = pd.DataFrame(dict(group=["c", "m"], mouse=["a", "a"], h=[1.0, 2.0]))
    with pytest.raises(ValueError):
        compare_groups_by_mouse(d, "h", "group", "mouse", "c", "m")
