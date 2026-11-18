"""Comparisons from the phenotype statistics table, with the noise-adjusted columns (no Qt needed)."""
import pandas as pd

from napari_cochlea_phenotyping._results import phenotype_comparisons


def _table(diff):
    return pd.DataFrame([dict(fields="flagged fields left out", turn="all turns", tested=True, family="bundle panel: median", cell_type="OHC1",
                              measure="glcm_contrast_d1", control_mean=5.0, mut_mean=5.0 + diff, mut_difference=diff, mut_P=0.02, mut_P_adj=0.1,
                              P_three_group=0.02, robustness="robust")])


def test_phenotype_comparisons_with_adjustment():
    c = phenotype_comparisons(_table(1.6), _table(1.0))
    assert len(c) == 1 and c[0].group == "mut" and c[0].difference == 1.6 and c[0].difference_adjusted == 1.0
    assert c[0].level == "partly checked" and any("noise adjustment" in w for w in c[0].warnings)


def test_phenotype_comparisons_unchanged_adjustment_has_no_note():
    c = phenotype_comparisons(_table(1.6), _table(1.6))
    assert not any("noise adjustment" in w for w in c[0].warnings)
