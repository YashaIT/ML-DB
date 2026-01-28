from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pick_two_regions(df: pd.DataFrame) -> tuple[str, str] | None:
    regs = [r for r in df["region"].dropna().astype(str).unique().tolist() if r.strip()]
    if len(regs) < 2:
        return None
    regs = sorted(regs)
    return regs[0], regs[1]


def mann_whitney_or_ttest(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Выбираем тест:
      - если обе выборки примерно нормальные (Shapiro p>0.05) и размеры достаточны -> t-test
      - иначе -> Mann–Whitney U
    """
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    out = {"n_x": int(len(x)), "n_y": int(len(y))}
    if len(x) < 8 or len(y) < 8:
        out["test"] = "insufficient_data"
        out["p_value"] = None
        return out

    # Shapiro (ограничение: лучше до 5000)
    xs = x[:5000]
    ys = y[:5000]
    p1 = stats.shapiro(xs).pvalue if len(xs) >= 8 else 0.0
    p2 = stats.shapiro(ys).pvalue if len(ys) >= 8 else 0.0

    normal = (p1 > 0.05) and (p2 > 0.05)

    if normal:
        res = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        out["test"] = "welch_ttest"
        out["p_value"] = float(res.pvalue)
        out["stat"] = float(res.statistic)
    else:
        res = stats.mannwhitneyu(x, y, alternative="two-sided")
        out["test"] = "mann_whitney_u"
        out["p_value"] = float(res.pvalue)
        out["stat"] = float(res.statistic)

    out["mean_x"] = float(np.mean(x)) if len(x) else None
    out["mean_y"] = float(np.mean(y)) if len(y) else None
    out["median_x"] = float(np.median(x)) if len(x) else None
    out["median_y"] = float(np.median(y)) if len(y) else None
    return out


def run_region_tests(df: pd.DataFrame, features: list[str], region_a: str, region_b: str) -> pd.DataFrame:
    a = df[df["region"].astype(str) == region_a]
    b = df[df["region"].astype(str) == region_b]

    rows = []
    for f in features:
        x = a[f].astype(float).to_numpy()
        y = b[f].astype(float).to_numpy()
        r = mann_whitney_or_ttest(x, y)
        r["feature"] = f
        r["region_a"] = region_a
        r["region_b"] = region_b
        rows.append(r)

    out = pd.DataFrame(rows).sort_values("p_value", na_position="last")
    return out
