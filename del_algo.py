from typing import List, Optional, Dict, Any, Set
import pandas as pd
import numpy as np
import os
import sys

try:
    from scipy.stats import pearsonr
except Exception:
    pearsonr = None

def _pearson_fallback(x: np.ndarray, y: np.ndarray):
    """Compute Pearson r, p-value fallback (p-value approximate via t-stat)."""
    x = x.astype(float)
    y = y.astype(float)
    if x.size < 2:
        return np.nan, np.nan
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom == 0:
        return np.nan, np.nan
    r = (xm * ym).sum() / denom
    n = x.size
    if n <= 2:
        return float(r), np.nan
    try:
        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
        from scipy.stats import t as t_dist
        p = 2 * t_dist.sf(abs(t_stat), df=n-2)
    except Exception:
        p = np.nan
    return float(r), float(p)

def compute_pearson(true: pd.Series, other: pd.Series, min_samples: int = 3):
    """
    Compute Pearson correlation between two pandas Series on paired non-NaN entries.
    Returns (r, p) or (np.nan, np.nan) if insufficient data.
    """
    mask = (~true.isna()) & (~other.isna())
    if mask.sum() < min_samples:
        return np.nan, np.nan
    x = true[mask].to_numpy(dtype=float)
    y = other[mask].to_numpy(dtype=float)
    if pearsonr is not None:
        try:
            r, p = pearsonr(x, y)
            return float(r), float(p)
        except Exception:
            return _pearson_fallback(x, y)
    else:
        return _pearson_fallback(x, y)

def select_features_for_targets(
    df: pd.DataFrame,
    targets: Optional[List[str]] = None,
    candidates: Optional[List[str]] = None,
    top_k: int = 5,
    threshold: Optional[float] = None,
    min_samples: int = 10,
    numeric_only: bool = True
) -> Dict[str, Any]:
    """
    For each target column, compute Pearson correlation with candidate columns and
    return sorted lists of correlations.

    Returns:
      {
        "per_target": { target: DataFrame(columns=["feature","r","p","n_pairs"]) , ... },
        "selected_features": set(...)   # union of chosen features for all targets
      }
    """
    if numeric_only:
        df_num = df.select_dtypes(include=[np.number]).copy()
    else:
        df_num = df.copy()

    if df_num.shape[1] == 0:
        raise ValueError("No numeric columns found in DataFrame (or numeric_only=False requested).")

    if targets is None:
        targets = [c for c in df_num.columns]
    else:
        targets = [t for t in targets if t in df_num.columns]
        if not targets:
            raise ValueError("No targets left after filtering to numeric columns.")

    if candidates is None:
        candidates = [c for c in df_num.columns]
    else:
        candidates = [c for c in candidates if c in df_num.columns]

    per_target: Dict[str, pd.DataFrame] = {}
    selected_features: Set[str] = set()

    for tgt in targets:
        rows = []
        for cand in candidates:
            if cand == tgt:
                continue
            r, p = compute_pearson(df_num[tgt], df_num[cand], min_samples=min_samples)
            n_pairs = int((~df_num[tgt].isna() & ~df_num[cand].isna()).sum())
            rows.append({"feature": cand, "r": r, "p": p, "n_pairs": n_pairs})
        df_corr = pd.DataFrame(rows)
        df_corr = df_corr.dropna(subset=["r"]).copy()
        if df_corr.empty:
            per_target[tgt] = df_corr
            continue
        df_corr["abs_r"] = df_corr["r"].abs()
        df_corr = df_corr.sort_values(by="abs_r", ascending=False).reset_index(drop=True)
        df_corr = df_corr[["feature", "r", "p", "n_pairs"]]

        if threshold is not None:
            chosen = df_corr[df_corr["r"].abs() >= threshold]["feature"].tolist()
        else:
            chosen = df_corr["feature"].head(top_k).tolist()

        per_target[tgt] = df_corr
        selected_features.update(chosen)

    return {"per_target": per_target, "selected_features": selected_features}

def _print_per_target_results(res: Dict[str, Any], topn: int = 10, out_dir: Optional[str] = None):
    per_target = res.get("per_target", {})
    print("\n=== DEL selection results ===\n")
    for tgt, df_corr in per_target.items():
        print(f"Target: {tgt}")
        if df_corr.empty:
            print("  (не достаточно данных для расчёта корреляций)\n")
            continue
        print(df_corr.head(topn).to_string(index=False))
        print()
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.join(out_dir, f"del_corr_{tgt}.csv")
            df_corr.to_csv(fname, index=False)
            print(f"  saved: {fname}\n")
    sel = res.get("selected_features", set())
    print("Union of selected features (across targets):")
    print(sorted(list(sel)))
    print()

def main():
    """
    Simplified main: read 'medical_dataset.csv' from current directory (no args),
    run selection with defaults and print results.
    """
    csv_path = "result.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}", file=sys.stderr)
        print("Place 'medical_dataset.csv' in the current directory and re-run.", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV '{csv_path}': {e}", file=sys.stderr)
        return 3

    print(f"Loaded '{csv_path}' shape={df.shape}")

    # Default parameters: use all numeric columns as targets, top_k=5, min_samples=10
    try:
        res = select_features_for_targets(
            df=df,
            targets=None,        # all numeric
            candidates=None,     # all numeric
            top_k=5,
            threshold=None,
            min_samples=10,
            numeric_only=True
        )
    except Exception as e:
        print("Error during selection:", e, file=sys.stderr)
        return 4

    _print_per_target_results(res, topn=5, out_dir=None)
    return 0

if __name__ == "__main__":
    sys.exit(main())