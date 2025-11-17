import pandas as pd
import numpy as np
from missing_fill_methods import (
    fill_linear_window_multi_iter,  # новое: регрессия с окном
    zet_fill_window,                # новое: zet с окном
    fill_remaining_gaps
)

def make_regression_dataset(df, features, n_rounds=5, window=2, min_filled=5):
    """
    Восстанавливает пропуски итеративно регрессией с учетом ±window соседей.
    """
    df_reg = df.copy()
    for _ in range(n_rounds):
        na_sum_before = df_reg[features].isna().sum().sum()
        for target in features:
            other_feats = [f for f in features if f != target]
            # <<< ВАЖНО! Это обновляет всю таблицу каждый раз, не одну колонку >>>
            df_reg = fill_linear_window_multi_iter(df_reg, target, other_feats, window=window, min_filled=min_filled)
        na_sum_after = df_reg[features].isna().sum().sum()
        if na_sum_after == na_sum_before:
            break
    #df_reg = fill_remaining_gaps(df_reg, fill_strategy="mean", cols=features)
    return df_reg

def make_zet_dataset(df, features, n_rounds=5, window=2, min_neighbors=1):
    """
    Восстанавливает пропуски итеративно zet-алгоритмом с учетом ±window соседей.
    """
    df_zet = df.copy()
    for _ in range(n_rounds):
        changed_any = False
        for col in features:
            df_zet, changed = zet_fill_window(df_zet, col, window=window, min_neighbors=min_neighbors)
            changed_any = changed_any or changed
        if not changed_any:
            break
    #df_zet = fill_remaining_gaps(df_zet, fill_strategy="mean", cols=features)
    return df_zet

if __name__ == "__main__":
    df = pd.read_csv("isodata_blocks.csv", sep=",")
    features = [
        "Пол",
        "Год_выдачи_паспорта",
        "СНИЛС_цифр",
        "Симптомы_код",
        "Врач_код",
        "Часы_до_визита",
        "Анализы_код",
        "Часы_до_анализа",
        "Стоимость",
        "Банк_код"
    ]
    window = 2
    min_filled = 1
    min_neighbors = 1
    df_regression = make_regression_dataset(df, features, n_rounds=5, window=window, min_filled=min_filled)
    df_regression.to_csv("dataset_regression.csv", index=False)

    df_zet = make_zet_dataset(df, features, n_rounds=5, window=window, min_neighbors=min_neighbors)
    df_zet.to_csv("dataset_zet.csv", index=False)