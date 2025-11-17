import pandas as pd
import numpy as np
from missing_fill_methods import (
    fill_linear_window_multi_iter,  # новое: регрессия с окном
    zet_fill_window,                # новое: zet с окном
    fill_remaining_gaps
)

def make_regression_dataset(df, features, n_rounds=5, window=2):
    """
    Восстанавливает пропуски МАКСИМАЛЬНО регрессией с учетом ±window соседей, затем дополняет средними.
    Проходит несколько итераций, чтобы новые значения могли помочь заполнить больше.
    """
    df_reg = df.copy()
    for r in range(n_rounds):
        # Сначала строки с одним пробелом
        for target in features:
            other_feats = [f for f in features if f != target]
            df_one_empty = df_reg[df_reg[target].isna() & df_reg[other_feats].notnull().all(axis=1)]
            if not df_one_empty.empty:
                df_reg.loc[df_one_empty.index, :] = fill_linear_window_multi_iter(
                    df_one_empty, target, other_feats, window=window
                )
        # Затем строки с несколькими пробелами
        for target in features:
            other_feats = [f for f in features if f != target]
            # Есть хотя бы один заполненный признак
            df_multi_empty = df_reg[
                df_reg[target].isna() & (df_reg[other_feats].notnull().sum(axis=1) >= 1)
            ]
            if not df_multi_empty.empty:
                df_reg.loc[df_multi_empty.index, :] = fill_linear_window_multi_iter(
                    df_multi_empty, target, other_feats, window=window
                )
    # Финальное заполнение средними/модой
    #df_reg = fill_remaining_gaps(df_reg, fill_strategy="mean", cols=features)
    return df_reg

def make_zet_dataset(df, features, n_rounds=5, window=2):
    """
    Восстанавливает пропуски МАКСИМАЛЬНО zet-алгоритмом с учетом ±window соседей, затем дополняет средними.
    Проходит несколько итераций, чтобы новые значения могли помочь заполнить больше.
    """
    df_zet = df.copy()
    for r in range(n_rounds):
        for col in features:
            df_zet = zet_fill_window(df_zet, col, window=window)
    df_zet = fill_remaining_gaps(df_zet, fill_strategy="mean", cols=features)
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
    df_regression = make_regression_dataset(df, features, n_rounds=5, window=window)
    df_regression.to_csv("dataset_regression.csv", index=False)

    #df_zet = make_zet_dataset(df, features, n_rounds=5, window=window)
    df_zet.to_csv("dataset_zet.csv", index=False)