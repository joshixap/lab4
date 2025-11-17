import pandas as pd
import numpy as np
from missing_fill_methods import fill_linear_multi_iter, zet_fill_simple, fill_remaining_gaps

def make_regression_dataset(df, features, n_rounds=5):
    """
    Восстанавливает пропуски МАКСИМАЛЬНО регрессией (без zet), затем дополняет средними.
    Проходит несколько итераций восстановления, чтобы новые значения могли помочь заполнить больше.
    """
    df_reg = df.copy()
    for r in range(n_rounds):
        # Сначала строки с одним пробелом
        for target in features:
            df_one_empty = df_reg[df_reg[target].isna() & df_reg[[f for f in features if f != target]].notnull().all(axis=1)]
            if not df_one_empty.empty:
                df_reg.loc[df_one_empty.index, :] = fill_linear_multi_iter(
                    df_one_empty, target, [f for f in features if f != target]
                )
        # Затем строки с несколькими пробелами, если по признакам можно что-то восстановить
        for target in features:
            df_multi_empty = df_reg[
                df_reg[target].isna() & (df_reg[[f for f in features if f != target]].notnull().sum(axis=1) >= 1)
            ]
            if not df_multi_empty.empty:
                df_reg.loc[df_multi_empty.index, :] = fill_linear_multi_iter(
                    df_multi_empty, target, [f for f in features if f != target]
                )
    # Остаточные пропуски - средними
    df_reg = fill_remaining_gaps(df_reg, fill_strategy="mean", cols=features)
    return df_reg

def make_zet_dataset(df, features, n_rounds=5):
    """
    Восстанавливает пропуски МАКСИМАЛЬНО zet-алгоритмом (по верх/низ соседам), затем дополняет средними.
    Проходит несколько итераций восстановления, чтобы новые значения могли помочь заполнить больше.
    """
    df_zet = df.copy()
    for r in range(n_rounds):
        for col in features:
            df_zet = zet_fill_simple(df_zet, col)
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
    # Построение максимально восстановленного датасета только регрессией
    df_regression = make_regression_dataset(df, features, n_rounds=5)
    df_regression.to_csv("dataset_regression.csv", index=False)

    # Построение максимально восстановленного датасета только zet-алгоритмом
    df_zet = make_zet_dataset(df, features, n_rounds=5)
    df_zet.to_csv("dataset_zet.csv", index=False)