import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def neighbors_window(arr, i, window=2):
    n = len(arr)
    neighbors = []
    for w in range(1, window+1):
        if i-w >= 0 and not pd.isnull(arr[i-w]):
            neighbors.append(arr[i-w])
    for w in range(1, window+1):
        if i+w < n and not pd.isnull(arr[i+w]):
            neighbors.append(arr[i+w])
    return neighbors

def zet_fill_window_full(df, features, window=2, min_neighbors=1, n_rounds=3):
    df = df.copy()
    for _ in range(n_rounds):
        changed_any = False
        # Проходим по всем строкам и признакам
        for col in features:
            arr = df[col].values.copy()
            is_numtype = pd.api.types.is_numeric_dtype(df[col])
            n = len(arr)
            for i in range(n):
                if pd.isnull(arr[i]):
                    neighbors = neighbors_window(arr, i, window=window)
                    if len(neighbors) >= min_neighbors:
                        if is_numtype:
                            arr[i] = np.mean([float(x) for x in neighbors])
                        else:
                            mode = pd.Series(neighbors).mode()
                            arr[i] = mode.iloc[0] if not mode.empty else neighbors[0]
                        changed_any = True
            df[col] = arr
        if not changed_any:
            break
    return df

def add_window_features(df, col, window=2):
    feats = pd.DataFrame(index=df.index)
    arr = df[col].values
    for w in range(1, window+1):
        feats[f"{col}_prev{w}"] = pd.Series(arr).shift(w)
        feats[f"{col}_next{w}"] = pd.Series(arr).shift(-w)
    return feats

def fill_linear_window_multi_iter_full(df, features, window=2, min_filled=1, n_rounds=3):
    df = df.copy()
    for _ in range(n_rounds):
        na_sum_before = df[features].isna().sum().sum()
        for target_col in features:
            feature_cols = [f for f in features if f != target_col]
            window_feats = add_window_features(df, target_col, window)
            full_features = pd.concat([df[feature_cols], window_feats], axis=1)
            mask_train = (
                df[target_col].notnull() & (full_features.notnull().sum(axis=1) >= min_filled)
            )
            if mask_train.sum() < 2:
                continue
            X_train = full_features.loc[mask_train]
            y_train = df.loc[mask_train, target_col]
            X_train = X_train.fillna(X_train.mean())
            model = LinearRegression()
            model.fit(X_train, y_train)
            mask_missing = (
                df[target_col].isnull() & (full_features.notnull().sum(axis=1) >= min_filled)
            )
            if mask_missing.sum() == 0:
                continue
            X_missing = full_features.loc[mask_missing]
            X_missing = X_missing.fillna(X_train.mean())
            new_values = model.predict(X_missing)
            for idx, value in zip(mask_missing.index, new_values):
                if pd.isnull(df.at[idx, target_col]):
                    df.at[idx, target_col] = value
        na_sum_after = df[features].isna().sum().sum()
        if na_sum_after == na_sum_before:
            break
    return df

def fill_remaining_gaps(df, fill_strategy="mean", cols=None):
    df = df.copy()
    if cols is None:
        cols = df.columns
    for col in cols:
        is_na = df[col].isna()
        if is_na.sum() == 0:
            continue
        is_numtype = pd.api.types.is_numeric_dtype(df[col])
        if is_numtype and fill_strategy == "mean":
            fill_value = df[col].dropna().mean()
        elif is_numtype and fill_strategy == "median":
            fill_value = df[col].dropna().median()
        else:
            mode = df[col].dropna().mode()
            fill_value = mode.iloc[0] if not mode.empty else None
        df.loc[is_na, col] = fill_value
    return df

if __name__ == "__main__":
    df = pd.read_csv("isodata.csv", sep=",")
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
    min_neighbors = 1
    min_filled = 1     # максимальное заполнение
    n_rounds = 7

    # 1. Zet по всему df — несколько полных проходов
    df = zet_fill_window_full(df, features, window=window, min_neighbors=min_neighbors, n_rounds=n_rounds)

    # 2. Линейная регрессия по всему df — несколько полных проходов
    df = fill_linear_window_multi_iter_full(df, features, window=window, min_filled=min_filled, n_rounds=n_rounds)

    # 3. Финальное заполнение средними/модой ВСЕХ ячеек
    df = fill_remaining_gaps(df, fill_strategy="mean", cols=features)

    for col in features:
        # только если колонка числовая и имеет дробные значения
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = np.ceil(df[col]).astype(int)

    df.to_csv("isodata_filled.csv", index=False)