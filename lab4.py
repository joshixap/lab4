import sys
import pandas as pd
import numpy as np
import re
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSpinBox,
    QLabel, QFileDialog, QComboBox, QTableWidget, QTableWidgetItem, QTextEdit, QTabWidget, QGroupBox, QLineEdit
)
from PyQt5.QtCore import Qt

# ======= DIGITIZATION FUNCTIONS =======

import math
from datetime import datetime

def digitize_fio(fio):
    parts = fio.upper().split()
    if any(p.endswith("А") or p.endswith("Я") or p.endswith("ВНА") or p.endswith("ИНА") for p in parts):
        return 1
    return 0

def digitize_passport(passport):
    digits = re.sub(r"\D", "", passport if pd.notnull(passport) else "")
    if len(digits) >= 4:
        year = int(digits[2:4])
        return 2000 + year
    return np.nan

def digitize_snils(snils):
    snils_digits = re.sub(r"\D", "", snils if pd.notnull(snils) else "")
    if len(snils_digits) >= 11:
        num = snils_digits[:9]
        ctrl = snils_digits[-2:]
        s = sum(int(num[i]) * (9 - i) for i in range(9))
        if s < 100:
            chk = f"{s:02d}"
        elif s in [100, 101]:
            chk = "00"
        elif s > 101:
            rem = s % 101
            chk = f"{rem:02d}" if rem < 100 else "00"
        else:
            chk = "00"
        return int(chk + ctrl)
    return np.nan

def digitize_symptoms(symptoms, symptoms_csv):
    group_weights = [0.42, 0.28, 0.14, 0.14]
    symptoms_map = {}
    for idx, line in enumerate(symptoms_csv, 1):
        items = line.split(";")
        if len(items) >= 3:
            name, _, group = items
            symptoms_map[name.strip().lower()] = (idx, int(group))
    input_symptoms = [s.strip().lower() for s in (symptoms if pd.notnull(symptoms) else "").split(",")]
    digitsum = 0
    last_group = 1
    for s in input_symptoms:
        if s in symptoms_map:
            idx, group = symptoms_map[s]
            weight = group_weights[group - 1]
            digitsum += idx * weight
            last_group = group
    return int(math.floor(digitsum)) * 10 + last_group

def digitize_doctor(doctor, specialties_csv):
    group_weights = [0.4, 0.37, 0.15, 0.03, 0.05]
    specialties_map = {}
    for idx, line in enumerate(specialties_csv, 1):
        items = line.split(";")
        if len(items) >= 2:
            name, group = items
            specialties_map[name.strip().lower()] = (idx, int(group))
    doctor_clean = (doctor if pd.notnull(doctor) else "").strip().lower()
    if doctor_clean in specialties_map:
        idx, group = specialties_map[doctor_clean]
        weight = group_weights[group - 1]
        value = int(math.floor(idx * weight)) * 10 + group
        return value
    return np.nan

def digitize_visit_date(dt_str):
    try:
        if pd.isnull(dt_str):
            return np.nan
        visit_date = datetime.strptime(str(dt_str)[:16], "%Y-%m-%dT%H:%M")
        start_date = datetime(2022, 1, 1, 0, 0)
        diff = visit_date - start_date
        hrs = int(diff.total_seconds() // 3600)
        return hrs
    except Exception:
        return np.nan

def digitize_analyses(analyses, analyses_csv):
    group_weights = [0.51, 0.34, 0.15]
    analyses_map = {}
    for idx, line in enumerate(analyses_csv, 1):
        items = line.split(";")
        if len(items) >= 2:
            name, group = items
            analyses_map[name.strip().lower()] = (idx, int(group))
    input_analyses = [a.strip().lower() for a in (analyses if pd.notnull(analyses) else "").split(",")]
    digitsum = 0
    last_group = 1
    for a in input_analyses:
        if a in analyses_map:
            idx, group = analyses_map[a]
            weight = group_weights[group - 1]
            digitsum += idx * weight
            last_group = group
    return int(math.floor(digitsum)) * 10 + last_group

def digitize_test_date(dt_str):
    try:
        if pd.isnull(dt_str):
            return np.nan
        test_date = datetime.strptime(str(dt_str)[:16], "%Y-%m-%dT%H:%M")
        start_date = datetime(2022, 1, 1, 0, 0)
        diff = test_date - start_date
        hrs = int(diff.total_seconds() // 3600)
        return hrs
    except Exception:
        return np.nan

def digitize_cost(cost):
    try:
        return float(cost)
    except Exception:
        return np.nan

def digitize_bank(card_num):
    digits = (card_num if pd.notnull(card_num) else "").replace(" ", "")[:6]
    if digits.startswith("2202") or digits.startswith("2200"):
        return 1
    elif digits.startswith("5228") or digits.startswith("5389") or digits.startswith("5211") or digits.startswith("5112"):
        return 2
    elif digits.startswith("4039") or digits.startswith("4377") or digits.startswith("4986") or digits.startswith("4306"):
        return 3
    else:
        return 4

def digitize_row(row, symptoms_csv, specialties_csv, analyses_csv):
    return [
        digitize_fio(row[0]),                          # ФИО -> пол (0/1)
        digitize_passport(row[1]),                     # Паспорт -> год выдачи
        digitize_snils(row[2]),                        # СНИЛС -> контрольное+2
        digitize_symptoms(row[3], symptoms_csv),       # Симптомы -> код
        digitize_doctor(row[4], specialties_csv),      # Врач -> код
        digitize_visit_date(row[5]),                   # Дата посещения -> часы
        digitize_analyses(row[6], analyses_csv),       # Анализы -> код
        digitize_test_date(row[7]),                    # Дата анализов -> часы
        digitize_cost(row[8]),                         # Стоимость -> float
        digitize_bank(row[9]),                         # Карта -> код банка
    ]

def digitize_dataset(df, symptoms_csv, specialties_csv, analyses_csv):
    cols = [
        "Пол", "Год_выдачи_паспорта", "СНИЛС_цифр", "Симптомы_код",
        "Врач_код", "Часы_до_визита", "Анализы_код", "Часы_до_анализа",
        "Стоимость", "Банк_код"
    ]
    digitized = []
    for _, row in df.iterrows():
        vals = digitize_row(row.tolist(), symptoms_csv, specialties_csv, analyses_csv)
        digitized.append(vals)
    return pd.DataFrame(digitized, columns=cols)

# ======= RESTORATION AND ANALYSIS FUNCTIONS (NUMERIC ONLY) =======

def fill_by_linear_regression_iter(df, target_col, feature_col):
    df = df.copy()
    while True:
        prev_na = df[target_col].isna().sum()
        df = fill_by_linear_regression(df, target_col, feature_col)
        now_na = df[target_col].isna().sum()
        if now_na == 0 or now_na == prev_na:
            break
    return df

def fill_by_linear_regression(df, target_col, feature_col):
    mask = df[target_col].notnull() & df[feature_col].notnull()
    try:
        train_X = df.loc[mask, feature_col].astype(float).values.reshape(-1, 1)
        train_y = df.loc[mask, target_col].astype(float).values
    except Exception:
        return df
    if len(train_X) < 2:
        return df
    coeff = np.polyfit(train_X.flatten(), train_y, 1)
    a, b = coeff
    missing_mask = df[target_col].isnull() & df[feature_col].notnull()
    try:
        fill_vals = a * df.loc[missing_mask, feature_col].astype(float) + b
        df.loc[missing_mask, target_col] = fill_vals
    except Exception:
        pass
    return df

def zet_fill(df, col):
    arr = df[col].values.copy()
    n = len(arr)
    for i in range(n):
        if pd.isnull(arr[i]):
            neighbors = []
            for d in [-2, -1, 1, 2]:
                idx = i + d
                if 0 <= idx < n and not pd.isnull(arr[idx]):
                    neighbors.append(arr[idx])
            if len(neighbors) >= 2:
                arr[i] = np.mean(neighbors)
    df[col] = arr
    return df

def zet_fill_iter(df, col):
    df = df.copy()
    for _ in range(10):
        na_before = df[col].isna().sum()
        df = zet_fill(df, col)
        na_after = df[col].isna().sum()
        if na_after == 0 or na_after == na_before:
            break
    return df

def remove_blocks_2d(df, percent, block_shape=(2, 2), seed=42):
    np.random.seed(seed)
    df = df.copy()
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    target_missing = int((percent / 100) * total_cells)
    if total_cells == 0 or target_missing == 0:
        return df
    missing_count = int(np.isnan(df.values).sum())
    while missing_count < target_missing:
        br, bc = block_shape
        if n_rows - br + 1 <= 0 or n_cols - bc + 1 <= 0:
            continue
        row_start = np.random.randint(0, n_rows - br + 1)
        col_start = np.random.randint(0, n_cols - bc + 1)
        for r in range(row_start, row_start + br):
            for c in range(col_start, col_start + bc):
                df.iat[r, c] = np.nan
        missing_count = int(np.isnan(df.values).sum())
    return df


def minkowski_distance(a, b, p=2):
    return (np.sum(np.abs(a - b) ** p)) ** (1/p)

def nearest_neighbor_distance(cluster1, cluster2, p=2):
    return np.min([[minkowski_distance(x, y, p) for y in cluster2] for x in cluster1])

def cluster_dispersion(Xc, center, p=2):
    # Среднее расстояние до центра по выбранной метрике
    return np.mean([minkowski_distance(x, center, p) for x in Xc])

def isodata(X, k_init=10, max_iter=100, p=2, min_cluster_size=None, split_threshold=None, merge_threshold=None, random_state=42):
    """
    ISODATA c методами split/merge/delete, и инициализацией центров.
    Приоритетные признаки: ['Стоимость', 'Анализы_код', 'Врач_код', 'Симптомы_код']
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    k = min(k_init, n_samples)
    indices = np.random.choice(n_samples, k, replace=False)
    centers = X[indices]
    labels = np.zeros(n_samples, dtype=int)

    for iter_count in range(max_iter):
        # Step 1: Assign to clusters by Minkowski
        distances = np.array([[minkowski_distance(x, c, p) for c in centers] for x in X])
        new_labels = np.argmin(distances, axis=1)

        # Step 2: If labels converged, break
        if np.all(labels == new_labels):
            break
        labels = new_labels

        # Step 3: Recalculate centers
        new_centers = []
        for i in range(k):
            members = X[labels == i]
            if len(members) > 0:
                new_centers.append(np.mean(members, axis=0))
            else:
                # If empty, reinit to a random sample
                new_centers.append(X[np.random.randint(0, n_samples)])
        centers = np.array(new_centers)

        # Step 4: Calculate dispersions
        dispersions = []
        for i in range(k):
            members = X[labels == i]
            if len(members) > 0:
                disp = cluster_dispersion(members, centers[i], p)
            else:
                disp = 0
            dispersions.append(disp)
        mean_disp = np.mean([d for d in dispersions if d > 0])

        # Step 5: Calculate nearest neighbor distances
        nn_distances = []
        for i in range(k):
            for j in range(i + 1, k):
                members_i = X[labels == i]
                members_j = X[labels == j]
                if len(members_i) > 0 and len(members_j) > 0:
                    d = nearest_neighbor_distance(members_i, members_j, p)
                    nn_distances.append(d)
        mean_nn_dist = np.mean(nn_distances) if nn_distances else 0

        # Initial thresholds if not set
        if split_threshold is None:
            split_threshold = 1.2 * mean_disp
        if merge_threshold is None:
            merge_threshold = 0.8 * mean_nn_dist
        if min_cluster_size is None:
            min_cluster_size = max(5, n_samples // 100)

        # --- SPLIT (разделение кластеров по большой дисперсии) ---
        split_idxs = [i for i, d in enumerate(dispersions) if d > split_threshold]
        new_centers_list = []
        remove_idxs = set()
        for i in range(k):
            members = X[labels == i]
            if i in split_idxs and len(members) > 2 * min_cluster_size:
                # Split by feature with highest std
                feat_std = np.std(members, axis=0)
                split_feat = np.argmax(feat_std)
                delta = feat_std[split_feat] / 2 if feat_std[split_feat] > 0 else 1
                center1 = np.array(centers[i])
                center2 = np.array(centers[i])
                center1[split_feat] -= delta
                center2[split_feat] += delta
                new_centers_list.extend([center1, center2])
                remove_idxs.add(i)
            else:
                new_centers_list.append(centers[i])
        centers = np.array(new_centers_list)
        k = len(centers)

        # --- MERGE (объединение кластера по ближайшему соседу) ---
        merged = set()
        i = 0
        while i < k:
            if i in merged:
                i += 1
                continue
            members_i = X[labels == i]
            j = i + 1
            while j < k:
                if j in merged:
                    j += 1
                    continue
                members_j = X[labels == j]
                if len(members_i) > 0 and len(members_j) > 0:
                    d = nearest_neighbor_distance(members_i, members_j, p)
                    if d < merge_threshold:
                        new_center = (centers[i] + centers[j]) / 2
                        centers[i] = new_center
                        merged.add(j)
                        # Re-assign j's members to i
                        labels[labels == j] = i
                j += 1
            i += 1
        # Remove merged centers
        centers = np.array([centers[i] for i in range(k) if i not in merged])
        unique_label_map = {old: new for new, old in enumerate([i for i in range(k) if i not in merged])}
        labels = np.array([unique_label_map.get(l, l) for l in labels])
        k = len(centers)

        # --- DELETE (удаление малых кластеров) ---
        sizes = [np.sum(labels == i) for i in range(k)]
        delete_idxs = [i for i, sz in enumerate(sizes) if sz < min_cluster_size]
        if delete_idxs:
            keep_idxs = [i for i in range(k) if i not in delete_idxs]
            centers = centers[keep_idxs]
            label_map = {old: new for new, old in enumerate(keep_idxs)}
            labels = np.array([label_map.get(l, l) for l in labels])
            k = len(centers)

        # If less than 2 clusters remain, stop
        if k < 2:
            break

    return labels, centers

# Подготовка X для вызова isodata:
# df_numeric -- ваш датафрейм
# informative_features = ["Стоимость", "Анализы_код", "Врач_код", "Симптомы_код"]
# X = df_numeric[informative_features].dropna().values
# labels, centers = isodata(X, k_init=10, max_iter=100, p=2)

def cluster_compactness(X, labels):
    s = 0
    for c in set(labels):
        Xc = X[labels == c]
        if len(Xc) == 0:
            continue
        center = Xc.mean(axis=0)
        s += np.sum((Xc - center) ** 2)
    return s

def feature_selection_del(X, k_clusters, target_feature_count):
    feats = list(range(X.shape[1]))
    while len(feats) > target_feature_count:
        best_compact = None
        worst_feat = None
        for idx in feats:
            sub_feats = [f for f in feats if f != idx]
            _labels, _ = isodata(X[:, sub_feats], k=k_clusters)
            compact = cluster_compactness(X[:, sub_feats], _labels)
            if (best_compact is None) or (compact < best_compact):
                best_compact = compact
                worst_feat = idx
        feats.remove(worst_feat)
    return feats

def nearest_neighbor_distance(cluster1, cluster2, dist_func):
    return np.min([[dist_func(x, y) for y in cluster2] for x in cluster1])

# ====== MAIN GUI CODE ======

class DatasetGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Восстановление и анализ количественного датасета")
        self.resize(1200, 700)
        self.df_raw = pd.DataFrame()          # исходный (текстовый)
        self.df_numeric = pd.DataFrame()      # цифровой (числовой) датасет
        self.symptom_table = []               # подразумевается список строк из symptoms.csv
        self.specialties_table = []           # подразумевается список строк из specialties.csv
        self.analyses_table = []              # подразумевается список строк из analyses.csv

        # UI setup
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        left_panel = self.init_left_panel()
        right_panel = self.init_right_panel()
        main_layout.addLayout(left_panel)
        main_layout.addWidget(right_panel, stretch=2)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def init_left_panel(self):
        left_layout = QVBoxLayout()
        # Block removal group
        remove_group = QGroupBox("Вставить пропуски (блоками)")
        remove_layout = QVBoxLayout()
        h_percent = QHBoxLayout()
        self.spin_percent = QSpinBox()
        self.spin_percent.setRange(1, 90)
        self.spin_percent.setValue(5)
        h_percent.addWidget(QLabel("Процент:"))
        h_percent.addWidget(self.spin_percent)
        remove_layout.addLayout(h_percent)
        h_block = QHBoxLayout()
        h_block.addWidget(QLabel("Блок (r x c):"))
        self.edit_block = QLineEdit("2x2")
        h_block.addWidget(self.edit_block)
        remove_layout.addLayout(h_block)
        btn_remove = QPushButton("Вставить блоки NaN")
        btn_remove.clicked.connect(self.remove_blocks)
        remove_layout.addWidget(btn_remove)
        remove_group.setLayout(remove_layout)
        # Restoration group
        method_group = QGroupBox("Восстановление пропусков")
        method_layout = QVBoxLayout()
        self.method_box = QComboBox()
        self.method_box.addItems([
            "Линейная регрессия", "Zet-алгоритм"
        ])
        btn_restore = QPushButton("Восстановить")
        btn_restore.clicked.connect(self.restore)
        method_layout.addWidget(self.method_box)
        method_layout.addWidget(btn_restore)
        method_group.setLayout(method_layout)
        # Clustering group
        cluster_group = QGroupBox("Кластеризация ISODATA")
        cluster_layout = QVBoxLayout()
        self.spin_clusters = QSpinBox()
        self.spin_clusters.setRange(2, 20)
        self.spin_clusters.setValue(3)
        btn_cluster = QPushButton("Кластеризовать")
        btn_cluster.clicked.connect(self.clusterize)
        cluster_layout.addWidget(QLabel("Число кластеров:"))
        cluster_layout.addWidget(self.spin_clusters)
        cluster_layout.addWidget(btn_cluster)
        cluster_group.setLayout(cluster_layout)
        # Feature selection group
        feature_group = QGroupBox("Отбор признаков (Del)")
        feature_layout = QVBoxLayout()
        self.spin_feats = QSpinBox()
        self.spin_feats.setRange(1, 10)
        self.spin_feats.setValue(1)
        btn_fs = QPushButton("Отобрать признаки")
        btn_fs.clicked.connect(self.select_features)
        feature_layout.addWidget(QLabel("Оставить признаков:"))
        feature_layout.addWidget(self.spin_feats)
        feature_layout.addWidget(btn_fs)
        feature_group.setLayout(feature_layout)
        # Load csv button
        self.btn_load = QPushButton("Загрузить CSV")
        self.btn_load.clicked.connect(self.load_csv)
        # Stat group
        stat_group = QGroupBox("Быстрая статистика:")
        self.stat_text = QTextEdit()
        self.stat_text.setReadOnly(True)
        stat_layout = QVBoxLayout()
        stat_layout.addWidget(self.stat_text)
        stat_group.setLayout(stat_layout)
        # Layout population
        left_layout.addWidget(remove_group)
        left_layout.addWidget(method_group)
        left_layout.addWidget(cluster_group)
        left_layout.addWidget(feature_group)
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(stat_group)
        left_layout.addStretch()
        return left_layout

    def init_right_panel(self):
        tabs = QTabWidget()
        self.table_numeric = QTableWidget()
        tabs.addTab(self.table_numeric, "Цифровой датасет")
        self.stat_box = QTextEdit()
        self.stat_box.setReadOnly(True)
        tabs.addTab(self.stat_box, "Статистика чисел")
        self.compact_box = QTextEdit()
        self.compact_box.setReadOnly(True)
        tabs.addTab(self.compact_box, "Компактность кластеров")
        return tabs

    def load_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Загрузить CSV", "", "CSV Files (*.csv);;All Files (*)")
        if filename:
            colnames = [
                "ФИО", "Паспорт", "СНИЛС", "Симптомы", "Врач",
                "Дата_посещения", "Анализы", "Дата_анализов", "Стоимость", "Карта_оплаты"
            ]
            self.df_raw = pd.read_csv(filename, sep=';', header=None, dtype=str, names=colnames)
            self.df_numeric = digitize_dataset(
                self.df_raw,
                self.symptom_table,
                self.specialties_table,
                self.analyses_table
            )
            self.refresh_table()
            self.show_basic_stats()

    def refresh_table(self):
        df = self.df_numeric
        if df.empty:
            self.table_numeric.clear()
            return
        self.table_numeric.setColumnCount(len(df.columns))
        self.table_numeric.setRowCount(len(df))
        self.table_numeric.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i, row in df.iterrows():
            for j, v in enumerate(row):
                val = "" if pd.isna(v) else str(round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v)
                item = QTableWidgetItem(val)
                if pd.isna(v):
                    item.setBackground(Qt.gray)
                self.table_numeric.setItem(i, j, item)

    def show_basic_stats(self):
        df = self.df_numeric
        if df.empty:
            self.stat_box.clear()
            self.stat_text.clear()
            return
        nrows = len(df)
        nmiss = int(df.isna().sum().sum())
        pmiss = 100 * nmiss / df.size if df.size > 0 else 0
        stat = f"Записей: {nrows}\nПропусков: {pmiss:.2f}%\n"
        numstats = []
        for col in df.columns:
            coldata = df[col]
            if pd.api.types.is_numeric_dtype(coldata):
                m = coldata.mean()
                md = coldata.median()
                mo = coldata.mode()
                mo_val = mo.iloc[0] if not mo.empty else "—"
                numstats.append(
                    f"{col}: среднее={m:.2f}, медиана={md:.2f}, мода={mo_val}"
                )
        self.stat_text.setText(stat)
        self.stat_box.setText("\n".join(numstats))

    def remove_blocks(self):
        df = self.df_numeric
        if df.empty:
            return
        try:
            percent = int(self.spin_percent.value())
            block_shape_str = self.edit_block.text().lower().replace(" ", "")
            if "x" in block_shape_str:
                br, bc = block_shape_str.split("x")
                block_shape = (max(2, int(br)), max(2, int(bc)))
            else:
                block_shape = (2, 2)
        except Exception:
            block_shape = (2, 2)
        self.df_numeric = remove_blocks_2d(self.df_numeric, percent, block_shape)
        self.refresh_table()
        self.show_basic_stats()

    def restore(self):
        df = self.df_numeric
        if df.empty:
            return
        col = "Стоимость"
        method = self.method_box.currentText()
        if col not in df.columns or df[col].isna().sum() == 0:
            return
        # Use row index as a feature for regression (can be replaced with any other useful numeric feature)
        if method == "Линейная регрессия":
            self.df_numeric["__row__"] = np.arange(len(df))
            self.df_numeric = fill_by_linear_regression_iter(self.df_numeric, col, "__row__")
            self.df_numeric = self.df_numeric.drop(columns=["__row__"])
        elif method == "Zet-алгоритм":
            self.df_numeric = zet_fill_iter(self.df_numeric, col)
        self.refresh_table()
        self.show_basic_stats()

    def clusterize(self):
        df = self.df_numeric
        if df.empty:
            return

        informative_features = ["Стоимость", "Анализы_код", "Врач_код", "Симптомы_код"]
        used_features = [f for f in informative_features if f in df.columns]
        X = df[used_features].dropna(axis=0).values
        if len(X) == 0 or len(used_features) == 0:
            self.compact_box.setText("Нет данных для кластеризации (информативные признаки пусты)")
            return
        
        labels, centers = isodata(X, k_init=10, max_iter=100, p=2)
        df_clust = df.dropna(subset=used_features).copy()
        df_clust['__cluster__'] = labels
        self.compact_box.setText(
            f"Кластеры (ISODATA, Минковски p=2):\n"
            f"Компактность кластеров: {cluster_compactness(X, labels):.4f}\n"
            f"Центры кластеров:\n{centers}"
        )
        self.df_numeric.loc[df_clust.index, '__cluster__'] = labels
        self.refresh_table()

    def select_features(self):
        df = self.df_numeric
        if df.empty:
            return
        num_feats = self.spin_feats.value()
        k_clust = max(2, self.spin_clusters.value())
        # Only use numeric columns (ignore __cluster__ and other service columns)
        data_cols = [c for c in df.columns if df[c].dtype.kind in 'iufc' and not c.startswith('__')]
        X = df[data_cols].dropna(axis=0).values
        cols_keep = data_cols
        if len(data_cols) > num_feats and len(X) >= k_clust:
            best_features_idx = feature_selection_del(X, k_clust, num_feats)
            cols_keep = [data_cols[i] for i in best_features_idx]
            self.df_numeric = df[cols_keep].copy()
            self.stat_box.setText(f"Оставлены признаки: {', '.join(cols_keep)}")
        else:
            self.stat_box.setText(f"Отбор не выполнен: недостаточно данных. Ост. признаки: {', '.join(cols_keep)}")
        self.compact_box.append("Проведен отбор признаков методом Del.")
        self.refresh_table()
        self.show_basic_stats()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DatasetGUI()
    # Перед стартом GUI присвойте gui.symptom_table, gui.specialties_table и gui.analyses_table!
    with open('symptoms.csv', encoding='utf8') as f:
        gui.symptom_table = [line.strip() for line in f if line.strip()]
    with open('specialties.csv', encoding='utf8') as f:
        gui.specialties_table = [line.strip() for line in f if line.strip()]
    with open('analyses.csv', encoding='utf8') as f:
        gui.analyses_table = [line.strip() for line in f if line.strip()]
    gui.show()
    sys.exit(app.exec_())