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

# ======= NUMERIC FEATURE FUNCTIONS =======

def extract_unique_symptoms(symptom_table):
    unique_symptoms = set()
    for line in symptom_table:
        items = line.strip().split(';')
        if len(items) >= 3:
            symptoms_str = items[2]
        else:
            symptoms_str = items[-1]
        for s in symptoms_str.split(','):
            s_clean = s.strip().lower()
            if s_clean:
                unique_symptoms.add(s_clean)
    return list(unique_symptoms)

def encode_symptoms(df, symptoms_list):
    for symp in symptoms_list:
        pattern = re.escape(symp)
        df[f'симп_{symp}'] = df["Симптомы"].str.lower().fillna("").apply(
            lambda x: int(bool(re.search(pattern, x)))
        )
    return df

def extract_unique_doctors(symptom_table):
    unique_doctors = set()
    for line in symptom_table:
        items = line.strip().split(';')
        if len(items) >= 2:
            doc = items[1].strip()
            if doc:
                unique_doctors.add(doc)
    return list(unique_doctors)

def encode_doctor_onehot(df, doctor_list):
    df['Врач_clean'] = df['Врач'].apply(lambda x: x.strip() if pd.notnull(x) else 'прочее')
    for doc in doctor_list:
        df[f'врач_{doc}'] = (df['Врач_clean'] == doc).astype(int)
    df.drop('Врач_clean', axis=1, inplace=True)
    return df

def encode_dates(df):
    df['Дата_посещения'] = pd.to_datetime(df['Дата_посещения'], format='%Y-%m-%dT%H', errors="coerce")
    df['Дата_анализов'] = pd.to_datetime(df['Дата_анализов'], format='%Y-%m-%dT%H', errors="coerce")
    df['visit_ts']     = df['Дата_посещения'].astype(np.int64) // 10 ** 9
    df['test_ts']      = df['Дата_анализов'].astype(np.int64) // 10 ** 9
    df['days_to_test'] = (df['Дата_анализов'] - df['Дата_посещения']).dt.days
    df['weekday_visit'] = df['Дата_посещения'].dt.weekday
    return df

def extract_unique_analyses(analyses_table):
    unique_analyses = set()
    for line in analyses_table:
        for item in line.strip().split(';'):
            item = item.strip()
            if ',' in item:
                name = item.split(',')[0].strip().lower()
                if name:
                    unique_analyses.add(name)
    return list(unique_analyses)

def encode_analyses(df, analyses_list):
    for test in analyses_list:
        pattern = re.escape(test)
        df[f'анализ_{test}'] = df["Анализы"].str.lower().fillna("").apply(
            lambda x: int(bool(re.search(pattern, x)))
        )
    return df

def encode_cost(df):
    df["Стоимость"] = pd.to_numeric(df["Стоимость"], errors="coerce")
    return df

def get_numeric_dataset(df, symptom_table, analyses_table):
    symptoms_list = extract_unique_symptoms(symptom_table)
    doctor_list = extract_unique_doctors(symptom_table)
    analyses_list = extract_unique_analyses(analyses_table)

    df_num = df.copy()
    df_num = encode_symptoms(df_num, symptoms_list)
    df_num = encode_doctor_onehot(df_num, doctor_list)
    df_num = encode_dates(df_num)
    df_num = encode_analyses(df_num, analyses_list)
    df_num = encode_cost(df_num)

    features = (
        [f'симп_{s}' for s in symptoms_list]
        + [f'врач_{d}' for d in doctor_list]
        + ['visit_ts', 'test_ts', 'days_to_test', 'weekday_visit']
        + [f'анализ_{a}' for a in analyses_list]
        + ['Стоимость']
    )
    return df_num[features]

# ======= YOUR ORIGINAL CODE =======
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
    is_numtype = pd.api.types.is_numeric_dtype(df[col])
    n = len(arr)
    for i in range(n):
        if pd.isnull(arr[i]):
            neighbors = []
            for d in [-2, -1, 1, 2]:
                idx = i + d
                if 0 <= idx < n and not pd.isnull(arr[idx]):
                    neighbors.append(arr[idx])
            if len(neighbors) >= 2:
                if is_numtype:
                    try:
                        arr[i] = np.mean([float(x) for x in neighbors])
                    except Exception:
                        pass
                else:
                    try:
                        arr[i] = pd.Series(neighbors).mode()[0]
                    except Exception:
                        arr[i] = neighbors[0]
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

def remove_blocks_2d(df, percent, block_shape_range=((2,2), (4,4)), seed=42):
    """
    block_shape_range: ((min_rows, min_cols), (max_rows, max_cols))
    Если параметр задан как tuple вида (r, c), то будет фиксированный размер блока.
    """
    np.random.seed(seed)
    df = df.copy()
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    
    # Исправленная логика диапазона/фиксированного блока
    if (isinstance(block_shape_range, tuple) and len(block_shape_range) == 2
        and all(isinstance(x, int) for x in block_shape_range)):
        min_block_rows = max_block_rows = block_shape_range[0]
        min_block_cols = max_block_cols = block_shape_range[1]
    elif (isinstance(block_shape_range, tuple) and len(block_shape_range) == 2
          and isinstance(block_shape_range[0], tuple) and isinstance(block_shape_range[1], tuple)):
        min_block_rows, min_block_cols = block_shape_range[0]
        max_block_rows, max_block_cols = block_shape_range[1]
    else:
        min_block_rows = max_block_rows = 2
        min_block_cols = max_block_cols = 2

    est_block_rows = (min_block_rows + max_block_rows) // 2
    est_block_cols = (min_block_cols + max_block_cols) // 2
    block_size = est_block_rows * est_block_cols    
    n_blocks = max(int(np.ceil((total_cells * percent / 100) / block_size)), 2)

    for _ in range(n_blocks):
        br = np.random.randint(min_block_rows, max_block_rows + 1)
        bc = np.random.randint(min_block_cols, max_block_cols + 1)
        # Следим, чтобы блок не выходил за границы
        if n_rows - br + 1 <= 0 or n_cols - bc + 1 <= 0:
            continue
        row_start = np.random.randint(0, n_rows - br + 1)
        col_start = np.random.randint(0, n_cols - bc + 1)
        for r in range(row_start, row_start + br):
            for c in range(col_start, col_start + bc):
                df.iat[r, c] = np.nan
    return df

def minkowski_distance(a, b, p=2):
    return (np.sum(np.abs(a - b) ** p)) ** (1/p)

def isodata(X, k=3, max_iter=100, p=2):
    X = np.array(X)
    n = X.shape[0]
    if n < k:
        k = n
    np.random.seed(42)
    centers = X[np.random.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        distances = np.array([[minkowski_distance(x, c, p) for c in centers] for x in X])
        new_labels = np.argmin(distances, axis=1)
        if np.all(labels == new_labels):
            break
        labels = new_labels
        for i in range(k):
            if np.any(labels == i):
                centers[i] = X[labels == i].mean(axis=0)
    return labels, centers

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

class DatasetGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Восстановление и анализ датасета")
        self.resize(1200, 700)
        self.df = pd.DataFrame()
        self.df_initial = pd.DataFrame()
        self.k_clusters = 3

        # Numeric data storage
        self.df_numeric = pd.DataFrame()
        self.symptom_table = []   # Must be filled with external data (list of str)
        self.analyses_table = []  # Must be filled with external data (list of str)

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        left_panel = self.init_left_panel()
        right_panel = self.init_right_panel()
        main_layout.addLayout(left_panel)
        main_layout.addWidget(right_panel, stretch=2)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def init_left_panel(self):
        # ... unchanged ...
        # Keep your original left_panel code as before!
        # (Due to size, omitted for brevity - assume unchanged as in your original file above)
        # ... unchanged ...
        left_layout = QVBoxLayout()
        remove_group = QGroupBox("Удалить блоками (матрицами)")
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
        btn_remove = QPushButton("Удалить (вставить пропуски)")
        btn_remove.clicked.connect(self.remove_blocks)
        remove_layout.addWidget(btn_remove)
        remove_group.setLayout(remove_layout)
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
        self.btn_load = QPushButton("Загрузить CSV")
        self.btn_load.clicked.connect(self.load_csv)
        stat_group = QGroupBox("Быстрая статистика:")
        self.stat_text = QTextEdit()
        self.stat_text.setReadOnly(True)
        stat_layout = QVBoxLayout()
        stat_layout.addWidget(self.stat_text)
        stat_group.setLayout(stat_layout)
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
        self.table = QTableWidget()
        tabs.addTab(self.table, "Датасет")
        self.stat_box = QTextEdit()
        self.stat_box.setReadOnly(True)
        tabs.addTab(self.stat_box, "Числовая статистика")
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
            self.df = pd.read_csv(filename, sep=';', header=None, dtype=str, names=colnames)
            # Преобразуем ТОЛЬКО "Стоимость" в число
            if "Стоимость" in self.df.columns:
                self.df["Стоимость"] = pd.to_numeric(self.df["Стоимость"], errors="coerce")
            self.df_initial = self.df.copy()

            # ======= ЦиФРОВЫЙ ДАТАСЕТ (numeric features) =======
            # You must fill self.symptom_table and self.analyses_table externally!
            # Example: self.symptom_table = [...] ; self.analyses_table = [...]
            self.df_numeric = get_numeric_dataset(self.df, self.symptom_table, self.analyses_table)

            self.refresh_table()
            self.show_basic_stats()

    def refresh_table(self):
        if self.df.empty:
            self.table.clear()
            return
        self.table.setColumnCount(len(self.df.columns))
        self.table.setRowCount(len(self.df))
        self.table.setHorizontalHeaderLabels([str(c) for c in self.df.columns])
        for i, row in self.df.iterrows():
            for j, v in enumerate(row):
                val = "" if pd.isna(v) else str(v)
                item = QTableWidgetItem(val)
                if pd.isna(v):
                    item.setBackground(Qt.gray)
                self.table.setItem(i, j, item)

    def show_basic_stats(self):
        if self.df.empty:
            self.stat_box.clear()
            self.stat_text.clear()
            return
        nrows = len(self.df)
        nmiss = int(self.df.isna().sum().sum())
        pmiss = 100 * nmiss / self.df.size if self.df.size > 0 else 0
        stat = f"Записей: {nrows}\nПропусков: {pmiss:.2f}%\n"
        numstats = []
        # Только "Стоимость"
        if "Стоимость" in self.df.columns:
            coldata = self.df["Стоимость"]
            if pd.api.types.is_numeric_dtype(coldata):
                m = coldata.mean()
                md = coldata.median()
                mo = coldata.mode()
                mo_val = mo.iloc[0] if not mo.empty else "—"
                numstats.append(
                    f"Стоимость: среднее={m:.2f}, медиана={md:.2f}, мода={mo_val}"
                )
        self.stat_text.setText(stat)
        self.stat_box.setText("\n".join(numstats))

    # ... остальные методы неизменны ...
    def remove_blocks(self):
        if self.df.empty:
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
        self.df = remove_blocks_2d(self.df, percent, block_shape)
        self.refresh_table()
        self.show_basic_stats()

    def restore(self):
        if self.df.empty or "Стоимость" not in self.df.columns:
            return
        col = "Стоимость"
        method = self.method_box.currentText()
        if self.df[col].isna().sum() == 0:
            return
        if method == "Линейная регрессия":
            self.df["__row__"] = np.arange(len(self.df))
            self.df = fill_by_linear_regression_iter(self.df, col, "__row__")
            self.df = self.df.drop(columns=["__row__"])
        elif method == "Zet-алгоритм":
            self.df = zet_fill_iter(self.df, col)
        self.refresh_table()
        self.show_basic_stats()

    def clusterize(self):
        if self.df.empty or "Стоимость" not in self.df.columns:
            return
        k = self.spin_clusters.value()
        X = self.df[["Стоимость"]].dropna(axis=0).values
        if len(X) < k or X.shape[1] == 0:
            self.compact_box.setText("Слишком мало данных/числовых признаков для кластеризации")
            return
        labels, centers = isodata(X, k=k, p=2)
        self.df.loc[self.df["Стоимость"].notnull(), '__cluster__'] = labels
        self.compact_box.setText(
            f"Кластеры (ISODATA, Минковски p=2):\n"
            f"Компактность кластеров: {cluster_compactness(X, labels):.4f}\n"
            f"Центры кластеров:\n{centers}"
        )
        self.refresh_table()

    def select_features(self):
        if self.df.empty or "Стоимость" not in self.df.columns:
            return
        self.df = self.df[["Стоимость"]].copy()
        self.refresh_table()
        self.show_basic_stats()
        self.stat_box.setText("Оставлен только признак 'Стоимость'")
        self.compact_box.append("Проведен отбор признаков методом Del (оставлен 'Стоимость').")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DatasetGUI()
    # Перед стартом GUI присвойте gui.symptom_table и gui.analyses_table!
    with open('symptoms.csv', encoding='utf8') as f:
        gui.symptom_table = [line.strip() for line in f if line.strip()]

    with open('analyses.csv', encoding='utf8') as f:
        gui.analyses_table = [line.strip() for line in f if line.strip()]
    gui.show()
    sys.exit(app.exec_())