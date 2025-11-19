import sys
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QTableWidget, QTableWidgetItem, QTextEdit, QTabWidget,
    QGroupBox
)
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt

from digitization import *
from isodata import *
from remove_blocks import *
from missing_fill_methods import *
from make_full_datasets import *
from compactness import *


def ceil_numeric_df(df, features):
    df = df.copy()
    for col in features:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = np.ceil(df[col]).astype(int)
    return df

class Lab4App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Восстановление и анализ датасета - lab4")
        self.resize(1300, 750)

        # Датасеты для работы
        self.df_loaded = pd.DataFrame()
        self.df_digitized = pd.DataFrame()
        self.df_isodata = pd.DataFrame()
        self.df_na = pd.DataFrame()
        self.df_restored_zet = pd.DataFrame()
        self.df_restored_regr = pd.DataFrame()
        self.symptoms_csv = []
        self.specialties_csv = []
        self.analyses_csv = []

        self.init_gui()

    def init_gui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()

        btn_load = QPushButton("Загрузить CSV")
        btn_load.clicked.connect(self.load_csv)
        left_panel.addWidget(btn_load)

        remove_group = QGroupBox("Удалить пропуски блоками")
        remove_layout = QVBoxLayout()
        self.percent_box = QComboBox()
        self.percent_box.addItems(["5", "10", "15", "30"])
        remove_layout.addWidget(QLabel("Процент пропусков:"))
        remove_layout.addWidget(self.percent_box)
        btn_remove = QPushButton("Удалить")
        btn_remove.clicked.connect(self.on_remove_blocks)
        remove_layout.addWidget(btn_remove)
        remove_group.setLayout(remove_layout)
        left_panel.addWidget(remove_group)

        restore_group = QGroupBox("Восстановление пропусков двумя способами")
        restore_layout = QVBoxLayout()
        btn_restore = QPushButton("Восстановить (2 способами)")
        btn_restore.clicked.connect(self.on_restore_dual)
        restore_layout.addWidget(btn_restore)
        restore_group.setLayout(restore_layout)
        left_panel.addWidget(restore_group)

        isodata_group = QGroupBox("ISODATA-кластеризация")
        isodata_layout = QVBoxLayout()
        btn_isodata = QPushButton("Применить ISODATA (на Zet)")
        btn_isodata.clicked.connect(self.on_isodata_zet)
        isodata_layout.addWidget(btn_isodata)
        btn_isodata2 = QPushButton("Применить ISODATA (на Регрессия)")
        btn_isodata2.clicked.connect(self.on_isodata_regr)
        isodata_layout.addWidget(btn_isodata2)
        isodata_group.setLayout(isodata_layout)
        left_panel.addWidget(isodata_group)

        btn_report = QPushButton("Сформировать отчет")
        btn_report.clicked.connect(self.show_report)
        left_panel.addWidget(btn_report)

        left_panel.addStretch()

        self.stat_box = QTextEdit()
        self.stat_box.setReadOnly(True)
        self.stat_box.setMinimumHeight(180)
        left_panel.addWidget(QLabel("Статистика / сообщения:"))
        left_panel.addWidget(self.stat_box)

        self.tabs = QTabWidget()
        self.table_loaded = QTableWidget()
        self.tabs.addTab(self.table_loaded, "Исходный")
        self.table_digitized = QTableWidget()
        self.tabs.addTab(self.table_digitized, "Цифровой")
        self.table_na = QTableWidget()
        self.tabs.addTab(self.table_na, "После удаления")
        self.table_restored_zet = QTableWidget()
        self.tabs.addTab(self.table_restored_zet, "Восстановленный Zet")
        self.table_restored_regr = QTableWidget()
        self.tabs.addTab(self.table_restored_regr, "Восстановленный Регрессия")
        self.table_isodata = QTableWidget()
        self.tabs.addTab(self.table_isodata, "ISODATA")
        self.compact_box = QTextEdit()
        self.compact_box.setReadOnly(True)
        self.tabs.addTab(self.compact_box, "Компактность кластеров")

        # Новая вкладка для отчета
        self.table_report = QTableWidget()
        self.tabs.addTab(self.table_report, "Отчет")

        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.tabs, 3)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def log_stat(self, message):
        self.stat_box.append(message)
        self.stat_box.moveCursor(QTextCursor.End)

    def _show_df(self, table_widget, df):
        table_widget.clear()
        if df.empty:
            table_widget.setRowCount(0)
            table_widget.setColumnCount(0)
            return
        table_widget.setColumnCount(len(df.columns))
        table_widget.setRowCount(len(df))
        table_widget.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i, row in df.iterrows():
            for j, v in enumerate(row):
                item = QTableWidgetItem("" if pd.isna(v) else str(v))
                if pd.isna(v):
                    item.setBackground(Qt.gray)
                table_widget.setItem(i, j, item)

    def _basic_stats(self, df):
        if df.empty:
            return "Нет данных."
        nrows = len(df)
        nmiss = int(df.isna().sum().sum())
        pmiss = 100 * nmiss / df.size if df.size > 0 else 0
        stat = f"Записей: {nrows}\nПропусков: {nmiss} ({pmiss:.2f}%)\n"
        numstats = []
        for col in df.select_dtypes(include=["number"]).columns:
            m = df[col].mean()
            md = df[col].median()
            mo = df[col].mode()
            mo_val = mo.iloc[0] if not mo.empty else "—"
            numstats.append(
                f"{col}: среднее={m:.2f}, медиана={md:.2f}, мода={mo_val}"
            )
        return stat + "\n".join(numstats)

    def load_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Загрузить CSV", "", "CSV Files (*.csv);;All Files (*)")
        if filename:
            colnames = [
                "ФИО", "Паспорт", "СНИЛС", "Симптомы", "Врач",
                "Дата_посещения", "Анализы", "Дата_анализов", "Стоимость", "Карта_оплаты"
            ]
            self.df_loaded = pd.read_csv(filename, sep=';', header=None, dtype=str, names=colnames)
            try:
                with open('symptoms.csv', encoding='utf8') as f:
                    self.symptoms_csv = [line.strip() for line in f if line.strip()]
                with open('specialties.csv', encoding='utf8') as f:
                    self.specialties_csv = [line.strip() for line in f if line.strip()]
                with open('analyses.csv', encoding='utf8') as f:
                    self.analyses_csv = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                self.symptoms_csv, self.specialties_csv, self.analyses_csv = [], [], []
            self.df_digitized = digitize_dataset(
                self.df_loaded, self.symptoms_csv, self.specialties_csv, self.analyses_csv
            )
            self.df_na = pd.DataFrame()
            self.df_restored_zet = pd.DataFrame()
            self.df_restored_regr = pd.DataFrame()
            self.df_isodata = pd.DataFrame()
            self._show_df(self.table_loaded, self.df_loaded)
            self._show_df(self.table_digitized, self.df_digitized)
            self.log_stat("Загружен исходный датасет:\n" + self._basic_stats(self.df_loaded))
            self.compact_box.setText("")
            self._show_df(self.table_na, pd.DataFrame())
            self._show_df(self.table_restored_zet, pd.DataFrame())
            self._show_df(self.table_restored_regr, pd.DataFrame())
            self._show_df(self.table_isodata, pd.DataFrame())
            self.table_report.clear()

    def on_remove_blocks(self):
        if self.df_digitized.empty:
            return
        percent = int(self.percent_box.currentText())
        df_na = remove_blocks_2d(
            self.df_digitized, percent=percent, block_shape_range=((2,2), (4,4)), seed=42
        )
        self.df_na = df_na
        self._show_df(self.table_na, self.df_na)
        self.log_stat(f"Удалены блоки пропусков ({percent}%):\n" + self._basic_stats(self.df_na))
        self.df_restored_zet = pd.DataFrame()
        self.df_restored_regr = pd.DataFrame()
        self._show_df(self.table_restored_zet, pd.DataFrame())
        self._show_df(self.table_restored_regr, pd.DataFrame())
        self._show_df(self.table_isodata, pd.DataFrame())
        self.compact_box.setText("")
        self.table_report.clear()

    def on_restore_dual(self):
        if self.df_na.empty:
            return
        features = [c for c in self.df_digitized.select_dtypes(include=["number"]).columns]
        window = 2
        min_filled = 1
        min_neighbors = 1
        n_rounds = 20

        # Zet recovery
        restored_zet = self.df_na.copy()
        for _ in range(n_rounds):
            changed_any = False
            for col in features:
                restored_zet, changed = zet_fill_window(restored_zet, col, window=window, min_neighbors=min_neighbors)
                changed_any = changed_any or changed
            if not changed_any:
                break
        restored_zet = fill_remaining_gaps(restored_zet, fill_strategy="ffill", cols=features)
        restored_zet = ceil_numeric_df(restored_zet, features) 
        self.df_restored_zet = restored_zet

        # Linear regression recovery
        restored_regr = self.df_na.copy()
        for _ in range(n_rounds):
            na_sum_before = restored_regr[features].isna().sum().sum()
            for target in features:
                other_feats = [f for f in features if f != target]
                restored_regr = fill_linear_window_multi_iter(restored_regr, target, other_feats, window=window, min_filled=min_filled)
            na_sum_after = restored_regr[features].isna().sum().sum()
            if na_sum_after == na_sum_before:
                break
        restored_regr = fill_remaining_gaps(restored_regr, fill_strategy="ffill", cols=features)
        restored_regr = ceil_numeric_df(restored_regr, features)
        self.df_restored_regr = restored_regr

        self._show_df(self.table_restored_zet, self.df_restored_zet)
        self._show_df(self.table_restored_regr, self.df_restored_regr)
        self.log_stat("Восстановленные датасеты (Zet и Регрессия):\n"
            + "Zet: " + self._basic_stats(self.df_restored_zet) + "\n\n"
            + "Регрессия: " + self._basic_stats(self.df_restored_regr)
        )
        self._show_df(self.table_isodata, pd.DataFrame())
        self.compact_box.setText("")
        self.table_report.clear()

    def on_isodata_zet(self):
        if self.df_restored_zet.empty:
            self.compact_box.setText("Нет восстановленного Zet датасета.")
            return
        self._run_isodata_on(self.df_restored_zet)

    def on_isodata_regr(self):
        if self.df_restored_regr.empty:
            self.compact_box.setText("Нет восстановленного Регрессия датасета.")
            return
        self._run_isodata_on(self.df_restored_regr)

    def _run_isodata_on(self, df_for_iso):
        FEATURES = [
            "Стоимость", "Анализы_код", "Врач_код", "Симптомы_код"
        ]
        features_present = [f for f in FEATURES if f in df_for_iso.columns]
        if len(df_for_iso) == 0 or len(features_present) == 0:
            self.compact_box.setText("Данных для кластеризации недостаточно.")
            return

        # Быстрая ISODATA
        K_INIT = 5
        P = 2
        np.random.seed(42)
        points = df_for_iso[features_present].to_numpy()
        initial_idxs = np.random.choice(len(points), min(K_INIT, len(points)), replace=False)
        centers = points[initial_idxs]
        labels = assign_clusters_fast(points, centers)
        converged, iter_count = False, 0
        while not converged and iter_count < 100:
            iter_count += 1
            center_dists = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = minkowski_dist(centers[i], centers[j], P)
                    center_dists.append(dist)
                    merge_threshold = 0.8 * np.mean(center_dists) if center_dists else 0
                    if dist < merge_threshold:
                        centers = merge_clusters_fast(centers, i, j)
                        converged = False
                        labels = assign_clusters_fast(points, centers)
                        break
                else:
                    continue
                break
            centers, labels = delete_clusters_fast(points, labels, centers)
            new_labels = assign_clusters_fast(points, centers)
            if np.array_equal(labels, new_labels):
                converged = True
            labels = new_labels

        df_for_iso = df_for_iso.copy()
        df_for_iso['Cluster'] = labels
        self.df_isodata = df_for_iso
        self._show_df(self.table_isodata, self.df_isodata)
        self.log_stat("ISODATA выполнен (fast):\n" + self._basic_stats(self.df_isodata))
        self.compact_box.setText(self._cluster_compactness_text(self.df_isodata, features_present))

    def _cluster_compactness_text(self, df, features):
        if "Cluster" not in df.columns:
            return "Нет кластеров."
        clusters = df['Cluster'].unique()
        P = 2
        txt = "Компактность по каждому кластеру:\n"
        total_ssd = 0
        total_pts = 0
        for cluster in clusters:
            members = df[df['Cluster'] == cluster]
            points = members[features].to_numpy()
            n_points = len(points)
            if n_points > 0:
                center = np.mean(points, axis=0)
                ssd = np.sum([minkowski_dist(pt, center, P)**2 for pt in points])
                compactness = ssd / n_points if n_points > 0 else 0
                txt += f"Кластер {cluster}: compactness = {compactness:.2f}, size = {n_points}\n"
                total_ssd += ssd
                total_pts += n_points
            else:
                txt += f"Кластер {cluster}: пустой кластер (size = 0)\n"
        compactness_avg = total_ssd / total_pts if total_pts > 0 else 0
        txt += f"\nСредняя компактность кластеризации: {compactness_avg:.2f}\n"
        return txt

    def show_report(self):
        # Проверяем, что все датасеты сформированы
        if (
            self.df_digitized.empty or self.df_na.empty
            or self.df_restored_zet.empty or self.df_restored_regr.empty
        ):
            self.log_stat("Для формирования отчета нужны все датасеты!")
            return

        numeric_cols = [c for c in self.df_digitized.select_dtypes(include=["number"]).columns]

        def mode_safe(arr):
            arr = pd.Series(arr).dropna()
            m = arr.mode()
            return m.iloc[0] if not m.empty else np.nan

        def sum_relative_error(true, pred):
            mask = (~pd.isna(true)) & (~pd.isna(pred)) & (true != 0)
            errors = np.abs(pred[mask] - true[mask]) / np.abs(true[mask])
            return errors.sum() * 100

        # Для каждого признака считаем статистику и Δ для восстановления
        results = []
        for col in numeric_cols:
            orig = self.df_digitized[col].dropna()
            gap = self.df_na[col].dropna()
            zet = self.df_restored_zet[col].dropna()
            regr = self.df_restored_regr[col].dropna()
            mean_vals = [
                np.mean(orig) if len(orig) else np.nan,
                np.mean(gap) if len(gap) else np.nan,
                np.mean(zet) if len(zet) else np.nan,
                np.mean(regr) if len(regr) else np.nan
            ]
            median_vals = [
                np.median(orig) if len(orig) else np.nan,
                np.median(gap) if len(gap) else np.nan,
                np.median(zet) if len(zet) else np.nan,
                np.median(regr) if len(regr) else np.nan
            ]
            mode_vals = [
                mode_safe(orig),
                mode_safe(gap),
                mode_safe(zet),
                mode_safe(regr)
            ]
            missing_mask = self.df_na[col].isna()
            true_missing = self.df_digitized[col][missing_mask]
            gap_missing = self.df_na[col][missing_mask].fillna(0)
            zet_missing = self.df_restored_zet[col][missing_mask]
            regr_missing = self.df_restored_regr[col][missing_mask]
            delta_gap = sum_relative_error(true_missing, gap_missing)
            delta_zet = sum_relative_error(true_missing, zet_missing)
            delta_regr = sum_relative_error(true_missing, regr_missing)

            results.append(
                [col] +
                [f"{mean_vals[0]:.3f}", f"{mean_vals[1]:.3f}", f"{mean_vals[2]:.3f}", f"{mean_vals[3]:.3f}"] +
                [f"{median_vals[0]:.3f}", f"{median_vals[1]:.3f}", f"{median_vals[2]:.3f}", f"{median_vals[3]:.3f}"] +
                [str(mode_vals[0]), str(mode_vals[1]), str(mode_vals[2]), str(mode_vals[3])] +
                [f"{delta_gap:.2f}", f"{delta_zet:.2f}", f"{delta_regr:.2f}"]
            )

        col_headers = [
            "Признак",
            "mean-оригинал", "mean-с пропусками", "mean-Zet", "mean-Регрессия",
            "median-оригинал", "median-с пропусками", "median-Zet", "median-Регрессия",
            "mode-оригинал", "mode-с пропусками", "mode-Zet", "mode-Регрессия",
            "Суммарная Δ c пропусками (%)", "Суммарная Δ Zet (%)", "Суммарная Δ Регрессия (%)"
        ]
        self.table_report.clear()
        self.table_report.setColumnCount(len(col_headers))
        self.table_report.setRowCount(len(results) + 1)
        self.table_report.setHorizontalHeaderLabels(col_headers)

        for i, row in enumerate(results):
            for j, val in enumerate(row):
                self.table_report.setItem(i, j, QTableWidgetItem(val))

        total_gap = sum(float(row[-3]) for row in results)
        total_zet = sum(float(row[-2]) for row in results)
        total_regr = sum(float(row[-1]) for row in results)
        self.table_report.setItem(len(results), 0, QTableWidgetItem("ИТОГ: Суммарная Δ (%)"))
        self.table_report.setItem(len(results), len(col_headers)-3, QTableWidgetItem(f"{total_gap:.2f}"))
        self.table_report.setItem(len(results), len(col_headers)-2, QTableWidgetItem(f"{total_zet:.2f}"))
        self.table_report.setItem(len(results), len(col_headers)-1, QTableWidgetItem(f"{total_regr:.2f}"))

        best = min((total_gap, "Пропуски"), (total_zet, "Zet"), (total_regr, "Регрессия"), key=lambda x: x[0])[1]
        self.tabs.setCurrentWidget(self.table_report)
        self.log_stat(f"Отчет сформирован. Наиболее эффективный метод заполнения пропусков: {best}")

        # --- PCA графики ---
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        if not hasattr(self, "pca_panel"):
            self.pca_panel = QWidget()
            self.pca_layout = QHBoxLayout()
            self.pca_panel.setLayout(self.pca_layout)
            self.tabs.addTab(self.pca_panel, "PCA-графики")

        clear_layout(self.pca_layout)

        datasets = [
            ("Оригинал", self.df_digitized),
            ("С пропусками", self.df_na.fillna(0)),
            ("Zet", self.df_restored_zet),
            ("Регрессия", self.df_restored_regr)
        ]
        for name, df in datasets:
            subset = df[numeric_cols].dropna()
            if subset.shape[1] < 2 or subset.shape[0] < 2:
                continue
            pca = PCA(n_components=2)
            try:
                X_pca = pca.fit_transform(subset)
            except Exception:
                continue
            fig, ax = plt.subplots(figsize=(4,4))
            ax.scatter(X_pca[:,0], X_pca[:,1], s=10, alpha=0.6)
            ax.set_title(f"PCA: {name}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            canvas = FigureCanvas(fig)
            self.pca_layout.addWidget(canvas)
            plt.close(fig)

        self.tabs.setCurrentWidget(self.pca_panel)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = Lab4App()
    gui.show()
    sys.exit(app.exec_())