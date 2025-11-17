import sys
import pandas as pd
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QTableWidget, QTableWidgetItem, QTextEdit, QTabWidget,
    QGroupBox
)
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt

# Импорт всех функций из ваших файлов!
from digitization import *
from isodata import *
from remove_blocks import *
from missing_fill_methods import *
from make_full_datasets import *
from compactness import *

class Lab4App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Восстановление и анализ датасета - lab4")
        self.resize(1200, 700)

        self.df_loaded = pd.DataFrame()
        self.df_digitized = pd.DataFrame()
        self.df_isodata = pd.DataFrame()
        self.df_na = pd.DataFrame()
        self.df_restored = pd.DataFrame()
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

        restore_group = QGroupBox("Восстановление пропусков")
        restore_layout = QVBoxLayout()
        self.restore_method_box = QComboBox()
        self.restore_method_box.addItems([
            "Линейная регрессия (дозаполнение средним/мединой)",
            "Zet-алгоритм (дозаполнение средним/мединой)"
        ])
        restore_layout.addWidget(QLabel("Метод восстановления:"))
        restore_layout.addWidget(self.restore_method_box)
        btn_restore = QPushButton("Восстановить")
        btn_restore.clicked.connect(self.on_restore)
        restore_layout.addWidget(btn_restore)
        restore_group.setLayout(restore_layout)
        left_panel.addWidget(restore_group)

        isodata_group = QGroupBox("ISODATA-кластеризация")
        isodata_layout = QVBoxLayout()
        btn_isodata = QPushButton("Применить ISODATA")
        btn_isodata.clicked.connect(self.on_isodata)
        isodata_layout.addWidget(btn_isodata)
        isodata_group.setLayout(isodata_layout)
        left_panel.addWidget(isodata_group)

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
        self.table_restored = QTableWidget()
        self.tabs.addTab(self.table_restored, "Восстановленный")
        self.table_isodata = QTableWidget()
        self.tabs.addTab(self.table_isodata, "ISODATA")
        self.compact_box = QTextEdit()
        self.compact_box.setReadOnly(True)
        self.tabs.addTab(self.compact_box, "Компактность кластеров")

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
            self.df_restored = pd.DataFrame()
            self.df_isodata = pd.DataFrame()
            self._show_df(self.table_loaded, self.df_loaded)
            self._show_df(self.table_digitized, self.df_digitized)
            self.log_stat("Загружен исходный датасет:\n" + self._basic_stats(self.df_loaded))
            self.compact_box.setText("")
            self._show_df(self.table_na, pd.DataFrame())
            self._show_df(self.table_restored, pd.DataFrame())
            self._show_df(self.table_isodata, pd.DataFrame())

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
        self.df_restored = pd.DataFrame()
        self._show_df(self.table_restored, pd.DataFrame())
        self._show_df(self.table_isodata, pd.DataFrame())
        self.compact_box.setText("")

    def on_restore(self):
        if self.df_na.empty:
            return
        method_text = self.restore_method_box.currentText()
        features = [
            "Пол", "Год_выдачи_паспорта", "СНИЛС_цифр", "Симптомы_код", "Врач_код",
            "Часы_до_визита", "Анализы_код", "Часы_до_анализа", "Стоимость", "Банк_код"
        ]
        if method_text.startswith("Линейная регрессия"):
            restored = make_regression_dataset(self.df_na, features, n_rounds=5)
        elif method_text.startswith("Zet-алгоритм"):
            restored = make_zet_dataset(self.df_na, features, n_rounds=5)
        else:
            restored = self.df_na.copy()
        self.df_restored = restored
        self._show_df(self.table_restored, self.df_restored)
        self.log_stat("Восстановленный датасет:\n" + self._basic_stats(self.df_restored))
        self._show_df(self.table_isodata, pd.DataFrame())
        self.compact_box.setText("")

    def on_isodata(self):
        if self.df_restored.empty:
            return
        df_for_iso = self.df_restored.copy()
        FEATURES = [
            "Стоимость", "Анализы_код", "Врач_код", "Симптомы_код"
        ]
        features_present = [f for f in FEATURES if f in df_for_iso.columns]
        if len(df_for_iso) == 0 or len(features_present) == 0:
            self.compact_box.setText("Данных для кластеризации недостаточно.")
            return

        # Быстрая ISODATA с новыми функциями!
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
            # Merging (используйте minkowski_dist или клид для расстояния между центрами)
            center_dists = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = minkowski_dist(centers[i], centers[j], P)
                    center_dists.append(dist)
                    # Для merge используем быструю функцию, по примеру
                    merge_threshold = 0.8 * np.mean(center_dists) if center_dists else 0
                    if dist < merge_threshold:
                        centers = merge_clusters_fast(centers, i, j)
                        converged = False
                        labels = assign_clusters_fast(points, centers)
                        break
                else:
                    continue
                break

            # Splitting/other heuristics, если нужны (можете доработать)
            # Удаляем лишние кластеры
            centers, labels = delete_clusters_fast(points, labels, centers)
            new_labels = assign_clusters_fast(points, centers)
            if np.array_equal(labels, new_labels):
                converged = True
            labels = new_labels

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = Lab4App()
    gui.show()
    sys.exit(app.exec_())