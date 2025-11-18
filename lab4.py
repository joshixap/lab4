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

from digitization import *
from isodata import *
from remove_blocks import *
from missing_fill_methods import *
from make_full_datasets import *
from compactness import *

METRICS = [
    ("mean", np.mean),
    ("median", np.median),
    ("std", np.std)
]

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

        # Собираем метрики для всех датасетов
        results = []
        for col in numeric_cols:
            orig = self.df_digitized[col].dropna()
            hole = self.df_na[col].dropna()
            zet = self.df_restored_zet[col].dropna()
            regr = self.df_restored_regr[col].dropna()
            row = [col]
            for metric_name, metric_fun in METRICS:
                # Везде считаем метрику
                m_orig = metric_fun(orig) if len(orig) > 0 else np.nan
                m_hole = metric_fun(hole) if len(hole) > 0 else np.nan
                m_zet = metric_fun(zet) if len(zet) > 0 else np.nan
                m_regr = metric_fun(regr) if len(regr) > 0 else np.nan
                # Считаем ошибки по формуле Δ
                err_hole = abs(m_orig - m_hole) / abs(m_orig) * 100 if m_orig != 0 and not np.isnan(m_hole) else 0
                err_zet = abs(m_orig - m_zet) / abs(m_orig) * 100 if m_orig != 0 and not np.isnan(m_zet) else 0
                err_regr = abs(m_orig - m_regr) / abs(m_orig) * 100 if m_orig != 0 and not np.isnan(m_regr) else 0
                row += [
                    f"{m_orig:.3f}", f"{m_hole:.3f}", f"{m_zet:.3f}", f"{m_regr:.3f}",
                    f"{err_hole:.2f}", f"{err_zet:.2f}", f"{err_regr:.2f}"
                ]
            results.append(row)

        # Суммарная ошибка по каждому методу (по всем метрикам и столбцам)
        sum_err_zet = 0
        sum_err_regr = 0
        metric_err_zet = []
        metric_err_regr = []
        for row in results:
            # В каждой строке есть сдвиг для каждой метрики
            # Δ по Z = индексы 7, 7+7 и пр. Δ по Рег = индексы 8, 8+7 и пр.
            for i in range(len(METRICS)):
                sum_err_zet += float(row[6 + i * 7])
                sum_err_regr += float(row[7 + i * 7])
                metric_err_zet.append(float(row[6 + i * 7]))
                metric_err_regr.append(float(row[7 + i * 7]))

        # Выбор лучшего метода
        best_method = None
        if sum_err_zet < sum_err_regr:
            best_method = "Zet"
        else:
            best_method = "Регрессия"

        # Формирование таблицы отчета
        n_metrics = len(METRICS)
        col_headers = ["Признак",]
        for metric_name, _ in METRICS:
            col_headers += [
                f"{metric_name}-оригинал",
                f"{metric_name}-дырявый",
                f"{metric_name}-Zet",
                f"{metric_name}-Регрессия",
                f"Δ {metric_name}-дыр.",
                f"Δ {metric_name}-Zet",
                f"Δ {metric_name}-Регрессия"
            ]
        self.table_report.clear()
        self.table_report.setColumnCount(len(col_headers))
        self.table_report.setRowCount(len(results) + 2)
        self.table_report.setHorizontalHeaderLabels(col_headers)

        for i, row in enumerate(results):
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                # Подсветить лучший Δ по минимальному значению (кроме дырявого)
                if j % 7 in (5,6):  # Δ Zet/Reg
                    if best_method == "Zet" and j % 7 == 5:
                        item.setBackground(Qt.yellow)
                    if best_method == "Регрессия" and j % 7 == 6:
                        item.setBackground(Qt.yellow)
                self.table_report.setItem(i, j, item)

        # Итоговые строки
        avg_err_zet = sum_err_zet / (len(results)*n_metrics) if (len(results)*n_metrics) > 0 else 0
        avg_err_regr = sum_err_regr / (len(results)*n_metrics) if (len(results)*n_metrics) > 0 else 0
        self.table_report.setItem(len(results), 0, QTableWidgetItem("Суммарная Δ Zet"))
        self.table_report.setItem(len(results), 5, QTableWidgetItem(f"{sum_err_zet:.2f}"))
        self.table_report.setItem(len(results), 6, QTableWidgetItem(f"{avg_err_zet:.2f}"))
        self.table_report.setItem(len(results)+1, 0, QTableWidgetItem("Суммарная Δ Регрессия"))
        self.table_report.setItem(len(results)+1, 6, QTableWidgetItem(f"{sum_err_regr:.2f}"))
        self.table_report.setItem(len(results)+1, 7, QTableWidgetItem(f"{avg_err_regr:.2f}"))

        for idx in [len(results), len(results)+1]:
            for color_col, check_method in [(5, "Zet"), (7, "Регрессия")]:
                if best_method == check_method:
                    item = self.table_report.item(idx, color_col)
                    if item: item.setBackground(Qt.yellow)
        self.tabs.setCurrentWidget(self.table_report)
        self.log_stat(f"Отчет сформирован. Наиболее эффективный метод заполнения пропусков: {best_method}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = Lab4App()
    gui.show()
    sys.exit(app.exec_())