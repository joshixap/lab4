import numpy as np
import pandas as pd

def minkowski_dist(a, b, p=2):
    return np.power(np.sum(np.abs(a - b) ** p), 1.0/p)

def cluster_compactness(
    data: pd.DataFrame,
    cluster_col: str = "Cluster",
    features=None,
    p: int = 2
):
    """
    Вычисляет компактность кластеров для датасета pandas.DataFrame,
    возвращает текстовый отчет и среднюю компактность.
    Параметры:
      - data (DataFrame): ваш DataFrame с колонкой кластеров
      - cluster_col (str): имя колонки с метками кластеров ("Cluster")
      - features (list): признаки, по которым считать (list of str)
      - p (int): степень расстояния Минковского (часто 2 — Евклид)
    Возвращает:
      - report_text (str): текст, аналогичный print в исходнике
      - compactness_avg (float): средняя компактность по всем кластерам
    """
    if features is None:
        # Если признаки не заданы — пусть берутся все float/int
        features = [
            col for col in data.select_dtypes(include=[np.number]).columns
            if col != cluster_col
        ]

    clusters = data[cluster_col].unique()
    compactness_total = 0
    total_points = 0

    report_lines = ["Компактность по каждому кластеру:"]
    for cluster in clusters:
        members = data[data[cluster_col] == cluster]
        points = members[features].to_numpy()
        n_points = len(points)
        if n_points > 0:
            center = np.mean(points, axis=0)
            ssd = np.sum([minkowski_dist(pt, center, p)**2 for pt in points])
            compactness = ssd / n_points if n_points > 0 else 0
            report_lines.append(
                f"Кластер {cluster}: compactness = {compactness:.2f}, size = {n_points}"
            )
            compactness_total += ssd
            total_points += n_points
        else:
            report_lines.append(f"Кластер {cluster}: пустой кластер (size = 0)")
    compactness_avg = compactness_total / total_points if total_points > 0 else 0
    report_lines.append(f"\nСредняя компактность кластеризации: {compactness_avg:.2f}")
    report_text = "\n".join(report_lines)
    return report_text, compactness_avg

# Ниже пример использования:
# from compactness import cluster_compactness
# features = ['Стоимость', 'Анализы_код', 'Врач_код', 'Симптомы_код']
# report, avg = cluster_compactness(data, features=features, p=2)
# print(report)