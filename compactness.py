import pandas as pd
import numpy as np

FEATURES = ['Стоимость', 'Анализы_код', 'Врач_код', 'Симптомы_код']
P = 2  # Евклидово

data = pd.read_csv("isodata.csv")

def minkowski_dist(a, b, p=P):
    return np.power(np.sum(np.abs(a - b) ** p), 1.0/p)

clusters = data['Cluster'].unique()
compactness_total = 0
total_points = 0

print("Компактность по каждому кластеру:")
for cluster in clusters:
    members = data[data['Cluster'] == cluster]
    points = members[FEATURES].to_numpy()
    n_points = len(points)
    if n_points > 0:
        center = np.mean(points, axis=0)
        ssd = np.sum([minkowski_dist(pt, center, P)**2 for pt in points])
        compactness = ssd / n_points if n_points > 0 else 0
        print(f"Кластер {cluster}: compactness = {compactness:.2f}, size = {n_points}")
        compactness_total += ssd
        total_points += n_points
    else:
        print(f"Кластер {cluster}: пустой кластер (size = 0)")

compactness_avg = compactness_total / total_points if total_points > 0 else 0
print(f"\nСредняя компактность кластеризации: {compactness_avg:.2f}")