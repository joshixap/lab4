import pandas as pd
import numpy as np

# Параметры алгоритма
K_INIT = 5                  # начальное число кластеров
MIN_CLUSTER_SIZE = 100      # минимальное число точек для кластера (иначе удалить)
P = 2                       # параметр расстояния Минковского (евклидово)
FEATURES = ['Стоимость', 'Анализы_код', 'Врач_код', 'Симптомы_код']

# 1. Загрузка данных
data = pd.read_csv('result.csv')

# Функция расстояния Минковского
def minkowski_dist(a, b, p=P):
    return np.power(np.sum(np.abs(a - b) ** p), 1.0/p)

# Инициализация центров (случайный выбор точек)
np.random.seed(42)
initial_idxs = np.random.choice(len(data), K_INIT, replace=False)
centers = data.loc[initial_idxs, FEATURES].to_numpy()

def assign_clusters(data, centers):
    points = data[FEATURES].to_numpy()
    clusters = []
    for pt in points:
        dists = [minkowski_dist(pt, c, P) for c in centers]
        clusters.append(np.argmin(dists))
    return np.array(clusters)

def update_centers(data, clusters, k):
    new_centers = []
    for idx in range(k):
        members = data[clusters == idx][FEATURES]
        if len(members) > 0:
            new_centers.append(members.mean().to_numpy())
        else:
            # "пустой" кластер: просто повторяем центр
            new_centers.append(np.zeros(len(FEATURES)))
    return np.array(new_centers)

def cluster_variance(data, clusters, centers):
    variances = []
    for idx, center in enumerate(centers):
        members = data[clusters == idx][FEATURES].to_numpy()
        if len(members) > 0:
            var = np.sum([minkowski_dist(pt, center, P)**2 for pt in members])
        else:
            var = 0
        variances.append(var)
    return variances

def cluster_distances(centers):
    # попарные расстояния между центрами
    dists = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dists.append(minkowski_dist(centers[i], centers[j], P))
    return dists

def most_variant_coord(data, clusters, centers, idx):
    # Вычисляет номер координаты с максимальной дисперсией в кластере idx
    members = data[clusters == idx][FEATURES].to_numpy()
    if len(members) > 1:
        vars = np.var(members, axis=0)
        split_dim = np.argmax(vars)
        return split_dim, np.mean(members[:, split_dim])
    else:
        return 0, centers[idx][0]

def split_cluster(centers, idx, split_dim, split_val):
    center_a = centers[idx].copy()
    center_b = centers[idx].copy()
    delta = 1.0
    center_a[split_dim] -= delta
    center_b[split_dim] += delta
    centers = np.delete(centers, idx, axis=0)
    centers = np.vstack([centers, center_a, center_b])
    return centers

def merge_clusters(centers, idx_a, idx_b):
    merged = (centers[idx_a] + centers[idx_b]) / 2
    mask = np.ones(len(centers), dtype=bool)
    mask[[idx_a, idx_b]] = False
    centers = centers[mask]
    centers = np.vstack([centers, merged])
    return centers

def delete_clusters(data, clusters, centers):
    valid_idxs = []
    for idx in range(len(centers)):
        if np.sum(clusters == idx) >= MIN_CLUSTER_SIZE:
            valid_idxs.append(idx)
    centers = centers[valid_idxs]
    mapping = {old:new for new, old in enumerate(valid_idxs)}
    clusters = np.array([mapping[c] for c in clusters if c in valid_idxs])
    return centers, clusters

# Основной цикл
converged = False
iter_count = 0

while not converged and iter_count < 100:
    iter_count += 1

    clusters = assign_clusters(data, centers)
    new_centers = update_centers(data, clusters, len(centers))
    converged = np.allclose(centers, new_centers)
    centers = new_centers

    # Вычисление дисперсий и средних расстояний между центрами
    variances = cluster_variance(data, clusters, centers)
    avg_variance = np.mean(variances)
    split_threshold = 1.2 * avg_variance

    center_dists = cluster_distances(centers)
    if center_dists:
        avg_center_dist = np.mean(center_dists)
    else:
        avg_center_dist = 0
    merge_threshold = 0.8 * avg_center_dist

    # SPLIT clusters
    for idx, var in enumerate(variances):
        if var > split_threshold:
            split_dim, split_val = most_variant_coord(data, clusters, centers, idx)
            centers = split_cluster(centers, idx, split_dim, split_val)
            converged = False
            break  # чтобы не split всех сразу

    # MERGE clusters
    merged = False
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = minkowski_dist(centers[i], centers[j], P)
            if dist < merge_threshold:
                centers = merge_clusters(centers, i, j)
                converged = False
                merged = True
                break
        if merged:
            break

    # DELETE clusters
    centers, clusters = delete_clusters(data, clusters, centers)
    # После удаления может поменяться число кластеров!

    # Прекращение при неизменности меток
    new_clusters = assign_clusters(data, centers)
    if np.array_equal(clusters, new_clusters):
        converged = True

# Сохраняем результат
data['Cluster'] = assign_clusters(data, centers)
data.to_csv('isodata.csv', index=False)