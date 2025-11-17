import numpy as np

# Параметры по умолчанию
DEFAULT_K_INIT = 5
DEFAULT_MIN_CLUSTER_SIZE = 100
DEFAULT_P = 2

def minkowski_dist_matrix(X, centers, p=DEFAULT_P):
    diffs = np.abs(X[:, None, :] - centers[None, :, :])
    dists = np.sum(diffs ** p, axis=2)
    return dists ** (1.0/p)

def assign_clusters_fast(X, centers, p=DEFAULT_P):
    dists = minkowski_dist_matrix(X, centers, p)
    return np.argmin(dists, axis=1)

def update_centers_fast(X, labels, k):
    return np.array([
        X[labels == i].mean(axis=0) if np.any(labels == i) else np.zeros(X.shape[1])
        for i in range(k)
    ])

def cluster_variance_fast(X, labels, centers, p=DEFAULT_P):
    vars = []
    for i, center in enumerate(centers):
        pts = X[labels == i]
        if len(pts) > 0:
            dists = np.sum((pts - center) ** p, axis=1)
            var = np.sum(dists)
        else:
            var = 0
        vars.append(var)
    return vars

def cluster_distances_fast(centers, p=DEFAULT_P):
    diff = centers[None, :, :] - centers[:, None, :]
    dists = np.sum(np.abs(diff) ** p, axis=2) ** (1/p)
    idx = np.triu_indices_from(dists, k=1)
    return dists[idx]

def most_variant_coord_fast(X, labels, centers, idx):
    pts = X[labels == idx]
    if len(pts) > 1:
        vars = np.var(pts, axis=0)
        split_dim = np.argmax(vars)
        return split_dim, np.mean(pts[:, split_dim])
    else:
        return 0, centers[idx][0]

def split_cluster_fast(centers, idx, split_dim, split_val):
    center_a = centers[idx].copy()
    center_b = centers[idx].copy()
    delta = 1.0
    center_a[split_dim] -= delta
    center_b[split_dim] += delta
    centers = np.delete(centers, idx, axis=0)
    centers = np.vstack([centers, center_a, center_b])
    return centers

def merge_clusters_fast(centers, idx_a, idx_b):
    merged = (centers[idx_a] + centers[idx_b]) / 2
    mask = np.ones(len(centers), dtype=bool)
    mask[[idx_a, idx_b]] = False
    centers = centers[mask]
    centers = np.vstack([centers, merged])
    return centers

def delete_clusters_fast(X, labels, centers, min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE):
    valid_idxs = [i for i in range(len(centers)) if np.sum(labels == i) >= min_cluster_size]
    if not valid_idxs:
        return centers, labels
    centers = centers[valid_idxs]
    mapping = {old: new for new, old in enumerate(valid_idxs)}
    labels = np.array([mapping[c] for c in labels if c in valid_idxs])
    return centers, labels

def isodata_clustering(
    X,
    k_init=DEFAULT_K_INIT,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    p=DEFAULT_P,
    max_iter=300,
    split_delta=1.0,
    random_state=None
):
    np.random.seed(random_state if random_state is not None else 42)
    initial_idxs = np.random.choice(len(X), k_init, replace=False)
    centers = X[initial_idxs]
    converged = False
    iter_count = 0
    labels = assign_clusters_fast(X, centers, p)

    while not converged and iter_count < max_iter:
        iter_count += 1
        labels = assign_clusters_fast(X, centers, p)
        new_centers = update_centers_fast(X, labels, len(centers))
        converged = np.allclose(centers, new_centers)
        centers = new_centers

        variances = cluster_variance_fast(X, labels, centers, p)
        avg_variance = np.mean(variances)
        split_threshold = 1.2 * avg_variance

        center_dists = cluster_distances_fast(centers, p)
        avg_center_dist = np.mean(center_dists) if len(center_dists) > 0 else 0
        merge_threshold = 0.8 * avg_center_dist

        # SPLIT
        for idx, var in enumerate(variances):
            if var > split_threshold:
                split_dim, split_val = most_variant_coord_fast(X, labels, centers, idx)
                centers = split_cluster_fast(centers, idx, split_dim, split_val)
                converged = False
                break

        # MERGE
        merged = False
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < merge_threshold:
                    centers = merge_clusters_fast(centers, i, j)
                    converged = False
                    labels = assign_clusters_fast(X, centers, p)
                    merged = True
                    break
            if merged:
                break

        # DELETE
        centers, labels = delete_clusters_fast(X, labels, centers, min_cluster_size)
        new_labels = assign_clusters_fast(X, centers, p)
        if np.array_equal(labels, new_labels):
            converged = True
        labels = new_labels
    return labels, centers

# Вот так файл будет спокойно импортироваться!
# Пример использования:
# import pandas as pd
# from isodata_fast import isodata_clustering, assign_clusters_fast
# data = pd.read_csv('result.csv')
# points = data[FEATURES].values
# labels, centers = isodata_clustering(points)
# data['Cluster'] = labels