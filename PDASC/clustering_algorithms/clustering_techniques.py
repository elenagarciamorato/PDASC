from k_means_constrained import KMeansConstrained
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import numpy as np
import kmedoids as fast_kmedoids
from math import ceil
from sklearn.metrics import pairwise_distances
import warnings

def constrained_cluster_reorder(data_objects, tg, random_state=42):
    """
    Clusteriza data_objects en grupos de tamaño tg (excepto el último)
    y devuelve un array reordenado concatenando los objetos de cada cluster.

    - data_objects: np.ndarray, (n_samples, n_features)
    - tg: tamaño objetivo de cada cluster
    """
    n_samples = len(data_objects)
    n_clusters = int(np.ceil(n_samples / tg))
    print(f"Clustering en {n_clusters} grupos de tamaño máximo {tg}")

    # 1. Clustering con restricción de tamaño
    model = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=None,  # sin mínimo, excepto el último
        size_max=tg,  # límite máximo
        random_state=random_state
    )
    model.fit(data_objects)
    labels = model.labels_

    # 2. Reordenar los datos por cluster
    order = np.argsort(labels)
    data_reordered = data_objects[order]

    # 3. (Opcional) Devolver los índices agrupados también
    grouped_indices = [np.where(labels == i)[0] for i in range(n_clusters)]

    return data_reordered, grouped_indices, labels

def pca_chunking(X, tg):
    """
    Agrupa X en bloques de tamaño tg usando una proyección PCA 1D
    para ordenar los puntos por similitud, y los concatena uno tras otro.

    Devuelve:
      - X_reordered: array reordenado completo
      - groups: lista de arrays (bloques consecutivos de tamaño tg)
      - group_indices: índices originales por grupo
    """
    n_samples = len(X)
    n_clusters = int(np.ceil(n_samples / tg))

    # 1️⃣ Proyección PCA a 1D (la dirección de máxima varianza)
    pca = PCA(n_components=1, random_state=42)
    X_proj = pca.fit_transform(X).ravel()

    # 2️⃣ Ordenamos los puntos según esa proyección
    order = np.argsort(X_proj)
    X_reordered = X[order]

    # 3️⃣ Dividimos en grupos consecutivos de tamaño tg
    groups = [X_reordered[i:i + tg] for i in range(0, len(X_reordered), tg)]

    # 4️⃣ (Opcional) guardamos también los índices originales por grupo
    group_indices = [order[i:i + tg] for i in range(0, len(order), tg)]

    print(f"Dividido en {len(groups)} grupos de tamaño {tg} (último puede ser menor)")

    return X_reordered, groups

def minibatch_cluster_and_concatenate(X, tg, random_state=42):
    """
    Agrupa X en clusters de tamaño aproximado tg usando MiniBatchKMeans
    y devuelve los datos concatenados grupo a grupo (similares consecutivos).

    - X: np.ndarray (n_samples, n_features)
    - tg: tamaño deseado del grupo
    """
    n_samples = len(X)
    n_clusters = int(np.ceil(n_samples / tg))
    print(f"Creando {n_clusters} clusters de tamaño objetivo {tg}...")

    # 1️⃣ Clustering rápido
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=1024,
        max_iter=100
    ).fit(X)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 2️⃣ Agrupar índices por cluster
    grouped_indices = []
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue  # evita clusters vacíos
        # Ordenar dentro del cluster por distancia al centroide
        dists = np.linalg.norm(X[idx] - centers[k], axis=1)
        sorted_idx = idx[np.argsort(dists)]
        grouped_indices.append(sorted_idx)

    # 3️⃣ Concatenar todos los clusters uno tras otro
    order = np.concatenate(grouped_indices)
    X_reordered = X[order]

    # 4️⃣ (Opcional) dividir en grupos exactos de tamaño tg
    groups = [X_reordered[i:i+tg] for i in range(0, len(X_reordered), tg)]

    print(f"Hecho ✅  → {len(groups)} grupos creados (último puede ser menor).")

    return X_reordered, groups


def hierarchical_clusters_by_size(X, tg, batch_size=1000, random_state=None):
    """
    Implementación propuesta por Javi
    Divide X en clusters de tamaño máximo tg usando MiniBatchKMeans de forma jerárquica.
    Ajusta automáticamente el número de clusters iniciales.
    Devuelve el array reordenado y los índices originales por grupo.
    """
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    n = len(X)
    n_clusters = int(np.ceil(n / tg))
    n_clusters_initial = min(20, n_clusters)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters_initial, batch_size=batch_size,
                             random_state=random_state)
    labels = kmeans.fit_predict(X)

    grouped_indices = []
    for i in range(n_clusters_initial):
        idx = np.where(labels == i)[0]
        points = X[idx]
        if len(points) <= tg:
            grouped_indices.append(idx)
        else:
            n_sub = int(np.ceil(len(points) / tg))
            kmeans_sub = MiniBatchKMeans(n_clusters=n_sub, batch_size=batch_size,
                                         random_state=random_state)
            sub_labels = kmeans_sub.fit_predict(points)
            for j in range(n_sub):
                idx_sub = idx[np.where(sub_labels == j)[0]]
                if len(idx_sub) > 0:
                    grouped_indices.append(idx_sub)

    order = np.concatenate(grouped_indices)
    X_reordered = X[order]

    print(f"Dividido en {len(grouped_indices)} grupos de tamaño máximo {tg} (último puede ser menor)")
    return X_reordered, grouped_indices

"""
def hierarchical_clusters_by_size_ordered(X, tg, batch_size=1000, random_state=None):
    # Divide X en clusters de tamaño máximo tg usando MiniBatchKMeans de forma jerárquica.
    #Ordena los grupos por proximidad entre sus centroides.
    #Devuelve el array reordenado y los índices originales por grupo.

    n = len(X)
    n_clusters = int(np.ceil(n / tg))
    n_clusters_initial = min(20, n_clusters)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters_initial, batch_size=batch_size,
                             random_state=random_state)
    labels = kmeans.fit_predict(X)

    grouped_indices = []
    centroids = []
    for i in range(n_clusters_initial):
        idx = np.where(labels == i)[0]
        points = X[idx]
        if len(points) <= tg:
            grouped_indices.append(idx)
            centroids.append(points.mean(axis=0))
        else:
            n_sub = int(np.ceil(len(points) / tg))
            kmeans_sub = MiniBatchKMeans(n_clusters=n_sub, batch_size=batch_size,
                                         random_state=random_state)
            sub_labels = kmeans_sub.fit_predict(points)
            for j in range(n_sub):
                idx_sub = idx[np.where(sub_labels == j)[0]]
                if len(idx_sub) > 0:
                    grouped_indices.append(idx_sub)
                    centroids.append(X[idx_sub].mean(axis=0))

    # Ordenar los grupos por recorrido greedy de centroides
    centroids = np.array(centroids)
    n_groups = len(grouped_indices)
    visited = np.zeros(n_groups, dtype=bool)
    order_groups = []
    current = 0
    order_groups.append(0)
    visited[0] = True
    for _ in range(1, n_groups):
        dists = np.linalg.norm(centroids[current] - centroids, axis=1)
        dists[visited] = np.inf
        next_group = np.argmin(dists)
        order_groups.append(next_group)
        visited[next_group] = True
        current = next_group

    # Concatenar los grupos en el orden calculado
    ordered_indices = [grouped_indices[i] for i in order_groups]
    order = np.concatenate(ordered_indices)
    X_reordered = X[order]

    #print(f"Dividido en {n_groups} grupos ordenados por proximidad de centroides (máx {tg} elementos por grupo)")

    return X_reordered, order
"""

def hierarchical_clusters_by_size_ordered(X, tg, method='kmeans', distance_function='euclidean',
                                          batch_size=1000, random_state=None):
    """
    Divide X en clusters jerárquicos de tamaño máximo tg y concatena
    los clusters en un orden calculado por proximidad entre sus representativos.

    Parámetros:
        X : np.ndarray
        tg : int, tamaño máximo de cada grupo
        method : {'fastpam', 'kmeans'}
        distance_function : métrica de distancia usada para clustering
        batch_size : int, solo para kmeans
        random_state : int o None
    Retorna:
        X_reordered : np.ndarray
        order : np.ndarray de índices respecto a X
    """
    X = np.asarray(X)
    n = len(X)
    if n == 0:
        return np.empty((0, X.shape[1])), np.array([], dtype=int)

    # 1️⃣ determinar número de clusters inicial
    n_clusters = int(ceil(n / tg))
    n_clusters_initial = min(20, n_clusters)

    # 2️⃣ función auxiliar para clustering
    def cluster_fit(data, n_c):
        if n_c <= 1:
            return np.zeros(len(data), dtype=int), np.reshape(data.mean(axis=0), (1, -1))
        if method == 'fastpam':

            model = fast_kmedoids.KMedoids(n_clusters=n_c, method='fasterpam', metric=distance_function)
            model.fit(data)
            labels = model.labels_
            reprs = data[model.medoid_indices_.astype(int)]
            return labels, reprs
        elif method == 'kmeans':
            if distance_function != 'euclidean':
                raise ValueError("KMeans solo admite euclidean")
            km = MiniBatchKMeans(n_clusters=n_c, batch_size=batch_size, random_state=random_state)
            labels = km.fit_predict(data)
            return labels, km.cluster_centers_
        else:
            raise ValueError("Método desconocido. Usa 'fastpam' o 'kmeans'.")

    # 3️⃣ clustering inicial
    labels_init, repr_init = cluster_fit(X, n_clusters_initial)

    grouped_indices = []
    repr_vectors = []

    # 4️⃣ subdividir clusters si son mayores que tg
    for i in range(n_clusters_initial):
        idx = np.where(labels_init == i)[0]
        if len(idx) == 0:
            continue
        points = X[idx]
        if len(points) <= tg:
            grouped_indices.append(idx)
            repr_vectors.append(repr_init[i] if i < len(repr_init) else points.mean(axis=0))
        else:
            n_sub = int(ceil(len(points) / tg))
            sub_labels, sub_repr = cluster_fit(points, n_sub)
            for j in range(n_sub):
                idx_sub = idx[sub_labels == j]
                if len(idx_sub) > 0:
                    grouped_indices.append(idx_sub)
                    repr_vectors.append(sub_repr[j])

    repr_vectors = np.vstack(repr_vectors)
    n_groups = len(grouped_indices)

    # 5️⃣ calcular distancias entre representativos para ordenar
    try:
        if distance_function.lower() == 'jaccard':
            repr_bool = (repr_vectors != 0).astype(bool)
            D = pairwise_distances(repr_bool, metric='jaccard')
        else:
            D = pairwise_distances(repr_vectors, metric=distance_function)
    except Exception:
        warnings.warn("No se pudo usar pairwise_distances con metric='{metric}', se usa euclidean",
                      RuntimeWarning)
        D = pairwise_distances(repr_vectors, metric='euclidean')

    # 6️⃣ recorrido greedy para determinar orden
    overall_mean = X.mean(axis=0)
    start_idx = int(np.argmin(np.linalg.norm(repr_vectors - overall_mean, axis=1)))
    visited = np.zeros(n_groups, dtype=bool)
    order_groups = [start_idx]
    visited[start_idx] = True
    current = start_idx

    for _ in range(1, n_groups):
        dists = D[current].copy()
        dists[visited] = np.inf
        next_group = int(np.argmin(dists))
        order_groups.append(next_group)
        visited[next_group] = True
        current = next_group

    # 7️⃣ concatenar
    ordered_indices = [grouped_indices[i] for i in order_groups]
    order = np.concatenate(ordered_indices)
    X_reordered = X[order]

    return X_reordered, order

