from k_means_constrained import KMeansConstrained
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import numpy as np

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


def hierarchical_clusters_by_size_ordered(X, tg, batch_size=1000, random_state=None):
    """
    Divide X en clusters de tamaño máximo tg usando MiniBatchKMeans de forma jerárquica.
    Ordena los grupos por proximidad entre sus centroides.
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
