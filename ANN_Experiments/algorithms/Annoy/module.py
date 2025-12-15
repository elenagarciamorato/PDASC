import annoy
import numpy as np
import os

class annoy_knn():
    """Euclidean distance (squared) or cosine similarity (using the squared distance of the normalized vectors)
    Works better if you don’t have too many dimensions (like <100) but seems to perform surprisingly well even up to 1,000 dimensions"""
    def __init__(self, n_trees, k_search, metric):
        self._n_trees = int(n_trees)
        self._metric = metric
        self._k_search = int(k_search)

    def annoy_nn_index(self, X):
        """
        Construye el índice Annoy a partir de los vectores X.
        """
        if self._metric == "cosine":
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            metric_used = "euclidean"  # Annoy no tiene 'cosine'
        else:
            metric_used = self._metric

        self.index = annoy.AnnoyIndex(X.shape[1], metric=metric_used)
        for i, vec in enumerate(X):
            self.index.add_item(i, vec.tolist())
        self.index.build(self._n_trees)
        return self
        

    def annoy_nn_search(self, queries, k):
        """
        Busca los k vecinos más cercanos para cada query en xq.
        queries: numpy array de forma (n_queries, dim)
        k: número de vecinos a buscar
        Retorna: (distancias, índices)
        """
        if self._metric == "cosine":
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        lista_indices, lista_dists = [], []
        
        for vec in queries:
            
            idxs, dists = self.index.get_nns_by_vector(vec.tolist(), k, self._k_search,include_distances=True)
            lista_indices.append(idxs)
            lista_dists.append(dists)
        return np.array(lista_indices), np.array(lista_dists), 0  # Retornamos 0 para n_distances ya que Annoy no proporciona esta información

    def pickleable_index(self, dataset):

        path = f"./ANN_Experiments/NearestNeighbors/{dataset}/indexes/ANNOY_index.ann"
        self.index.save(path)

        with open(path, "rb") as f:
            pickeable_index = f.read()
        f.close()
        os.remove(path)

        return pickeable_index

