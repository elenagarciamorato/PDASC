import faiss
import numpy as np
import sklearn.preprocessing


class FaissLSH():

    def __init__(self, metric, n_bits):
        self._n_bits = int(n_bits)
        self.index = None
        self._metric = metric
    

    def LSH_nn_index(self, X):
        """
        Crea un IndexLSH y añade los datos.
        - Para 'cosine'/'angular' normaliza L2 (recomendado).
        - Para 'euclidean' -> se normaliza también y se busca por coseno (aproximación).
        """
        if self._metric in ("cosine", "angular", "ip", "inner_product", "dot"):
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")
        elif self._metric in ("euclidean", "l2"):
            # Aviso: LSH de FAISS es angular; normalizamos para aproximar.
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")
        else:
            raise ValueError(
                f"distance {self._metric} not accepted. "
                "Use 'euclidean'/'l2' o 'angular'/'cosine'/'ip'/'inner_product'/'dot'"
            )

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self._d = X.shape[1]
        
        self.index = faiss.IndexLSH(self._d, self._n_bits)

        self.index.train(X)
        self.index.add(X)
        return self


    def LSH_nn_search(self, queries, k):
        """
        Realizar búsqueda y calcular estadísticas de distancias computadas
        """

        print("Entro por aqui nn search")

        
        if self._metric in ("cosine", "angular", "ip", "inner_product", "dot", "euclidean", "l2"):
            queries = sklearn.preprocessing.normalize(queries, axis=1, norm="l2")

        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)



        lista_indices, lista_dists, lista_n_distances = [], [], []
        nq=queries.shape[0]
        for i in range(nq):
            Di, Ii = self.index.search(queries[i:i+1,:], k)
            lista_indices.append(Ii[0])
            lista_dists.append(Di[0])
            #n_distances = int(self.index.ntotal)
            # Nr distancias son todas las del dataset
            #lista_n_distances.append(n_distances)

        lista_n_distances=np.nan

        
        return np.array(lista_dists), np.array(lista_indices), np.array(lista_n_distances)
