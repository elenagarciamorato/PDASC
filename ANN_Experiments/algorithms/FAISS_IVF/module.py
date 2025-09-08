import faiss
import numpy as np
import sklearn.preprocessing


class FaissIVF():
    def __init__(self, metric, n_list, n_probe):
        self._n_list = int(n_list)
        self._metric = metric
        self._n_probe = int(n_probe)

    def IVF_nn_index(self, X):
        """
        Construye un IndexIVFFlat de FAISS según self._metric y self._n_list.
        - Métricas admitidas: "euclidean"/"l2" y "cosine"/"angular".
        - Para "cosine"/"angular" se normaliza y se usa METRIC_INNER_PRODUCT.
        """
        #print(faiss.__version__)

        if self._metric in ("cosine", "angular"):
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        d = X.shape[1]

        if self._metric in ("euclidean", "l2"):
            self.quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(self.quantizer, d, self._n_list, faiss.METRIC_L2)
            print("Entro por aqui nn index con metric L2")
        elif self._metric in ("cosine", "angular", "ip", "inner_product", "dot"):
            self.quantizer = faiss.IndexFlatIP(d)
            print("Entro por aqui nn index con metric angular o cosine")
            self.index = faiss.IndexIVFFlat(self.quantizer, d, self._n_list, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(
                f"distance {self._metric} not accepted. Use 'euclidean'/'l2' or 'angular'/'cosine' or 'ip'/'inner_product'/'dot'")

        self.index.train(X)
        self.index.add(X)
        return self

    def IVF_nn_search(self, queries, k):
        """
        Realizar búsqueda y calcular estadísticas de distancias computadas
        """
        print("Entro por aqui nn search")
        print(f"shape of queries: {queries.shape}")
        lista_indices, lista_dists, lista_n_distances = [], [], []

        # Preparar queries según la métrica
        if self._metric in ("cosine", "angular"):
            queries = sklearn.preprocessing.normalize(queries, axis=1, norm="l2")

        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)

        # Realizar la búsqueda
        nq = queries.shape[0]
        self.index.nprobe = self._n_probe
        for i in range(nq):
            # Resetear estadísticas antes de la búsqueda
            faiss.cvar.indexIVF_stats.reset()
            Di, Ii = self.index.search(queries[i:i + 1, :], k)
            lista_indices.append(Ii[0])
            lista_dists.append(Di[0])
            lista_n_distances.append(faiss.cvar.indexIVF_stats.ndis)

        return np.array(lista_dists), np.array(lista_indices), np.array(lista_n_distances)
