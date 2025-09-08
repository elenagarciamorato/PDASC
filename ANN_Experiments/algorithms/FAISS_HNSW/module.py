import faiss
import numpy as np
import sklearn.preprocessing


# from ..FAISS.module import Faiss


class FaissHNSW():
    def __init__(self, metric, M, efConstruction=200, efSearch=50):
        self._metric = metric
        self._M = M
        self._efConstruction = efConstruction
        self._efSearch = efSearch

    # For binary data (0/1), check if the data is packed (uint8 with values > 1 in some cell)
    def _is_packed_binary(self, X):
        # Empaquetado: matriz de uint8 con valores > 1 en alguna celda (bytes con 8 bits mezclados)
        return X.dtype == np.uint8 and np.any(X > 1)

    def _ensure_binary_packed(self, X):
        """
        If X is not packed binary (bool or 0/1), convert to packed binary (uint8 with bits).
        If X is already packed binary, do nothing.
        Return (X_bytes, d_bits) where d_bits is the original number of bits (not bytes).
        """
        if self._is_packed_binary(X):
            d_bits = X.shape[1] * 8
            return X, d_bits

        # Caso no empaquetado: bool o 0/1
        if X.dtype != np.uint8:
            X = X.astype(np.uint8)  # bool -> {0,1}, float -> {0,1} si ya venía binario

        n, d_bits = X.shape
        pad = (-d_bits) % 8
        if pad:
            X = np.pad(X, ((0, 0), (0, pad)), mode="constant", constant_values=0)
            d_bits += pad

        X_bytes = np.packbits(X, axis=1)
        return X_bytes, d_bits

    def HNSW_nn_index(self, X):
        """Construye el índice HNSW con la métrica elegida."""

        if self._metric in ("cosine", "angular"):
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")
        if self._metric in ("hamming"):
            # Tu HDF5 trae bool con shape (*, 256) -> empaquetar a bytes
            Xb, d_bits = self._ensure_binary_packed(X)
            self._d_bits = d_bits
            X = Xb

        elif X.dtype != np.float32:
            X = X.astype(np.float32)

        d = X.shape[1]
        M = self._M

        if self._metric in ("euclidean", "l2"):
            self.index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
            print("Entro por aqui nn index con metric L2")
        elif self._metric in ("cosine", "angular", "ip", "inner_product", "dot"):
            print("Entro por aqui nn index con metric angular/cosine o ip/inner_product /dot")
            self.index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        elif self._metric in ("hamming"):
            print("Entro por aqui nn index con metric hamming")
            self.index = faiss.IndexBinaryHNSW(d_bits, M)
        else:
            raise ValueError(
                f"distance {self._metric} not accepted. Use 'euclidean'/'l2' or 'angular'/'cosine' or 'ip'/'inner_product'/'dot'")

        self.index.hnsw.efConstruction = self._efConstruction
        self.index.add(X)
        faiss.omp_set_num_threads(1)
        return self

    def HNSW_nn_search(self, queries, k):
        """
        Realizar búsqueda y calcular estadísticas de distancias computadas
        """
        print("Entro por aqui nn search")
        faiss.omp_set_num_threads(1)

        # Preparar queries según la métrica
        if self._metric in ("cosine", "angular"):
            queries = sklearn.preprocessing.normalize(queries, axis=1, norm="l2")

        if self._metric in ("hamming"):
            if self._d_bits is None:
                raise RuntimeError("Índice binario no inicializado (falta d_bits).")

            Qb, q_bits = self._ensure_binary_packed(queries)
            if q_bits != self._d_bits:
                raise ValueError(f"Dimensión en bits de queries ({q_bits}) != índice ({self._d_bits}).")

            self.index.hnsw.efSearch = self._efSearch
            distances, indices = self.index.search(Qb, k)  # Hamming (int)
            queries = Qb  # para consistencia en el chequeo de tipo abajo

        elif queries.dtype != np.float32:
            queries = queries.astype(np.float32)

        # Realizar la búsqueda con efSearch configurado
        self.index.hnsw.efSearch = self._efSearch

        # Configurar listas para resultados
        lista_indices, lista_dists, lista_n_distances = [], [], []
        nq = queries.shape[0]
        for i in range(nq):
            # Resetear estadísticas antes de la búsqueda
            faiss.cvar.hnsw_stats.reset()
            Di, Ii = self.index.search(queries[i:i + 1, :], k)
            lista_indices.append(Ii[0])
            lista_dists.append(Di[0])
            # Obtener estadísticas de distancias computadas

            #print(dir(faiss.cvar.hnsw_stats))
            #n1 = faiss.cvar.hnsw_stats.n1  # Setup/inicialización por query
            #print(n1)
            #n2 = faiss.cvar.hnsw_stats.n2  # Operaciones de nivel superior
            #print(n2)
            n3 = faiss.cvar.hnsw_stats.ndis  # Operaciones principales (distancias/comparaciones)

            n_distances = n3

            lista_n_distances.append(n_distances)

        # Obtener estadísticas de distancias computadas => Normalmente no es el contador bueno
        # self._last_ndis = faiss.cvar.hnsw_stats.ndis

        # print(f"  - Distancias computadas: {n_distances}")
        # print(f"Distancias computadas (faiss): {self._last_ndis}")

        return np.array(lista_dists), np.array(lista_indices), np.array(lista_n_distances)
