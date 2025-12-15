import nmslib
import numpy as np
import scipy


def sparse_matrix_to_str(matrix):
    result = []
    matrix = matrix.tocsr()
    matrix.sort_indices()
    for row in range(matrix.shape[0]):
        arr = [k for k in matrix.indices[matrix.indptr[row]: matrix.indptr[row + 1]]]
        result.append(" ".join([str(k) for k in arr]))
    return result


def dense_vector_to_str(vector):
    if vector.dtype == np.bool_:
        indices = np.flatnonzero(vector)
    else:
        indices = vector
    result = " ".join([str(k) for k in indices])
    return result


class NmslibHNSW():   
    def __init__(self, metric, M, efConstruction=200, efSearch=50, post=0, coords_in_degrees="False"):
        self._space = self._map_space(metric)
        self._M = M
        self._efConstruction = efConstruction
        self._efSearch = efSearch
        self._post = post
        self._cooords_in_degrees = coords_in_degrees
        #self.num_threads = int(self.method_param.get("num_threads", 1))


    def _map_space(self, metric):
        m = metric.lower()
        if m in ("euclidean", "l2"):
            return "l2"
        if m in ("cosine", "angular"):
            return "cosinesimil"        # NMSLIB usa 1 - cos
        if m in ("ip", "inner_product", "dot"):
            return "negdotprod"         # minimiza -dot
        if m == "hamming":
            return "bit_hamming"
        if m == "haversine":
            return "angulardist"
        if m == "jaccard":
            return "jaccard_sparse"
        raise ValueError(f"distance {metric} not accepted. Use 'euclidean'/'l2', 'angular'/'cosine', 'ip'/'inner_product'/'dot', 'hamming', 'jaccard' or 'haversine'")

    def HNSW_nn_index(self, X):
        """Construye el índice HNSW con la métrica elegida."""

        if str(self._space) == "haversine" and self._cooords_in_degrees:
            print("Es que entro por aqui ya esta en rads")
            X = np.deg2rad(X.astype(np.float32, copy=False))

        if self._space == "jaccard_sparse":
            self.index = nmslib.init(
                space=self._space,
                method="hnsw",
                data_type=nmslib.DataType.OBJECT_AS_STRING,
            )
            if type(X) == list:
                sizes = [len(x) for x in X]
                n_cols = max([max(x) for x in X]) + 1
                sparse_matrix = scipy.sparse.csr_matrix((len(X), n_cols), dtype=np.float32)
                sparse_matrix.indices = np.hstack(X).astype(np.int32)
                sparse_matrix.indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
                sparse_matrix.data = np.ones(sparse_matrix.indices.shape[0], dtype=np.float32)
                sparse_matrix.sort_indices()
            # else:
            #     sparse_matrix = scipy.sparse.csr_matrix(X)
            # string_data = sparse_matrix_to_str(sparse_matrix)
            string_data = sparse_matrix_to_str(X)
            self.index.addDataPointBatch(string_data)
        else:
            if X.dtype != np.float32:
                X = X.astype(np.float32)
            print(f"starting index creation with space={self._space}")
            self.index = nmslib.init(method="hnsw", space=self._space)
            self.index.addDataPointBatch(X)

        # self.index = nmslib.init(method="hnsw", space=self._space)
        # self.index.addDataPointBatch(X)
        self.index.createIndex({
            "M": self._M,
            "efConstruction": self._efConstruction,
            "post": self._post
        }, print_progress=True)

        return self

    def HNSW_nn_search(self, metric, train_set, queries, k):
        """
        Devuelve: distances, indices, n_distance_computations
        Nota: 'metric' aquí solo se valida; la métrica real está fijada al construir.
        """
        print("NMSLIB HNSW -> search")

        # Configurar efSearch
        self.index.setQueryTimeParams({"efSearch": int(self._efSearch)})

        lista_indices, lista_coords, lista_dists = [], [] , []

        # Preparar queries según la métrica
        if self._space == "haversine" and (self._cooords_in_degrees == "True"):
            queries = np.deg2rad(queries.astype(np.float32, copy=False))

        nq = queries.shape[0]
        print(f"NMSLIB HNSW -> nº queries: {nq}")
        if self._space == "jaccard_sparse":
            # nq = len(queries)
            print(f"NMSLIB HNSW -> nº queries: {nq}")
            for i in range(nq):
                row = queries[i]  # 1xD CSR
                if scipy.sparse.issparse(row):
                    idx = row.indices  # índices no nulos (0-based)
                else:
                    # Si viniera como lista de ints:
                    idx = np.unique(np.asarray(queries[i], dtype=np.int64))

                v_string = " ".join(str(j) for j in idx)
                Ii, Di = self.index.knnQuery(v_string, k=int(k))
                lista_indices.append(Ii)
                lista_dists.append(Di)
                # v = np.array(queries[i])
                # # v_string = dense_vector_to_str(v)

                # # Realizar la búsqueda
                # Ii, Di = self.index.knnQuery(v_string, k=int(k))
                # lista_indices.append(Ii)
                # lista_dists.append(Di)
        else:

            if queries.dtype != np.float32:
                queries = queries.astype(np.float32)
            for i in range(nq):
                # Realizar la búsqueda
                Ii, Di = self.index.knnQuery(queries[i, :], k=int(k))
                lista_indices.append(Ii)
                lista_dists.append(Di)

                Ci=train_set[Ii]
                lista_coords.append(Ci)

                # Completar Ii con elementos vacios hasta k si es necesario
                if len(Ii)<k:
                    Ii = np.pad(Ii, (0, k - len(Ii)), 'constant', constant_values=-1)
                    Di = np.pad(Di, (0, k - len(Di)), 'constant', constant_values=np.inf)
                    Ci = np.vstack([Ci, np.full((k - len(Ci), train_set.shape[1]), np.nan)])
                    lista_coords[-1] = Ci
                    lista_indices[-1] = Ii
                    lista_dists[-1] = Di

                if len(Ii)<k:
                    print(f"Warning: for query {i}, only found {len(Ii)} neighbors (requested {k})")

        # Store results
        if metric == "jaccard":
            # Get coordinates of the neighbors
            idx2d = np.asarray(lista_indices, dtype=int)
            # 2) Toma las coordenadas (lista de listas de listas)
            coords = [[train_set[int(j)] for j in row] for row in idx2d]
        #else:
            #coords = train_set[lista_indices]
        
        
        return lista_indices, lista_coords, lista_dists, 0
