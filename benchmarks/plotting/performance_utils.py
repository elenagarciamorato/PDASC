from benchmarks.neighbors_utils import *
import logging
import pandas as pd


# Recall Benchmark
def recall(dataset_name, d, method, k, same_train_test=False, file_name_le=None, file_name=None):

    # Recall in Exhaustive Point Query (query points are the same from training set)
    if same_train_test:

        # Load neighbors obtained through the method choosen
        indices_mc, coords_mc, dists_mc = load_neighbors(file_name)

        '''
        hit = 0.0
        for i in range(indices_mc.shape[0]):
            if indices_mc[i] == i:
                hit = hit + 1
        '''

        # Count number of 1-neighbor which are the same as the point searched
        #hit = map(lambda x, y: x == y, list(indices_mc), range(indices_mc.shape[0])).count(True)


    # Recall in query points different from training set
    else:
        # Load neighbors obtained through linear exploration
        indices_le, coords_le, dists_le = load_neighbors(file_name_le)

        # Load neighbors obtained through the method choosen
        indices_mc, coords_mc, dists_mc = load_neighbors(file_name)

        '''
        hit = 0.0
        for i in range(indices_mc.shape[0]):
            hit = hit + len(np.intersect1d(indices_mc[i].astype(int), indices_le[i]))
        '''

        # Count number of 1-neighbor which are the same as the point searched
        hit = sum(map(lambda x, y: len(np.intersect1d(x.astype(int), y)), list(indices_mc), list(indices_le)))

    # Recall: %  hit returned vs number of points
    rec = hit / indices_mc.size * 100


    # Show percentage of hit/miss on screen and save information on log file
    '''
    print ("---- Case " + str(k) + " nn applying " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Correct neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    print("Hit percentage: " + str(rec) + "%\n\n")
    logging.info("---- Case " + str(k) + " nn applying " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    '''
    logging.info("Correct neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    logging.info("Hit percentage: " + str(rec) + "%\n\n")

    return rec


# Compare intersection percentage between neighbors found by two different methods
def compare(dataset_name, d, method1, method2, knn, file_name1=None, file_name2=None):

    # Load neighbors obtained through first method
    indices_m1, coords_m1, dists_m1 = load_neighbors(file_name1)

    # Load neighbors obtained through the second method choosen
    indices_m2, coords_m2, dists_m2 = load_neighbors(file_name2)

    # Count number of 1-neighbor which are calculated as the same by both methods
    hit = sum(map(lambda x, y: len(np.intersect1d(x.astype(int), y)), list(indices_m2), list(indices_m1)))


    # Compare: %  hit returned vs number of points
    ip = hit/indices_m2.size * 100

    # Show percentage of hit/miss on screen an save information on a log file
    print ("---- Case " + str(knn) + " nn within " + method1 + " and " + method2 + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Same neighbors rate: " + str(hit) + "/" + str(float(indices_m2.size)))
    print("Intersection percentage: " + str(ip) + "%\n\n")
    logging.info("---- Case " + str(knn) + " nn within " + method1 + " and " + method2 + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    logging.info("Same neighbors rate: " + str(hit) + "/" + str(float(indices_m2.size)))
    logging.info("Intersection percentage: " + str(ip) + "%\n\n")

    return ip

def get_recall(dataset, distance, method, k, baseline, algorithm=None, implementation=None, r=None, tg=None, nc=None):

    # Set logging info
    logging.basicConfig(filename='./benchmarks/logs/' + dataset + '/' + dataset + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_GDASC_" + str(algorithm) + "_" + str(implementation) +'_recall.log',
                        filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO, force=True)
    logging.info('------------------------------------------------------------------------')
    logging.info('                            %s Dataset RECALL', dataset)
    logging.info('------------------------------------------------------------------------')
    logging.info('Search of k nearest neighbors over a choosen dataset, using different methods, run_benchmarks.py')
    logging.info('------------------------------------------------------------------------\n')

    logging.info('Distance: %s ', distance)
    logging.info('Method: %s', method)
    logging.info('k: %s', k)
    logging.info('GDASC params: tg=%s, nc=%s, r=%s, algorithm=%s, implementation=%s\n', tg, nc, r, algorithm, implementation)

    # From a chosen dataset, calculate recall for each benchmarks  (k-dataset-distance-method combination)

    logging.info('-- %s method --\n', method)


    file_name_le = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(
        k) + "_" + distance + "_" + baseline + "_auto.hdf5"

    if method == 'GDASC':
        file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(
            k) + "_" + distance + "_" + method + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_" + str(algorithm) + "_" + str(implementation) + ".hdf5"

    elif method == 'Exact':
        file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(
            k) + "_" + distance + "_" + method + "_" + algorithm + ".hdf5"
    else:
        file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(
            k) + "_" + distance + "_" + method + ".hdf5"

    if not os.path.isfile(file_name):
        rec = np.nan
    else:
        rec = recall(dataset, distance, method, k, False, file_name_le, file_name)

    logging.shutdown()
    return rec

def get_recall_new(dataset, k, distance, indices, coords, distances):

    baseline_file = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(
            k) + "_" + distance + "_Exact_auto.hdf5"

    if not os.path.isfile(baseline_file):
        rec = np.nan
    else:

        indices_baseline, coords_baseline, dists_baseline = load_neighbors(baseline_file)

        # Count number of 1-neighbor which are the same as the point searched
        hit = sum(map(lambda x, y: len(np.intersect1d(x.astype(int), y)), list(indices), list(indices_baseline)))

        # Recall: %  hit returned vs number of points
        rec = hit / indices_baseline.size * 100

    return rec

def get_avgRecall(datasets, distances, methods, knn, gdasc_algorithm, gdasc_implementation, baseline):

    # Once we have obtained the recall for each experiment (each k-dataset-distance-method combination)
    recalls = get_recall(datasets, distances, methods, knn, gdasc_algorithm, gdasc_implementation, baseline)

    # Obtain mean Average Points for each dataset-distance-method combination
    avgRecall = recalls.groupby(['Dataset', 'Distance', 'Method'])['Recall'].mean().reset_index()
    #print("average Recall (avgRecall):\n\n " + str(avgRecall))
    print(avgRecall)

    return avgRecall
