from benchmarks.plotting.draw_benchmark_plots import print_avgRecall_pointplot
import benchmarks.plotting.performance_utils as benchmark_metrics


if __name__ == '__main__':

    # Set var for benchmarks:
    # datasets = ['municipios', 'MNIST', 'NYtimes', 'GLOVE']
    datasets = ['wdbc']
    # distances = ['manhattan', 'euclidean', 'chebyshev', 'cosine']
    distances = ['euclidean']
    methods = ['GDASC']
    # methods = ['FLANN', 'PYNN', 'GDASC']
    baseline = 'Exact'
    knn = [5]
    # knn = [5, 10, 15]

    mask_algorithm = 'kmeans'
    mask_implementation = 'kclust'

    gdasc_algorithm = 'kmedoids'
    gdasc_implementation = 'fastkmedoids'

    # Show results on a graph

    # Obtain recall of each (k=5,10,15) experiment
    #recalls = benchmark_metrics.get_recall(datasets, distances, methods, knn, gdasc_algorithm, gdasc_implementation, baseline)

    # Print recall graph
    # print_compare_recall_graph(recalls)
    # print_compare_recall_boxplots(recalls)
    # print_recall_heatmap(datasets, distances, methods, knn, recalls)

    # Obtain average Recall of all (k=5,10,15) experiments
    avgRecall = benchmark_metrics.get_avgRecall(datasets, distances, methods, knn, gdasc_algorithm, gdasc_implementation, baseline)

    # Print avgRecall graph
    # print_avgRecall_barplot(datasets, distances, methods, avgRecall) #barplot
    #print_avgRecall_pointplot(datasets, distances, methods, avgRecall) #pointplot

exit(0)
