import numpy as np
import matplotlib.pyplot as plt
import data.load_train_test_set as load_train_test_set
import seaborn as sns


#######################          Recall plots       ###########################
# Build a graph to show recall results
def print_recall_graph(dataset, distances, methods, k, recalls):

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), sharey=True)
    fig.subplots_adjust(top=0.75)

    for i in range(len(distances)):
        for j in range(len(methods)):
            axs[i].plot(k, recalls[i][j], label=methods[j], marker='o')
            axs[i].set_title(distances[i], pad=7)

    fig.legend(methods, loc='center right', title='Method')
    fig.suptitle(dataset + " dataset - Recall (%)", fontsize=20, y= 0.95)
    plt.ylim([0, 105])
    plt.show()

def print_recall_heatmap(datasets, distances, methods, k, recalls):

    plt.rcParams['figure.dpi'] = 300

    for dataset in datasets:
        re_ma = np.asarray(recalls.loc[recalls['Distance'] == "manhattan", 'Recall'].tolist())
        re_eu = np.asarray(recalls.loc[recalls['Distance'] == "euclidean", 'Recall'].tolist())
        re_ch = np.asarray(recalls.loc[recalls['Distance'] == "chebyshev", 'Recall'].tolist())
        re_co = np.asarray(recalls.loc[recalls['Distance'] == "cosine", 'Recall'].tolist())

        # setting the dimensions of the plot
        fig, ax = plt.subplots(figsize=(16, 10.5))

        # Create a mask to hide null (np.nan) values from heatmap
        mask = [re_ma, re_eu, re_ch, re_co] == np.nan

        # Heatmap
        # h = sns.heatmap([re_ma, re_eu, re_ch, re_co], annot=True, annot_kws={"size": 20}, fmt='.3g', yticklabels=distances, xticklabels=k+k+k, cmap="icefire", mask=mask, vmin=0, vmax=100)
        h = sns.heatmap([re_ma, re_eu, re_ch, re_co], annot=True, annot_kws={"size": 20}, fmt='.3g', yticklabels=distances,
                        xticklabels=k + k + k, cmap="Oranges", mask=mask, vmin=0, vmax=100)

        # Colorbar
        h.collections[0].colorbar.set_label('Recall (%)', labelpad=30, fontsize=25)
        h.collections[0].colorbar.ax.tick_params(labelsize=20)

        # Title
        if dataset == "municipios":
            dataset = "municipalities"
        h.axes.set_title(str(dataset + " dataset"), fontsize=30, pad=35)

        # Axis x and y (knn and distance)
        h.set_xlabel("k-nearest neighbors", fontsize=25, labelpad=30)
        h.set_ylabel("Distance", fontsize=25, labelpad=40)
        h.tick_params(axis='both', which='major', labelsize=20)

        # Axis twin (method)
        hb = h.twiny()
        hb.set_xticks(range(len(methods)))
        # hb.set(xticklabels=methods)
        hb.set_xticklabels(methods, ha='center')
        hb.set_aspect(aspect=0.75)
        hb.set_xlabel("Method", fontsize=25, labelpad=30)
        hb.tick_params(axis='both', which='major', labelsize=20)

        # Show heatmap
        plt.show()

# Build a graph to compare recall results
def print_compare_recall_graph(recalls):

    #recalls = recalls[recalls['k'] == 10] #to keep only knn=10 benchmarks

    # Adding three columns (dataset n and d) to dataframe in order to get some statistics

    recalls.insert(loc=len(recalls.columns), column='n', value=0)
    recalls.insert(loc=len(recalls.columns), column='d', value=0)
    recalls.insert(loc=len(recalls.columns), column='n/d', value=0)

    datasets = recalls['Dataset'].unique()

    for dataset in datasets:
        # Regarding the dataset name, set the file name to load the train and test set
        file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
        train_set, test_set = load_train_test_set.load_train_test_h5py(file_name)
        size = train_set.shape[0] + test_set.shape[0]
        dim = train_set.shape[1]
        recalls.loc[recalls['Dataset'] == dataset, 'n'] = size
        recalls.loc[recalls['Dataset'] == dataset, 'd'] = dim
        recalls.loc[recalls['Dataset'] == dataset, 'n/d'] = size/dim

    # Compare recalls by n and dim

    # Keep only columns refering dataset features & drop duplicates
    dataset_info = recalls.loc[:, ['Dataset', 'n', 'd', 'n/d']].drop_duplicates()


    fig, axes = plt.subplots(1, 3, figsize=(35, 15), sharey=True)

    sns.lineplot(data=recalls, x="n", y="Recall", hue='Method', ax=axes[0], palette=['lightblue', 'orange', 'r'])
    sns.lineplot(data=recalls, x="d", y="Recall", hue='Method', ax=axes[1], palette=['lightblue', 'orange', 'r'])
    sns.lineplot(data=recalls, x="n/d", y="Recall", hue='Method', ax=axes[2], palette=['lightblue', 'orange', 'r'])

    #for index, row in dataset_info.iterrows():
    #    axes[0].annotate(text=row['Dataset'], xy=(row['n'], 20), ha='center')
    #    axes[1].annotate(text=row['Dataset'], xy=(row['d'], 20), ha='center')

    axes[0].set_title("Recall regarding dataset's size", fontsize=25)
    axes[1].set_title("Recall regarding dataset's dimensionality", fontsize=25)
    axes[2].set_title("Recall regarding dataset's ratio size-dimensionality", fontsize=25)


    '''
    # Compare recalls by distance
    fig, axes = plt.subplots(1, 3, figsize=(23, 5), sharey=True)


    for i, each in enumerate(distances):
        sns.lineplot(data=recalls.loc[recalls['Distance'] == each], ax=axes[i], x='n', y='Recall', hue='Method')

    '''

    plt.show()


# Build a boxplots graph to compare recall results
def print_compare_recall_boxplots(recalls):

    #recalls = recalls[recalls['k'] == 10] #to keep only knn=10 benchmarks

    # Adding two columns (dataset n and d) to dataframe in order to get some statistics

    recalls.insert(loc=len(recalls.columns), column='n', value=0)
    recalls.insert(loc=len(recalls.columns), column='d', value=0)

    # Get only datasets name for the graph
    datasets = recalls['Dataset'].unique()

    for dataset in datasets:
        # Regarding the dataset name, set the file name to load the train and test set
        file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"
        train_set, test_set = load_train_test_set.load_train_test_h5py(file_name)
        size = train_set.shape[0] + test_set.shape[0]
        dim = train_set.shape[1]
        recalls.loc[recalls['Dataset'] == dataset, 'n'] = size
        recalls.loc[recalls['Dataset'] == dataset, 'd'] = dim

    # Compare recalls by n and dim

    # Keep only columns refering dataset features & drop duplicates
    dataset_info = recalls.loc[:, ['Dataset', 'n', 'd']].drop_duplicates()


    fig, axes = plt.subplots(1, 2, figsize=(25, 15), sharey=True)

    sns.boxplot(data=recalls, x="n", y="Recall", hue='Method', ax=axes[0], palette=['lightblue', 'orange', 'r'])
    sns.swarmplot(data=recalls, x="d", y="Recall", hue='Method', ax=axes[1], palette=['lightblue', 'orange', 'r'])

    #for index, row in dataset_info.iterrows():
    #    axes[0].annotate(text=row['Dataset'], xy=(row['n'], 20), ha='center')
    #    axes[1].annotate(text=row['Dataset'], xy=(row['d'], 20), ha='center')

    axes[0].set_title("Recall regarding dataset's size", fontsize=35)
    axes[1].set_title("Recall regarding dataset's dimensionality", fontsize=35)

    plt.show()



#######################          Recall plots       ###########################

# Build a barplot to show average Recall results for each dataset provided
def print_avgRecall_barplot(datasets, distances, methods, avgRecall):

    plt.rcParams['figure.dpi'] = 300

    for dataset in datasets:

        # Bar plot
        fig, axes = plt.subplots(1, len(distances), figsize=(25, 15), sharey=True)
        axes[0].set_ylabel('Recall', fontsize=30)

        for i in range(0, len(distances)):
            distance_subset = avgRecall[avgRecall['Distance'] == distances[i]]
            ax=sns.barplot(data=distance_subset, x="Method", y="Recall", ax=axes[i], order=methods, palette=['lightblue', 'orange', 'r'],)

            # Set title and x legend for each subplot
            axes[i].set_title(distances[i], fontsize=40)
            axes[i].set_xlabel('Method', fontsize=30)
            axes[i].tick_params(axis='both', which='major', labelsize=25)

        # Set title and y legend for all subplots
        plt.suptitle(dataset, fontsize= 50,  ha='center')
        ax.set_ylim(0,100)
        ax.set_ylabel('Average Recall')

        # Show graph
        plt.show()


# Build a pointplot to show average Recall results for each dataset provided
def print_avgRecall_pointplot(datasets, distances, methods, avgRecall):

    # Set resolution of the figure
    plt.rcParams['figure.dpi'] = 300

    # Replace PYNN denomination by PyNN
    avgRecall = avgRecall.replace("PYNN", "PyNN")
    methods = list(map(lambda x: x.replace("PYNN", "PyNN"), methods))

    # Print one graph for each dataset
    for dataset in datasets:

        # Get the avgRecall for the dataset
        dataset_avgRecall = avgRecall.loc[avgRecall['Dataset'] == dataset]

        # Point plot
        # lineas gruesas y PDASC discontinuo
        sns.set_context("paper", font_scale=1.3)
        ax = sns.pointplot(data=dataset_avgRecall, x="Distance", y="Recall",hue="Method", hue_order= methods, order=distances, palette=['#58a5e0', '#fca92d', '#b970e0'], markersize=4.5, linestyles=["-", "-", "--"])
        ax.margins(x=0.07)
        sns.move_legend(ax, "lower right", )
        plt.setp(ax.get_legend().get_title(), fontsize='16')
        plt.setp(ax.get_legend().get_texts(), fontsize='13')
        ''' Title
        # Set title and y-axis limits (0-100)
        if dataset == "municipios":
            plt.suptitle("municipalities", fontsize=25, ha='center')
        else:
            plt.suptitle(dataset, fontsize=25, ha='center')
        '''

        ax.set_ylim(0,105)
        ax.set_ylabel('Average Recall')

        # Show graph and save it into a file
        plt.savefig("./benchmarks/figures/fig_" + dataset + "_avgRecall.png")
        plt.show()

        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()


def print_Recall_pointplot(dataset, results_dataset):

    # Drop from the df the entrys whose methods are Exact & replace PYNN by PyNN
    results_dataset = results_dataset[results_dataset['Method'] != 'Exact'].replace("PYNN", "PyNN").replace("PDASC", "PDASC")

    # Set resolution of the figure
    plt.rcParams['figure.dpi'] = 300

    distances=results_dataset['Distance'].unique()

    # reorder distances, putting manhattan, euclidean, chebyshev, cosine, and, if present, harversine
    distances = ['manhattan', 'euclidean', 'chebyshev', 'cosine'] if 'haversine' not in distances else ['manhattan', 'euclidean', 'chebyshev', 'cosine', 'haversine']
    methods = ['FLANN', 'PyNN', 'PDASC']

    # Point plot
    # lineas gruesas y PDASC discontinuo
    sns.set_context("paper", font_scale=1.3)
    ax = sns.pointplot(data=results_dataset, x="Distance", y="Recall(Av)", hue="Method", hue_order=methods,
                       order=distances, palette=['#58a5e0', '#fca92d', '#b970e0'], markersize=4.5,
                       linestyles=["-", "-", "--"])

    ax.margins(x=0.07)
    sns.move_legend(ax, "lower right", )
    plt.setp(ax.get_legend().get_title(), fontsize='16')
    plt.setp(ax.get_legend().get_texts(), fontsize='13')
    ''' Title
    # Set title and y-axis limits (0-100)
    if dataset == "municipios":
        plt.suptitle("municipalities", fontsize=25, ha='center')
    else:
        plt.suptitle(dataset, fontsize=25, ha='center')
    '''

    ax.set_ylim(0, 105)
    ax.set_ylabel('Recall')

    # Show graph and save it into a file
    plt.savefig("./benchmarks/figures/fig_" + dataset + "_Recall.png")
    #plt.show()

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

#######################          Execution Time plots       ###########################

# Build a graph to show execution time results
def print_extime_plot(dataset, distances, methods, k, ex_times):
    fig, axs = plt.subplots(2, 3, figsize=(9, 4), sharey=True)
    fig.subplots_adjust(top=0.75)

    for i in range(len(distances)):
        for j in range(len(methods)):
            axs[0][i].set_title(distances[i], pad=7)
            axs[0][0].set_ylabel('Indexation time')
            axs[1][0].set_ylabel('Search time')
            for z in range(2):
                aux_extimes = np.transpose(ex_times[i][j])
                axs[z][i].plot(k, aux_extimes[z], label=methods[j], marker='o')

    fig.legend(methods, loc='center right', title='Method')
    fig.suptitle(dataset + " dataset - Execution time avevarage (s)", fontsize=20, y=0.95)
    plt.show()


