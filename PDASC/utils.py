# coding=utf-8
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import distance
import math as m


def busca_dist_menor(distance_matrix):
    """
    Find the column index of the smallest element in the first row of a distance matrix, excluding the first element.

    Parameters:
    distance_matrix : numpy.ndarray
      A 2D distance matrix where the first row contains distances to be evaluated.

    Returns:
    int: The column index of the smallest element in the first row, excluding the first element.
    """
    # Find the minimum element in the first row, excluding the first element
    min_distance = distance_matrix[0, 1:].min()
    num_rows, num_columns = distance_matrix.shape
    found = False
    column_index = 1
    # Iterate through the columns to find the index of the minimum element
    while column_index < num_columns and not found:
        if min_distance == distance_matrix[0, column_index]:
            found = True
        else:
            column_index += 1
    return column_index


def argmin_diagonal_ignored(matrix):
    """
      Find the indices of the minimum element in a 2D matrix, ignoring the diagonal elements.

      Parameters:
      matrix (numpy.ndarray): A 2D numpy array (matrix).

      Returns:
      tuple: A tuple (row_index, column_index) of the minimum element in the matrix, ignoring the diagonal.
      """
    # Create a mask to ignore the diagonal elements
    mask = np.ones(matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)

    # Find the minimum element in the matrix, ignoring the diagonal
    min_element = matrix[mask].min()
    num_rows, num_columns = matrix.shape
    found = False
    row_index = 0

    # Iterate through the matrix to find the indices of the minimum element
    while row_index < num_rows and not found:
        column_index = row_index + 1
        while column_index < num_columns and not found:
            if min_element == matrix[row_index, column_index]:
                found = True
            column_index += 1
        row_index += 1
    return row_index - 1, column_index - 1


def obten_num_ptos(points_list, num_centroids):
    """
    Calculate the total number of points based on a list of points and the number of centroids.

    This function sums the number of points in `points_list`, adding `ncentroides` if an element
    exceeds `num_centroids`, otherwise adding the element itself.

    Parameters:
    points_list  : list of int
           A list containing the number of points.
    num_centroids : int
           The number of centroids.

    Returns:
    int: The total number of points.
    """
    total_points = 0
    for points in points_list:
        if points > num_centroids:
            total_points += num_centroids
        else:
            total_points += points
    return total_points


def divide_en_grupos(array, total_length, num_groups, group_size):
    """
    Divide an  array into groups of a specified size.

   Parameters:
    array : numpy array
        The array to be divided.
    total_length : int
        The length of the array.
    num_groups : int
        The number of groups to divide the array into.
    group_size : int
        The size of each group.


    Returns:
    numpy array: A 2D array where each sub-array represents a group.
    """
    if total_length == num_groups * group_size:
        # Exact division: Split the array into evenly sized groups
        array = np.array(np.split(array, num_groups))
    else:
        # Inexact division: Handle the remainder
        main_part = array[:group_size * (num_groups - 1)]
        remainder = array[group_size * (num_groups - 1):]

        # Split the main part into groups and add the remainder as the last group
        groups = np.split(main_part, num_groups - 1)
        groups.append(remainder)

        # Convert the list of groups to a numpy array
        array = np.array(groups)

    return array


def obten_idgrupo(point_id, group_sizes):
    """
    Determine the group ID to which a given point ID belongs.

    Parameters:
    point_id : int
        The ID of the point to be checked.
    group_sizes : list of int
        A list where each element represents the size of a group.

    Returns:
    int: The group ID to which the given point belongs.
    """
    num_groups = len(group_sizes)  # Total number of groups
    group_id = 0
    group_start = 0  # Start index of the current group
    group_end = group_sizes[0]  # End index of the current group
    found = False  # Flag to indicate if the group is found
    # Loop through the groups to find the one that contains the point ID
    while (group_id  < num_groups) and not (found):
        if point_id >= group_start and point_id < group_end:
            found = True  # The point is within the current group range
        else:
            group_id += 1  # Move to the next group
            group_start = group_end  # Update the start index to the end of the current group
            group_end = group_end + group_sizes[group_id]  # Update the end index to the new group's end

    return group_id


def myFunc(element):
    return len(element)


def busca_candidato(distance_matrix, group_index):
    """
    Find the index of the closest candidate in the distance matrix for a given group.

    This function searches for the minimum distance value in a specific row of the distance matrix
    `distance_matrix`, corresponding to the given group `group_id`. The row is sorted to find the smallest non-zero
    distance, which is assumed to be the closest candidate.

    Parameters:
    distance_matrix       : numpy array
        A 2D array where `distance_matrix[i, j]` represents the distance between the `i`-th and `j`-th points.
    group_index : int
        The index of the row in the distance matrix to search.

    Returns:
    int: The index of the closest candidate in the specified row.
    """

    # Convert the row of the distance matrix for the specified group to a list
    row = distance_matrix[group_index, :].tolist()
    # Sort the list to find the smallest value more easily
    sorted_row = sorted(row)
    # The minimum value excluding the first element (which is always zero or not relevant)
    minimum_distance = sorted_row[1]

    # Find the index of this minimum value in the original row
    closest_index = row.index(minimum_distance)
    return closest_index


def hay_grupos_peq(groups, size_threshold):
    """
    return all(len(group) < tam_grupo for group in vector)
    Check if all groups in a list of groups are smaller than a specified size.

    Parameters:
    groups : list of lists
        A list where each element is a group (which is itself a list).
    size_threshold : int
        The size threshold to check against.

    Returns:
    bool: True if all groups in `vector` are smaller than `tam_grupo`, otherwise False.
    """
    all_smaller = False
    count = 0
    for group in groups:
        if (len(group) < size_threshold):
            count += 1

    if count == len(groups):
        all_smaller = True

    return all_smaller


def funcdist(point, points_vector, dimensions):
    """
    Calculate the Euclidean distance between a given point and each point in a vector.

    Parameters:
    point : list of list of floats
        A 2D list representing a single point with dimensions.
    points_vector : list of list of floats
        A list of points where each point is a list of coordinates.
    dimensions : int
       The number of dimensions for each point.

    Returns:
    numpy.ndarray: A 1D array containing the Euclidean distances from the given point to each point in the vector.
    """
    distances = np.empty(len(points_vector), float)
    for i in range(len(points_vector)):
        sum_of_squares= 0.0
        for n in range(dimensions):
            coordinate_difference = (point[0][n] -points_vector[i][n]) * (point[0][n] - points_vector[i][n])
            sum_of_squares += coordinate_difference
        distances[i] = m.sqrt(sum_of_squares)
    return distances

def get_distances(point1, point2, metric):
    """
    Calculate the distance between two points using the specified metric.

    Parameters:
    point1 : array-like
        The first point.
    point2 : array-like
        The second point.
    metric : str
        The distance metric to use.

    Returns:
    float: The calculated distance.
    """
    if metric == 'haversine':
        return pairwise_distances(point1, point2, metric=metric)[0]
    elif metric == 'manhattan':
        return distance.cdist(point1, point2, metric='cityblock')[0]
    else:
        return distance.cdist(point1, point2, metric=metric)[0]



def calculate_numcapas(total_points, group_size, num_centroids):
    """
    Calculate the number of layers (num_layers) required for a hierarchical clustering process.

    Parameters:
    total_points : int
            The total number of points.
    group_size : int
            The size of each group.
    num_centroids : int
            The number of centroids per group.

    Returns:
    int: The number of layers (num_layers) required.
    """
    # If the total points are less than the group size or group size is equal to the number of centroids
    if total_points < group_size or group_size == num_centroids:
        num_layers = 1
    else:
        # Initial calculation for the first layer
        quotient = total_points // group_size
        remainder = total_points % group_size
        num_groups = quotient + (1 if remainder > 0 else 0)
        new_points = num_groups * num_centroids
        num_layers = 1

        # Iteratively calculate the number of layers until new_points is less than or equal to num_centroids
        while new_points > num_centroids:
            quotient = new_points // group_size
            remainder = new_points % group_size
            num_groups = quotient + (1 if remainder > 0 else 0)
            new_points = num_groups * num_centroids

            # Increase the number of layers if new_points is still greater than or equal to num_centroids
            if new_points >= num_centroids:
                num_layers += 1

    return num_layers


def built_estructuras_capa(total_points, group_size, num_centroids, num_layers, dimensions):
    """
    Construct hierarchical structures for multiple layers of data points and centroids.

    Parameters:
    total_points : int
        The total number of points.
    group_size : int
        The size of each group.
    num_centroids : int
        The number of centroids per group.
    num_layers : int
        The number of layers.
    dimensions : int
        The dimensionality of the data points.

    Returns:
    tuple: A tuple containing three numpy arrays:
        - points_per_layer: Array of data points for each layer.
        - labels_per_layer: Array of labels for each layer.
        - groups_per_layer: Array of group sizes for each layer.
    """
    labels_per_layer = np.empty(num_layers, object)
    points_per_layer = np.empty(num_layers, object)
    groups_per_layer = np.empty(num_layers, object)

    # Calculate the initial number of groups and the remainder
    num_groups = int(total_points / group_size)
    rest = total_points % group_size

    for capa in range(num_layers):

        if rest != 0:
            # If there is a remainder, increase the number of groups by one
            num_groups = num_groups + 1
            group_labels = np.empty(num_groups, object)
            for num in range(num_groups - 1):
                group_labels[num] = np.zeros(group_size, dtype=int)
            group_labels[num_groups - 1] = np.zeros(rest, dtype=int)
            labels_per_layer[capa] = group_labels
            if (rest >= num_centroids):
                group_points = np.zeros((num_groups, num_centroids, dimensions), dtype=float)
                new_rest = (num_groups * num_centroids) % group_size
                new_num_groups= int((num_groups * num_centroids) / group_size)
            else:
                group_points = np.empty(num_groups, object)
                for num in range(num_groups - 1):
                    group_points[num] = np.zeros((num_groups - 1, num_centroids, dimensions))
                group_points[num_groups - 1] = np.zeros((1, rest, dimensions))
                new_rest = ((num_groups - 1) * num_centroids + rest) % group_size
                new_num_groups = int(((num_groups - 1) * num_centroids + rest) / group_size)
            points_per_layer[capa] = group_points
            groups_per_layer[capa] = np.zeros(num_groups, dtype=int)
        else:
            points_per_layer[capa] = np.zeros((num_groups, num_centroids, dimensions), dtype=object)
            labels_per_layer[capa] = np.zeros((num_groups, group_size), dtype=int)
            groups_per_layer[capa] = np.zeros(num_groups, dtype=int)
            new_rest = (num_groups * num_centroids) % group_size
            new_num_groups = int((num_groups * num_centroids) / group_size)
        # Update the remainder and number of groups for the next layer
        rest = new_rest
        num_groups = new_num_groups

    return points_per_layer, labels_per_layer, groups_per_layer


def built_estructuras_capa_new(total_points, group_size, num_centroids, num_layers, dimensions):
    """
    Construct hierarchical structures for multiple layers of data points and centroids.

    Parameters:
    total_points : int
        The total number of points.
    group_size : int
        The size of each group.
    num_centroids : int
        The number of centroids per group.
    num_layers : int
        The number of layers.
    dimensions : int
        The dimensionality of the data points.

    Returns:
    tuple: A tuple containing three numpy arrays:
        - points_per_layer: Array of data points for each layer.
        - labels_per_layer: Array of labels for each layer.
        - groups_per_layer: Array of group sizes for each layer.
    """
    labels_per_layer = np.empty(num_layers, object)
    points_per_layer = np.empty(num_layers, object)
    groups_per_layer = np.empty(num_layers, object)


    # Calculate the initial number of groups and the remainder
    num_groups = int(total_points / group_size)
    rest = total_points % group_size

    for capa in range(num_layers):

        if rest != 0:
            # If there is a remainder, increase the number of groups by one
            num_groups = num_groups + 1
            group_labels = np.empty(num_groups, object)
            for num in range(num_groups - 1):
                group_labels[num] = np.zeros(group_size, dtype=int)
            group_labels[num_groups - 1] = np.zeros(rest, dtype=int)
            labels_per_layer[capa] = group_labels
            if (rest >= num_centroids):
                group_points = np.zeros((num_groups, num_centroids, dimensions), dtype=float)
                new_rest = (num_groups * num_centroids) % group_size
                new_num_groups = int((num_groups * num_centroids) / group_size)
            else:
                group_points = np.empty(num_groups, object)
                for num in range(num_groups - 1):
                    group_points[num] = np.zeros((num_groups - 1, num_centroids, dimensions))
                group_points[num_groups - 1] = np.zeros((1, rest, dimensions))
                new_rest = ((num_groups - 1) * num_centroids + rest) % group_size
                new_num_groups = int(((num_groups - 1) * num_centroids + rest) / group_size)
            points_per_layer[capa] = group_points
            groups_per_layer[capa] = np.zeros(num_groups, dtype=int)
        else:
            points_per_layer[capa] = np.zeros((num_groups, num_centroids, dimensions), dtype=float)
            labels_per_layer[capa] = np.zeros((num_groups, group_size), dtype=int)
            groups_per_layer[capa] = np.zeros(num_groups, dtype=int)
            new_rest = (num_groups * num_centroids) % group_size
            new_num_groups = int((num_groups * num_centroids) / group_size)
        # Update the remainder and number of groups for the next layer
        rest = new_rest
        num_groups = new_num_groups

    promoted_points =  np.empty(groups_per_layer[0].shape, object)

    return points_per_layer, labels_per_layer, groups_per_layer, promoted_points


def built_lista_pos(group_id, compressed_group_sizes, position_list):
    """
    Calculate the position in a compressed group structure.

    This function computes the position in a compressed group structure by adding the
    displacement of previous groups to the provided position.

    Parameters:
    group_id : int
        The index of the group for which the position is being calculated.
    compressed_group_sizes : list of int
        A list containing the sizes of each group in the compressed structure.
    position_list : int
        The position within the current group.

    Returns:
    int: The computed position in the overall compressed group structure.
    """
    displacement = 0
    # Calculate the total displacement by summing the sizes of all groups before the current group
    for id in range(group_id):
        displacement += compressed_group_sizes[id]
    # Add the displacement to the provided position
    absolute_position = position_list + displacement
    return absolute_position


def search_near_centroid(group_id, centroid_id, last_neighbor_id, examined_centroids, points_per_layer, labels_per_layer,
                         groups_per_layer, neighbor_ids, original_points, metric):
    """
    Search for the nearest point to a centroid within a group and update examination status.

    This function finds the nearest point to a specified centroid within a group, considering points
    that have not been previously examined. It updates the status of centroids and points as necessary.

    Parameters:
    group_id : int
        The index of the group being examined.
    centroid_id : int
        The index of the centroid within the group.
    last_neighbor_id : int
        The index of the last neighbor point considered.
    examined_centroids : np.ndarray
        A 2D array indicating whether centroids have been examined (1) or not (0).
    points_per_layer : np.ndarray
        A 2D array containing points in the current layer.
    labels_per_layer: np.ndarray
        A 2D array containing labels for points in the current layer.
    groups_per_layer : np.ndarray
        A 2D array containing the sizes of groups in the current layer.
    neighbor_ids : np.ndarray
        An array of indices of neighbor points already considered.
    original_points : np.ndarray
        The original set of points.
    metric: str
        The metric to use for distance calculation.

    Returns:
    tuple: A tuple containing the index of the nearest point, the distance to the nearest point,
           the group index, and the updated centroid index.
    """
    distances = pairwise_distances(points_per_layer[0][group_id], metric=metric)
    second_smallest_distance = np.partition(distances[centroid_id], 1)[1]
    nearest_centroid_id = (np.argwhere(distances[centroid_id] == second_smallest_distance)).ravel()
    # Check if the nearest centroid has been examined
    if examined_centroids[group_id][nearest_centroid_id] == 0:
        position_list = np.argwhere(labels_per_layer[0][group_id][:] == nearest_centroid_id)
        position_list= built_lista_pos(group_id, groups_per_layer[0][:],  position_list)
        position_list= position_list.ravel()

        new_position_list = np.setdiff1d(position_list, neighbor_ids)
        if len(new_position_list) >= 1:
            selected_points = np.array(original_points[new_position_list])
            saved_neighbor = (np.array(original_points[last_neighbor_id])).reshape(1, 2)
            distance_points = np.concatenate([saved_neighbor, selected_points])
            distances = pairwise_distances(distance_points, metric=metric)
            closest_column = busca_dist_menor(distances)
            point_id = new_position_list[closest_column - 1]
            distance = distances[0, closest_column]

            if len(new_position_list) == 1:

                examined_centroids[group_id][nearest_centroid_id] = 1
        else:

            next_best = 2
            exit_loop = False
            while (examined_centroids[group_id][nearest_centroid_id] == 1) and (next_best < len(distances[centroid_id])) \
                    and (not exit_loop):
                second_smallest_distance = np.partition(distances[centroid_id], next_best)[next_best]
                nearest_centroid_id = (np.argwhere(distances[centroid_id] == second_smallest_distance)).ravel()

                position_list = np.argwhere(labels_per_layer[0][group_id][:] == nearest_centroid_id)
                position_list = built_lista_pos(group_id, groups_per_layer[0][:], position_list)
                position_list = position_list.ravel()

                new_position_list = np.setdiff1d(position_list, neighbor_ids)
                if len(new_position_list) >= 1:
                    selected_points = np.array(original_points[new_position_list])
                    saved_neighbor = (np.array(original_points[last_neighbor_id])).reshape(1, 2)
                    distance_points = np.concatenate([saved_neighbor, selected_points])
                    distances= pairwise_distances(distance_points, metric=metric)
                    closest_column = busca_dist_menor(distances)
                    point_id = new_position_list[closest_column - 1]
                    distance = distances[0, closest_column]
                    exit_loop = True
                    if len(new_position_list) == 1:

                        examined_centroids[group_id][nearest_centroid_id] = 1
                else:
                    next_best += 1

    else:
        # Si el centroide ha sido examinado, tengo que buscar el siguiente más cercano
        next_best = 2
        max_length = len(distances[centroid_id])
        exit_loop = False
        while (examined_centroids[group_id][nearest_centroid_id] == 1) and (next_best < max_length) \
                and (not exit_loop):
            second_smallest_distance = np.partition(distances[centroid_id], next_best)[next_best]
            nearest_centroid_id = (np.argwhere(distances[centroid_id] == second_smallest_distance)).ravel()

            position_list = np.argwhere(labels_per_layer[0][group_id][:] == nearest_centroid_id)
            position_list = built_lista_pos(group_id, groups_per_layer[0][:], position_list)
            position_list = position_list.ravel()

            new_position_list = np.setdiff1d(position_list, neighbor_ids)
            if len(new_position_list) >= 1:
                selected_points = np.array(original_points[new_position_list])
                saved_neighbor = (np.array(original_points[last_neighbor_id])).reshape(1, 2)
                distance_points = np.concatenate([saved_neighbor, selected_points])
                distances = pairwise_distances(distance_points, metric=metric)
                closest_column = busca_dist_menor(distances)
                point_id = new_position_list[closest_column - 1]
                distance= distances[0, closest_column]
                exit_loop = True
                if len(new_position_list) == 1:
                    examined_centroids[group_id][nearest_centroid_id] = 1
            else:
                next_best += 1

    return point_id, distance, group_id, nearest_centroid_id


def find_centroid_group(inheritage, grupos_capa, n_centroids, subgroup):
    """
    Determines the group of a centroid in a hierarchical tree structure.

    Parameters:
    inheritage : list
        List of ancestor groups leading to the current group.
    grupos_capa : list
        Number of points in each group at each layer.
    n_centroids : int
        Number of centroids in each group.
    subgroup : int
        Subgroup index within the current group.

    Returns:
    int
        The group index of the centroid.
    """
    # Number of branches depends on the number of centroids and the number of points in the group
    n_branches = grupos_capa[-1][0] // n_centroids

    # The group of the centroid depends on the number of branches, the current layer and the subgroup (0 or 1) index
    return inheritage[-1] * n_branches + subgroup




def get_dynamic_radius_list(n_layers, initial_radius, dataset):
    """
    Generate a list that allows to assign different radius values to be used at each layer.

    This function provides three alternatives for adjusting the radius values:
    1. Static radius values based on the 10th-nearest neighbor CDF percentiles.
    2. Dynamic radius values reduced over layers based on a heuristic.
    3. Dynamic radius values equally reduced over layers based on percentile values of the 10th-nearest neighbor CDF distribution.

    Parameters:
    n_layers (int): The number of layers.
    initial_radius (float): The initial radius value.
    dataset (str): The name of the dataset. Supported datasets are "wdbc", "municipios", "NYtimes", "MNIST", and "GLOVE".

    Returns:
    numpy.ndarray: A list of radius values for each layer.
    """

    # 1st Radius Adjustment Alternative: Statical Values based on 10th-nn CDF percentiles
    # Equivalent to the legacy implementation (static radius): just assign the same radius to all the layers
    dynamic_radius_list = [np.round(initial_radius, 2)] * n_layers

    # 2nd Radius Adjustment Alternative: Dynamic radius which value is reduced over layers based on an heuristic
    # (Reducing the radius by a fixed percentage at each layer, a 20% in this case)
    # dynamic_radius_list = [np.round(initial_radius * (0.80 ** layer), 2) for layer in range(int(n_layers)-1, -1, -1)]

    # 3rd Radius Adjustment Alternative: Dynamic radius which value is equally reduced over layers
    # regarding a max and min percentile value of the 10th-nn CDF distribution (0.8 and 1 in this case)

    # Define the interval values regarding the dataset used:

    # The starting value for the radius interval will be the provided initial radius,
    # corresponding to the value for percentile 1 of the 10th-nn CDF
    start = initial_radius

    # The ending value for the radius interval will depend on the dataset used,
    # corresponding to the value for percentile 0.8 of their 10th-nn CDF

    if dataset == "wdbc":
        end = 119.0
    elif dataset == "municipios":
        end = 0.42
    elif dataset == "NYtimes":
        end = 1.29
    elif dataset == "MNIST":
        end = 2005.0
    elif dataset == "GLOVE":
        end = 6.7
    else:
        print("Dataset not found")
        return None

    # The number of partitions will correspond to the number of layers
    partitions = n_layers

    # Partition equally the interval between the start and end values in n_layers partitions
    #dynamic_radius_list = np.round(np.linspace(end, start, partitions), 2)

    # Print the values of the radius for each layer
    # print(f" radius values are = {dynamic_radius_list}")

    return dynamic_radius_list

def insert_candidate_neighbour(candidates_array, distance):
    """
    Insert a number into the array while maintaining sorted order.
    If the array is full, replace the largest element if the new number is smaller.
    """
    if len(candidates_array) < candidates_array.size:
        # Add the number and sort
        candidates_array.append(distance)
        candidates_array.array.sort()
    else:
        # Replace the largest number if the new number is smaller
        if distance < candidates_array[-1]:
            candidates_array[-1] = distance
            candidates_array.sort()

    return candidates_array


def explore_centroid_dynamicradius(punto_buscado, current_layer, inheritage, current_centroid_id, current_centroid_distance, coords_puntos_capas, puntos_capas, grupos_capa, promoted_points, n_centroides, metrica, neighbours, distances_computed, dynamic_radius_list):
    """
    Explores the hierarchical tree structure to find centroids within a given radius.

    This function is an optimised version of `explore_centroid` that avoids redundant distance computation in the
    case of centroids that have only one child prototype (which would be actually itself)

    In case that one prototype is associated to a sole prototype in the last layer (an actual point of the dataset)
    the after distance computation is avoided, returning the id of that point and alsa its distance to the query point

    This optimisation still conserves the possibility of use an adaptively radius at each layer


    Parameters:
    punto_buscado : numpy.ndarray
        The query point for which the nearest neighbors are to be found.
    current_layer : int
        The current layer in the hierarchical tree being explored.
    inheritage : list
        List of ancestor groups leading to the current group.
    current_centroid_id : int
        The ID of the current centroid being explored.
    current_centroid_distance : float
        The distance from the query point to the current centroid.
    coords_puntos_capas : list
        Coordinates of points at each layer.
    puntos_capas : list
        Cluster centroids at each layer.
    grupos_capa : list
        Number of points in each group at each layer.
    n_centroides : int
        Number of centroids in each group.
    metrica : str
        The distance metric to use for calculating distances.
    radio : float
        Search radius for the approximate search.
    neighbours : list
        List to store the nearest neighbors found during the search.
    distances_computed : list
        List to store the count of distance computations.
    dynamic_radius_list : list
        List containing the radius value to be applied at each layer.

    Returns:
    list
        Updated list of nearest neighbors.
    """

    # Calculate the group onto the layer which the centroid belongs to
    prototype_group = inheritage[-1]

    # Get the radius value to be used in this layer in order to the comulative kde function
    radius = dynamic_radius_list[current_layer-1]

    # Obtain the IDs of prototypes from the layer below
    id_prototypes_layer_down = puntos_capas[current_layer-1][prototype_group]

    # Obtain the prototypes of the layer below which are mapped by this prototype
    id_associated_prototypes_layer_down = np.where(id_prototypes_layer_down == current_centroid_id)[0]

    # Explore each associated prototype in the layer below and store it into a list
    associated_prototypes_layer_down = np.empty((len(id_associated_prototypes_layer_down), 4), dtype=object)

    for i in range(len(id_associated_prototypes_layer_down)):
        subgroup = id_associated_prototypes_layer_down[i] // n_centroides
        group = find_centroid_group(inheritage, grupos_capa, n_centroides, subgroup)
        id_associated_prototypes_layer_down[i] = id_associated_prototypes_layer_down[i] % n_centroides

        associated_prototypes_layer_down[i, 0] = id_associated_prototypes_layer_down[i]
        associated_prototypes_layer_down[i, 1] = group
        associated_prototypes_layer_down[i, 2] = None if current_layer-1 == 0 else coords_puntos_capas[current_layer-2][group][id_associated_prototypes_layer_down[i]]
        associated_prototypes_layer_down[i, 3] = current_centroid_distance if current_layer-1 == 0 else None

    # If the centroid explored is in the last layer of the index
    if current_layer == 1:

        # Lets take into account that at this point we do not restrict by radius, but explore all the points mapped by the current prototype
        for i in range(len(associated_prototypes_layer_down)):
            neighbour_id = n_centroides * associated_prototypes_layer_down[i, 1] + associated_prototypes_layer_down[i, 0]
            tam_grupo = grupos_capa[0][0]
            group_id = neighbour_id // tam_grupo

            if len(associated_prototypes_layer_down) == 1 or promoted_points[group_id][neighbour_id % tam_grupo]:
                #print(f'Neighbour found: {neighbour_id} and distance: {current_centroid_distance}')
                neighbours.append((neighbour_id, current_centroid_distance))

                # Print the radius value at this step
                # print(f'Current prototype distance: {current_centroid_distance}')
                # print(f'Radius value at this step: {radius}')

            else:
                neighbours.append(neighbour_id)

        return neighbours, distances_computed

    # If the prototype has only one associated prototype below
    if len(associated_prototypes_layer_down) == 1:

        if current_centroid_distance < radius:
            associated_prototypes_layer_down[0, 3] = current_centroid_distance
            explorable_prototypes = associated_prototypes_layer_down
        else:
            explorable_prototypes = []

    else:
        coordinates_bottomed_prototypes = np.vstack(associated_prototypes_layer_down[:, 2])

        for i in range(len(coordinates_bottomed_prototypes)):
            if np.isnan(coordinates_bottomed_prototypes[i]).any():
                associated_prototypes_layer_down[i, 3] = current_centroid_distance
            else:
                associated_prototypes_layer_down[i, 3] = get_distances(punto_buscado, coordinates_bottomed_prototypes[i].reshape(1, -1), metrica)[0]
                #associated_prototypes_layer_down[i, 3] = distance.pdist([punto_buscado[0], coordinates_bottomed_prototypes[i]], metric=metrica)[0]
                #associated_prototypes_layer_down[i, 3] = pairwise_distances(punto_buscado, coordinates_bottomed_prototypes[i].reshape(1, -1), metric=metrica)[0][0]
                #distances_computed.append(associated_prototypes_layer_down[i, 3])
                distances_computed += 1

        explorable_prototypes_indices = np.where(associated_prototypes_layer_down[:, 3] <= radius)[0]
        explorable_prototypes = associated_prototypes_layer_down[explorable_prototypes_indices]

    for i in range(len(explorable_prototypes)):
        centroid = explorable_prototypes[i]
        neighbours, distances_computed = explore_centroid_dynamicradius(punto_buscado, current_layer-1, inheritage + [centroid[1]], centroid[0], centroid[3], coords_puntos_capas, puntos_capas, grupos_capa, promoted_points, n_centroides, metrica, neighbours, distances_computed, dynamic_radius_list)

    return neighbours, distances_computed


def explore_centroid_pruning(vector_original, punto_buscado, current_layer, inheritage, current_centroid_id, current_centroid_distance, coords_puntos_capas, puntos_capas, grupos_capa, promoted_points, n_centroides, metrica, neighbours, distances_computed, candidate_neighbours):

    # Rather than identify all the points that are neighbors of the query point and, at the end of the index exploration,
    # obtain the distance for all of them, we will compute the distance for each point when reaching it
    # This alternative allows the traditional approach to work as it should but is more expensive in terms of time

    # Calculate the group onto the layer which the centroid belongs to
    prototype_group = inheritage[-1]

    # Obtain the IDs of prototypes from the layer below
    id_prototypes_layer_down = puntos_capas[current_layer-1][prototype_group]

    # Obtain the prototypes of the layer below which are mapped by this prototype
    id_associated_prototypes_layer_down = np.where(id_prototypes_layer_down == current_centroid_id)[0]

    # Explore each associated prototype in the layer below and store it into a list
    associated_prototypes_layer_down = np.empty((len(id_associated_prototypes_layer_down), 4), dtype=object)

    for i in range(len(id_associated_prototypes_layer_down)):
        subgroup = id_associated_prototypes_layer_down[i] // n_centroides
        group = find_centroid_group(inheritage, grupos_capa, n_centroides, subgroup)
        id_associated_prototypes_layer_down[i] = id_associated_prototypes_layer_down[i] % n_centroides

        associated_prototypes_layer_down[i, 0] = id_associated_prototypes_layer_down[i]
        associated_prototypes_layer_down[i, 1] = group
        associated_prototypes_layer_down[i, 2] = None if current_layer-1 == 0 else coords_puntos_capas[current_layer-2][group][id_associated_prototypes_layer_down[i]]
        associated_prototypes_layer_down[i, 3] = current_centroid_distance if current_layer-1 == 0 else None

    # If the centroid explored is in the last layer of the index
    if current_layer == 1:

        # Lets take into account that at this point we do not restrict by radius, but explore all the points mapped by the current prototype
        for i in range(len(associated_prototypes_layer_down)):
            neighbour_id = n_centroides * associated_prototypes_layer_down[i, 1] + associated_prototypes_layer_down[i, 0]
            tam_grupo = grupos_capa[0][0]
            group_id = neighbour_id // tam_grupo

            if len(associated_prototypes_layer_down) == 1 or promoted_points[group_id][neighbour_id % tam_grupo]:

                if current_centroid_distance <= candidate_neighbours[-1]:
                    neighbours.append((neighbour_id, current_centroid_distance))

                    # Print the radius value at this step
                    # print(f'Current prototype distance: {current_centroid_distance}')
                    #print(f'Radius value at this step: {radius}')
                    candidate_neighbours = insert_candidate_neighbour(candidate_neighbours, current_centroid_distance)
                    #print(candidate_neighbours)

            else:
                coords_neighbour = vector_original[neighbour_id]
                #print(punto_buscado)
                #print(coords_neighbour)
                neig_distance = get_distances(punto_buscado, coords_neighbour.reshape(1, -1), metrica)[0]
                #neig_distance = distance.pdist([punto_buscado[0], coords_neighbour], metric=metrica)[0]

                if neig_distance <= candidate_neighbours[-1]:
                    neighbours.append((neighbour_id, neig_distance))
                    candidate_neighbours = insert_candidate_neighbour(candidate_neighbours, neig_distance)

        # return neighbours, candidates
        return neighbours

    # If the prototype has only one associated prototype below
    if len(associated_prototypes_layer_down) == 1:
        associated_prototypes_layer_down[0, 3] = current_centroid_distance

        if current_centroid_distance < candidate_neighbours[-1]:
            associated_prototypes_layer_down[0, 3] = current_centroid_distance
            explorable_prototypes = associated_prototypes_layer_down
        else:
            explorable_prototypes = []

    else:
        coordinates_bottomed_prototypes = np.vstack(associated_prototypes_layer_down[:, 2])
        for i in range(len(coordinates_bottomed_prototypes)):
            if np.isnan(coordinates_bottomed_prototypes[i]).any():
                associated_prototypes_layer_down[i, 3] = current_centroid_distance
            else:
                associated_prototypes_layer_down[i, 3] = get_distances(punto_buscado, coordinates_bottomed_prototypes[i].reshape(1, -1), metrica)[0]
                # associated_prototypes_layer_down[i, 3] = distance.pdist([punto_buscado[0], coordinates_bottomed_prototypes[i]], metric=metrica)[0]
                distances_computed.append(associated_prototypes_layer_down[i, 3])

                #print(f'Radius value at this step: {radius}')
                #print(f'Distance to the child prototype: {associated_prototypes_layer_down[i, 3]}')
                #candidate_neighbours = insert_candidate_neighbour(candidate_neighbours, associated_prototypes_layer_down[i, 3])
                #print(candidate_neighbours)

        explorable_prototypes_indices = np.where(associated_prototypes_layer_down[:, 3] <= candidate_neighbours[-1])[0]
        explorable_prototypes = associated_prototypes_layer_down[explorable_prototypes_indices]

    for i in range(len(explorable_prototypes)):
        centroid = explorable_prototypes[i]
        explore_centroid_pruning(vector_original, punto_buscado, current_layer - 1, inheritage + [centroid[1]], centroid[0], centroid[3], coords_puntos_capas, puntos_capas, grupos_capa, promoted_points, n_centroides, metrica, neighbours, distances_computed, candidate_neighbours)

