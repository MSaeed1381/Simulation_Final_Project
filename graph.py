import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math
import random
from scipy.stats import norm
import scipy
import sys


sys.setrecursionlimit(10000)


def adjacency_eigen_values(graph):
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    return np.linalg.eig(adj_matrix)[0]


def adjacency_eigen_vectors(graph):
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    return np.linalg.eig(adj_matrix)[1]


def laplacian_eigen_values(graph):
    laplacian_matrix = nx.laplacian_matrix(graph).toarray()
    return np.linalg.eig(laplacian_matrix)[0]


def laplacian_eigen_vectors(graph):
    laplacian_matrix = nx.laplacian_matrix(graph).toarray()
    return np.linalg.eig(laplacian_matrix)[1]


def spectral_radius(graph):
    eigenvalues = adjacency_eigen_values(graph)
    eigenvalues.sort()
    return np.round(eigenvalues[-1], 10)


def spectral_gap(graph):
    eigenvalues = adjacency_eigen_values(graph)
    eigenvalues.sort()

    if len(eigenvalues) >= 2:
        return np.round(eigenvalues[-1] - eigenvalues[-2], 10)

    return 0


def algebraic_connectivity(graph):
    eigenvalues = laplacian_eigen_values(graph)
    eigenvalues.sort()
    return abs(np.round(eigenvalues[1], 10))


def natural_connectivity(graph):
    eigenvalues = adjacency_eigen_values(graph)

    eig_sum = 0
    n = len(eigenvalues)
    for eig in eigenvalues:
        eig_sum += math.exp(eig) / n

    return np.round(np.log(eig_sum), 10)


def symmetry_ratio(graph):
    if not nx.is_connected(graph):
        return 0  # TODO

    d = nx.diameter(graph)
    e = len(set(adjacency_eigen_values(graph)))
    return e / (d + 1)


def energy(graph):
    eigenvalues = adjacency_eigen_values(graph)
    s = 0
    for eig in eigenvalues:
        s += abs(eig)

    return np.round(s, 10)


def laplacian_energy(graph):
    eigenvalues = laplacian_eigen_values(graph)
    s = 0
    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    for eig in eigenvalues:
        s += abs(eig - (2 * m / n))

    return np.round(s, 10)


def draw_degree_distribution(graph):
    # Get the degrees of nodes
    degrees = [d for n, d in graph.degree()]

    # Plotting the histogram
    n, bins, patches = plt.hist(degrees, 60, density=1, edgecolor='black', alpha=0.5)

    # Customizing the plot
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')

    # Display the plot
    # plt.legend()
    plt.show()


def RSRBG(n, d1, d2):
    LIMIT = 10
    color_map = []
    rsrbg_graph = nx.Graph()

    n1 = (n * d2) // (d1 + d2)
    n2 = (n * d1) // (d1 + d2)
    if n1 + n2 != n:
        return rsrbg_graph, color_map

    rsrbg_graph.add_nodes_from(list(range(1, n1 + 1)))
    rsrbg_graph.add_nodes_from(list(range(n1 + 1, n + 1)))

    for i in range(n):
        if i < n1:
            color_map.append('red')
        else:
            color_map.append('blue')

    list1 = list(range(1, n1 + 1)) * d1
    list2 = list(range(n1 + 1, n1 + n2 + 1)) * d2
    random.shuffle(list1)
    random.shuffle(list2)

    edges_added = set()

    limit = LIMIT

    while list1 and list2:

        if limit < 0:
            return RSRBG(n, d1, d2)

        node1 = np.random.choice(list1)
        node2 = np.random.choice(list2)

        if (node1, node2) not in edges_added and (node2, node1) not in edges_added:
            limit = LIMIT
            rsrbg_graph.add_edge(node1, node2)
            edges_added.add((node1, node2))

            list1.remove(node1)
            list2.remove(node2)

        else:
            limit -= 1

    return rsrbg_graph, color_map


def RSRG(p, n, d1, d2):
    LIMIT = 10
    color_map = []
    rsrg_graph = nx.Graph()

    n1 = (n * d2) // (d1 + d2)
    n2 = (n * d1) // (d1 + d2)
    if n1 + n2 != n:
        return rsrg_graph, color_map

    rsrg_graph.add_nodes_from(list(range(1, n1 + 1)))
    rsrg_graph.add_nodes_from(list(range(n1 + 1, n + 1)))

    for i in range(n):
        if i < n1:
            color_map.append('red')
        else:
            color_map.append('blue')

    ls = list(range(1, n1 + 1)) * d1 + list(range(n1 + 1, n1 + n2 + 1)) * d2

    random.shuffle(ls)

    edges_added = set()

    limit = LIMIT

    while ls:
        if limit < 0:
            return RSRG(p, n, d1, d2)

        node1 = np.random.choice(ls)
        node2 = np.random.choice(ls)

        if (node1, node2) not in edges_added and (node2, node1) not in edges_added and node1 != node2:
            limit = LIMIT
            if random.uniform(0, 1) < p:
                rsrg_graph.add_edge(node1, node2)
                edges_added.add((node1, node2))

            ls.remove(node1)
            ls.remove(node2)
        else:
            limit -= 1

    return rsrg_graph, color_map