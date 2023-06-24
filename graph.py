import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math
import random
from scipy.stats import norm
import scipy
import sys
import seaborn as sns

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
    return np.round(abs(eigenvalues[-1]), 10)


def spectral_gap(graph):
    eigenvalues = adjacency_eigen_values(graph)
    eigenvalues.sort()

    if len(eigenvalues) >= 2:
        return abs(np.round(eigenvalues[-1] - eigenvalues[-2], 10))

    return 0


def algebraic_connectivity(graph):
    eigenvalues = laplacian_eigen_values(graph)
    eigenvalues.sort()

    return abs(np.round(abs(eigenvalues[1]), 10))


def natural_connectivity(graph):
    eigenvalues = adjacency_eigen_values(graph)

    eig_sum = 0
    n = len(eigenvalues)
    for eig in eigenvalues:
        if eig > 0:
            aeig = abs(eig)
        else:
            aeig = -abs(eig)

        eig_sum += math.exp(aeig) / n

    return np.round(np.log(abs(eig_sum)), 10)


def symmetry_ratio(graph):
    if not nx.is_connected(graph):
        d = nx.global_efficiency(graph)
    else:
        d = nx.diameter(graph)

    e = len(set(np.round(complex_abs(adjacency_eigen_values(graph)), 10)))
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
    degrees = [d for n, d in graph.degree()]
    sns.displot(degrees, kde=True, color="black")

    plt.xlabel("degree")
    plt.ylabel("frequency")
    plt.title("Degree Distribution")


def draw_eigen_values_distribution(graph):
    sns.displot(adjacency_eigen_values(graph), kde=True, color="black")

    plt.xlabel("eigen value")
    plt.ylabel("frequency")
    plt.title("Eigen Values Distribution")


def draw_distribution(data, name):
    sns.displot(data, kde=True, color="black")

    plt.xlabel(name)
    plt.ylabel("frequency")
    plt.title(f"{name} Distribution")


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


def calculate_statistics_parameters(data):
    mean, std = np.mean(data), np.std(data)
    print(f"mean is {mean}")
    print(f"standard deviation is {std}")
    e = (1.96 * np.std(data)) / math.sqrt(len(data))
    ci = (np.mean(data) - e), np.mean(data) + e
    print(f"CI is {ci}")
    plt.boxplot(data)
    plt.ylabel("box plot")
    plt.title("values box plot")
    return mean, std, e, ci


def complex_abs(a):
    res = []
    for x in a:
        if x > 0:
            res.append(np.abs(x))
        else:
            res.append(-np.abs(x))
    return res


def box_plot(data, index, title):
    # Creating dataset
    np.random.seed(10)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(data, patch_artist=True,
                    notch='True', vert=0)

    colors = ['#0000FF', '#00FF00',
              '#FFFF00', '#FF00FF']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B',
                    linewidth=1.5,
                    linestyle=":")

    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color='#8B008B',
                linewidth=2)

    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color='red',
                   linewidth=3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color='#e7298a',
                  alpha=0.5)

    # x-axis labels
    ax.set_yticklabels(index)

    # Adding title
    plt.title(title)

    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # show plot
    plt.show()
