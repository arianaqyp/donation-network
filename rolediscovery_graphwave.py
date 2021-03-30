from typing import Union, Optional, Dict
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import networkx as nx
import pyvis
import karateclub
import sklearn
import graphrole
import pandas as pd
import snap
import umap
from networkx import Graph
from pandas import DataFrame, Series
from snap import snap
from sklearn.decomposition import PCA


def read_pajek(path):
    """
    Takes path to a pajek file and return a networkx Graph.

    Parameters
    ----------
    path: str
        path for pajek file.

    Returns
    -------
    G: nx.classes.graph.Graph
        a networkx Graph

    """
    G: nx.classes.graph.Graph = nx.Graph(nx.read_pajek(path))

    return G


if __name__ == '__main__':

    graph_merged_path = './graph_merged.out'
    graph_merged = read_pajek(graph_merged_path)

    # label_id_dict = nx.get_node_attributes(graph_merged, 'id')
    # intlabel_id_dict = {int(ID): int(ID) for ID in label_id_dict.values()}
    #
    # graph_merged_intlabel = nx.relabel_nodes(graph_merged, label_id_dict)

    graph_merged_intlabel = nx.convert_node_labels_to_integers(graph_merged)

    gw = karateclub.GraphWave()
    gw.fit(graph_merged_intlabel)
    embeddings = gw.get_embedding()

    # pca = PCA(n_components=2)
    # pca_embeddings = pca.fit_transform(embeddings)
    #
    # reducer = umap.UMAP(n_components=2)
    # umap_embeddings = reducer.fit_transform(embeddings)
    #
    # plt.figure()
    # plt.scatter(pca_embeddings[:,0], pca_embeddings[:,1])
    # plt.title("PCA Embeddings")
    # plt.show()
    #
    # plt.figure()
    # plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1])
    # plt.title("UMAP Embeddings")
    # plt.show()

    ap = sklearn.cluster.AffinityPropagation(random_state=42, max_iter=2000, convergence_iter=30)
    ap.fit(embeddings)
    labels = ap.labels_

    print(labels)

    node_roles = {node: label for (node,label) in zip(graph_merged.nodes, labels)}

    unique_roles = sorted(set(node_roles.values()))
    color_map_ego = sns.color_palette('Reds', n_colors=1)[0]
    color_map_hex = sns.color_palette('Paired', n_colors=len(unique_roles)).as_hex()
    color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
    # map roles to colors
    role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
    # build list of colors for all nodes in G
    node_colors = [role_colors[node_roles[node]] if node != 'Ariana' else color_map_ego for node in graph_merged.nodes]

    role_colors_hex = {role: color_map_hex[i] for i, role in enumerate(unique_roles)}
    # build list of colors for all nodes in G
    node_colors_hex = [role_colors_hex[node_roles[node]] for node in graph_merged.nodes]

    plt.figure()

    with warnings.catch_warnings():
        # catch matplotlib deprecation warning
        warnings.simplefilter('ignore')
        nx.draw(
            graph_merged,
            pos=nx.kamada_kawai_layout(graph_merged),
            with_labels=True,
            node_color=node_colors,
        )

    plt.show()

    node_colors_hex_dict = {}
    counter = 0
    for node in graph_merged.nodes:
        node_colors_hex_dict[node] = node_colors_hex[counter]
        counter += 1

    nx.set_node_attributes(graph_merged, node_colors_hex_dict, 'color')
    nx.set_node_attributes(graph_merged, {'Ariana': '#FF0000'}, 'color')
    graph_merged_pyvis = pyvis.network.Network()
    graph_merged_pyvis.from_nx(graph_merged)
    graph_merged_pyvis.show_buttons()
    graph_merged_pyvis.show('graph_merged.html')


    print("Execute Main Program...")