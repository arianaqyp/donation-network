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
from graphrole import RecursiveFeatureExtractor
from networkx import Graph
from pandas import DataFrame, Series
from snap import snap

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


def extract_features(G: nx.classes.Graph):
    """
    Takes an input graph and recursively extract features from graph. Returns features extracted.

    Parameters
    ----------
    G: nx.classes.Graph
        Input graph.

    Returns
    -------
    features: Union[DataFrame, Series]
        Extracted features.

    feature_extractor: graphrole.RecursiveFeatureExtractor
        graphrole package feature extractor.s

    """

    feature_extractor = graphrole.RecursiveFeatureExtractor(G)
    features = feature_extractor.extract_features()

    return features, feature_extractor


def extract_roles(features, n_roles=None):
    """

    Parameters
    ----------
    features: Union[DataFrame, Series]
        features extracted by graphrole.RecursiveFeatureExtractor.
    n_roles: int
        number of roles to be extracted.
    Returns
    -------
    node_roles: Optional[Dict[Union[int, str], float]]
        Dictionary of roles extracted.

    role_extractor: graphrole.RecursiveRoleExtractor
        graphrole.RecursiveFeatureExtractor object.

    """
    role_extractor = graphrole.RoleExtractor(n_roles)
    role_extractor.extract_role_factors(features)
    node_roles = role_extractor.roles

    return node_roles, role_extractor


if __name__ == '__main__':

    graph_merged_path = './graph_merged.out'
    graph_merged = read_pajek(graph_merged_path)

    features, feature_extractor = extract_features(graph_merged)
    node_roles, role_extractor = extract_roles(features, n_roles=4)

    # build color palette for plotting
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