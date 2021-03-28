#! import
from snap import snap
import numpy as np
import pandas as pd
import functools
import operator
import matplotlib.pyplot as plt
import seaborn as sns
import random
import operator
import functools
import csv
import networkx as nx

# set random seed
random.seed(42)

# create hash table for TIntStrH
h = snap.TIntStrH()  # create empty table

# Groups of friends
#! fully connected
mech_list = ['Ariana', 'Chen', 'Clinton', 'Elvin', 'Jian An', 'Kean Wee', 'Ram', 'Shaun', 'Wen Hao', 'Vinura',
             'Wee Lis',
             'Weldon', 'Yu Son', 'Gaven', 'Adha', 'Annecia', 'Faris', 'Hui Hong', 'Jaden', 'Jun Kang', 'Kapil',
             'Keat Jin', 'Zheng Yang', 'Sean Morais', 'Shermit', 'Si Zhe', 'Teng Kang', 'Tyen Likh', 'Vela', 'Krystal',
             'Ji Feng', 'De Liang', 'Lennie', 'Adib', 'Ameen', 'Octson', 'Mubarak', 'Daraab', 'Jeong Han']
college_list = ['Ariana', 'Saran', 'Vanessa', 'Jia Khai', 'Kai Sheng']
uni_senior_list = ['Ariana', 'Kim Meng', 'Jerene', 'You Wei']
gym_list = ['Ariana', 'Ji Hin', 'Uncle Heng', 'Jing Poh']
#! need to add 'Ariana' node
ee_list = ['Jin Yi', 'Jin Hong', 'Ravi', 'Zac', 'Mahmoud', 'Yap Ying', 'Samuel', 'Denise', 'Nabilah', 'Ming Hui',
           'Tiffany']
alvina_list = ['Alvina', 'Jia Wen', 'Esther']
#! Individual generation
stranger_list = ['anon1', 'anon2', 'anon3', 'anon4']
mum_list = ['Ariana', 'Mum', 'Alison', 'Yvonne', 'PL', 'Etsumi']
individual_list = ['Ariana', 'Jeremy', 'Ody', 'Sue', 'Emilie', 'WRB', '138']


def create_hashtable(hashtb: snap.TIntStrH, stringlist):
    """
    creates a hashtable for node labelling
    :param hashtb: hashtable of Int and Str
    :param stringlist: list of strings
    """
    for id in range(len(stringlist)):
        hashtb[id] = stringlist[id]

    return hashtb


# Create hashtable for each groups
mechHT = snap.TIntStrH()
create_hashtable(mechHT, mech_list)
uni_seniorHT = snap.TIntStrH()
create_hashtable(uni_seniorHT, uni_senior_list)
eeHT = snap.TIntStrH()
create_hashtable(eeHT, ee_list)
gymHT = snap.TIntStrH()
create_hashtable(gymHT, gym_list)
collegeHT = snap.TIntStrH()
create_hashtable(collegeHT, college_list)
alvinaHT = snap.TIntStrH()
create_hashtable(alvinaHT, alvina_list)
strangerHT = snap.TIntStrH()
create_hashtable(strangerHT, stranger_list)
mumHT = snap.TIntStrH()
create_hashtable(mumHT, mum_list)
individualHT = snap.TIntStrH()
create_hashtable(individualHT, individual_list)

list_of_ht = [mechHT, uni_seniorHT, eeHT, gymHT, collegeHT, alvinaHT, strangerHT, mumHT, individualHT]
list_of_groupnames = ['mech', 'uni_senior', 'ee', 'gym', 'college', 'alvina', 'stranger', 'mum', 'individual']

#! fully connected graphs
graph_college = snap.GenFull(snap.TUNGraph, len(collegeHT))
graph_college.SavePajek('pajek_college.out', collegeHT)

graph_uni_senior = snap.GenFull(snap.TUNGraph, len(uni_seniorHT))
graph_uni_senior.SavePajek('pajek_uni_senior.out', uni_seniorHT)

graph_gym = snap.GenFull(snap.TUNGraph, len(gymHT))
graph_gym.SavePajek('pajek_gym.out', gymHT)

#! fully connected then add 'Ariana' node
# ee_list = ['Jin Yi', 'Jin Hong', 'Ravi', 'Zac', 'Mahmoud', 'Yap Ying', 'Samuel', 'Denise', 'Nabilah', 'Ming Hui',
#            'Tiffany']

graph_ee = snap.GenFull(snap.TUNGraph, len(eeHT))
eeArianaNId = len(eeHT)
eeHT[eeArianaNId] = 'Ariana'
graph_ee.AddNode(eeArianaNId)
# Add edges
for i in range(3):
    graph_ee.AddEdge(eeArianaNId, i)      # jy, jh, ravi
graph_ee.SavePajek('pajek_ee.out', eeHT)


#! Add Ariana to fully connected Alvina graph
# alvina_list = ['Alvina', 'Jia Wen', 'Esther']
graph_alvina = snap.GenFull(snap.TUNGraph, len(alvinaHT))
alvinaArianaNId = len(alvinaHT)
alvinaHT[alvinaArianaNId] = 'Ariana'
graph_alvina.AddNode(alvinaArianaNId)
# Add edges
graph_alvina.AddEdge(alvinaArianaNId, 0)      # ariana - alvina
graph_alvina.SavePajek('pajek_alvina.out', alvinaHT)

#! Add Jy, Jh, Ravi to fully connected Mech
graph_mech = snap.GenFull(snap.TUNGraph, len(mechHT))
ee_mech_list = ['Jin Yi', 'Jin Hong', 'Ravi']
len_mech = len(mechHT)
for i in range(len(ee_mech_list)):
    mechHT[len_mech+i] = ee_mech_list[i]
    graph_mech.AddNode(len_mech+i)
for i in range(len(ee_mech_list)):
    for j in range(len_mech):
        graph_mech.AddEdge(len_mech+i, j)
graph_mech.SavePajek('pajek_mech.out', mechHT)

# ! generate individual graph: stranger, mum, individual and save in Pajek
# strangers not connected to anyone (individual nodes)
# stranger_list = ['anon1', 'anon2', 'anon3']
graph_stranger = snap.TUNGraph.New()
for i in range(len(stranger_list)):
    graph_stranger.AddNode(i)
graph_stranger.SavePajek('pajek_stranger.out', strangerHT)

# mum connected to all (source graph)
# mum_list = ['Ariana', 'Mum', 'Alison', 'Yvonne', 'PL', 'Etsumi']
graph_mum = snap.TUNGraph.New()
for i in range(len(mum_list)):
    graph_mum.AddNode(i)
for i in range(len(mum_list)):
    if i != 1:
        graph_mum.AddEdge(1, i)
graph_mum.SavePajek('pajek_mum.out', mumHT)

# Ariana connected to all
# individual_list = ['Ariana', 'Jeremy', 'Ody', 'Sue', 'Emilie', 'WRB', '138']
graph_individual = snap.TUNGraph.New()
for i in range(len(individual_list)):
    graph_individual.AddNode(i)
for i in range(len(individual_list)):
    if i != 0:
        graph_individual.AddEdge(0, i)
# Add Mum and Alvina to connect to Sue and Emilie
fam_individual_list = ['Alvina', 'Mum']
len_individual = len(individualHT)
for i in range(len(fam_individual_list)):
    individualHT[len_individual+i] = fam_individual_list[i]
    graph_individual.AddNode(len_individual+i)
for i in range(len(fam_individual_list)):
    graph_individual.AddEdge(3, len_individual+i)
    graph_individual.AddEdge(4, len_individual+i)
graph_individual.SavePajek('pajek_individual.out', individualHT)


#! output filename in a list
output_file_list = []
for filename in list_of_groupnames:
    output_file_list.append('pajek_{}.out'.format(filename))


#! read pajek using nx
graph_list = []
for file in output_file_list:
    G = nx.Graph(nx.read_pajek(file))
    graph_list.append(G)
    # nx.draw(G, with_labels=True)
    # plt.show()

graph_ht_dict = {graph_list[i]: list_of_ht[i] for i in range(len(list_of_ht))}
nx_graph_list = [nx.relabel_nodes(graph, {n: ht[int(n)] for n in graph.nodes()}) for (graph, ht) in
                 graph_ht_dict.items()]
nx.draw(nx_graph_list[0], with_labels=True)
plt.show()

# graph_merged = functools.reduce(nx.compose, nx_graph_list)
graph_merged = nx.compose_all(nx_graph_list)
nx.draw(graph_merged, with_labels=True, font_size=5, node_size=200, pos=nx.kamada_kawai_layout(graph_merged, scale=10))

node_label_id_dict = nx.get_node_attributes(graph_merged, 'id')
nodeid_list = [i for i in range(len(node_label_id_dict))]
node_label_id_dict_uniqueId = {key: i for i, key in zip(nodeid_list, node_label_id_dict.keys())}
nx.set_node_attributes(graph_merged, node_label_id_dict_uniqueId, 'id')

graph_merged_file = './graph_merged.out'
nx.write_pajek(graph_merged, graph_merged_file)

plt.show()

# ---------------------------------------------------------------------------------------

import warnings
import graphrole

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


# graph_merged_path = './graph_merged.out'
# graph_merged = read_pajek(graph_merged_path)
nx.draw(graph_merged, with_labels=True, font_size=5, node_size=200,
        pos=nx.kamada_kawai_layout(graph_merged, scale=10))

features, feature_extractor = extract_features(graph_merged)
node_roles, role_extractor = extract_roles(features, n_roles=4)

# build color palette for plotting
unique_roles = sorted(set(node_roles.values()))
color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
# map roles to colors
role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
# build list of colors for all nodes in G
node_colors = [role_colors[node_roles[node]] for node in graph_merged.nodes]

plt.figure()

with warnings.catch_warnings():
    # catch matplotlib deprecation warning
    warnings.simplefilter('ignore')
    nx.draw(
        graph_merged,
        pos=nx.kamada_kawai_layout(graph_merged),
        with_labels=True,
        node_color=node_colors, font_size=5, node_size=200
    )

plt.show()



print("End of Program")

# test_graph_out = nx.relabel_nodes(graph_stranger_nx, {n: strangerHT[int(n)] for n in graph_stranger_nx.nodes()})