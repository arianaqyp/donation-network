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
             'Wee Lis', 'Weldon', 'Yu Son', 'Gaven', 'Adha', 'Annecia', 'Faris', 'Hui Hong', 'Jaden', 'Jun Kang', 'Kapil',
             'Keat Jin', 'Zheng Yang', 'Sean Morais', 'Shermit', 'Si Zhe', 'Teng Kang', 'Tyen Likh', 'Vela', 'Krystal',
             'Ji Feng', 'De Liang', 'Lennie', 'Adib', 'Ameen', 'Octson', 'Mubarak', 'Daraab', 'Jeong Han', 'Josh']
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
individual_list = ['Ariana', 'Jeremy', 'Ody', 'Sue', 'Emilie', 'WRB', '138', 'Anna']


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

# create dictionary to relabel nodes
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


# set edges attributes (add weightage to edges)
# dict = {'Ariana': 0}
#! full connected weights
unigroup1 = ['Adha', 'Adib', 'Ameen', 'Faris', 'Mubarak']       # 3
unigroup2 = ['Chen', 'Elvin', 'Kean Wee', 'Ram', 'Shaun', 'Vinura', 'Yuson', 'Jun Kang', 'Kapil', 'Zheng Yang',
             'Sean Morais', 'Shermit', 'Teng Kang', 'Tyen Likh', 'Ji Feng', 'Lennie', 'Octson', 'Daarab', 'Jin Yi',
             'Jin Hong', 'Ravi']       # 3
unigroup3 = ['Jian An', 'De Liang', 'Wen Hao', 'Jeong Han', 'Wee Lis', 'Ming Hui', 'Jerene']        # 3
unigroup4 = ['Weldon', 'Gaven', 'Jaden', 'Krystal']     #  3
unigroup5 = ['Hui Hong', 'Si Zhe']      # 3
unigroup6 = ['Annecia', 'Keat Jin']     # 3
unigroup7 = ['Denise', 'Nabilah']       # 3
unigroup8 = ['Yap Ying', 'Samuel']      # 5
unigroup9 = ['Tiffany', 'Zac']         # 5
collegegroup= ['Ariana', 'Saran', 'Vanessa', 'Jia Khai']        # 4
#! self connected weights [ 'Ariana - XX ]
self_uni = ['Ariana', 'Jeong Han', 'Wen Hao', 'Jin Hong']        # 3
self_three = ['Ariana', 'Jihin', 'Jing Poh', 'WRB']        # 3
self_four = ['Ariana', 'Andrea', 'Kapil', 'Jerene', 'Emilie', 'Saran', 'Jia Khai', 'Vanessa']      # 4
self_fam = ['Ariana', 'Jin Yi', 'Alvina', 'Mum']        # 5


def equal_dict_weight(group_list: list, weight: int):
    '''
    Takes a list of names (node labels) and specific weight and,
    creates and returns a dictionary of labels as keys and weights as value.
    Parameters
    ----------
    group_list: list
    weight: int

    Returns: a dictionary of labels as keys with equal weights as value
    -------

    '''
    pair_tuple_list = []
    for i in range(len(group_list)):
        for j in range(len(group_list)):
            if i != j:
                pair_tuple_list.append((group_list[i], group_list[j]))
            else:
                continue
    removed_dup_list = {frozenset(tup) for tup in pair_tuple_list}
    weight_dict = {key: weight for key in removed_dup_list}

    return weight_dict


# create dict for full connected weight groups
unigroup1_dict = equal_dict_weight(unigroup1, 3)
unigroup2_dict = equal_dict_weight(unigroup2, 3)
unigroup3_dict = equal_dict_weight(unigroup3, 3)
unigroup4_dict = equal_dict_weight(unigroup4, 3)
unigroup5_dict = equal_dict_weight(unigroup5, 3)
unigroup6_dict = equal_dict_weight(unigroup6, 3)
unigroup7_dict = equal_dict_weight(unigroup7, 3)
unigroup8_dict = equal_dict_weight(unigroup8, 5)
unigroup9_dict = equal_dict_weight(unigroup9, 5)
collegegroup_dict = equal_dict_weight(collegegroup, 4)

# dict for other groups
#self_uni = ['Ariana', 'Jeong Han', 'Wen Hao', 'Jin Hong', 'WRB']
self_uni_pair = []
for i in range(1, len(self_uni)):
    self_uni_pair.append((self_uni[0], self_uni[i]))
self_uni_dict = {key: 3 for key in self_uni_pair}

#self_three = ['Ariana', 'Jihin', 'Jing Poh', 'WRB']
self_three_pair = []
for i in range(1, len(self_three)):
    self_three_pair.append((self_three[0], self_three[i]))
self_three_dict = {key: 3 for key in self_three_pair}

# self_four = ['Ariana', 'Andrea', 'Kapil', 'Jerene', 'Emilie', 'Saran', 'Jia Khai', 'Vanessa']
self_four_pair = []
for i in range(1, len(self_four)):
    self_four_pair.append((self_four[0], self_four[i]))
self_four_dict = {key: 4 for key in self_four_pair}

# self_fam = ['Ariana', 'Jin Yi', 'Alvina', 'Mum']
self_fam_pair = []
for i in range(1, len(self_fam)):
    self_fam_pair.append((self_fam[0], self_fam[i]))
self_fam_dict = {key: 5 for key in self_fam_pair}


weight_dict_list = [unigroup1_dict, unigroup2_dict, unigroup3_dict, unigroup4_dict, unigroup5_dict, unigroup6_dict,
                    unigroup7_dict, unigroup8_dict, unigroup9_dict, collegegroup_dict, self_uni_dict, self_three_dict,
                    self_four_dict, self_fam_dict]


# loop through each dict and cast weight
for weight_group in weight_dict_list:
    nx.set_edge_attributes(graph_merged, weight_group, 'weight')


graph_merged_file = './graph_merged.out'
nx.write_pajek(graph_merged, graph_merged_file)

plt.show()

print("End of Program")


# test_graph_out = nx.relabel_nodes(graph_stranger_nx, {n: strangerHT[int(n)] for n in graph_stranger_nx.nodes()})