import networkx as nx
import random


# create graph map of the game for optimisation
G = nx.grid_2d_graph(4, 4)
G2 = nx.DiGraph()
for i in G.nodes:
    if i in [(0,0), (0,1), (0,2), (3,3), (2,1), (1,2)]:
        city =random.choice(['c_1', 'c_2'])
        G2.add_node(i, type=["friendly_city", city])
    elif i in [(1, 1), (2,2), (3,2)]:
        G2.add_node(i, type=["wood", "resource", "coal_researched", "uranium_researched"])
    else:
        G2.add_node(i, type=["empty"])

G2.add_edges_from(G.edges)
for s, t in G2.edges():
    G2.add_edge(t,s)

for s, t in G2.edges():
    weight=3
    weight_ac=3
    G2[s][t]['weight'] = weight
    G2[s][t]['weight_ac'] = weight_ac



source = (3,3)
#Calculate the length of paths from source to all other nodes
lengths=nx.single_source_dijkstra_path_length(G2, source, weight="weight_ac")
paths = nx.single_source_dijkstra_path(G2, source, weight="weight_ac")
#We are only interested in a particular type of node
subnodes = [name for name, d in G2.nodes(data=True) if ("friendly_city" in d['type'])]
subnode_type = [d['type'][1] for name, d in G2.nodes(data=True) if ("friendly_city" in d['type'])]
subdict = {k: v for k, v in lengths.items() if k in subnodes}
dist_array = sorted([(subnode_type[subnodes.index(k)], v) for (k,v) in subdict.items()], reverse=True)
dist_dic = {i[0]:i[1] for i in dist_array}

