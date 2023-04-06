import networkx as nx
from networkx.classes import DiGraph, MultiDiGraph

def metr_hast_weights(G, copy=True, weight="weight"):
    if nx.number_of_selfloops(G)< nx.number_of_nodes(G):
        raise NotImplementedError("[metr_hast_weights] Not implemented for graphs without self loops.")
    if nx.is_directed(G):
        raise NotImplementedError("[metr_hast_weights] Not implemented for directed graphs.")
    if copy:
        G = DiGraph(G).to_directed()
    degree = dict(G.out_degree())
    for u, v, d in G.edges(data=True):
        if degree[u] == 0:
            d[weight] = 0
        else:
            if not u==v:
                d[weight] = 1/ (max(degree[u], degree[v]) - 1 + 1) # The -1 at the denominator accounts for the self-loop
            else:
                d[weight] = 0 # actual value is given in next loop
    for u, v, d in nx.selfloop_edges(G, data=True):
        d[weight] = 1 - G.out_degree(u, weight=weight)

    return G