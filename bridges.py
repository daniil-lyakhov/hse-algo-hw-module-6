from typing import List, Tuple

import numpy as np
import networkx as nx

from graph_gen import (
    get_random_simple_Gnp_graph,
    get_random_Gnp_digraph,
    get_Euler_digraph,
    get_random_Gnm_digraph,
    get_random_simple_Gnm_graph,
    get_hypercube_digraph)


def main():
    graph = get_random_simple_Gnm_graph(50, 50)
    res = compute_bridges_determ(graph)
    a = 6

# детерминированный алгоритм для поиска мостов
# на вход поступает граф представленный списком смежности
# список представлен как словарь(хеш-таблица) списков
# выход представляет собой список ребер, являющихся мостами
def compute_bridges_determ(adj_list):
    # этот код нужно заменить на Ваш
    G = nx.Graph(adj_list)
    tin = {} # input time for each node
    fup = {} # first in upper node except parent
    bridges = []
    t = 0

    def dfs(v, parent=-1):
        nonlocal t
        tin[v] = t
        fup[v] = t
        t += 1
        for child in G[v]:
            if child == parent: # If child is parent - skip it
                continue
            elif child in tin: # If child visited
                fup[v] = min(fup[v], tin[child]) # Recompute current first in upper node except parent
            else: # If child not visited and not parent
                dfs(child, v) # Compute fup for child
                fup[v] = min(fup[v], fup[child]) # Recompute current first in upper node except parent
                if tin[v] < fup[child]: # Check if the earliest ancestor except v is child of v
                    bridges.append(tuple(sorted((child, v)))) # If it is thus such edge is bridge
        return bridges

    for node in G.nodes():
        if node not in tin:
            dfs(node)
    return bridges


def assemble_matrix(edges: List[Tuple[int, int]], verts_len: int ) -> np.array:
    matrix = np.zeros((verts_len, len(edges)))
    for idx, edge in enumerate(edges):
        matrix[edge[0]][idx] = matrix[edge[1]][idx] = 1
    return matrix

# рандомизированный алгоритм для поиска мостов
# на вход поступает граф представленный списком смежности
# список представлен как словарь(хеш-таблица) списков
# выход представляет собой список ребер, являющихся мостами с большой вероятностью
def compute_bridges_rand(adj_list: dict):
    # Compute vertex / edges matrix
    #edges = set()
    #for node, childs in adj_list.items():
    #    for child in childs:
    #        edges.add(tuple(sorted((node, child))))

    ## Assemble matrix
    #matrix = assemble_matrix(list(edges), len(adj_list))
    G = nx.Graph(adj_list)
    bfs_tree_edges = list()
    visited = set()

    def bfs(v):
        visited.add(v)
        for child in G[v]:
            if child not in visited:
                bfs_tree_edges.append((v, child))
                bfs(child)

    bfs(0)
    other_edges = set([tuple(sorted(edge)) for edge in G.edges]) - set(bfs_tree_edges)
    other_edges = list(other_edges)
    mask_non_tree = np.random.randint(2, size=len(other_edges), dtype=np.int64)
    mask_tree = np.empty((len(bfs_tree_edges),), dtype=np.int64)
    # Propagate masks
    for in_e, out_e in bfs_tree_edges[::-1]:





    return list(nx.algorithms.bridges(G))


def compute_2bridges_rand(*args):
   pass


if __name__ == '__main__':
    main()
