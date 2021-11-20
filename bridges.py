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

    #bfs(0)
    #other_edges = set([tuple(sorted(edge)) for edge in G.edges]) - set(bfs_tree_edges)
    #other_edges = list(other_edges)
    #mask_non_tree = np.random.randint(2, size=len(other_edges), dtype=np.int64)
    #mask_tree = np.empty((len(bfs_tree_edges),), dtype=np.int64)
    ## Propagate masks
    #for in_e, out_e in bfs_tree_edges[::-1]:





    #return list(nx.algorithms.bridges(G))


def binary_gauss(A):
    n = A.shape[0]
    order = np.arange(A.shape[1])

    for i in range(0, n):
        # Search for maximum in this column
        max_column = -1
        for k in range(i, n):
            if A[i][k] == 1:
                max_column = k
                break

        # In case maximum column value is zero
        if max_column == -1:
            # Swap row
            tmp = A[i, :].copy()
            A[i, :] = A[-1, :]
            A[-1, :] = tmp
            # Skip iter
            continue

        # Swap maximum row with current row (column by column)
        if max_column != i:
            tmp = A[:, i].copy()
            A[:, i] = A[:, max_column]
            A[:, max_column] = tmp
            order[i], order[max_column] = order[max_column], order[i]

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            if A[k][i] != 0:
                A[k, :] ^= A[i, :]

    range_ = 0
    for k in range(n - 1, -1, -1):
        if A[k][k] != 0:
            range_ = k + 1
            break

    return A, range_, order


def sample_solutions(A, range_, order):
    #sample = np.random.randint(-2**31, 2**31-1, size=(A.shape[0],))
    #sample = np.random.randint(0, 2, size=(A.shape[0],))
    sample = np.zeros(A.shape[1], dtype=np.int8)
    free_vars = A.shape[1] - range_
    sample[-free_vars:] = np.random.randint(0, 2, size=(free_vars,), dtype=np.uint64)
    for i in range(range_ - 1, -1, -1):
        #sample[i] = sample.dot(A[i, :])
        sample[i] = np.bitwise_xor.reduce(sample & A[i, :])

    return sample.take(order, axis=0)


def compute_2bridges_rand(*args):
   pass

def test():
    a = np.array([[1, 1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 1, 0, 1, 1],
                  [0, 1, 1, 0, 0, 0]], dtype=np.int8)
    for _ in range(10):
        res = binary_gauss(a.copy())
        res = sample_solutions(*res)
        print('*'* 10)
        print(res)
        print(a @ res % 2)

if __name__ == '__main__':
    test()
