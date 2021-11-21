from typing import List, Tuple

import numpy as np
import galois
import networkx as nx

from graph_gen import (
    get_random_simple_Gnp_graph,
    get_random_simple_Gnp_graph_edges,
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


def assemble_matrix(edges: List[Tuple[int, int]], verts_len: int) -> np.array:
    matrix = np.zeros((verts_len, len(edges)), dtype=np.int8)
    for idx, edge in enumerate(edges):
        matrix[edge[0]][idx] = matrix[edge[1]][idx] = 1
    return matrix

# рандомизированный алгоритм для поиска мостов
# на вход поступает граф представленный списком смежности
# список представлен как словарь(хеш-таблица) списков
# выход представляет собой список ребер, являющихся мостами с большой вероятностью
def compute_bridges_rand(adj_list: dict):
    G = nx.Graph(adj_list)
    edges = G.edges
    matrix = assemble_matrix(edges, len(G)).astype(np.uint8)
    _, rank, order = binary_gauss(matrix)
    solutions = sample_64_bit_solutions(matrix.astype(np.uint64), rank)
    bridges_idxs = np.where(solutions == 0)
    bridges = np.array(edges).take(order, axis=0)[bridges_idxs].tolist()
    return {tuple(x) for x in bridges}


def binary_gauss(A):
    n = A.shape[0]
    order = np.arange(A.shape[1])

    stop_iters = False
    for i in range(0, n):

        # Search for maximum in this column
        while True:
            max_column = -1
            for k in range(i, A.shape[1]):
                if A[i][k] == 1:
                    max_column = k
                    break

            # In case not all rows are zero
            if max_column != -1:
                # Go next
                break
            # Find non zero row
            non_zero_row_idx = -1
            for m in range(n-1, i, -1):
                if A[m, :].any():
                    non_zero_row_idx = m
                    break
            # If all rows zeros
            # stop iterations
            if non_zero_row_idx == -1:
                stop_iters = True
                break
            # Else swap rows and
            # repeat iter
            tmp = A[i, :].copy()
            A[i, :] = A[non_zero_row_idx, :].copy()
            A[non_zero_row_idx, :] = tmp
            # Break infinite loop
            if not A[i, :].any():
                break

        if stop_iters:
            break

        # Swap maximum row with current row (column by column)
        if max_column != i:
            tmp = A[:, i].copy()
            A[:, i] = A[:, max_column].copy()
            A[:, max_column] = tmp
            order[i], order[max_column] = order[max_column], order[i]

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            if A[k][i] != 0:
                A[k, :] ^= A[i, :]

    rank = 0
    for k in range(min(A.shape) - 1, -1, -1):
        if A[k][k] != 0:
            rank = k + 1
            break

    return A, rank, order


def sample_64_bit_solutions(A, rank):
    sample = np.zeros(A.shape[1], dtype=np.uint64)
    A[np.where(A == 1)] = np.iinfo(np.uint64).max
    free_vars = A.shape[1] - rank
    if free_vars == 0:
        return sample

    sample[-free_vars:] = np.random.randint(0, 2**64-1, size=(free_vars,), dtype=np.uint64)
    for i in range(rank - 1, -1, -1):
        sample[i] = np.bitwise_xor.reduce(sample & A[i, :])

    return sample


def sample_solutions(A, rank):
    #sample = np.random.randint(-2**31, 2**31-1, size=(A.shape[0],))
    #sample = np.random.randint(0, 2, size=(A.shape[0],))
    sample = np.zeros(A.shape[1], dtype=np.int8).view(galois.GF2)
    free_vars = A.shape[1] - rank
    if free_vars == 0:
        return sample

    sample[-free_vars:] = galois.GF2.Random((free_vars,))#np.random.randint(0, 2, size=(free_vars,), dtype=np.uint64)
    for i in range(rank - 1, -1, -1):
        sample[i] = ((np.bitwise_xor.reduce(sample & A[i, :]) ^ A[i][i]) + np.array(1).view(galois.GF2))

    assert not (A @ sample).any()
    return sample


def compute_2bridges_rand(*args):
   pass

def test_simple():
    a = np.array([[1, 1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 1, 0, 1, 1],
                  [0, 1, 1, 0, 0, 0]], dtype=np.int8)
    a = a.view(galois.GF2)
    for _ in range(10):
        res = binary_gauss(a.copy())
        res = sample_solutions(*res)
        check = a @ res
        if check.any():
            print('*'* 10)
            print(res)
            print(check)


def draw_solution(graph, solution):
    for edge, sol in zip(graph.edges, solution):
        graph[edge[0]][edge[1]]['tree'] = int(sol)
    print(nx.nx_pydot.to_pydot(graph))


def test():
    for seed in range(1000):
        edges, nodes_count, graph = get_random_simple_Gnp_graph_edges(100, 200, seed)
        matrix = assemble_matrix(edges, nodes_count).view(galois.GF2)
        #print(matrix)
        for _ in range(5):
            res = binary_gauss(matrix.copy())
            _, rank, order = res
            res_to_print = np.array(res[0])
            matrix_to_print = np.array(matrix)
            res_to_print_ordered = res_to_print.take(order, axis=1).view(galois.GF2)
            res = sample_solutions(*res)
            matrix = matrix.take(order, axis=1)
            #draw_solution(graph, res)
            print('*'*100)
            if not (matrix @ res).any():
                print('OK')
                continue

            print(res)
            print(matrix @ res)
        a = 6


def test_gaus_for_non_singular_matrix():
    # Gen non singular matrix matrix
    print('start testing')
    iter = 1000
    for test_i in range(iter):
        if test_i % 10 == 0:
            print(f'iter {test_i}')
        n =5
        while True:
            #matrix = np.random.randint(0, 2, size=(n, n), dtype=np.uint64)
            matrix = galois.GF2.Random((n, n))
            if np.linalg.matrix_rank(matrix) == n:
                break
                solution = np.linalg.solve(matrix, np.zeros(n, dtype=np.uint64))
                if solution.any():
                    print(f'ref solution: {solution}')
                    break
        diag, range_, order = binary_gauss(matrix.copy())
        # Find the roots
        x = np.zeros(n, dtype=np.uint8).view(galois.GF2)
        for i in range(n - 1, -1, -1):
            x[i] = ((np.bitwise_xor.reduce(x & diag[i, :]) ^ diag[i][i]) + np.array(1).view(galois.GF2))
        #x = x.take(order, axis=0)

        if x.any():
            print('non trivial solution')
        res = matrix @ x
        if res.any():
            print('aa')
            assert False

    # Stop testing


if __name__ == '__main__':
    #test_gaus_for_non_singular_matrix()
    test()
