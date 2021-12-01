import galois
import numpy as np
import networkx as nx

from graph_gen import (
  get_random_simple_Gnp_graph,
  get_Euler_digraph,
  get_random_simple_Gnp_graph_edges,
  get_hypercube_digraph)
from bridges import (compute_bridges_determ,
                     compute_bridges_rand,
                     compute_2bridges_rand,
                     assemble_matrix,
                     binary_gauss,
                     sample_solution)
from euler import compute_Euler_circuit_digraph


def test_stress_test_bridges_determ(n=100, m=300, iterations_num=1000):
  for i in range(iterations_num):
    G = get_random_simple_Gnp_graph(n, m, i)
    briges_test = compute_bridges_determ(G)
    bridges_true = set(nx.algorithms.bridges(nx.Graph(G)))
    diff = list(bridges_true.symmetric_difference(briges_test))
    if len(diff) > 0:
      print(diff)
      raise Exception(f"Неверное решение в детерминированном алгоритме поиска мостов, n: {n}, m: {m}, seed: {i}")
  print(f"Стресс тест для детерминированного поиска мостов пройден!, n: {n}, m: {m}, iterations_num: {iterations_num}")


def test_stress_test_bridges_rand(n=100, m=200, iterations_num=1000):
  exp_err = m * (1/ (2**64))
  for i in range(iterations_num):
    G = get_random_simple_Gnp_graph(n, m, i)
    briges_test = compute_bridges_rand(G)
    bridges_true = set(nx.algorithms.bridges(nx.Graph(G)))
    diff = list(bridges_true.symmetric_difference(briges_test))
    if len(diff) > exp_err:
      raise Exception(f"Число ошибок в рандомизированном алгоритме поиска мостов превышено, n: {n}, m: {m}, errors_num: {len(diff)}, seed: {i}")
  print(f"Стресс тест для рандомизированного поиска мостов пройден!, n: {n}, m: {m}, iterations_num: {iterations_num}")


def test_stress_test_2bridges_rand(n=100, m=150, sort=np.argsort, iterations_num=1000):
  exp_err = (m*(m-1)/2) * (1/ (2**64))
  for iteration in range(iterations_num):
    err_num = 0
    G = nx.Graph(get_random_simple_Gnp_graph(n, m, iteration))
    bridges_test = compute_2bridges_rand(G, sort)
    for edge_group in bridges_test:
      for e1 in range(len(edge_group)):
        for e2 in range(e1+1,len(edge_group)):
          G_with_deleted_2bridge = G.copy()
          G_with_deleted_2bridge.remove_edge(*edge_group[e1])
          G_with_deleted_2bridge.remove_edge(*edge_group[e2])
          if nx.connected.number_connected_components(G) == nx.connected.number_connected_components(G_with_deleted_2bridge):
            ++err_num
    if err_num > exp_err:
      raise Exception(f"Число ошибок в рандомизированном алгоритме поиска мостов превышено, n: {n}, m: {m}, errors_num: {err_num}, sort: {sort}, seed: {iteration}")
  print(f"Стресс тест для рандомизированного поиска мостов пройден!, n: {n}, m: {m}, , sort: {sort}, iterations_num: {iterations_num}")


def Euler_circuit_test(G, test_circuit):
  edges_dict = {}
  m = 0
  for (v,neib) in G.items():
    for u in neib:
      edges_dict[(v,u)] = False
      m += 1

  l = len(test_circuit)
  if m != l:
    return False

  for i in range(1,m):
    if test_circuit[i-1][1] != test_circuit[i][0]:
      return False
  if test_circuit[l-1][1] != test_circuit[0][0]:
    return False

  for e in test_circuit:
    if edges_dict.get(e, True) == True:
      return False
    else:
      edges_dict[e] = True

  return True


# max_n должно быть строго больше чем 10
def test_stress_test_Euler_circuit_digraph(max_n=50, iterations_num=1000):
  for iteration in range(iterations_num):
    n = np.random.randint(10, max_n)
    k = np.floor(n/4)
    G = get_Euler_digraph(n,k)
    test_circuit = compute_Euler_circuit_digraph(G)
    if not Euler_circuit_test(G, test_circuit):
      raise Exception(f"Неправильный Эйлеров обход, функция генерации: get_Euler_digraph, n: {n}, k: {k}")
  print(f"Стресс тест для ориентированных Эйлеровых циклов пройден, max_n: {max_n}, iterations_num: {iterations_num}")


def test_euler_circuit_unit_tests():
  dims = [1, 2, 3, 4, 5, 6]
  for dim in dims:
    G = get_hypercube_digraph(dim)
    test_circuit = compute_Euler_circuit_digraph(G)
    assert(Euler_circuit_test(G,test_circuit))


def test_random_bridge_matrix_matrix_assemble():
    test_edges = ((0, 1), (0, 4), (3, 4), (1, 2), (2, 3), (1, 3))
    ref_matrix = np.array([[1., 1., 0., 0., 0., 0.],
                           [1., 0., 0., 1., 0., 1.],
                           [0., 0., 0., 1., 1., 0.],
                           [0., 0., 1., 0., 1., 1.],
                           [0., 1., 1., 0., 0., 0.]])
    res = assemble_matrix(test_edges, 5)

    assert (ref_matrix == res).all()


def test_euler_cycles(n=100, m=200, iterations=10):
    for seed in range(iterations):
        edges, nodes_count, graph = get_random_simple_Gnp_graph_edges(n, m, seed)
        matrix = assemble_matrix(edges, nodes_count).view(galois.GF2)
        for _ in range(5):
            res = binary_gauss(matrix.copy())
            _, rank, order = res
            res = sample_solution(*res[:-1])
            matrix = matrix.take(order, axis=1)
            if not (matrix @ res).any():
                continue

            print(res)
            print(matrix @ res)
            raise AssertionError('Computed solution isn\'t correct')


def test_euler_simple():
    a = np.array([[1, 1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 1, 0, 1, 1],
                  [0, 1, 1, 0, 0, 0]], dtype=np.int8)
    a = a.view(galois.GF2)
    for _ in range(10):
        diag, rank, order = binary_gauss(a.copy())
        res = sample_solution(diag, rank)
        check = a.take(order) @ res
        assert not check.any()

def test_random_bridges_simple():
    pass
    #test_edges = {0: [1, 4], 1: [0](3, 4), (1, 2), (2, 3), (1, 3))
    #test_edges = nx.convert.to_dict_of_lists(test_edges)
    #res = compute_bridges_rand(test_edges)
    #print(res)
