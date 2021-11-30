import networkx as nx
from collections import deque


def compute_Euler_circuit_digraph(adj_list):
    G = nx.DiGraph(adj_list)
    visited = set()
    path = []

    stack = deque()
    stack.append(list(G.edges)[0])
    while stack:
        edge = stack[-1]
        if edge in visited:
            stack.pop()
            path.append(edge)
            continue

        visited.add(edge)
        for neighbor_edge in G.edges(edge[1]):
            if neighbor_edge not in visited:
                stack.append(neighbor_edge)

    return path[::-1]
