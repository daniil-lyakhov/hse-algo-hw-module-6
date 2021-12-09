import networkx as nx
from collections import deque


def compute_Euler_circuit_digraph(adj_list):
    G = nx.DiGraph(adj_list)
    visited = dict()
    stack = deque()
    path = deque()

    def update_neighbors(edge, neighbors_edges):
        if neighbors_edges:
            visited[edge] = neighbors_edges[1:]
            if neighbors_edges[0] not in visited:
                stack.appendleft(neighbors_edges[0])
        return neighbors_edges

    stack.append(list(G.edges)[0])
    while stack:
        edge = stack[0]
        if edge not in visited:
            visited[edge] = []
            neighbors_edges = list(G.edges(edge[1]))
            update_neighbors(edge, neighbors_edges)
        else:
            if not update_neighbors(edge, visited[edge]):
                edge = stack.popleft()
                path.appendleft(edge)

    return path


def compute_Euler_circuit_digraph_recursive(adj_list):
    G = nx.DiGraph(adj_list)
    visited = set()
    path = []
    def edge_dfs(edge, parent=-1):
        visited.add(edge)
        for neighbor_edge in G.edges(edge[1]):
            if neighbor_edge not in visited:
                edge_dfs(neighbor_edge, edge)
        path.append(edge)

    edge_dfs(list(G.edges)[0])
    return path[::-1]