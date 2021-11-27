import networkx as nx


def compute_Euler_circuit_digraph(adj_list):
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
