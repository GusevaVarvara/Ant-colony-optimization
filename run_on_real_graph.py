import json
import time
import random
import networkx as nx
from collections import defaultdict
from dynamic_ac_algorithm import run_ant_colony_dynamic


def load_params_from_json(filename: str = "optimized_params.json") -> dict:
    """
    Loads optimized parameters from a JSON file.

    Args:
        filename (str): Path to the JSON file.

    Returns:
        dict: Loaded parameters.
    """
    with open(filename, "r") as f:
        return json.load(f)


def load_ny_road_graph(file_path: str, base_fraction: float, dynamic_fraction: float) -> tuple[nx.Graph, dict]:
    """
    Loads a part of the New York road graph and generates dynamic changes.

    Args:
        file_path (str): Path to the USA-road-d.NY.gr file.
        base_fraction (float): Fraction of edges to include in the initial graph.
        dynamic_fraction (float): Fraction of remaining edges used for dynamic changes.

    Returns:
        tuple[nx.Graph, dict]: The initial graph and a dictionary of dynamic changes.
    """
    G = nx.Graph()
    dynamic_changes = defaultdict(list)
    all_edges = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("a"):
                _, u, v, w = line.strip().split()
                weight = int(w) / 100
                all_edges.append((int(u), int(v), weight))

    cutoff = int(len(all_edges) * base_fraction)
    base_edges = all_edges[:cutoff]

    for u, v, w in base_edges:
        G.add_edge(u, v, weight=w)

    print(f"\nInitial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    future_edges = all_edges[cutoff:]
    num_changes_total = int(len(future_edges) * dynamic_fraction)
    batch_size = max(1, num_changes_total // 100)
    selected_for_changes = random.sample(future_edges, num_changes_total)

    step = 0
    for i in range(0, len(selected_for_changes), batch_size):
        batch = selected_for_changes[i:i + batch_size]
        for u, v, w in batch:
            dynamic_changes[step].append((u, v, w))
        step += 1

    print(f"\nTotal number of dynamic steps: {len(dynamic_changes)}")
    average_changes_per_step = int(sum(len(batch) for batch in dynamic_changes.values()) / len(dynamic_changes))
    print(f"Average number of changes per step: {average_changes_per_step}")

    return G, dynamic_changes


def select_connected_nodes(graph: nx.Graph, min_length: float, max_length: float) -> tuple[int, int]:
    """
    Selects a pair of connected nodes with a path length within specified bounds.

    Args:
        graph (nx.Graph): The graph.
        min_length (float): Minimum acceptable path length.
        max_length (float): Maximum acceptable path length.

    Returns:
        tuple[int, int]: A pair of connected nodes.
    """
    nodes = list(graph.nodes)
    while True:
        start = random.choice(nodes)
        end = random.choice(nodes)
        if start != end and nx.has_path(graph, start, end):
            try:
                path_length = nx.dijkstra_path_length(graph, start, end, weight='weight')
                if min_length <= path_length <= max_length:
                    return start, end
            except nx.NetworkXNoPath:
                continue


if __name__ == "__main__":
    graph_path = "USA-road-d.NY.gr"
    graph, dynamic_changes = load_ny_road_graph(graph_path, base_fraction=0.5, dynamic_fraction=0.5)

    params = load_params_from_json()

    start_node, end_node = select_connected_nodes(graph, min_length=100, max_length=300)
    print(f"\nSearching for the shortest path from {start_node} to {end_node}...")

    start_time = time.time()

    paths, length = run_ant_colony_dynamic(
        graph=graph,
        num_ants_start=params["base_params"]["num_ants_start"],
        num_ants_end=params["extended_params"]["num_ants_end"],
        num_iterations=50,
        start_node=start_node,
        end_node=end_node,
        dynamic_changes=dynamic_changes,
        num_threads=4,
        alpha_start=params["base_params"]["alpha_start"],
        beta_start=params["base_params"]["beta_start"],
        alpha_end=params["extended_params"]["alpha_end"],
        beta_end=params["extended_params"]["beta_end"],
        min_rate=params["base_params"]["min_rate"],
        max_rate=params["extended_params"]["max_rate"]
    )

    elapsed_time = time.time() - start_time

    real_length = length * 100

    print(f"\nResult:")
    print(f"  Shortest path length: {real_length:.2f}")
    print(f"  Found paths: {paths}")
    print(f"  Execution time: {elapsed_time:.2f} seconds")
