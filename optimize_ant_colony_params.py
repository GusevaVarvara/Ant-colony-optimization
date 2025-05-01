import json
import time
import random
import optuna
import networkx as nx

from dynamic_ac_algorithm import run_ant_colony_dynamic


def load_ny_road_graph(file_path: str, edge_fraction: float) -> nx.Graph:
    """
    Loads a part of the New York road graph and scales the edge weights.

    Args:
        file_path (str): Path to the USA-road-d.NY.gr file.
        edge_fraction (float): Fraction of edges to include in the initial graph.

    Returns:
        nx.Graph: The loaded subgraph.
    """
    G = nx.Graph()
    all_edges = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("a"):
                _, u, v, w = line.strip().split()
                weight = int(w) / 100
                all_edges.append((int(u), int(v), weight))

    print(f"Total number of edges in file: {len(all_edges)}")

    cutoff = int(len(all_edges) * edge_fraction)
    base_edges = all_edges[:cutoff]

    for u, v, w in base_edges:
        G.add_edge(u, v, weight=w)

    print(f"Number of edges in subgraph: {len(base_edges)}")
    print(f"Number of nodes in subgraph: {G.number_of_nodes()}")

    return G


def select_connected_nodes(graph: nx.Graph, min_length: float, max_length: float) -> tuple[int, int]:
    """
    Selects a pair of nodes that are connected and have a path length within specified bounds.

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


def select_best_by_length_and_time(trials):
    """
    Selects the best trial: first by minimum path length, then by minimum execution time.

    Args:
        trials (List[optuna.trial.FrozenTrial]): List of trials.

    Returns:
        optuna.trial.FrozenTrial: The best trial found.
    """
    min_length = min(t.values[0] for t in trials)
    best_by_length = [t for t in trials if t.values[0] == min_length]
    return min(best_by_length, key=lambda t: t.values[1])


def optimize_base_params(graph: nx.Graph, start_node: int, end_node: int) -> dict:
    """
    Optimizes the base parameters for the dynamic ant colony algorithm.

    Args:
        graph (nx.Graph): The graph.
        start_node (int): Starting node.
        end_node (int): Ending node.

    Returns:
        dict: Optimized base parameters.
    """
    def objective(trial):
        alpha_start = trial.suggest_float("alpha_start", 0.5, 2.0)
        beta_start = trial.suggest_float("beta_start", 1.0, 3.0)
        min_rate = trial.suggest_float("min_rate", 0.05, 0.3)
        num_ants_start = trial.suggest_int("num_ants_start", 400, 800)

        alpha_end = alpha_start
        beta_end = beta_start
        max_rate = 0.8
        num_ants_end = num_ants_start // 2
        num_iterations = 50
        num_threads = 4

        dynamic_changes = {}

        start_time = time.time()
        _, best_length = run_ant_colony_dynamic(
            graph=graph,
            num_ants_start=num_ants_start,
            num_ants_end=num_ants_end,
            num_iterations=num_iterations,
            start_node=start_node,
            end_node=end_node,
            dynamic_changes=dynamic_changes,
            num_threads=num_threads,
            alpha_start=alpha_start,
            beta_start=beta_start,
            alpha_end=alpha_end,
            beta_end=beta_end,
            min_rate=min_rate,
            max_rate=max_rate,
        )
        elapsed = time.time() - start_time
        print(f"[BASE Trial {trial.number}] Path length: {100 * best_length:.2f}, Time: {elapsed:.2f} seconds")
        return best_length, elapsed

    print("\n=== Multi-objective optimization: base parameters ===\n")
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=15)

    best = select_best_by_length_and_time(study.best_trials)
    print("\n== Best base parameters ==")
    print("  Path length:", 100 * best.values[0])
    print("  Time:", best.values[1])
    print("  Parameters:", best.params)
    return best.params


def optimize_extended_params(graph: nx.Graph, start_node: int, end_node: int, base_params: dict) -> dict:
    """
    Optimizes the extended parameters for the dynamic ant colony algorithm.

    Args:
        graph (nx.Graph): The graph.
        start_node (int): Starting node.
        end_node (int): Ending node.
        base_params (dict): Base parameters.

    Returns:
        dict: Optimized extended parameters.
    """
    def objective(trial):
        alpha_end = trial.suggest_float("alpha_end", 0.5, 2.0)
        beta_end = trial.suggest_float("beta_end", 1.0, 3.0)
        max_rate = trial.suggest_float("max_rate", 0.7, 1.0)
        num_ants_end = trial.suggest_int("num_ants_end", 200, 500)

        alpha_start = base_params["alpha_start"]
        beta_start = base_params["beta_start"]
        min_rate = base_params["min_rate"]
        num_ants_start = base_params["num_ants_start"]

        num_iterations = 50
        num_threads = 4

        dynamic_changes = {}

        start_time = time.time()
        _, best_length = run_ant_colony_dynamic(
            graph=graph,
            num_ants_start=num_ants_start,
            num_ants_end=num_ants_end,
            num_iterations=num_iterations,
            start_node=start_node,
            end_node=end_node,
            dynamic_changes=dynamic_changes,
            num_threads=num_threads,
            alpha_start=alpha_start,
            beta_start=beta_start,
            alpha_end=alpha_end,
            beta_end=beta_end,
            min_rate=min_rate,
            max_rate=max_rate,
        )
        elapsed = time.time() - start_time
        print(f"[EXT Trial {trial.number}] Path length: {100 * best_length:.2f}, Time: {elapsed:.2f} seconds")
        return best_length, elapsed

    print("\n=== Multi-objective optimization: extended parameters ===\n")
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=15)

    best = select_best_by_length_and_time(study.best_trials)
    print("\n== Best extended parameters ==")
    print("  Path length:", 100 * best.values[0])
    print("  Time:", best.values[1])
    print("  Parameters:", best.params)
    return best.params


def save_params_to_json(base: dict, extended: dict, filename: str = "optimized_params.json") -> None:
    """
    Saves the optimized parameters to a JSON file.

    Args:
        base (dict): Base parameters.
        extended (dict): Extended parameters.
        filename (str): Output filename.
    """
    with open(filename, "w") as f:
        json.dump({
            "base_params": base,
            "extended_params": extended
        }, f, indent=4)
    print(f"\nParameters saved to {filename}")


if __name__ == "__main__":
    graph_path = "USA-road-d.NY.gr"
    graph = load_ny_road_graph(graph_path, edge_fraction=1)

    start_node, end_node = select_connected_nodes(graph, min_length=100, max_length=300)
    print(f"Selected nodes: start={start_node}, end={end_node}")

    base = optimize_base_params(graph, start_node, end_node)
    extended = optimize_extended_params(graph, start_node, end_node, base)

    save_params_to_json(base, extended)
