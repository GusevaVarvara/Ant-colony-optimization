import json
import time
import networkx as nx
import optuna
from dynamic_ac_algorithm import run_ant_colony_dynamic


def create_test_graph() -> nx.Graph:
    """
    Creates a small test graph.

    Returns:
        nx.Graph: A weighted undirected graph.
    """
    graph = nx.Graph()
    edges = [
        (1, 2, 2),
        (1, 3, 5),
        (2, 3, 3),
        (2, 4, 4),
        (3, 4, 1),
        (4, 5, 2),
        (3, 5, 6),
    ]
    graph.add_weighted_edges_from(edges)
    return graph


def select_best_by_length_and_time(trials):
    """
    Selects the best trial based on the minimum path length and execution time.

    Args:
        trials (list): List of Optuna trials.

    Returns:
        optuna.trial.FrozenTrial: The best trial.
    """
    min_length = min(t.values[0] for t in trials)
    best_by_length = [t for t in trials if t.values[0] == min_length]
    return min(best_by_length, key=lambda t: t.values[1])


def optimize_base_params():
    """
    Stage 1: Multi-objective optimization of the base parameters.

    Returns:
        dict: Best found base parameters.
    """
    def objective(trial):
        alpha_start = trial.suggest_float("alpha_start", 0.5, 3.0)
        beta_start = trial.suggest_float("beta_start", 0.5, 3.0)
        min_rate = trial.suggest_float("min_rate", 0.05, 0.5)
        num_ants_start = trial.suggest_int("num_ants_start", 10, 100)

        alpha_end = alpha_start
        beta_end = beta_start
        max_rate = 0.6
        num_ants_end = num_ants_start // 2
        num_iterations = 30
        num_threads = 4

        graph = create_test_graph()
        dynamic_changes = {10: [(1, 4, 7)], 20: [(2, 3, None)]}

        start_time = time.time()
        _, best_length = run_ant_colony_dynamic(
            graph=graph,
            num_ants_start=num_ants_start,
            num_ants_end=num_ants_end,
            num_iterations=num_iterations,
            start_node=1,
            end_node=5,
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
        print(f"[BASE Trial {trial.number}] Length: {best_length:.2f}, Time: {elapsed:.2f}s")
        return best_length, elapsed

    print("\n=== Base parameters optimization ===\n")
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=30)

    best = select_best_by_length_and_time(study.best_trials)
    print("\n== Best base parameters ==")
    print(f"  Path length: {best.values[0]}")
    print(f"  Time: {best.values[1]}")
    print(f"  Params: {best.params}")
    return best.params


def optimize_extended_params(base_params):
    """
    Stage 2: Multi-objective optimization of the extended parameters.

    Args:
        base_params (dict): Optimized base parameters.

    Returns:
        dict: Best found extended parameters.
    """
    def objective(trial):
        alpha_end = trial.suggest_float("alpha_end", 0.5, 3.0)
        beta_end = trial.suggest_float("beta_end", 0.5, 3.0)
        max_rate = trial.suggest_float("max_rate", 0.5, 1.0)
        num_ants_end = trial.suggest_int("num_ants_end", 5, 50)

        alpha_start = base_params["alpha_start"]
        beta_start = base_params["beta_start"]
        min_rate = base_params["min_rate"]
        num_ants_start = base_params["num_ants_start"]

        num_iterations = 30
        num_threads = 4

        graph = create_test_graph()
        dynamic_changes = {10: [(1, 4, 7)], 20: [(2, 3, None)]}

        start_time = time.time()
        _, best_length = run_ant_colony_dynamic(
            graph=graph,
            num_ants_start=num_ants_start,
            num_ants_end=num_ants_end,
            num_iterations=num_iterations,
            start_node=1,
            end_node=5,
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
        print(f"[EXT Trial {trial.number}] Length: {best_length:.2f}, Time: {elapsed:.2f}s")
        return best_length, elapsed

    print("\n=== Extended parameters optimization ===\n")
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=30)

    best = select_best_by_length_and_time(study.best_trials)
    print("\n== Best extended parameters ==")
    print(f"  Path length: {best.values[0]}")
    print(f"  Time: {best.values[1]}")
    print(f"  Params: {best.params}")
    return best.params


def save_params_to_json(base: dict, extended: dict, filename: str = "optimized_params.json"):
    """
    Saves the optimized parameters to a JSON file.

    Args:
        base (dict): Base parameters.
        extended (dict): Extended parameters.
        filename (str): File name to save the parameters.
    """
    with open(filename, "w") as f:
        json.dump({
            "base_params": base,
            "extended_params": extended
        }, f, indent=4)
    print(f"\nParameters saved to {filename}")


if __name__ == "__main__":
    base = optimize_base_params()
    extended = optimize_extended_params(base)
    save_params_to_json(base, extended)
