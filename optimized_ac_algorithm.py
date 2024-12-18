from concurrent.futures import ThreadPoolExecutor
import os
import random
import math
import time


def initialize_pheromones(graph, k):
    """
    Initialize pheromone levels for all edges in the graph.

    Args:
        graph (any): The graph object with weighted edges.
        k (float): Constant used to initialize pheromone levels inversely proportional to edge weights.

    Returns:
        dict[tuple[int, int], float]: Dictionary mapping edges to initial pheromone levels.
    """
    pheromones = {}
    for u, v in graph.edges:
        edge = (min(u, v), max(u, v))
        distance = graph[u][v]["weight"]
        pheromones[edge] = k / distance
    return pheromones


def update_pheromones(
    pheromones,
    paths,
    graph,
    iteration,
    total_iterations,
    min_rate,
    max_rate,
    max_pheromone=100,
):
    """
    Updates the pheromone levels based on evaporation, path density and contributions.

    Args:
        pheromones (Dict[Tuple[int, int], float]): Current pheromone levels for each edge.
        paths (List[List[int]]): List of paths taken by ants during the current iteration.
        graph (any): The graph representing the environment in which ants are moving.
        iteration (int): The current iteration number.
        total_iterations (int): The total number of iterations.
        min_rate (float): The minimum rate of evaporation.
        max_rate (float): The maximum rate of evaporation.
        max_pheromone (float, optional): The maximum pheromone level for an edge (default is 100).

    Returns:
        Dict[Tuple[int, int], float]: The updated pheromone levels for all edges.
    """
    evaporation_rate = min_rate + (max_rate - min_rate) * math.exp(
        -iteration / total_iterations
    )

    for edge in pheromones:
        pheromones[edge] *= 1 - evaporation_rate
        pheromones[edge] = max(pheromones[edge], 1e-5)

    path_lengths = []
    edge_usage = {edge: 0 for edge in pheromones}

    for path in paths:
        length = sum(
            graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
        )
        path_lengths.append(length)
        for i in range(len(path) - 1):
            edge = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            edge_usage[edge] += 1

    avg_length = sum(path_lengths) / len(path_lengths) if path_lengths else 1

    for path in paths:
        path_length = sum(
            graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
        )
        contribution = math.log(1 + avg_length / path_length) * 0.125
        for i in range(len(path) - 1):
            edge = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            density_factor = edge_usage[edge] / len(paths)
            pheromones[edge] += contribution / (1 + density_factor)
            pheromones[edge] = min(pheromones[edge], max_pheromone)

    return pheromones


def adjust_parameters(
    iteration, max_iterations, alpha_start, beta_start, alpha_end, beta_end
):
    """
    Smoothly adjusts the alpha and beta parameters over iterations.

    Args:
        iteration (int): The current iteration number.
        max_iterations (int): The total number of iterations.
        alpha_start (float): The initial value of alpha.
        beta_start (float): The initial value of beta.
        alpha_end (float): The final value of alpha.
        beta_end (float): The final value of beta.

    Returns:
        Tuple[float, float]: The adjusted values of alpha and beta.
    """
    alpha = alpha_start + (alpha_end - alpha_start) * (iteration / max_iterations)
    beta = beta_start + (beta_end - beta_start) * (iteration / max_iterations)
    return alpha, beta


def simulate_ant(start_node, end_node, graph, pheromones, alpha, beta):
    """
    Simulates an ant's movement from the start node to the end node.

    Args:
        start_node (int): The starting node for the ant.
        end_node (int): The destination node for the ant.
        graph (any): The graph in which the ant moves.
        pheromones (Dict[Tuple[int, int], float]): The pheromone levels for each edge.
        alpha (float): The influence of pheromone on decision-making.
        beta (float): The influence of distance on decision-making.

    Returns:
        Optional[List[int]]: The path taken by the ant, or None if no valid path is found.
    """
    path = [start_node]
    current_node = start_node
    visited_nodes = {start_node}

    while current_node != end_node:
        next_node = move_ant(
            current_node, visited_nodes, graph, pheromones, alpha, beta
        )
        if next_node is None:
            break
        path.append(next_node)
        visited_nodes.add(next_node)
        current_node = next_node

    return path if path[-1] == end_node else None


def move_ant(current_node, visited_nodes, graph, pheromones, alpha, beta):
    """
    Chooses the next node for the ant to visit based on pheromone levels and distance.

    Args:
        current_node (int): The current node the ant is at.
        visited_nodes (set): A set of nodes that the ant has already visited.
        graph (any): The graph in which the ant is moving.
        pheromones (Dict[Tuple[int, int], float]): The pheromone levels for each edge.
        alpha (float): The influence of pheromone on decision-making.
        beta (float): The influence of distance on decision-making.

    Returns:
        Optional[int]: The next node for the ant to visit, or None if no valid node is available.
    """
    unvisited_nodes = set(graph.neighbors(current_node)) - visited_nodes
    if not unvisited_nodes:
        return None

    if random.random() < 0.2:
        return random.choice(list(unvisited_nodes))

    probabilities = calculate_transition_probabilities(
        current_node, unvisited_nodes, graph, pheromones, alpha, beta
    )
    nodes, probs = zip(*probabilities)
    return random.choices(nodes, weights=probs, k=1)[0]


def calculate_transition_probabilities(
    current_node, unvisited_nodes, graph, pheromones, alpha, beta
):
    """
    Calculates the transition probabilities to unvisited neighboring nodes.

    Args:
        current_node (int): The current node the ant is at.
        unvisited_nodes (set): The set of unvisited neighboring nodes.
        graph (any): The graph in which the ant moves.
        pheromones (Dict[Tuple[int, int], float]): The pheromone levels for each edge.
        alpha (float): The influence of pheromone on decision-making.
        beta (float): The influence of distance on decision-making.

    Returns:
        List[Tuple[int, float]]: A list of tuples containing the node and the corresponding transition probability.
    """
    probabilities = []
    total = 0

    for node in unvisited_nodes:
        edge = (min(current_node, node), max(current_node, node))
        pheromone = pheromones.get(edge, 0)
        distance = graph[current_node][node]["weight"]
        total += (pheromone**alpha) * ((1 / distance) ** beta)

    for node in unvisited_nodes:
        edge = (min(current_node, node), max(current_node, node))
        pheromone = pheromones.get(edge, 0)
        distance = graph[current_node][node]["weight"]
        probability = (pheromone**alpha) * ((1 / distance) ** beta) / total
        probabilities.append((node, probability))

    return probabilities


def run_ant_colony(
    graph,
    num_ants_start,
    num_ants_end,
    num_iterations,
    start_node,
    end_node,
    num_threads=None,
    k=10,
    min_rate=0.1,
    max_rate=0.5,
    alpha_start=0.5,
    beta_start=5,
    alpha_end=2,
    beta_end=1,
):
    """
    Runs the ant colony optimization algorithm.

    Args:
        graph (any): The graph representing the network.
        num_ants_start (int): Initial number of ants per iteration.
        num_ants_end (int): Final number of ants per iteration.
        num_iterations (int): Number of iterations to perform.
        start_node (int): Starting node for all ants.
        end_node (int): Target node for all ants.
        num_threads (int | None): Number of threads to use for parallel execution.
        k (float): Constant for initial pheromone level calculation.
        min_rate (float): Minimum pheromone evaporation rate.
        max_rate (float): Maximum pheromone evaporation rate.
        alpha_start (float): Initial pheromone influence factor.
        beta_start (float): Initial distance influence factor.
        alpha_end (float): Final pheromone influence factor.
        beta_end (float): Final distance influence factor.

    Returns:
        tuple[list[list[int]], float]: The best paths found and their length.
    """
    pheromones = initialize_pheromones(graph, k)
    best_paths_set = set()
    best_path_length = float("inf")

    if not num_threads:
        num_threads = min(os.cpu_count() // 2, len(graph.nodes))

    for iteration in range(num_iterations):
        num_ants = (
            num_ants_end
            + (num_ants_start - num_ants_end)
            * (num_iterations - iteration)
            / num_iterations
        )
        num_ants = int(num_ants)

        alpha, beta = adjust_parameters(
            iteration, num_iterations, alpha_start, beta_start, alpha_end, beta_end
        )
        all_paths = []
        chunk_size = max(1, num_ants // num_threads)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for _ in range(num_threads):
                futures.append(
                    executor.submit(
                        lambda: [
                            simulate_ant(
                                start_node, end_node, graph, pheromones, alpha, beta
                            )
                            for _ in range(chunk_size)
                        ]
                    )
                )

            for future in futures:
                all_paths.extend(filter(None, future.result()))

        pheromones = update_pheromones(
            pheromones, all_paths, graph, iteration, num_iterations, min_rate, max_rate
        )

        for path in all_paths:
            length = sum(
                graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
            )
            path_tuple = tuple(path)
            if length < best_path_length:
                best_paths_set = {path_tuple}
                best_path_length = length
            elif length == best_path_length:
                best_paths_set.add(path_tuple)

    return list(best_paths_set), best_path_length


if __name__ == "__main__":
    import networkx as nx

    def create_test_graph():
        """
        Creates a predefined graph for testing the ant colony optimization algorithm.

        Returns:
            nx.Graph: The graph with nodes and weighted edges.
        """
        graph = nx.Graph()

        nodes = range(1, 21)
        graph.add_nodes_from(nodes)

        edges = [
            (1, 2, 2),
            (1, 3, 5),
            (1, 4, 3),
            (1, 5, 7),
            (2, 3, 3),
            (2, 6, 6),
            (3, 4, 1),
            (3, 7, 4),
            (4, 5, 2),
            (4, 8, 3),
            (5, 9, 4),
            (6, 7, 2),
            (6, 10, 5),
            (7, 8, 3),
            (7, 11, 1),
            (8, 9, 2),
            (8, 12, 6),
            (9, 13, 3),
            (10, 11, 4),
            (10, 14, 2),
            (11, 12, 5),
            (11, 15, 1),
            (12, 13, 4),
            (12, 16, 3),
            (13, 17, 2),
            (14, 15, 3),
            (14, 18, 6),
            (15, 16, 2),
            (15, 19, 4),
            (16, 17, 5),
            (16, 20, 1),
            (17, 20, 3),
            (18, 19, 2),
            (19, 20, 4),
        ]
        graph.add_weighted_edges_from(edges)

        return graph

    test_graph = create_test_graph()

    num_ants_start = 80
    num_ants_end = 20
    num_iterations = 50
    start_node = 1
    end_node = 20
    num_threads = 2

    start_time = time.time()
    best_paths, best_path_length = run_ant_colony(
        test_graph,
        num_ants_start,
        num_ants_end,
        num_iterations,
        start_node,
        end_node,
        num_threads,
    )
    ant_time = time.time() - start_time

    print("Best paths found:", best_paths)
    print("Shortest path length:", best_path_length)
    print("Time:", ant_time)
