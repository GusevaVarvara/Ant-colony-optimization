import networkx as nx
import random


ALPHA = 1  # Влияние феромонов
BETA = 2  # Влияние расстояния
EVAPORATION_RATE = 0.5
INITIAL_PHEROMONE = 1.0


def initialize_pheromones(graph):
    """
    Initializes pheromones for all edges in the graph.

    Args:
        graph (any): The graph object representing the network, which must have an 'edges' attribute and edge weights.

    Returns:
        Dict[Tuple[int, int], float]: A dictionary with edges as keys and initial pheromone levels as values.
    """
    pheromones = {}
    for u, v in graph.edges:
        edge = (min(u, v), max(u, v))
        pheromones[edge] = INITIAL_PHEROMONE
    return pheromones


def calculate_transition_probabilities(
    current_node, unvisited_nodes, graph, pheromones
):
    """
    Calculates transition probabilities for an ant to move from the current node to each of the unvisited nodes.

    Args:
        current_node (int): The node where the ant is currently located.
        unvisited_nodes (Set[int]): Nodes that the ant has not yet visited.
        graph (any): The graph in which the ant is moving.
        pheromones (Dict[Tuple[int, int], float]): Pheromone levels for each edge.

    Returns:
        List[Tuple[int, float]]: A list of tuples where each tuple contains a node and the probability of moving to that node.
    """
    probabilities = []
    total = 0

    for node in unvisited_nodes:
        if graph.has_edge(current_node, node):
            edge = (min(current_node, node), max(current_node, node))
            pheromone = pheromones.get(edge, 0)
            distance = graph[current_node][node]["weight"]
            total += (pheromone**ALPHA) * ((1 / distance) ** BETA)

    if total == 0:
        return [(node, 1 / len(unvisited_nodes)) for node in unvisited_nodes]

    for node in unvisited_nodes:
        if graph.has_edge(current_node, node):
            edge = (min(current_node, node), max(current_node, node))
            pheromone = pheromones.get(edge, 0)
            distance = graph[current_node][node]["weight"]
            probability = (pheromone**ALPHA) * ((1 / distance) ** BETA) / total
            probabilities.append((node, probability))

    return probabilities


def move_ant(current_node, unvisited_nodes, graph, pheromones):
    """
    Determines the next node for an ant to move to based on transition probabilities.

    Args:
        current_node (int): The current position of the ant.
        unvisited_nodes (Set[int]): The set of nodes not yet visited by the ant.
        graph (any): The graph in which the ant is moving.
        pheromones (Dict[Tuple[int, int], float]): Pheromone levels for each edge.

    Returns:
        int or None: The next node to move to, or None if no valid moves are available.
    """
    available_nodes = [
        node for node in unvisited_nodes if graph.has_edge(current_node, node)
    ]
    if not available_nodes:
        return None

    probabilities = calculate_transition_probabilities(
        current_node, available_nodes, graph, pheromones
    )
    nodes, probs = zip(*probabilities)
    next_node = random.choices(nodes, weights=probs, k=1)[0]

    return next_node


def update_pheromones(pheromones, paths, graph):
    """
    Updates the pheromone levels on all edges based on the paths taken by ants.

    Args:
        pheromones (Dict[Tuple[int, int], float]): Current pheromone levels for edges.
        paths (List[List[int]]): A list of paths taken by ants, where each path is a list of nodes.
        graph (any): The graph in which the ant is moving.

    Returns:
        Dict[Tuple[int, int], float]: Updated pheromone levels.
    """
    for edge in pheromones:
        pheromones[edge] *= 1 - EVAPORATION_RATE

    for path in paths:
        for i in range(len(path) - 1):
            edge = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            distance = graph[path[i]][path[i + 1]]["weight"]
            pheromones[edge] += 1 / distance
    return pheromones


def run_ant_colony(graph, num_ants, num_iterations, start_node):
    """
    Runs the ant colony optimization algorithm to find the shortest path in the graph.

    Args:
        graph (any): The graph representing the network.
        num_ants (int): The number of ants to use in each iteration.
        num_iterations (int): The number of iterations to perform.
        start_node (int): The starting node for all ants.

    Returns:
        Tuple[List[Tuple[int]], float]: The best paths found and their length.
    """
    pheromones = initialize_pheromones(graph)
    best_paths_set = set()
    best_path_length = float("inf")

    for _ in range(num_iterations):
        all_paths = []
        for _ in range(num_ants):
            path = [start_node]
            unvisited_nodes = set(graph.nodes()) - {start_node}
            current_node = start_node

            while unvisited_nodes:
                next_node = move_ant(current_node, unvisited_nodes, graph, pheromones)
                if next_node is None:
                    break
                path.append(next_node)
                unvisited_nodes.remove(next_node)
                current_node = next_node

            all_paths.append(path)

        pheromones = update_pheromones(pheromones, all_paths, graph)

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

    def create_graph():
        """
        Creates a predefined graph for testing the ant colony optimization algorithm.

        Returns:
            nx.Graph: The graph with nodes and weighted edges.
        """
        graph = nx.Graph()
        nodes = [1, 2, 3, 4, 5]
        edges = [
            (1, 2, 2),
            (1, 3, 5),
            (2, 3, 3),
            (2, 4, 4),
            (3, 4, 1),
            (4, 5, 2),
            (3, 5, 6),
        ]
        graph.add_nodes_from(nodes)
        graph.add_weighted_edges_from(edges)
        return graph

    graph = create_graph()
    num_ants = 20
    num_iterations = 50
    start_node = 1

    best_paths, best_path_length = run_ant_colony(
        graph, num_ants, num_iterations, start_node
    )
    print("Best paths found:", best_paths)
    print("Shortest path length:", best_path_length)
