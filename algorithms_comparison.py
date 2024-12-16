import time
import random
import networkx as nx
from dynamic_ac_algorithm import run_ant_colony_dynamic
from heapq import heappop, heappush


def dijkstra(graph, start_node, end_node):
    """
    Finds the shortest path between two nodes in a graph using Dijkstra's algorithm.

    Args:
        graph (any): The graph object with nodes and edges. Edges must have a 'weight' attribute.
        start_node (int): The starting node for the path.
        end_node (int): The destination node for the path.

    Returns:
        float: The length of the shortest path from `start_node` to `end_node`.
    """
    distances = {node: float("inf") for node in graph.nodes}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_distance, current_node = heappop(priority_queue)

        if current_node == end_node:
            break

        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]["weight"]
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heappush(priority_queue, (distance, neighbor))

    return distances[end_node]


def apply_dynamic_changes(graph, changes):
    """
    Applies dynamic changes to the graph, such as updating edge weights or removing edges.

    Args:
        graph (any): The graph object to modify.
        changes (List[Tuple[int, int, Optional[float]]]): A list of changes. Each change is a tuple `(u, v, weight)`:
            - `u` (int): The source node of the edge.
            - `v` (int): The target node of the edge.
            - `weight` (float or None): The new weight of the edge, or `None` to remove the edge.

    Returns:
        None: The graph is modified in place.
    """
    for u, v, weight in changes:
        if weight is None:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)
        else:
            graph.add_edge(u, v, weight=weight)


def generate_random_changes(graph, num_changes_per_iteration, max_weight=10):
    """
    Generates random changes for a graph, simulating a dynamic environment.

    Args:
        graph (any): The graph object on which changes will be based.
        num_changes_per_iteration (int): Number of changes to generate per iteration.
        max_weight (int): The maximum possible weight for an edge.

    Returns:
        Dict[int, List[Tuple[int, int, Optional[float]]]]:
            A dictionary where keys are iteration numbers and values are lists of changes.
            Each change is a tuple `(u, v, weight)` or `(u, v, None)` to remove an edge.
    """
    random.seed(50)
    changes = {}
    for iteration in range(1, 51):
        iteration_changes = []
        for _ in range(num_changes_per_iteration):
            u, v = random.sample(list(graph.nodes), 2)
            if random.random() < 0.8:
                weight = random.randint(1, max_weight)
                iteration_changes.append((u, v, weight))
            else:
                if graph.has_edge(u, v):
                    iteration_changes.append((u, v, None))
        changes[iteration] = iteration_changes
    return changes


def main():
    """
    Executes the main workflow to compare Dijkstra's algorithm and the ant colony algorithm on both static and dynamic graphs.

    Steps:
        1. Create a complete graph with 50 nodes and random edge weights.
        2. Run both algorithms on the static graph and compare results.
        3. Generate random dynamic changes to the graph.
        4. Run both algorithms on the dynamic graph and compare results.

    Returns:
        None: Prints results of the comparisons to the console.
    """
    random.seed(34)
    graph = nx.complete_graph(50)
    for u, v in graph.edges():
        graph[u][v]["weight"] = random.randint(1, 10)

    start_node, end_node = 0, 49

    start_time = time.time()
    ant_paths, ant_length = run_ant_colony_dynamic(
        graph,
        num_ants_start=40,
        num_iterations=50,
        start_node=start_node,
        end_node=end_node,
    )
    ant_time_1 = time.time() - start_time

    print(
        f"\nAnt colony algorithm on static graph: path {ant_paths}, length {ant_length}, time {ant_time_1:.4f} s"
    )

    start_time = time.time()
    dijkstra_length = dijkstra(graph, start_node, end_node)
    dijkstra_time_1 = time.time() - start_time

    print(
        f"Dijkstra's algorithm on static graph: length {dijkstra_length}, total time {dijkstra_time_1:.4f} s"
    )

    dynamic_changes = generate_random_changes(graph, num_changes_per_iteration=5)

    start_time = time.time()
    ant_paths, ant_length = run_ant_colony_dynamic(
        graph,
        num_ants_start=40,
        num_iterations=50,
        start_node=start_node,
        end_node=end_node,
        dynamic_changes=dynamic_changes,
    )
    ant_time_2 = time.time() - start_time

    print(
        f"\nAnt colony algorithm on dynamic graph with 5 changes per iteration: path {ant_paths}, length {ant_length}, time {ant_time_2:.4f} s"
    )

    dijkstra_time_2 = 0
    dijkstra_length = None
    graph_copy = graph.copy()

    for iteration in range(1, max(dynamic_changes.keys()) + 1):
        if iteration in dynamic_changes:
            apply_dynamic_changes(graph_copy, dynamic_changes[iteration])

        start_time = time.time()
        dijkstra_length = dijkstra(graph_copy, start_node, end_node)
        dijkstra_time_2 += time.time() - start_time

    print(
        f"Dijkstra's algorithm on dynamic graph with 5 changes per iteration: length {dijkstra_length}, total time {dijkstra_time_2:.4f} s"
    )


if __name__ == "__main__":
    main()
