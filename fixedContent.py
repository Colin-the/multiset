from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Set, Tuple
from typing import List, Dict, Set
from collections import defaultdict, deque
from collections import defaultdict

def generate_strings(k: int, m: int) -> List[List[int]]:
    """
    Generate all strings of length k using non-negative integers such that
    the sum of the symbols is exactly m, and only include unique strings
    under rotation, keeping the lexicographically greatest string in each class.
    """
    result = set()

    def is_max_rotation(seq: List[int]) -> bool:
        """
        Return True if seq is the lexicographically greatest rotation of itself.
        """
        return seq == max(seq[i:] + seq[:i] for i in range(len(seq)))

    def backtrack(current: List[int], remaining: int):
        if len(current) == k:
            if remaining == 0 and is_max_rotation(current):
                result.add(tuple(current))
            return

        for i in range(remaining + 1):
            current.append(i)
            backtrack(current, remaining - i)
            current.pop()

    backtrack([], m)
    return [list(t) for t in sorted(result, reverse=True)]

def shared_k_minus_2_consecutive(a: List[int], b: List[int], k: int) -> bool:
    n = len(a)
    if k - 2 <= 0:
        return True

    a_double = a + a
    b_double = b + b
    a_blocks = set()
    b_blocks = set()

    for start in range(n):
        block_a = tuple(a_double[start:start + k - 2])
        a_blocks.add(block_a)
        block_b = tuple(b_double[start:start + k - 2])
        b_blocks.add(block_b)

    return not a_blocks.isdisjoint(b_blocks)

def generate_rotation_tree(strings: List[List[int]], k: int) -> Dict[tuple, Set[tuple]]:
    graph = defaultdict(set)
    tuples = [tuple(s) for s in strings]
    n = len(tuples)

    for i in range(n):
        for j in range(i + 1, n):
            if shared_k_minus_2_consecutive(list(tuples[i]), list(tuples[j]), k):
                graph[tuples[i]].add(tuples[j])
                graph[tuples[j]].add(tuples[i])

    return graph

def group_level_key(node: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Create a grouping key based on the two highest values in the tuple.
    This helps group nodes with shared content like 4-1-*-*-*.
    """
    sorted_vals = sorted(node, reverse=True)
    return tuple(sorted_vals[:2])  # use top 2 values

def visualize_grouped_tree(graph: Dict[Tuple[int, ...], Set[Tuple[int, ...]]]):
    """
    Visualize the graph with nodes grouped by content similarity.
    Nodes with similar high values (like 4,1,...) are on the same level.
    """
    G = nx.Graph()
    for node, neighbors in graph.items():
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # Compute levels based on grouping key
    group_levels = defaultdict(list)
    for node in G.nodes:
        group_levels[group_level_key(node)].append(node)

    # Assign positions
    pos = {}
    for level, (group, nodes) in enumerate(sorted(group_levels.items(), reverse=True)):
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (i, -level)

    plt.figure(figsize=(14, 8))
    nx.draw(G, pos, with_labels=True, labels={n: str(n) for n in G.nodes()},
            node_size=500, node_color="lightblue", font_size=8)
    plt.title("Grouped Rotation Tree Visualization")
    plt.axis("off")
    plt.show()

def get_representative(seq: List[int]) -> Tuple[int, ...]:
    """Returns lexicographically greatest rotation of seq."""
    rotations = [seq[i:] + seq[:i] for i in range(len(seq))]
    max_rot = max(rotations)
    return tuple(max_rot)

def traverse_graph(start: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Traverses the entire graph using unit shifts between adjacent positions.
    Returns nodes in BFS order."""
    visited_order = []
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        visited_order.append(node)  # Record node when processed
        node_list = list(node)
        k = len(node_list)
        
        for i in range(k):
            if node_list[i] > 0:
                # Shift clockwise (i -> i+1)
                new_list1 = node_list.copy()
                new_list1[i] -= 1
                new_list1[(i + 1) % k] += 1
                rep1 = get_representative(new_list1)
                
                # Shift counter-clockwise (i -> i-1)
                new_list2 = node_list.copy()
                new_list2[i] -= 1
                new_list2[(i - 1) % k] += 1
                rep2 = get_representative(new_list2)
                
                if rep1 not in visited:
                    visited.add(rep1)
                    queue.append(rep1)
                if rep2 not in visited:
                    visited.add(rep2)
                    queue.append(rep2)
                    
    return visited_order

def compare_tuple_list(visited: List[Tuple[int, ...]], strings: List[List[int]]):
    """
    Compares visited (list of tuples) against strings (list of lists).
    Prints any elements in visited that are not found in strings (ignoring order).
    """
    set_visited = set(visited)
    set_strings = set(tuple(s) for s in strings)

    missing = set_visited - set_strings

    if not missing:
        print("All visited elements are present in the strings list.")
    else:
        print("The following visited elements are NOT found in the strings list:")
        for item in sorted(missing):
            print(item)


k = 5
m = 5
strings = generate_strings(k, m)
print("Total number of string: ",len(strings))
# for s in strings:
#     print(s)
graph = generate_rotation_tree(strings, k)

# Print graph edges:
# for node, neighbors in graph.items():
#     print(f"{node} -> {sorted(neighbors)}")
#print(graph)
visualize_grouped_tree(graph)

# Example usage:
start_node = (5, 0, 0, 0, 0)
all_nodes = traverse_graph(start_node)
print("Visited: ",all_nodes)
print("Strings: ",strings)
compare_tuple_list(all_nodes, strings)
print(f"Total nodes reached: {len(all_nodes)}")
