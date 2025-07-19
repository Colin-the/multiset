#!/usr/bin/env python3
"""
multiset_uc.py

Generate a universal cycle for k-multisets summing to m, under the
“shift one unit between adjacent positions + lexicographic-max rotation” rule.

Usage:
    python multiset_uc.py [k] [m]

Defaults:
    k = 5, m = 5
"""
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from typing import List, Tuple, Dict

def get_representative(seq: List[int]) -> Tuple[int, ...]:
    """
    Return the lexicographically greatest rotation of seq.
    """
    k = len(seq)
    # Build all rotations
    rotations = (tuple(seq[i:] + seq[:i]) for i in range(k))
    # Return the maximum one
    return max(rotations)

def build_content_tree(k: int, m: int) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    """
    Build a spanning tree on the rotation-representatives graph.
    Nodes: all k-tuples of non-negative ints summing to m, in their
           lexicographically greatest rotation form.
    Edges: defined by shifting 1 unit from any positive position i
           to i+1 or i-1 (mod k), then re-normalizing.
    Returns a dict: parent_node -> [child_node, ...]
    """
    start = (m,) + (0,) * (k - 1)
    root = get_representative(list(start))

    tree: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = defaultdict(list)
    visited = {root}
    queue = deque([root])

    while queue:
        node = queue.popleft()
        arr = list(node)
        for i in range(k):
            if arr[i] <= 0:
                continue    # nothing to shift
            for delta in (+1, -1):
                j = (i + delta) % k
                new_arr = arr.copy()
                new_arr[i] -= 1
                new_arr[j] += 1
                child = get_representative(new_arr)
                if child not in visited:
                    visited.add(child)
                    tree[node].append(child)
                    queue.append(child)

    return tree

def postorder_traversal(tree: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
                        root: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Perform a depth-first post-order traversal of the tree.
    Returns the list of nodes in the order visited, reversed so that
    root appears first, leaves last.
    """
    sequence: List[Tuple[int, ...]] = []
    def dfs(u: Tuple[int, ...]):
        for v in tree.get(u, []):
            dfs(v)
        sequence.append(u)

    dfs(root)
    return sequence[::-1]  # reverse so we get root → ... → leaves

def plot_tree(tree: Dict[Tuple[int, ...], List[Tuple[int, ...]]], root: Tuple[int, ...], filename="tree.png"):
    """
    Draws the tree in a hierarchical layout:
    - Root at the top (level 0)
    - Children one level down, evenly spaced horizontally.
    """
    # Build directed graph
    G = nx.DiGraph()
    for parent, children in tree.items():
        for child in children:
            G.add_edge(parent, child)

    # Compute depth (level) of each node via BFS
    depth = {root: 0}
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in tree.get(u, []):
            depth[v] = depth[u] + 1
            queue.append(v)

    # Group nodes by level
    levels = defaultdict(list)
    for node, d in depth.items():
        levels[d].append(node)

    # Assign positions: for each level, spread nodes across [-1,1] on x-axis
    pos = {}
    max_width = max(len(nodes) for nodes in levels.values())
    for d, nodes in levels.items():
        count = len(nodes)
        # horizontal spacing
        xs = [i/(count-1) * 2 - 1 if count > 1 else 0 for i in range(count)]
        y = -d  # root at y=0, children downward
        for x, node in zip(xs, nodes):
            pos[node] = (x, y)

    # Draw
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, labels={n: str(n) for n in G.nodes()},
            node_size=800, font_size=8, arrowsize=12)
    plt.title("Hierarchical Tree of Rotation Representatives")
    plt.axis('off')
    plt.savefig(filename)
    plt.show()

def main():
    import sys
    # Command-line args: k, m
    if len(sys.argv) >= 3:
        k = int(sys.argv[1])
        m = int(sys.argv[2])
    else:
        k, m = 5, 5

    print(f"Building tree for k={k}, m={m}...", file=sys.stderr)
    tree = build_content_tree(k, m)

    start = (m,) + (0,) * (k - 1)
    root = get_representative(list(start))

    print("Traversing tree in post-order to get universal cycle...", file=sys.stderr)
    cycle = postorder_traversal(tree, root)

    print("\n# Universal cycle sequence of representatives:")
    for tup in cycle:
        print(tup)

    # Optionally, produce a single cyclic string if you want:
    # Here we just flatten the tuples with a separator
    print("\n# Flattened cycle (strings):")
    flat = " → ".join("".join(str(x) for x in t) for t in cycle)
    print(flat)
    plot_tree(tree, root)

if __name__ == "__main__":
    main()
