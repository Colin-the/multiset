from itertools import combinations_with_replacement, permutations
from collections import deque, defaultdict
from math import comb
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter

def visualize_merge_tree(tree_edges, root):
    G = nx.DiGraph()
    for parent, child in tree_edges:
        G.add_edge(parent, child)

    # Compute BFS levels 
    levels = dict(nx.single_source_shortest_path_length(G, root))
    max_level = max(levels.values())

    # Group nodes by level
    level_nodes = {lvl: [] for lvl in range(max_level + 1)}
    for node, lvl in levels.items():
        level_nodes[lvl].append(node)

    # Assign x spaced evenly within each level, y = -level
    pos = {}
    for lvl, nodes in level_nodes.items():
        n = len(nodes)
        if n == 1:
            xs = [0.5]
        else:
            xs = [i / (n - 1) for i in range(n)]
        for x, node in zip(xs, nodes):
            pos[node] = (x, -lvl)

    # Draw the tree
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos,
            with_labels=True,
            node_size=600,
            font_size=8,
            font_weight='bold',
            arrows=False)
    plt.title("Top-Down Merge Tree of Content Vectors")
    plt.tight_layout()
    plt.savefig("chart")
    plt.show()
    
# Helper fuction to generate all possible contents for a given k and m
def generate_contents(k, m):
    if k == 1:
        return [(m,)]
    result = []
    for i in range(m + 1):
        for tail in generate_contents(k - 1, m - i):
            result.append((i,) + tail)
    return result

# Helper function to generate all permutations with fixed content
def multiset_permutations(content):
    symbols = []
    for val, count in enumerate(content):
        symbols.extend([str(val)] * count)
    return sorted(set(permutations(symbols)))

def build_fixed_content_uc(content):
    perms = multiset_permutations(content)
    used = set()
    uc = []
    window_size = sum(content)
    for p in perms:
        s = ''.join(p)
        if s in used:
            continue
        for i in range(window_size):
            rotation = s[i:] + s[:i]
            if rotation not in used:
                uc.append(rotation)
                used.add(rotation)
                break

    merged = uc[0]
    for next_str in uc[1:]:
        merged += next_str[-1]
    return merged

# Build merge graph based on 1-count transfers
# Vertices are content-vectors and edges between if differ by one symbol
def get_neighbors(content):
    neighbors = []
    k = len(content)
    for i in range(k):
        if content[i] == 0:
            continue
        for j in range(k):
            if i == j:
                continue
            new = list(content)
            new[i] -= 1
            new[j] += 1
            neighbors.append(tuple(new))
    return neighbors

# BFS to build a spanning tree
def build_merge_tree(contents, root):
    visited = set()
    queue = deque([root])
    tree_edges = []
    visited.add(root)
    while queue:
        current = queue.popleft()
        for neighbor in get_neighbors(current):
            if neighbor in contents and neighbor not in visited:
                visited.add(neighbor)
                tree_edges.append((current, neighbor))
                queue.append(neighbor)
    return tree_edges

# Helper function to find a shared substring of length k-2
def find_shared_substring(c1, c2, k_minus_2):
    substrs1 = set(c1[i:i+k_minus_2] for i in range(len(c1)-k_minus_2+1))
    substrs2 = set(c2[i:i+k_minus_2] for i in range(len(c2)-k_minus_2+1))
    shared = substrs1 & substrs2
    return next(iter(shared)) if shared else None

# Splice two cycles at a shared substring
def splice_cycles(cycle1, cycle2, overlap):
    i = cycle1.find(overlap)
    j = cycle2.find(overlap)
    if i == -1 or j == -1:
        raise ValueError("Overlap not found in both cycles")
    # Cut at first occurrence
    c1 = cycle1[i:] + cycle1[:i]
    c2 = cycle2[j+len(overlap):] + cycle2[:j+len(overlap)]
    return c1 + c2

def build_universal_multiset_cycle(k, m, return_tree=False):
    contents = generate_contents(k, m)
    print(contents)

    fixed_cycles = {c: build_fixed_content_uc(c) for c in contents}
    print(fixed_cycles)

    root = contents[0]
    tree = build_merge_tree(set(contents), root)

    # Merge in tree order
    final_cycle = fixed_cycles[root]
    for parent, child in tree:
        s1 = final_cycle
        s2 = fixed_cycles[child]
        overlap = find_shared_substring(s1, s2, k_minus_2=k-2)
        if not overlap:
            raise ValueError(f"No overlap found between {parent} and {child}")
        final_cycle = splice_cycles(s1, s2, overlap)

    if return_tree:
        return final_cycle, tree, root
    return final_cycle


if __name__ == '__main__':
    k, m = 4, 5
    uc, tree, root = build_universal_multiset_cycle(k, m, return_tree=True)
    print("UC:", uc)
    print("Len:", len(uc))

    visualize_merge_tree(tree, root)

