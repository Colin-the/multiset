from itertools import combinations_with_replacement, permutations
from itertools import combinations
from collections import deque, defaultdict
from math import comb
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter
import concatenationUC

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
# where k is the number of elements and m is the total number of elements
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

def combinationUtil(ind, r, data, result, arr):
    n = len(arr)

    # If size of current combination is r
    if len(data) == r:
        result.append(data.copy())
        return

    # Replace index with all possible elements
    for i in range(ind, n):

        # Current element is included
        data.append(arr[i])

        # Recur for next elements
        combinationUtil(i + 1, r, data, result, arr)

        # Backtrack to find other combinations
        data.pop()

# Function to find all combinations of size r
# in an array of size n
def findCombination(arr, r):
    n = len(arr)

    # to store the result
    result = []

    # Temporary array to store current combination
    data = []
    combinationUtil(0, r, data, result, arr)
    return result

def findCommonSubstring(a, b, size):
    # edge cases
    if size <= 0:
        return True   # "empty" substring always matches
    if size > len(a) or size > len(b):
        return False  # can't have a length-size substring
    
    # make them “circular” by doubling
    a2 = a + a
    b2 = b + b
    
    # collect all size-length substrings of each, starting only in the valid window
    a_subs = { a2[i:i+size] for i in range(len(a)) }
    b_subs = { b2[j:j+size] for j in range(len(b)) }
    
    # if any overlap, return True
    return not a_subs.isdisjoint(b_subs)

def findCommonSubSet(a, b, size):
    # edge cases
    if size <= 0:
        return {()}   # the “empty” subset always matches
    if size > len(a) or size > len(b):
        return set()  # impossible to pick size items

    # generate all unique k-sized subsets (as sorted tuples) from each string
    subs_a = {
        tuple(sorted(chars))
        for chars in combinations(a, size)
    }
    subs_b = {
        tuple(sorted(chars))
        for chars in combinations(b, size)
    }

    # return the intersection
    return subs_a & subs_b

def build_universal_multiset_cycle(k, m, return_tree=False):
    contents = generate_contents(k, m)
    #print(contents)
    canonical = { "".join(str(d) for d in sorted(t, reverse=True)) for t in contents }
    #print(canonical)
    nodes = list(canonical)
    nodes.sort(reverse=True)

    res = findCombination(nodes, 2)
    print("Cont: ",nodes)
    edges = []
    for combo in res:
        #print(combo[0], "and", combo[1])
        edge = findCommonSubSet(combo[0], combo[1], 2)

        if edge:
            edges.append((combo[0], combo[1]))


    root = nodes[0]

    return nodes, edges, root

if __name__ == '__main__':
    k, m = 4, 3
    nodes, edges, root = build_universal_multiset_cycle(k, m, return_tree=True)

    UCS = []
    #print("Main nodes:",nodes)
    for node in nodes:
        content = []
        for i in range(m + 1):
            content.append(node.count(str(i)))
        print("Node: ",node," content: ",content)
        cycles = concatenationUC.make_cycles(content)

        currentCycle = ""
        for cycle in cycles:
            for element in cycle:
                dig = int(element) - 1

                currentCycle += str(dig)
                #print(dig, end="")
            #print()
        #print("cc: ",currentCycle)
        UCS.append(currentCycle)

    print(UCS)
    print(edges)
    print(nodes)
    contentToUC = dict(zip(nodes, UCS))

    transformed = [tuple(contentToUC[x] for x in tup) for tup in edges]
    print(transformed)

    visualize_merge_tree(edges, root)
    visualize_merge_tree(transformed, contentToUC[root])
            
            

    

    






# k = 4, m = 6

# Content 2220
# UC: 3331

# Content 3210
# UC: 324134214231234124314321

# Both share {2, 0}
# And both share 31 as a substring

# Content 3111
# UC: 4222

# Content 3210
# UC: 324134214231234124314321

# Both share {3, 1}
# And both share 42 as a substring