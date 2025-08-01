import networkx as nx
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict
from collections import defaultdict, deque

def get_representative(seq):
    """Return lexicographically greatest rotation of seq."""
    k = len(seq)
    return max(tuple(seq[i:]+seq[:i]) for i in range(k))

def parent_of(rep):
    """
    Apply the parent-rule to rep, then re-normalize to lex-greatest rotation.
    """
    k = len(rep)
    arr = list(rep)
    i = max(idx for idx, v in enumerate(arr) if v > 0)
    arr[i] -= 1
    left = (i - 1) % k
    arr[left] += 1
    return get_representative(arr)

def build_parent_tree(k, m):
    all_tuples = itertools.product(range(m+1), repeat=k)
    reps = { get_representative(list(t)) for t in all_tuples if sum(t)==m }
    root = get_representative([m] + [0]*(k-1))
    tree = defaultdict(list)
    for rep in reps:
        if rep == root:
            continue
        par = parent_of(rep)
        tree[par].append(rep)
    return tree, root

def compute_subtree_sizes(tree: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
                          root: Tuple[int, ...]) -> Dict[Tuple[int, ...], int]:
    """Recursively count number of leaves under each node."""
    sizes = {}
    def dfs(u):
        children = tree.get(u, [])
        if not children:
            sizes[u] = 1
        else:
            sizes[u] = sum(dfs(v) for v in children)
        return sizes[u]
    dfs(root)
    return sizes


def get_representative(arr):
    k = len(arr)
    best = tuple(arr)
    for r in range(1, k):
        rot = tuple(arr[r:] + arr[:r])
        if rot > best:
            best = rot
    return best

def successor(rep):
    """
    Given a k-tuple rep summing to m, produce the next rep
    in the universal cycle by moving one unit right and re-rotating.
    """
    k = len(rep)
    arr = list(rep)
    # find rightmost positive
    j = max(i for i,v in enumerate(arr) if v > 0)
    # move one unit right
    arr[j]   -= 1
    arr[(j+1)%k] += 1
    # re-normalize
    return get_representative(arr)


def plot_tree(tree: Dict[Tuple[int, ...], List[Tuple[int, ...]]], root: Tuple[int, ...], filename="tree.png", shorthand = False):
    # Build directed graph
    G = nx.DiGraph()
    for parent, children in tree.items():
        for child in children:
            G.add_edge(parent, child)

    # Compute subtree sizes to know how much horizontal “room” each node needs
    sizes = compute_subtree_sizes(tree, root)

    # Assign positions by walking the tree
    pos = {}
    def assign(u, x_start, x_end, depth):
        # place u at the center of its interval
        x = (x_start + x_end) / 2
        pos[u] = (x, -depth * 0.05)
        children = tree.get(u, [])
        if children:
            # carve the interval [x_start, x_end] into chunks proportional to each child’s size
            total = sum(sizes[v] for v in children)
            acc = x_start
            for v in children:
                w = sizes[v] / total * (x_end - x_start)
                assign(v, acc, acc + w, depth+1)
                acc += w

    # root gets full width [0,1]
    assign(root, 0.0, 1.0, 0)

    # Draw
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=700,
        linewidths=1,
        edgecolors='gray'
    )
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=12,
        edge_color='gray'
    )
    if shorthand:
        nx.draw_networkx_labels(
            G, pos,
            labels={n: str(n)[:len(str(n))-4] +")" for n in G.nodes()},
            font_size=8
        )
    else:
        nx.draw_networkx_labels(
            G, pos,
            labels={n: str(n) for n in G.nodes()},
            font_size=8
        )

    plt.title("\"shorthand\" tree")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

# Example usage:
if __name__ == "__main__":
    k, m = 5, 5
    tree, root = build_parent_tree(k, m)
    plot_tree(tree, root,shorthand=True)




# rotate first symbol + following 0s to end and test if it is representiative

# the sucessor rule is what encodes how to traverse the tree and what end up yeilding the cycle


# this uses the missing symbol register