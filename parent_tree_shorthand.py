import itertools
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
import networkx as nx
import matplotlib.pyplot as plt


def full_form(short_seq: Tuple[int, ...], m: int) -> List[int]:
    """Restore full k-length vector from shorthand representation."""
    return list(short_seq) + [m - sum(short_seq)]


def to_shorthand(full_seq: List[int]) -> Tuple[int, ...]:
    """Convert full k-length vector to shorthand (drop last element)."""
    return tuple(full_seq[:-1])


def get_representative_shorthand(seq: Tuple[int, ...], m: int) -> Tuple[int, ...]:
    """
    Return the lexicographically greatest rotation of the full k-length vector,
    then convert back to shorthand.
    """
    full = full_form(seq, m)
    k = len(full)
    # generate all rotations of full
    rotations = (tuple(full[i:] + full[:i]) for i in range(k))
    best = max(rotations)
    return to_shorthand(list(best))


def parent_of_shorthand(rep: Tuple[int, ...], m: int) -> Tuple[int, ...]:
    """
    Apply the parent-rule to the full vector reconstructed from shorthand,
    then re-normalize (get representative) and return shorthand.
    """
    full = full_form(rep, m)
    k = len(full)
    arr = list(full)
    # find rightmost positive entry
    i = max(idx for idx, v in enumerate(arr) if v > 0)
    arr[i] -= 1
    left = (i - 1) % k
    arr[left] += 1
    # normalize and return
    return get_representative_shorthand(to_shorthand(arr), m)



def get_representative_shorthand_no_full(seq: Tuple[int,...], m: int) -> Tuple[int,...]:
    """
    Given a shorthand seq of length k-1, find the lex-greatest
    rotation of the *full* k-tuple (where last = m - sum(seq))
    *without actually constructing* the full list.
    """
    k1 = len(seq)
    sum_seq = sum(seq)
    # precompute implicit last of original
    last0 = m - sum_seq

    def rotation(i):
        # yield the rotated sequence of length k, one element at a time
        # first k-1 elements:
        for j in range(k1):
            yield seq[(i + j) % k1]
        # then the implicit last:
        # it depends on which elements of seq you actually rotated out vs in;
        # but in a cyclic rotation of length k, the sum of the first k-1
        # is still sum_seq, so the implicit last is the same:
        yield last0

    # compare two rotations lex order
    best_i = 0
    best_rot = tuple(rotation(0))
    for i in range(1, k1):
        rot_i = tuple(rotation(i))
        if rot_i > best_rot:
            best_rot, best_i = rot_i, i

    # best_rot is a length-k tuple; drop the last entry to get shorthand
    return best_rot[:-1]


def parent_of_shorthand_no_full(seq: Tuple[int,...], m: int) -> Tuple[int,...]:
    """
    Apply the parent-rule directly on the shorthand seq:
      - find rightmost positive in the *implicit* full vector
      - subtract 1 there, add 1 to its left neighbor (in the k-cycle)
      - renormalize by taking the lex-greatest rotation
    """
    k1 = len(seq)
    sum_seq = sum(seq)
    last = m - sum_seq

    # find rightmost i in 0..k-1 with value > 0
    # if i < k-1, that's seq[i]; if i == k-1, that's the implicit last
    # so build a small list of length k to scan
    values = list(seq) + [last]
    i = max(idx for idx, v in enumerate(values) if v > 0)

    # now build the *new* shorthand after doing the −1/+1
    new = list(seq)
    if i < k1:
        # decrement seq[i]
        new[i] -= 1
    else:
        # decrement the implicit last
        last -= 1

    # compute the “left” index in the k-cycle
    left = (i - 1) % (k1 + 1)
    if left < k1:
        new[left] += 1
    else:
        # left == k-1 refers to implicit last
        last += 1

    # (Note: you don't have to re-compute sum(new) here;
    #  it's automatically consistent because you did -1/+1 on
    #  either seq or last.)

    # re-normalize by rotating to lex-greatest
    return get_representative_shorthand_no_full(tuple(new), m)



def build_parent_tree_shorthand(k: int, m: int) -> Tuple[Dict[Tuple[int, ...], List[Tuple[int, ...]]], Tuple[int, ...]]:
    """
    Build the parent tree over shorthand representations of length k-1,
    where the sum of entries is <= m (the last entry is inferred).

    Returns:
      tree: mapping from parent shorthand to list of child shorthands
      root: the root shorthand (corresponding to [m,0,...,0])
    """
    tree: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = defaultdict(list)
    # All possible shorthand tuples of length k-1 whose sum <= m
    all_shorthand = itertools.product(range(m + 1), repeat=k-1)
    # Normalize each to its representative
    reps = {
        get_representative_shorthand_no_full(seq, m)
        for seq in all_shorthand
        if sum(seq) <= m
    }
    # Define root shorthand
    root_full = [m] + [0] * (k - 1)
    root = to_shorthand(root_full)

    # Build tree
    for rep in reps:
        if rep == root:
            continue
        par = parent_of_shorthand_no_full(rep, m)
        tree[par].append(rep)

    return tree, root


def compute_subtree_sizes(
    tree: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    root: Tuple[int, ...]
) -> Dict[Tuple[int, ...], int]:
    """
    Recursively count number of leaves under each node in the parent tree.

    Returns a dict mapping each shorthand tuple to its leaf-count.
    """
    sizes: Dict[Tuple[int, ...], int] = {}

    def dfs(u: Tuple[int, ...]) -> int:
        children = tree.get(u, [])
        if not children:
            sizes[u] = 1
        else:
            sizes[u] = sum(dfs(v) for v in children)
        return sizes[u]

    dfs(root)
    return sizes



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

    plt.title("Tree")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

if __name__ == "__main__":
    k, m = 5, 5
    tree, root = build_parent_tree_shorthand(k, m)
    plot_tree(tree, root,shorthand=False)
    sizes = compute_subtree_sizes(tree, root)

    print(f"Shorthand root: {root}")
    print(f"Total nodes: {len(sizes)}")
    print(f"Subtree sizes (node: leaf count):")
    # Print first 10 entries
    for i, (node, size) in enumerate(sorted(sizes.items(), reverse=True)):
        if i >= 10:
            break
        print(f"  {node}: {size}")

    # Optionally, show children of root
    print(f"\nChildren of root {root}:")
    for child in tree.get(root, []):
        print(f"  {child}")  
