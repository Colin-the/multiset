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

    plt.title("Tree")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

# add to parentTree.py (or run alongside it)

def find_successor_cycles(reps, successor):
    succ = {r: successor(r) for r in reps}
    visited = set()
    cycles = []
    for r in reps:
        if r in visited:
            continue
        path = []
        cur = r
        index = {}
        while cur not in index and cur not in visited:
            index[cur] = len(path)
            path.append(cur)
            cur = succ[cur]
        if cur in index:
            # found a cycle
            start = index[cur]
            cycle = path[start:]
            cycles.append(cycle)
        # mark whole path visited (cycles + trees feeding them)
        for v in path:
            visited.add(v)
    return cycles, succ

def build_cycle_index(cycles):
    # map rep -> (cycle_id, index_in_cycle)
    cidx = {}
    for cid, cyc in enumerate(cycles):
        for i, v in enumerate(cyc):
            cidx[v] = (cid, i)
    return cidx

def splice_cycles(cycles, cid_a, cid_b, idx_parent, idx_child):
    A = cycles[cid_a]
    B = cycles[cid_b]
    newA = A[:idx_parent+1] + B[idx_child:] + B[:idx_child] + A[idx_parent+1:]
    # remove the two cycles and append newA
    # we'll keep new cycle at index cid_a and delete cid_b (caller must handle)
    cycles[cid_a] = newA
    cycles[cid_b] = None

def build_universal_cycle_by_splicing(k, m):
    # build reps and tree using your functions
    tree, root = build_parent_tree(k, m)
    # reps set
    reps = set(tree.keys())
    for chs in tree.values():
        reps.update(chs)

    # find successor cycles
    cycles, succ = find_successor_cycles(reps, successor)
    cycles = [c for c in cycles]  # list of lists
    cidx = build_cycle_index(cycles)

    # process parent->child edges and splice when they connect distinct cycles
    changed = True
    while True:
        # rebuild index map each outer iteration (because cycles change)
        cycles = [c for c in cycles if c is not None]
        cidx = build_cycle_index(cycles)
        if len(cycles) <= 1:
            break

        merged = False
        # iterate parent tree edges looking for cross-cycle edges
        for parent, children in tree.items():
            for child in children:
                if parent not in cidx or child not in cidx:
                    continue
                cid_p, idx_p = cidx[parent]
                cid_c, idx_c = cidx[child]
                if cid_p != cid_c:
                    # splice child-cycle into parent-cycle
                    splice_cycles(cycles, cid_p, cid_c, idx_p, idx_c)
                    # mark the old place of cid_c as None
                    cycles[cid_c] = None
                    merged = True
                    break
            if merged:
                break
        if not merged:
            # no more cross-cycle parent->child edges found
            break

    # final cleanup
    cycles = [c for c in cycles if c is not None]
    if len(cycles) == 1:
        return cycles[0]
    else:
        # failed to merge into a single cycle
        return cycles  # return remaining cycles so you can inspect

if __name__ == "__main__":
    k, m = 4, 3
    tree, root = build_parent_tree(k, m)
    

    big_cycle = build_universal_cycle_by_splicing(k, m)
    if isinstance(big_cycle[0], tuple):
        print("Final cycle length (reps):", len(big_cycle))
        print(big_cycle[:20])
    else:
        print("Remaining cycles:", len(big_cycle))

    plot_tree(tree, root,shorthand=False)




# rotate first symbol + following 0s to end and test if it is representiative

# the sucessor rule is what encodes how to traverse the tree and what end up yeilding the cycle


# this uses the missing symbol register

# 0000001000011000101000111001001011001101001111010101110110111111