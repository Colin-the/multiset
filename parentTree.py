import networkx as nx
import matplotlib.pyplot as plt
import itertools
from itertools import combinations
from collections import defaultdict
from typing import List, Tuple, Dict
from collections import defaultdict, deque
from math import gcd, comb

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

def root_joe(m, k):
    if k < 0 or m < 0:
        raise ValueError("k and m must be nonnegative integers")
    if k == 0:
        if m == 0:
            return tuple()
        raise ValueError("no length-0 sequence can sum to a positive m")

    q, r = divmod(m, k)
    print("Root: ",tuple([q + 1] * r + [q] * (k - r)))
    # first r entries are q+1, rest are q
    return tuple([q + 1] * r + [q] * (k - r))

def parent_joe(rep):
    """
    Apply the parent-rule to rep, then re-normalize to lex-greatest rotation.
    """
    k = len(rep)
    arr = list(rep)
    i = min(idx for idx, v in enumerate(arr) if v > 0)
    arr[i] -= 1
    left = k - 1
    arr[left] += 1
    return get_representative(arr)

def num_representatives(m: int, k: int) -> int:
    if k <= 0:
        if m == 0:
            return 1
        return 0
    total = 0
    for r in range(k):
        t = gcd(k, r)
        L = k // t
        if m % L != 0:
            fix = 0
        else:
            S = m // L
            fix = comb(S + t - 1, t - 1)  # weak compositions of S into t parts
        total += fix
    return total // k

def buildRoot(m,k):
    return tuple([m]+[0]*(k-1))
    
def build_parent_tree(k, m):
    all_tuples = itertools.product(range(m+1), repeat=k)
    reps = { get_representative(list(t)) for t in all_tuples if sum(t)==m }
    root = buildRoot(m,k)
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

def compute_subtree_sizes(tree: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
                          root: Tuple[int, ...],
                          reachable: set[Tuple[int, ...]]) -> Dict[Tuple[int, ...], int]:
    """
    Compute a 'size' for each reachable node = 1 for leaves, otherwise sum of children's sizes.
    Detect cycles and raise ValueError if found.
    """
    memo = {}
    visiting = set()

    def dfs(u):
        if u in memo:
            return memo[u]
        if u in visiting:
            raise ValueError(f"Cycle detected in tree at node {u}")
        visiting.add(u)
        children = [v for v in tree.get(u, []) if v in reachable]
        if not children:
            s = 1
        else:
            s = 0
            for v in children:
                s += dfs(v)
        memo[u] = s
        visiting.remove(u)
        return s

    dfs(root)
    return memo

def plot_tree(tree: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
              root: Tuple[int, ...],
              filename="tree.png",
              shorthand=False):
    """
    Plot the subtree reachable from `root` contained in `tree`.
    `tree` can be any mapping parent -> list(children). Only nodes reachable
    from `root` will be drawn (so pos will be defined for each drawn node).
    """
    # 1) Compute reachable nodes from root (iterative DFS/BFS)
    reachable = set()
    stack = [root]
    while stack:
        u = stack.pop()
        if u in reachable:
            continue
        reachable.add(u)
        for v in tree.get(u, []):
            if v not in reachable:
                stack.append(v)

    if root not in reachable:
        raise ValueError("Root is not present / reachable in the provided tree.")

    print("We have ",len(reachable)," reachable nodes from the root out of ",num_representatives(m, k)," nodes")
    # 2) Build graph using only reachable edges
    G = nx.DiGraph()
    for parent in reachable:
        for child in tree.get(parent, []):
            if child in reachable:
                G.add_edge(parent, child)

    # 3) Compute subtree sizes (only using reachable nodes)
    sizes = compute_subtree_sizes(tree, root, reachable)

    # 4) Assign positions by walking the reachable tree
    pos = {}
    def assign(u, x_start, x_end, depth):
        x = (x_start + x_end) / 2.0
        pos[u] = (x, -depth * 0.1)   # vertical spacing per depth (tweak as desired)
        children = [v for v in tree.get(u, []) if v in reachable]
        if children:
            total = sum(sizes[v] for v in children)
            if total == 0:
                # fallback evenly
                total = len(children)
            acc = x_start
            for v in children:
                w = (sizes[v] / total) * (x_end - x_start)
                assign(v, acc, acc + w, depth + 1)
                acc += w

    assign(root, 0.0, 1.0, 0)

    # Sanity check: pos should now contain all nodes in G
    missing = [n for n in G.nodes() if n not in pos]
    if missing:
        # This shouldn't happen for reachable nodes; raise informative error
        raise RuntimeError(f"Internal error — positions missing for nodes: {missing}")

    # 5) Draw
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
        labels = {n: str(n)[:len(str(n))-4] + ")" for n in G.nodes()}
    else:
        labels = {n: str(n) for n in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title("Tree (subtree reachable from root)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

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

def compositions_length_k_sum_m(m: int, k: int):
    """
    Generator of all weak compositions of m into k parts using stars-and-bars.
    Yields tuples of length k.
    """
    if k == 0:
        if m == 0:
            yield ()
        return
    # think of placing k-1 dividers among m + k - 1 slots
    # choose positions of dividers from range(1, m+k)
    for divs in combinations(range(1, m + k), k - 1):
        parts = []
        prev = 0
        for d in divs:
            parts.append(d - prev - 1)
            prev = d
        parts.append(m + k - prev - 1)
        yield tuple(parts)

def list_representatives(m: int, k: int, reverse: bool = True) -> List[Tuple[int, ...]]:
    """
    Enumerate representatives: keep compositions whose lex greatest rotation equals themselves.
    Returns a list sorted lexicographically (descending if reverse=True).
    Warning: number of compositions = C(m+k-1, k-1); may be large.
    """
    reps = []
    for comp in compositions_length_k_sum_m(m, k):
        if comp == get_representative(comp):
            reps.append(comp)
    reps.sort(reverse=reverse)
    return reps

if __name__ == "__main__":
    k, m = 4, 4
    tree, root = build_parent_tree(k, m)
    #print(tree)
    reps = list_representatives(m, k)
    print("We have ",len(reps),"\n",reps)
    # big_cycle = build_universal_cycle_by_splicing(k, m)
    # if isinstance(big_cycle[0], tuple):
    #     print("Final cycle length (reps):", len(big_cycle))
    #     print(big_cycle[:20])
    # else:
    #     print("Remaining cycles:", len(big_cycle))

    plot_tree(tree, root,shorthand=False)

    cnt = 1
    node = (3,0,1,0,0,0)
    while cnt < 10:
        cnt += 1
        print("Node: ",node)
        node = parent_joe(node)
        


# rotate first symbol + following 0s to end and test if it is representiative

# the sucessor rule is what encodes how to traverse the tree and what end up yeilding the cycle


# this uses the missing symbol register

# 0000001000011000101000111001001011001101001111010101110110111111