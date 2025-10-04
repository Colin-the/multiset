# build_multiset_unicycle.py
from collections import defaultdict, deque

def all_L_tuples_sum_m(L, m):
    res = []
    def rec(i, rem, pref):
        if i == L-1:
            res.append(tuple(pref + [rem]))
            return
        for x in range(rem+1):
            rec(i+1, rem-x, pref + [x])
    rec(0, m, [])
    return res

def eulerian_on_edges(comp_edges, L):
    # Hierholzer on subgraph defined by comp_edges; returns list of edge-label tuples in circuit order
    adj = defaultdict(list)
    for e in comp_edges:
        u = e[:L-1]; v = e[1:]
        adj[u].append((v, e))
    adj_iter = {u: adj[u][:] for u in adj}
    start = next(iter(adj_iter))
    stack = [(start, None)]
    circuit_edges = []
    while stack:
        node, in_e = stack[-1]
        if adj_iter.get(node):
            nbr, lab = adj_iter[node].pop()
            stack.append((nbr, lab))
        else:
            stack.pop()
            if in_e is not None:
                circuit_edges.append(in_e)
    circuit_edges.reverse()
    return circuit_edges

def middle_substring(e):      # length L-2 substring used for joining
    return e[1:-1]

def join_cycles_by_middle(cycles):
    # Greedy splice cycles whenever two cycles contain an edge whose middle substring matches.
    cycles = [list(c) for c in cycles]
    while len(cycles) > 1:
        # build map mid -> list of (cycle_idx, edge_idx)
        mid_map = defaultdict(list)
        for ci, cyc in enumerate(cycles):
            for ei, e in enumerate(cyc):
                mid_map[middle_substring(e)].append((ci, ei))
        pair = None
        for mid, occ in mid_map.items():
            seen_cycles = {}
            for ci, ei in occ:
                if ci not in seen_cycles:
                    seen_cycles[ci] = ei
                    if len(seen_cycles) >= 2:
                        cidxs = list(seen_cycles.keys())[:2]
                        pair = (mid, cidxs[0], seen_cycles[cidxs[0]], cidxs[1], seen_cycles[cidxs[1]])
                        break
            if pair:
                break
        if not pair:
            break
        _, a_ci, a_ei, b_ci, b_ei = pair
        A = cycles[a_ci]; B = cycles[b_ci]
        A_rot = A[a_ei:] + A[:a_ei]
        B_rot = B[b_ei:] + B[:b_ei]
        # splice B into A immediately after the matched edge
        newA = A_rot[:1] + B_rot + A_rot[1:]
        # replace and delete (preserve indices)
        if a_ci < b_ci:
            cycles[a_ci] = newA
            del cycles[b_ci]
        else:
            cycles[b_ci] = newA
            del cycles[a_ci]
    return cycles[0] if cycles else []

def build_universal_cycle(k, m):
    L = k - 1
    edges = all_L_tuples_sum_m(L, m)           # every shorthand tuple is an "edge"
    # build undirected connectivity to get components
    undirected = defaultdict(set)
    for e in edges:
        u = e[:L-1]; v = e[1:]
        undirected[u].add(v); undirected[v].add(u)
    # find weakly connected components (on nodes)
    seen = set(); components_nodes = []
    for n in undirected.keys():
        if n in seen: continue
        comp = []; q = deque([n]); seen.add(n)
        while q:
            x = q.popleft(); comp.append(x)
            for nb in undirected[x]:
                if nb not in seen:
                    seen.add(nb); q.append(nb)
        components_nodes.append(comp)
    # collect edges per component
    components_edges = []
    nodeset_list = [set(comp) for comp in components_nodes]
    for comp_set in nodeset_list:
        comp_edges = [e for e in edges if e[:L-1] in comp_set]
        components_edges.append(comp_edges)
    # compute Eulerian cycle within each component
    per_comp_cycles = [eulerian_on_edges(comp_edges, L) for comp_edges in components_edges]
    # join them by middle-substring splicing
    final_cycle = join_cycles_by_middle(per_comp_cycles)
    return final_cycle


from collections import Counter
from math import comb

def all_L_tuples_sum_m(L, m):
    """All weak compositions (tuples) length L summing to m."""
    res = []
    def rec(i, rem, pref):
        if i == L-1:
            res.append(tuple(pref + [rem]))
            return
        for x in range(rem+1):
            rec(i+1, rem-x, pref + [x])
    rec(0, m, [])
    return res

def verify_ucycle(seq, K, M, token_mode='auto'):
    """
    Verify whether `seq` is a universal cycle (shorthand representation)
    for parameters K and M.

    Parameters
    - seq: either a list/iterable of integer tokens, or a string.
      If string:
        * token_mode='chars' -> every character is parsed as one integer token (e.g. '3020...')
        * token_mode='ints'  -> whitespace-separated integers (e.g. '3 0 2 0 ...')
        * token_mode='auto'  -> try whitespace-split ints first, else char-by-char
    - K: original multiset size; L = K-1 is shorthand length
    - M: total sum per shorthand
    Returns: (is_valid: bool, details: dict)
    details contains stats and diagnostics (missing tuples, duplicates, length problems).
    """
    # Parse tokens
    if isinstance(seq, str):
        if token_mode == 'ints':
            toks = [int(x) for x in seq.split()]
        elif token_mode == 'chars':
            toks = [int(ch) for ch in seq]
        else:
            parts = seq.split()
            if len(parts) > 1:
                try:
                    toks = [int(x) for x in parts]
                except:
                    toks = [int(ch) for ch in seq]
            else:
                toks = [int(ch) for ch in seq]
    else:
        toks = list(seq)

    # validate tokens
    try:
        toks = [int(x) for x in toks]
    except:
        return False, {"error":"tokens not integers"}
    if any(x < 0 for x in toks):
        return False, {"error":"negative token found"}

    L = K - 1
    expected_count = comb(M + L - 1, L - 1)
    n = len(toks)

    windows = []
    if n == expected_count:
        # cyclic representation: take wrap-around windows
        for i in range(n):
            windows.append(tuple(toks[(i + j) % n] for j in range(L)))
    elif n == expected_count + L - 1:
        # linear printed representation: windows seq[i:i+L] for i=0..expected_count-1 (no wrap)
        for i in range(expected_count):
            windows.append(tuple(toks[i + j] for j in range(L)))
    else:
        return False, {"error":"length mismatch", "len_tokens": n,
                       "expected_cyclic": expected_count,
                       "expected_linear": expected_count + L - 1}

    counts = Counter(windows)
    expected_set = set(all_L_tuples_sum_m(L, M))
    observed_set = set(counts.keys())

    missing = expected_set - observed_set
    extra = observed_set - expected_set
    duplicates = {w: c for w, c in counts.items() if c != 1}

    is_valid = (len(missing) == 0) and (len(extra) == 0) and (len(duplicates) == 0)
    details = {
        "n_tokens": n,
        "expected_count": expected_count,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "duplicates": duplicates,
        "missing_examples": list(missing)[:6],
        "extra_examples": list(extra)[:6]
    }
    return is_valid, details



if __name__ == "__main__":
    k = 5
    m = 5
    cycle = build_universal_cycle(k, m)
    print("Cycle length (edges):", len(cycle))
    for i, e in enumerate(cycle):
        print(f"{i:2d}: {''.join(str(x) for x in e)}")
    # also output compact cyclic shorthand string (first edge fully, then last digit of each subsequent edge)
    if cycle:
        L = k-1
        cyclic = ''.join(str(x) for x in cycle[0])
        for e in cycle[1:]:
            cyclic += str(e[-1])
        print("Cyclic shorthand string:", cyclic)
        print("Length of cyclic shorthand string:", len(cyclic))

    valid, details = verify_ucycle(cyclic, k, m, token_mode='chars')
    print("Test k=",k,"m=",m,"len(seq)=",len(cyclic),"valid:",valid)
    print(" details:",details)


# 30202102121131122011301400320023001400000541010310102102 