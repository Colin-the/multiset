# Self-contained program to build a universal cycle of k-tuples summing to m
# using the parent rule and a "missing symbol register" (MSR) to decide when
# to splice child-cycles into parent-cycles.

from collections import defaultdict
import itertools
from typing import List, Tuple, Dict, Set

def get_representative(arr: List[int]) -> Tuple[int, ...]:
    """Return lexicographically greatest rotation of arr."""
    k = len(arr)
    best = tuple(arr)
    for r in range(1, k):
        rot = tuple(arr[r:] + arr[:r])
        if rot > best:
            best = rot
    return best

def parent_of(rep: Tuple[int, ...]) -> Tuple[int, ...]:
    """Parent rule: move 1 from the right-most positive entry to its left neighbor,
    then re-normalize (lex greatest rotation)."""
    k = len(rep)
    arr = list(rep)
    i = max(idx for idx, v in enumerate(arr) if v > 0)
    arr[i] -= 1
    left = (i - 1) % k
    arr[left] += 1
    return get_representative(arr)

def build_parent_tree(k: int, m: int):
    """Build the parent tree mapping parent_rep -> list of child_reps and return root rep."""
    all_tuples = itertools.product(range(m+1), repeat=k)
    reps = { get_representative(list(t)) for t in all_tuples if sum(t) == m }
    root = get_representative([m] + [0]*(k-1))
    tree = defaultdict(list)
    for rep in reps:
        if rep == root:
            continue
        par = parent_of(rep)
        tree[par].append(rep)
    return tree, root, reps

def successor(rep: Tuple[int, ...]) -> Tuple[int, ...]:
    """Successor rule: move one unit from rightmost positive entry to its right neighbor,
    then re-normalize by lex-greatest rotation."""
    k = len(rep)
    arr = list(rep)
    j = max(i for i,v in enumerate(arr) if v > 0)
    arr[j] -= 1
    arr[(j+1) % k] += 1
    return get_representative(arr)

def find_successor_cycles(reps: Set[Tuple[int, ...]]):
    """Find cycles under the successor map among the given reps.
    Returns list of cycles (each a list) and the successor dict."""
    succ = {r: successor(r) for r in reps}
    visited = set()
    cycles = []
    for r in reps:
        if r in visited:
            continue
        path = []
        index = {}
        cur = r
        while cur not in index and cur not in visited:
            index[cur] = len(path)
            path.append(cur)
            cur = succ[cur]
        if cur in index:
            start = index[cur]
            cycle = path[start:]
            cycles.append(cycle)
        for v in path:
            visited.add(v)
    return cycles, succ

def build_cycle_index(cycles):
    cidx = {}
    for cid, cyc in enumerate(cycles):
        for i, v in enumerate(cyc):
            cidx[v] = (cid, i)
    return cidx

def splice_cycles(cycles, cid_parent, cid_child, idx_parent, idx_child):
    A = cycles[cid_parent]
    B = cycles[cid_child]
    newA = A[:idx_parent+1] + B[idx_child:] + B[:idx_child] + A[idx_parent+1:]
    cycles[cid_parent] = newA
    cycles[cid_child] = None

def validate_universal_cycle(cycle, reps, succ_map):
    cycset = set(cycle)
    if len(cycle) != len(reps):
        print(f"Validation failed: cycle length {len(cycle)} != number of reps {len(reps)}")
        return False
    if len(cycset) != len(reps):
        print("Validation failed: cycle has duplicate elements or missing elements.")
        return False
    n = len(cycle)
    for i in range(n):
        a = cycle[i]
        b = cycle[(i+1)%n]
        if succ_map[a] != b:
            print(f"Validation failed: successor({a}) = {succ_map[a]} but next in cycle is {b}")
            return False
    return True

def build_universal_cycle_using_msr(k: int, m: int, verbose: bool=False):
    tree, root, reps = build_parent_tree(k, m)
    cycles, succ = find_successor_cycles(reps)
    cycles = [c for c in cycles]  # mutable

    if verbose:
        print(f"Initial cycles: {len(cycles)}")

    cidx = build_cycle_index(cycles)
    msr = { parent: set() for parent in tree.keys() }
    for parent, children in tree.items():
        for child in children:
            if parent in cidx and child in cidx and cidx[parent][0] != cidx[child][0]:
                msr[parent].add(child)

    while True:
        cycles = [c for c in cycles if c is not None]
        if len(cycles) <= 1:
            break
        cidx = build_cycle_index(cycles)
        merged_any = False
        for cid, cyc in enumerate(cycles):
            for idx_p, node in enumerate(cyc):
                if node in tree and node in msr and msr[node]:
                    selected_child = None
                    for ch in list(msr[node]):
                        if ch in cidx:
                            selected_child = ch
                            break
                        else:
                            msr[node].discard(ch)
                    if selected_child is None:
                        continue
                    cid_child, idx_child = cidx[selected_child]
                    if cid_child == cid:
                        msr[node].discard(selected_child)
                        continue
                    if verbose:
                        print(f"Splicing child {selected_child} (cycle {cid_child}) into parent {node} (cycle {cid}) at indices {idx_child}, {idx_p}")
                    splice_cycles(cycles, cid, cid_child, idx_p, idx_child)
                    for p in list(msr.keys()):
                        if selected_child in msr[p]:
                            msr[p].discard(selected_child)
                    merged_any = True
                    break
            if merged_any:
                break
        if not merged_any:
            break

    cycles = [c for c in cycles if c is not None]
    if len(cycles) == 1:
        final = cycles[0]
        ok = validate_universal_cycle(final, reps, succ)
        return final, ok, tree, msr
    else:
        return cycles, False, tree, msr

# Quick tests
def run_tests():
    cases = [(3,2), (3,3), (4,3), (4,4), (5,5)]
    for k,m in cases:
        print(f"\nBuilding universal cycle for k={k}, m={m}")
        final, ok, tree, msr = build_universal_cycle_using_msr(k,m, verbose=False)
        if ok:
            print(f"Success! Final cycle length = {len(final)}. Validated = {ok}")
        else:
            print(f"Failed to build/validate for k={k}, m={m}. Remaining cycles or error shown above.")

if __name__ == "__main__":
    run_tests()
