from itertools import combinations_with_replacement, permutations
from itertools import combinations
from collections import deque, defaultdict
from math import comb
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter
import concatenationUC

def visualize_merge_tree(tree_edges, root, filename="graph"):
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
    plt.savefig(filename)
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
    return a_subs & b_subs

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
    #print("Cont: ",nodes)
    edges = []
    edgesWithShared = []
    for combo in res:
        #print(combo[0], "and", combo[1])
        #edge = findCommonSubSet(combo[0], combo[1], 2)
        edge = findCommonSubstring(combo[0], combo[1], 2)
        #print(edge)
        if edge:
            edges.append((combo[0], combo[1]))
            edgesWithShared.append((combo[0], combo[1], edge))


    root = nodes[0]

    return nodes, edges, root, edgesWithShared

#def findAllOccurrences(string, substring):
    
def circularFind(text: str, sub: str, start: int):
    extended_text = text + text
    index = extended_text.find(sub, start)
    if index >= len(text):
        return index - len(text)
    else:
        return index
    
def rotate(string: str, substring: str):
    #print("\tFor ",string," ",substring)
    """
    Will return a list of all possible strings that
    start with the given substring
    """
    strings = []

    # Base case when we can't possibly have any matches
    if len(string) < len(substring):
        return strings
    
    start = 0
    while True:
        start = circularFind(string, substring, start)
        #print("Start: ",start)
        if start == -1:
            break
        rotatedString = string[start:] + string[:start]
        #print("string: ",rotatedString)

        # If we have come accross the string before then we are done
        if rotatedString in strings:
            break

        # If this is our first time Encountering the string add it to the list
        else:
            strings.append(rotatedString)
            start += 1
    return strings

def unrotate(rotated, candidates):
    """
    Given a rotated string and a dict of original-string keys,
    return the original key that can be rotated to form `rotated`.
    If no match is found, returns None.
    """
    for original in candidates:
        if len(original) != len(rotated):
            continue
        # A rotated version of original must be a substring of original+original
        if rotated in (original + original):
            return original
    return None

def searchForUC(remaining: dict, adj: dict, current: str, goalLen: int, commonSubstringSize: int, currentSequence: list, allSequence: list):
    """
    Function that is designed to look threw all of the diffrent
    fixed content UC's to try and find some way to join them
    togther. It does this by treating it like a graph problem
    and searching for all possible ways to join the fixed 
    content's togther.
    """

    # Extract the remaining symbols form the current node 
    currentNode = unrotate(current, remaining)
    ptr = remaining[currentNode]
    #print("Current: ", current, " sequ: ",currentSequence, " ptr: ",ptr, " current node: ",currentNode)
    print("Sequence: ","".join(currentSequence)," length: ",len(currentSequence))

    # If this node has no symbols left, we can't proceed
    if not remaining[currentNode]:
        print("Out of symbols in the current node")
        return None

    # Extract the current symbol that we are looking at
    for symbol in ptr:
        #print("Sym: ",symbol)
        currentSequence.append(symbol)
        #print("Sequence: ",currentSequence," len ",len(currentSequence))
        
        # Remove the first char as this is the symbol we are currently looking at
        remaining[currentNode] = remaining[currentNode][1:]

        #print("New rem ",remaining)
        # If we reach the desired length then we have found our cycle
        if len(currentSequence) == goalLen:
            return currentSequence
        
        # If we have seen more then k - 2 symbols then we could possibly 
        # make a jump to some other node via an edge
        if len(currentSequence) >= commonSubstringSize:
            # extract the last k-2 symbols from the current string
            possibleJoin = currentSequence[commonSubstringSize * -1:]
            possibleJoin = "".join(possibleJoin)
            #print("Join on: ",possibleJoin)

            # Look at all of our edges coming out of the current node 
            #print(current," list ",adj)
            edges = adj[currentNode]         
            for edge in edges:
                # Find out the k-2 symbols that these two nodes share
                joiningSubstring = edge[1]

                # Compare the last k-2 symbols from the current string with the
                # edge to determine if we CAN go from one node to the next
                if joiningSubstring == possibleJoin:

                    #print("Can join on: ",possibleJoin, "using ", joiningSubstring," via edge ",edge)
                    # At this point we need to branch and consider the possibility where we 
                    # jump from the current node to this new node and the possibility that we do not

                    # First we need to check if the node we are trying to jump to still contains the 
                    # substring that would allow us to make the jump

                    # Extract the oringal node from the edge
                    destinationNode = edge[0]
                    
                    # Update it with what we have currently used in the cycle
                    destinationString = remaining[destinationNode] 

                    #print("Dest: ",destinationNode)
                    # Find out what the node we are jumping to looks like under rotation
                    possibleRotations = rotate(destinationString, joiningSubstring)
                    #print("Will join to: ",possibleRotations)

                    if possibleRotations == []:
                        continue

                    for rotatedString in possibleRotations:
                        # Make a new dictorary so we don't mess up the old one
                        remainingSubCase = remaining.copy()
                        subSequence = currentSequence.copy()
                        remainingSubCase[destinationNode] = rotatedString[2:]

                        # And then calling the function recursively and then examining what it returns
                        seq = searchForUC(remainingSubCase, adj, destinationNode, goalLen, commonSubstringSize, subSequence, allSequence)
                        #print("Seq: ",seq)
                        if seq == None:
                            continue
                        else:

                            #print("Found seq by searching: ",seq, " seq has len ",len(seq))
                            return seq
    return None


def verifyMultisetUC(cycle: str, k: int, m: int) -> bool:
    """
    Verify that `cycle` is a valid universal cycle of all size-k 
    multisets with max-multiplicity m.
    
    Returns True if it’s valid, False otherwise.
    """
    N = len(cycle)
    # 1) Check length
    expected_length = comb(m + k - 1, m) 
    if N != expected_length:
        print(f"Wrong length: got {N}, expected {expected_length}")
        return False

    # 2) Generate the set of all target multisets as sorted tuples
    #    e.g. all ways to pick k items from {0,..,n-1} with rep ≤ m
    targets = set()
    def gen_multisets(prefix, start, left):
        if left == 0:
            targets.add(tuple(prefix))
            return
        for x in range(start, m):
            if prefix.count(x) < m:
                gen_multisets(prefix + [x], x, left - 1)
    gen_multisets([], 0, k)
    #print(targets)

    # 3) Slide a window of length k around the cycle (wrap‑around)
    seen = set()
    for i in range(N):
        window = [cycle[(i + j) % N] for j in range(k)]
        # convert window to a multiset: sort
        ms = tuple(window)
        if ms in seen:
            print("Duplicate window:", ms)
            print(window)
            return False
        seen.add(ms)

    # 4) Compare seen vs. targets
    if seen != targets:
        missing = targets - seen
        extras  = seen    - targets
        print("Missing multisets:", missing)
        print("Unexpected windows:", extras)
        return False

    return True

if __name__ == '__main__':
    # k is number of symbols in the multi-set
    # m is what the content has to sum up to
    k, m = 4, 4
    nodes, edges, root, edgesWithShared = build_universal_multiset_cycle(k, m, return_tree=True)
    #print("EWS:",edgesWithShared)
    UCS = []
    #print("Main nodes:",nodes)
    for node in nodes:
        content = []
        for i in range(m + 1):
            content.append(node.count(str(i)))
        #print("Node: ",node," content: ",content)
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

    #print(UCS)
    #print(edges)
    #print(nodes)
    contentToUC = dict(zip(nodes, UCS))

    transformed = [tuple(contentToUC[x] for x in tup) for tup in edges]

    edgeListWithCommon = []
    for tup in edgesWithShared:
        edge = []
        for item in tup:
            # This occurs for the contents that are encoded as strings 
            if type(item) == str:
                edge.append(contentToUC[item])

            # This is for the common substring that the contents share
            elif type(item) == set:
                common = ""
                for obj in item:
                    for substr in obj:
                        #print("obj: ",substr)
                        common += str(substr)

                edge.append(str(common))

        edgeListWithCommon.append(edge)


    #print("New list: ",edgeListWithCommon)
    #print("NOdes: ",nodes)

    UClength = comb(k + m - 1, m)
    print("UC should be ",UClength," long")

    currentSequence = []
    allSequence = []

    # Build adjacency list
    adj = {n: [] for n in UCS}
    for u, v, overlap in edgeListWithCommon:
        adj[u].append((v, overlap))
        adj[v].append((u, overlap))

    #print("Formed adj: ",adj)
    remaining = {n:n for n in UCS}
    #print(remaining)

    # Pick some node to be the root. It doesn't matter what node
    # we use as the final sequence will be the same under rotation 
    root = UCS[0]
    for node in UCS:

        current = node
        cycle = searchForUC(remaining, adj, current, UClength, k-2, currentSequence, allSequence)
        if cycle != None:
            cycle = "".join(cycle)

            print("Found cycle: ",cycle)
            print("This cycle has length ",len(cycle))

            verifyMultisetUC(cycle, k, m)

    visualize_merge_tree(edges, root, filename="content")
    visualize_merge_tree(transformed, contentToUC[root], filename="cycles")
            
            

    

    






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

#__________________________________________#

# 3000, 120020102100, 1110

# 0300, 002010210012
# 0300 2010210012
# 03002010210012
# 01203002010210, 1011
# 0120300201021011


# 03002010210012
# 0120300201021011
# 012, 120, 203, 030, 300, 002, 020, 201, 102, 021, 210, 101, 011, 110, 101

# 3002010210012

# 2100123002010, 1011 
# 0123002010210, 1011 <- will fail as we have 101 twice

# 21001230020101011 
# 210, 100, 001, 012, 123, 230, 300, 002, 020, 201, 010, 101, 010, 101, 011, 112, 121


# 123121321
# 123, 231, 312, 121

from typing import List, Tuple, Set, Optional


def search_for_uc(
    nodes: List[str],
    edges: List[Tuple[str, str, str]],
    goal_len: int,
) -> Optional[str]:
    """
    Find a universal cycle of length `goal_len` by DFS over the graph.

    - `nodes`: list of node-strings.
    - `edges`: list of tuples (u, v, overlap).
    Returns the cycle-string if found, else None.
    """
    # Build adjacency list
    adj = {n: [] for n in nodes}
    for u, v, overlap in edges:
        adj[u].append((v, overlap))
        adj[v].append((u, overlap))

    

    def rotate_to_prefix(s: str, sub: str) -> str:
        idx = s.find(sub)
        if idx == -1:
            raise ValueError(f"substring {sub} not in {s}")
        return s[idx:] + s[:idx]

    def dfs(current: str, seq: str, visited: Set[Tuple[str, str, str]]) -> Optional[str]:
        if len(seq) == goal_len:
            return seq
        # Try each edge out of current
        for nxt, overlap in adj[current]:
            # Build a consistent edge key
            edge_key = (min(current, nxt), max(current, nxt), overlap)
            if edge_key in visited:
                continue
            # Can we transition? seq must end with overlap
            if not seq.endswith(overlap):
                continue
            # Rotate next so that overlap is prefix
            rotated = rotate_to_prefix(nxt, overlap)
            tail = rotated[len(overlap):]
            if not tail:
                continue
            visited.add(edge_key)
            res = dfs(nxt, seq + tail, visited)
            if res:
                return res
            visited.remove(edge_key)
        return None

    # Try each node as a starting point
    print("adj:", adj)
    for start in nodes:
        print("Start: ",start)
        result = dfs(start, start, set())
        print("Res: ",result)
        if result:
            return result
    return None

# Example usage
# if __name__ == '__main__':
#     nodes = ['4000', '130030103100', '202200', '112012102110', '1']
#     edges = [
#         ('4000', '130030103100', '00'),
#         ('4000', '202200', '00'),
#         ('130030103100', '202200', '00'),
#         ('130030103100', '112012102110', '10'),
#         ('202200', '112012102110', '02'),
#         ('112012102110', '1', '11'),
#     ]
#     uc = search_for_uc(nodes, edges, goal_len=20)
#     print('Found UC:', uc)
