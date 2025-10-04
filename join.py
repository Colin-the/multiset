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

def buildRoot(m,k):
    return tuple([m]+[0]*(k-1))

def lengthOfUC(m,k):
    """Length of the unique cycle in the join graph."""
    return comb(m+k-1, m)

def msr(str, m:int):
    """return the missing symbol in the sequence."""
    return m - sum(str)

def nextRotation(seq):
    """Return the next rotation of seq."""
    k = len(str("".join(map(str,seq))))
    newStr = []
    for i in seq[1:]:
        newStr.append(i)
    newStr.append(seq[0])
    return tuple(newStr)

def jump(str:List[int]):
    """Apply the jump rule to str."""
    k = len(str)
    x = msr(str[:k-1], m)
    # If the sum in our current string is m then we cannot jump
    if x == 0:
        return "No jump"
    # Now we have to look at what we would be jumping to
    else:
        # Drop the first symbol from the current window
        newnode = list(str[1:k-1])
        # Add a decremented version of the new missing symbol to the end of the window
        newnode.append(x-1)
        newnode.append(msr(newnode, m))
        print("Newnode", newnode)

        # Now we can only make a jump to this node IF it is a child/parent of the current node
        n1, n2 = get_representative(str), get_representative(newnode)
        if parent_of(n1) == n2 or parent_of(n2) == n1:
            return get_representative(newnode)
        return "Not related in tree"


def findDecPos(str):
    """Find the rightmost position of the rightmost decrease in a sequence."""
    k = len(str)
    for i in range(k-1, -1, -1):
        if str[i] > str[(i+1)%k]:
            return i
    return -1


if __name__ == "__main__":
    k, m = 4, 4
    UClen = lengthOfUC(m, k)
    root = buildRoot(m, k)

    # Declare our cycle and load our root into the first part of it
    uc = [0]*UClen
    j = 0
    for i in root:
        uc[j] = i
        j+=1
    
    x = (2,1,1,0)
    x = nextRotation(x)
    x = nextRotation(x)
    print("Root:", root)
    print("Node:", x)
    print("Length of UC:", UClen)
    print(jump(x))
    print("UC:",uc)