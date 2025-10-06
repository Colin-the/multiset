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

def parent_of(rep, m, k):
    """
    Apply the parent-rule to rep, then re-normalize to lex-greatest rotation.
    """
    # As the root has no parent
    if rep == buildRoot(m, k):
        return None
    
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

def jump(str:List[int], m, k):
    """Apply the jump rule to str."""
    k = len(str)
    x = msr(str[:k-1], m)

    # Now Based on what the missing symbol is we can potentially jump from the 
    # current node to a parent or child of the current node
    # If the sum in our current string is m then we cannot jump
    if x == 0:
        print("x = 0 no jump for ", str)
        return None
    
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
        p1,p2 = parent_of(n1, m, k), parent_of(n2, m, k)

        print("n1: ",n1," n2: ",n2)
        print("p1: ",p1," p2: ",p2)

        if ((p1 is None)):
            if (list(p2) == list(n1)):
                return x-1
            else:
                print("n1 is root and no jmp")
                return None
        elif ((p2 is None)):
            if (list(p1) == list(n2)):
                return x-1
            else:
                print("n2 is root and no jmp")
                return None

            
        
        

        # print(list(parent_of(n2)), " and ",list(n1))

        if (list(p1) == list(n2)) or (list(p2) == list(n1)):
            return x-1
        
        print("Not related in tree", str, " and ",newnode)
        return None


def findDecPos(str):
    """Find the rightmost position of the rightmost decrease in a sequence."""
    k = len(str)
    for i in range(k-1, -1, -1):
        if str[i] > str[(i+1)%k]:
            return i
    return -1


if __name__ == "__main__":
    k, m = 4, 4
    currentLen = 0
    UClen = lengthOfUC(m, k)
    root = buildRoot(m, k)

    # Declare our cycle and load our root into the first part of it
    uc = [0]*UClen
    for i in root:
        # As we want to leave one symbol out initialy
        if currentLen < k - 1:
            uc[currentLen] = i
            currentLen+=1

    currentNode = root
    while currentLen < UClen - 20:
        print("LP start: ", currentNode)
        # Check and see if we need to jump to another cycle or if we can contine on this one
        next = jump(currentNode, m, k)

        # If we are going to be staying on our current cycle
        if next is None:
            # append the missing symbol in the current cycle to the UC
            print("Appilying MSR to ",uc[currentLen-k+1:currentLen])
            uc[currentLen] = msr(uc[currentLen-k+1:currentLen],m)
            currentLen+=1 

            # Now we will still be on the same node but just rotated one pos
            currentNode = nextRotation(currentNode)   

        # If we are going to be jumping to some other node
        else:
            # Insert our next symbol into the cycle
            uc[currentLen] = next
            currentLen+=1

            # The node we are jumping to is the last k-1 symbols of the current cycle
            # and we can appliy the msr on this to find the full node
            newNode = uc[currentLen-k+1:currentLen]
            newNode.append(msr(newNode, m))
            currentNode = tuple(newNode)

        print("Current UC:",uc[:currentLen],"\n")


    # x = (0,0,3,1) 
    # # x = (2,1,1,0)
    # # x = nextRotation(x)
    # # x = nextRotation(x)
    # print("Root:", root)
    # print("Node:", x)
    # print("Length of UC:", UClen)
    # print(jump(x, m, k))
    # print("UC:",uc)