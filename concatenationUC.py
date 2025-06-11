import sys

sys.setrecursionlimit(10**7)

# Globals
N = 0
K = 0
a = []  # 1-based: a[1]..a[N]

def necklace():
    p = 1
    for i in range(2, N+1):
        if a[i-p] > a[i]:
            return 0
        if a[i-p] < a[i]:
            p = i
    return p if (N % p == 0) else 0

def gen(t, output):
    """
    Recursive cool-lex generator.  Whenever we 'visit' a necklace,
    we append it to `output` instead of printing.
    """
    i = t
    while a[i] != a[1]:
        while a[i] == a[i-1]:
            i -= 1
        for j in range(i, t+1):
            # swap a[j-1]<->a[j]
            a[j-1], a[j] = a[j], a[j-1]
            if necklace() != 0:
                gen(j-1, output)
            else:
                # undo the sequence of swaps from j back to i
                for s in range(j, i-1, -1):
                    a[s-1], a[s] = a[s], a[s-1]
                # **visit**: collect the current necklace
                p = necklace()
                output.append([a[k] for k in range(p, 0, -1)])
                return
        # undo the block of swaps from t down to i
        for j in range(t, i-1, -1):
            a[j-1], a[j] = a[j], a[j-1]
        i -= 1
    # final visit when a[i]==a[1]
    p = necklace()
    output.append([a[k] for k in range(p, 0, -1)])

def make_cycles(counts):
    """
    counts: a list [N1, N2, …, NK]
    returns: a list of all the cool-lex necklaces (each is a list of symbols)
    """
    global N, K, a
    K = len(counts)
    N = sum(counts)
    a = [None]  # pad so that a[1]..a[N] are valid
    # build the initial content
    for symbol, cnt in enumerate(counts, start=1):
        a.extend([symbol] * cnt)

    output = []
    gen(N, output)
    return output

if __name__ == "__main__":
    # example usage: user inputs
    K = int(input("Enter K: "))
    counts = [int(input(f"N_{i+1}: ")) for i in range(K)]
    cycles = make_cycles(counts)

    # now you have them in a list — print or otherwise process:
    print(f"Generated {len(cycles)} necklaces:")
    print(cycles)
    for cyc in cycles:
        print(" ".join(map(str, cyc)))
