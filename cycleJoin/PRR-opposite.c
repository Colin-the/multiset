//----------------------------------------------------------------------------------------
// Shift (successor) rules to construct binary de Bruijn sequences based on
// the Pure Run-length Register (PRR) for n>2. This includes the prefer-opposite
// greedy sequence and related.
//
// RESEARCH BY: Joe Sawada
// PROGRAMMED BY: Joe Sawada -- 2019, udpdated for debruijnsequence.org 2020
//----------------------------------------------------------------------------------------
#include<stdio.h>
#include<math.h>
#define N_MAX 50

int n;

// =============================================================================
// Compute the RLE of a[s..m] in run[1..r], returning r = run length
// =============================================================================
int RLE(int a[], int run[], int s, int m) {
    int i,j,r,old;
    
    old = a[m+1];
    a[m+1] = 1 - a[m];
    r = j = 0;
    for (i=s; i<=m; i++) {
        if (a[i] == a[i+1]) j++;
        else {  run[++r] = j+1;  j = 0;  }
    }
    a[m+1] = old;
    return r;
}
// ===============================================================================
// Check if a[1..n] is a "special" RL representative: the RLE of a[1..n] is of
// the form 1 x^j y where y > x and j is odd. Eg. 12224, 1111113 (PCR-related)
// ===============================================================================
int Special(int a[]) {
    int i,r,rle[N_MAX];
    
    r = RLE(a,rle,1,n);
    if (r%2 == 0) return 0;
    for (i=3; i<r; i++) if (rle[i] != rle[2]) return 0;
    if  (a[1] == 1 && a[2] == 0 && i == r && rle[r] > rle[2]) return 1;
    return 0;
}
// =============================================================================
// Apply PRR^{t} to a[1..n] to get b[1..n], where t is the length of the
// prefix in a[1..n] before the first 01 or 10 in a[2..n]
// =============================================================================
int Shift(int a[], int b[]) {
    int i,t=1;
    
    while (a[t+1] == a[t+2] && t < n-1) t++;
    for (i=1; i<=n; i++) b[i] = a[i];
    for (i=1; i<=n; i++) b[i+n] = (b[i] + b[i+1] + b[n+i-1]) % 2;
    for (i=1; i<=n; i++) b[i] = b[i+t];
    return t;
}
// =============================================================================
// Test if b[1..len] is the lex smallest rep (under rotation), if so, return the
// period p; otherwise return 0. Eg. (114114, p=3)(11244, p=5)(124114, p=0).
// =============================================================================
int IsSmallest(int b[], int len) {
    int i, p=1;
    for (i=2; i<=len; i++) {
        if (b[i-p] > b[i]) return 0;
        if (b[i-p] < b[i]) p = i;
    }
    if (len % p != 0) return 0;
    return p;
}
// =============================================================================
// Membership testers with special case for 111111...1  (run length for a[2..n])
// =============================================================================
int RLrep(int a[]) {
    int p,r,rle[N_MAX];
    
    r = RLE(a,rle,2,n);
    if (r == 1) return 1;       // Special case: a[1..n] = 000..0 or 111..1
    if (a[1] == a[2]) return 0;
    p = IsSmallest(rle,r);

    if (a[1] == a[n] && p > 0 && (p == r || a[1] == 0 || p%2 == 0)) return 1;  //PCR-related
    if (a[1] != a[n] && p > 0 && a[1] == 0) return 1;  // CCR-related
    return 0;
}
// =============================================================================
int LCrep(int a[]) {
    int t,b[N_MAX];
    
    if (a[1] == a[2]) return 0;
    t = Shift(a,b);
    return RLrep(b);
}
// =============================================================================
int OppRep(int a[]) {
    int b[N_MAX];
    
    Shift(a,b);
    if (Special(a) || (LCrep(a) && !Special(b))) return 1;
    return 0;
}
// =============================================================================
// Repeatedly apply the Prefer Opp or LC or RL successor rule starting with 1^n
// =============================================================================
void DB(int type) {
    int i,j,v,a[N_MAX],REP;

    // Initial string
    for (i=1; i<=n; i+=2) a[i] = 0;
    for (i=2; i<=n; i+=2) a[i] = 1;
    
    for (j=1; j<=pow(2,n); j++) {
        printf("%d", a[1]);
        
        v = (a[1] + a[2] + a[n]) % 2;
        REP = 0;
        // Membership testing of a[1..n]
        if (type == 1 && OppRep(a)) REP = 1;
        if (type == 2 && LCrep(a)) REP = 1;
        if (type == 3 && RLrep(a)) REP = 1;
        
        // Membership testing of conjugate of a[1..n]
        a[1] = 1 - a[1];
        if (type == 1 && OppRep(a)) REP = 1;
        if (type == 2 && LCrep(a)) REP = 1;
        if (type == 3 && RLrep(a)) REP = 1;

        // Shift String and add next bit
        for (i=1; i<n; i++) a[i] = a[i+1];
        if (REP) a[n] = 1 - v;
        else a[n] = v;
    }
}
//------------------------------------------------------
int main(int argc, char **argv) {
    int type;
    
    printf("Enter (1) Prefer-opposite (2) LC2 (3) Run-length2: ");  scanf("%d", &type);
    printf("Enter n>2: ");   scanf("%d", &n);

    DB(type);
}