
//----------------------------------------------------------------------------------------
// Shift (successor) rules to construct binary de Bruijn sequences based on
// the Pure Run-length Register (PRR) for n>2.  This includes the prefer-same
// greedy sequence and one by concatenating lexicographic compositions
//
// RESEARCH BY: Joe Sawada, Evan Sala
// PROGRAMMED BY: Joe Sawada -- 2019, udpdated for debruijnsequence.org 2020
//----------------------------------------------------------------------------------------

#include<stdio.h>
#include<math.h>
#define N_MAX 100

int n;

// =============================================================================
// Compute the RLE of a[1..m] in run[1..r], returning r = run length
// =============================================================================
int RLE(int a[], int run[], int m) {
    int i,j,r,old;
    
    old = a[m+1];
    a[m+1] = 1 - a[m];
    r = j = 0;
    for (i=1; i<=m; i++) {
        if (a[i] == a[i+1]) j++;
        else {  run[++r] = j+1;  j = 0;  }
    }
    a[m+1] = old;
    return r;
}
// ===============================================================================
// Check if a[1..n] is a "special" RL representative.  It must be that a[1] = a[n]
// and the RLE of a[1..n] is of the form (21^j)^s1^t where j is even, s >=2, t>=2
// ===============================================================================
int Special(int a[]) {
    int i,j,r,s,t,run[N_MAX];
    
    if (a[1] != 0 || a[n] != 0) return 0;
    
    r = RLE(a,run,n);
    
    // Compute j of prefix 21^j
    if (run[1] != 2) return 0;
    j = 0;
    while (run[j+2] == 1 && j+2 <= r) j++;
    
    // Compute s of prefix (21^j)^s
    s = 1;
    while (s <= r/(1+j) -1 && run[s*(j+1)+1] == 2) {
        for (i=1; i<=j; i++) if (run[s*(j+1)+1+i] != 1) return 0;
        s++;
    }
    
    // Test remainder of string is (21^j)^s is 1^t
    for (i=s*(j+1)+1; i<=r; i++) if (run[i] != 1) return 0;
    t = r - s*(1+j);
    
    if (s >= 2 && t >= 2 && j%2 == 0) return 1;
    return 0;
}
// =============================================================================
// Apply PRR^{t+1} to a[1..n] to get b[1..n], where t is the length of the
// prefix before the first 00 or 11 in a[2..n] up to n-2
// =============================================================================
int Shift(int a[], int b[]) {
    int i,t = 0;
    
    while (a[t+2] != a[t+3] && t < n-2) t++;
    for (i=1; i<=n; i++) b[i] = a[i];
    for (i=1; i<=n; i++) b[i+n] = (b[i] + b[i+1] + b[n+i-1]) % 2;
    for (i=1; i<=n; i++) b[i] = b[i+t+1];
    return t;
}
// =============================================================================
// Test if b[1..len] is the lex largest rep (under rotation), if so, return the
// period p; otherwise return 0. Eg. (411411, p=3)(44211, p=5) (411412, p=0).
// =============================================================================
int IsLargest(int b[], int len) {
    int i, p=1;
    for (i=2; i<=len; i++) {
        if (b[i-p] < b[i]) return 0;
        if (b[i-p] > b[i]) p = i;
    }
    if (len % p != 0) return 0;
    return p;
}
// =============================================================================
// Membership testers not including the cycle containing 0101010...
// =============================================================================
int RLrep(int a[]) {
    int p,r,rle[N_MAX];
    
    r = RLE(a,rle,n-1);
    p = IsLargest(rle,r);

    // PCR-related cycle
    if (a[1] == a[n]) {
        if (r == n-1 && a[1] == 1) return 0;  // Ignore root a[1..n] = 1010101..
        if (r == 1) return 1;  // Special case: a[1..n] = 000..0 or 111..1
        if (p > 0 && a[1] != a[n-1] && (a[1] == 1 || p%2 == 0)) return 1;
    }
    // CCR-related cycle
    if (a[1] != a[n]) {
        if (p > 0 && a[1] == 1 && a[n-1] == 1) return 1;
    }
    return 0;
}
// =============================================================================
int LCrep(int a[]) {
    int b[N_MAX];
    
    if (a[1] != a[2]) return 0;
    Shift(a,b);
    return RLrep(b);
}
// =============================================================================
int SameRep(int a[]) {
    int b[N_MAX];
    
    Shift(a,b);
    if (Special(a) || (LCrep(a) && !Special(b))) return 1;
    return 0;
}
// =============================================================================
// Repeatedly apply the Prefer Same or LC or RL successor rule starting with 1^n
// =============================================================================
void DB(int type) {
    int i,j,v,a[N_MAX],REP;

    for (i=1; i<=n; i++) a[i] = 1;  // Initial string
    
    for (j=1; j<=pow(2,n); j++) {
        printf("%d", a[1]);
        
        v = (a[1] + a[2] + a[n]) % 2;
        REP = 0;
        // Membership testing of a[1..n]
        if (type == 1 && SameRep(a)) REP = 1;
        if (type == 2 && LCrep(a)) REP = 1;
        if (type == 3 && RLrep(a)) REP = 1;
        
        // Membership testing of conjugate of a[1..n]
        a[1] = 1 - a[1];
        if (type == 1 && SameRep(a)) REP = 1;
        if (type == 2 && LCrep(a)) REP = 1;
        if (type == 3 && RLrep(a)) REP = 1;

        // Shift String and add next bit
        for (i=1; i<n; i++) a[i] = a[i+1];
        if (REP) a[n] = 1 - v;
        else a[n] = v;
    }
}
//------------------------------------------------------
int main() {
    int type;
    
    printf("Enter (1) Prefer-same (2) LC (3) Run-length: ");  scanf("%d", &type);
    printf("Enter n>2: ");   scanf("%d", &n);
 
    DB(type);
}