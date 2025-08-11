//----------------------------------------------------------------------------------------
// Shift (successor) rules to construct binary de Bruijn sequences based on
// the PCR and CCR feedback shift registers.
//
// RESEARCH BY: Joe Sawada, Dennis Wong, Aaron Willams, Daniel Gabric
// PROGRAMMED BY: Joe Sawada -- 2016, udpdated for debruijnsequence.org 2019
//----------------------------------------------------------------------------------------

#include<stdio.h>
#define MAX 100

// =====================
// Test if a[1..n] = 0^n
// =====================
int Zeros(int a[], int n) {
    for (int i=1; i<=n; i++) if (a[i] == 1) return 0;
    return 1;
}
// =============================
// Test if b[1..n] is a necklace
// =============================
int IsNecklace(int b[], int n) {
    int i, p=1;

    for (i=2; i<=n; i++) {
        if (b[i-p] > b[i]) return 0;
        if (b[i-p] < b[i]) p = i;
    }
    if (n % p != 0) return 0;
    return 1;
}
// ====================================
// Returns true if x[1..n] > y[1..n]
// ====================================
int Larger(int x[], int y[], int n) {
    int i;
    
    for (i=1; i<=n; i++) {
        if (x[i] < y[i]) return 0;
        if (x[i] > y[i]) return 1;
    }
    return 0;
}
// ===========================================================
// Compute number of 01 and 10 substrings in x[1..n],(1-x[1])
// ===========================================================
int Diff(int x[], int n) {
    int i,d=0;
    
    for (i=1; i<n; i++) if (x[i] != x[i+1]) d++;
    if (x[n] == x[1]) d++;
    return d;
}
// ===========================================
// Necklace Successor Rules
// ===========================================
int GrandDaddy(int a[], int n) {
    int i,j,b[MAX];
    
    j = 2;
    while (j<=n && a[j] == 1) j++;
    for (i=j; i<=n; i++) b[i-j+1] = a[i];
    b[n-j+2] = 0;
    for (i=2; i<j; i++) b[n-j+i+1] = a[i];
    
    if (IsNecklace(b,n)) return 1-a[1];
    return a[1];
}
// -------------------------------
int GrandMamma(int a[], int n) {
    int i,j,k,b[MAX];
    
    j = 1;
    while (j<n && a[n-j+1] == 0) b[j++] = 0;
    b[j] = 1;
    k = 2;
    for (i=j+1; i<=n; i++) b[i] = a[k++];
    
    if (IsNecklace(b,n)) return 1-a[1];
    return a[1];
}
// -------------------------------
int Williams(int a[], int n) {
    int i,b[MAX];
    
    b[1] = 0;
    for (i=2; i<=n; i++) b[i] = a[i];
    
    if (IsNecklace(b,n)) return 1-a[1];
    return a[1];
}
// -------------------------------
int Wong(int a[], int n) {
    int i,b[MAX];
    
    for (i=1; i<n; i++) b[i] = a[i+1];
    b[n] = 1;
    
    if (IsNecklace(b,n)) return 1-a[1];
    return a[1];
}
// ===========================================
// Co-necklace Successor Rules
// ===========================================
int Sawada(int a[], int n) {
    int i,j,b[MAX],c=1;
    
    for (i=2; i<=n; i++) if (a[i] == 0) break;
    for (j=i; j<=n; j++) b[c++] = a[j];
    b[c++] = 1;
    for (j=2; j<i; j++)  b[c++] = 1-a[j];
    for (i=1; i<=n; i++) b[n+i] = 1-b[i];

    if (IsNecklace(b,2*n)) return a[1];
    return 1-a[1];
}
// -------------------------------
int S2(int a[], int n) {
    int i,j,b[MAX],c=1;
    
    i = n;
    while(a[i] == 0 && i >=1) i--;
    if(i == 0) i = n;
    for (j=i+1; j<=n; j++) b[c++] = 0;
    b[c++] = 1;
    for (j=2; j<=i; j++) b[c++] = 1-a[j];
    for (j=1; j<=n; j++) b[n+j] = 1-b[j];
    
    if (IsNecklace(b,2*n)) return a[1];
    return 1-a[1];
}
// -------------------------------
int Gabric(int a[], int n) {
    int i,b[MAX];
    
    for (i=1; i<n; i++) b[i] = a[i+1];
    b[n] = 0;
    for (i=1; i<=n; i++) b[n+i] = 1-b[i];
    
    if (IsNecklace(b,2*n) && !Zeros(b,n)) return a[1];
    return 1-a[1];
}
// -------------------------------
int Huang(int a[], int n) {
    int gamma[MAX],i,j,d1,d2,d3,sigma[MAX];
   
    gamma[1] = 0;
    for (i=2; i<=n; i++)   gamma[i] = a[i];
    for (i=1; i<=2*n; i++) gamma[i+n] = 1-gamma[i];
    
    d1 = Diff(gamma,n);
    gamma[1] = 1;
    d2 = Diff(gamma,n);
    gamma[1] = 0;

    // Special case x11..1
    for (i=2; i<=n; i++) if (a[i] == 0) break;
    if (i > n) return 1 - a[1];
    
    // Condition P1
    if (d2 < d1) {
        for (i=1; i<2*n; i++) {
            for (j=1; j<=n; j++) sigma[j] = gamma[i+j];
            sigma[1] = 1 - sigma[1];
            d3 = Diff(sigma,n);
            sigma[1] = 0;
            if (d3 < d1 && Larger(sigma,gamma,n)) return 1-a[1];
        }
        return a[1];
    }
    // Condition P2
    if (d1 == d2) {
        for (i=1; i<2*n; i++) {
            for (j=1; j<=n; j++) sigma[j] = gamma[i+j];
            sigma[1] = 1 - sigma[1];
            d3 = Diff(sigma,n);
            sigma[1] = 0;
            if ((d3 < d1  || (d3 == d1  && Larger(sigma,gamma,n) == 1))) return 1-a[1];
        }
        return a[1];
    }
    return 1-a[1];
}
// =====================================================================
// Generate de Bruijn sequences by iteratively applying a successor rule
// =====================================================================
void DB(int seq, int n) {
    int i, new_bit, a[MAX];

    for (i=1; i<=n; i++)  a[i] = 0;   // First n bits
    do {
        printf("%d", a[1]);
        switch(seq) {
            case 1: new_bit = GrandDaddy(a,n); break;
            case 2: new_bit = GrandMamma(a,n); break;
            case 3: new_bit = Wong(a,n); break;
            case 4: new_bit = Williams(a,n); break;
            case 5: new_bit = Sawada(a,n); break;
            case 6: new_bit = S2(a,n); break;
            case 7: new_bit = Gabric(a,n); break;
            case 8: new_bit = Huang(a,n); break;

            default: break;
        }
        for (i=1; i<=n; i++) a[i] = a[i+1];
        a[n] = new_bit;
    } while (!Zeros(a,n));
}
// ===========================================
int main() {
    int i, n;
    
    printf("Enter n: ");    scanf("%d", &n);
    for (i=1; i<=8; i++) {
        switch(i) {
            case 1: printf("PCR1 GrandDaddy (lex minimal):\n"); break;
            case 2: printf("PCR2 GrandMamma:\n"); break;
            case 3: printf("PCR3 Wong:\n"); break;
            case 4: printf("PCR4 Williams:\n");  break;
            case 5: printf("CCR1 Sawada:\n"); break;
            case 6: printf("CCR2 S2:\n"); break;
            case 7: printf("CCR3 Gabric:\n"); break;
            case 8: printf("CCR4 Huang:\n"); break;
            default: break;
        }
        DB(i,n);
        printf("\n\n");
    }
}