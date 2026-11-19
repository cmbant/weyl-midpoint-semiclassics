#!/usr/bin/env python3
"""Flux normalization of the Hahn/3j and Racah/6j chains as an exact Gauss-weight identity (rem:flux-gauss: the intrinsic route to delta_fl = 0; its semiclassical step is open).
# log: verify_gauss_weight_normalization.txt

Flux defect.  For the physical solution Y_k(x) (natural gauge) the complex Darboux branch has flux J = 1/2 + h^2 delta(x) + O(h^4), delta = c + J_2(0,x)
(lem:pullback-flux, thm:general-phase-exactness); alpha = -J_2 iff delta = 0.  Leading-order dictionary (Weyl / Gauss-Christoffel): for the
orthonormal eigenvector family Y(x_j) = lambda_j v^(j) of a finite chain with a hard edge, sum_k Y_k(x_j)^2 = lambda_j^2, and
    J(x_j) = lambda_j^2 w_j / 4 (1 + O(h^2)),   w_j := (x_{j-1} - x_{j+1})/2  (the node gap seen by the weight),
so delta = 0 is the statement  lambda_j^2 w_j = 2 (1 + O(h^4)).  For the 3j and 6j chains this holds EXACTLY with w_j the centered node gap:
    3j:  x = 2 mu/N, lambda^2 = sum_L (2L+1) Phi_L^2 = N, w = 2/N;          6j:  lambda^2 = 4 A B/(2 C), w = C/(A B)  (C = 2c+1 over 2).
Checks (exact rational arithmetic, small N): (A) 3j sum rule and lambda^2 w = 2 at every node; (B) 6j sum rule and lambda^2 w = 2 at every node;
(C) control: a truncated chain (hard edge moved into the oscillatory region) violates lambda^2 w = 2 -- orthonormality of a truncation does not give the
flux normalization; the identity is a property of the full hard-edge chain with centered Gauss weights.
Classification: exact (finite N).  The step lambda^2 w = 2 => delta = 0 (the O(h^2) semiclassical Weyl--Gauss lemma) is NOT proved here (open step).
"""
import sys
import sympy as sp
from sympy.physics.wigner import wigner_3j, wigner_6j

R = sp.Rational


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


print('(A) 3j: Phi_L = (-1)^{...} sqrt(N) 3j(L, jm, jp; M, mu-M/2, -mu-M/2), nodes x = 2 mu/N', flush=True)
for N, M_, S_ in [(7, 1, 0), (8, 2, 1), (9, 1, 1), (10, 2, 0)]:
    jm, jp = R(N - 1 - S_, 2), R(N - 1 + S_, 2)
    off = (jm + R(M_, 2)) % 1
    mus = [off + k for k in range(-N, N + 1) if abs(off + k - R(M_, 2)) <= jm and abs(off + k + R(M_, 2)) <= jp]
    kmin = max(abs(M_), abs(S_)); Ls = [kmin + j for j in range(int(jm + jp) - kmin + 1)]
    lam2 = {}
    for mu in mus:
        lam2[mu] = sum((2*L + 1)*N*wigner_3j(L, jm, jp, M_, mu - R(M_, 2), -mu - R(M_, 2))**2 for L in Ls)
    xs = [2*m/N for m in sorted(mus)]                                    # nodes x = 2 mu/N
    ok = all(sp.simplify(lam2[m] - N) == 0 for m in mus) and all(
        sp.simplify(lam2[sorted(mus)[i]]*(xs[i + 1] - xs[i - 1])/2 - 2) == 0 for i in range(1, len(xs) - 1))
    check(ok, '(A) 3j (M,sigma)=(%d,%d) N=%d: lambda^2 = N and lambda^2 (x_{j+1}-x_{j-1})/2 = 2 at all %d nodes' % (M_, S_, N, len(mus)))

print('\n(B) 6j: Psi_f = (-1)^{...} 2 sqrt(AB) {6j}, x = (A^2+B^2-C^2)/(2AB), C = c + 1/2', flush=True)
for (m_, n_), (A_, B_) in [((1, 0), (R(7, 2), R(9, 2))), ((2, 1), (R(9, 2), R(11, 2)))]:
    a6, b6 = A_ - R(n_ + 1, 2), B_ - R(m_ + 1, 2)
    Cs = [R(c, 2) for c in range(2, 26)]
    def lam(Cc):
        c6 = Cc - R(1, 2); tot = 0
        fmin = max(abs(R(m_)), abs(R(n_)))
        for f in range(int(fmin), int(min(2*a6 + n_, 2*b6 + m_)) + 1):
            tot += (2*f + 1)*4*A_*B_*wigner_6j(a6, b6, c6, b6 + m_, a6 + n_, f)**2
        return sp.nsimplify(sp.simplify(tot))
    good = []
    for Cc in Cs:
        c6 = Cc - R(1, 2)
        if not (abs(a6 - b6) <= c6 <= a6 + b6 and abs(a6 + n_ - b6 - m_) <= c6 <= a6 + n_ + b6 + m_ and c6.q in (1, 2)):
            continue
        try:
            l2 = lam(Cc)
        except ValueError:
            continue
        w = ((A_**2 + B_**2 - (Cc - 1)**2) - (A_**2 + B_**2 - (Cc + 1)**2))/(4*A_*B_)       # (x_{C-1} - x_{C+1})/2
        good.append(sp.simplify(l2 - 4*A_*B_/(2*Cc)) == 0 and sp.simplify(l2*w - 2) == 0)
    check(len(good) > 3 and all(good), '(B) 6j offsets (%d,%d) A=%s B=%s: lambda^2 = 4AB/(2C) and lambda^2 (x_{C-1}-x_{C+1})/2 = 2 (%d nodes)' % (m_, n_, A_, B_, len(good)))

print('\n(C) control: truncation (hard edge inside the oscillatory region)', flush=True)
N, M_, S_ = 9, 1, 1
jm, jp = R(N - 1 - S_, 2), R(N - 1 + S_, 2); off = (jm + R(M_, 2)) % 1
mu = off + 0
kmin = 1; Ls = [kmin + j for j in range(int(jm + jp) - kmin + 1)]
full = sum((2*L + 1)*N*wigner_3j(L, jm, jp, M_, mu - R(M_, 2), -mu - R(M_, 2))**2 for L in Ls)
trunc = sum((2*L + 1)*N*wigner_3j(L, jm, jp, M_, mu - R(M_, 2), -mu - R(M_, 2))**2 for L in Ls[:len(Ls)//2])
print('   central node: full sum = %s, half-truncated sum = %s' % (full, sp.nsimplify(trunc)), flush=True)
check(sp.simplify(full - N) == 0 and sp.simplify(trunc - N) != 0 and abs(float(trunc) - N) > 0.1,
      '(C) a truncated chain does not satisfy lambda^2 = N: the Gauss-weight identity is a hard-edge property')
print('ALL CHECKS PASSED')
