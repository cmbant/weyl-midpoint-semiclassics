#!/usr/bin/env python3
"""Decay of the physical solution beyond the rank turn (prop:forbidden-decay) for the Hahn/3j and centered Racah/6j chains.
# log: verify_forbidden_side_decay.txt

Setting (gauge tilde Y = chi_tau Y > 0 on the hyperbolic side): c_k tY_k = A_{k+1} tY_{k+1} + A_k tY_{k-1}, c_k = tau (x - B_k), hard edge
A_{kmax+1} = 0, ratios rho_k = tY_{k+1}/tY_k from the backward continued fraction rho_{k-1} = A_k/(c_k - A_{k+1} rho_k), rho_kmax = 0.
Theorem proved in the text: with u(s) = tau (x - beta_0 s)/e(s), Theta = arcosh u (increasing on [sigma_*, sigma_max)), kappa = (1 - tau x)/sigma_*,
q_m^2 = sigma_max, theta_max = Theta(sigma_e) at the end of the window, and the explicit constants
  rho_1 = 4 (a^2 + b^2 + 1/3)/sigma_*,  rho_2 = beta_0 (1 + 2(m^2+n^2))/4 + 4|mn|/sigma_*,
  D_3 = beta_0 q_m + beta_0/4 + rho_2 + rho_1 cosh(theta_max),   D_4 = q_m + D_3/kappa,   D_5 = D_4 Theta_max/(2 q_*),
for every rank j in the window,  y_j := exp(-Theta(q_j^2 - D_4 h))  satisfies
  (a) F_i(y_j) = A_{i+1} y_j + A_i/y_j <= c_i for ALL i >= j up to the hard edge   (hence rho_{j-1} <= y_j, by backward induction);
  (b) prod_{j=k1+1}^{e} rho_{j-1} <= exp(-N int_{q_k1}^{q_e} Theta(q^2) dq + D_5)   (the decay is at least the frozen rate, up to a constant).
Checks (exact coefficients in mpmath, inequalities in double precision with the margins stated): (a), the consequence rho_{j-1} <= y_j, and (b);
(C) negative control: for a non-concave chain the hypothesis F_i(1) <= c_i fails beyond the turn.
Classification: finite-grid checks of inequalities proved in the text (the proof is the argument, not the scan); (D) numerical evidence that
the backward continued fraction tracks the frozen rate to relative O(h), which is not needed for the proof.
"""
import sys
import os
import numpy as np
import mpmath as mp
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_airy2_numerics import Chain, setup as make_chain  # noqa: E402

mp.mp.dps = 30
R = sp.Rational


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


def prepare(kind, m_, n_, ab, x, N):
    ch = make_chain(kind, sp.N(m_, 30), sp.N(n_, 30), N, ab)
    t = ch.turn(x)
    if t is None:
        return None
    qs, tau = t
    kmax = ch.N*mp.sqrt(ch.edge_sigma()) - 1
    ks = [ch.kmin + j for j in range(int(mp.floor(kmax - ch.kmin + mp.mpf('1e-12'))) + 1)]
    A = {k: ch.A(k) for k in ks + [ks[-1] + 1]}
    c = {k: tau*(x - ch.B(k)) for k in ks}
    rho = {ks[-1]: mp.mpf(0)}
    for k in reversed(ks[1:]):
        rho[k - 1] = A[k]/(c[k] - A[k + 1]*rho[k])
    return ch, qs, tau, ks, A, c, rho


def constants(ch, qs, tau, x, qe):
    sg = qs*qs
    beta0 = mp.mpf(0) if ch.kind == 'hahn' else 1/(4*ch.ab[0]*ch.ab[1])
    kappa = (1 - tau*x)/sg
    rho1 = 4*(ch.m**2 + ch.n**2 + mp.mpf(1)/3)/sg
    rho2 = beta0*(1 + 2*(ch.m**2 + ch.n**2))/4 + 4*abs(ch.m*ch.n)/sg
    qm = mp.sqrt(ch.edge_sigma())
    u = lambda s: tau*(x - beta0*s)/ch.e(s)
    Th = lambda s: mp.acosh(u(s))
    thmax = Th(qe*qe)
    D3 = beta0*qm + beta0/4 + rho2 + rho1*mp.cosh(thmax)
    D4 = qm + D3/kappa
    D5 = D4*thmax/(2*qs)
    return beta0, kappa, qm, u, Th, D3, D4, D5


print('(a)-(b) exact continued fraction on the whole hyperbolic side up to the hard edge', flush=True)
CASES = [('hahn', 2, 1, None, mp.mpf('0.5')), ('hahn', 1, 0, None, mp.mpf('-0.45')), ('hahn', 0, 0, None, mp.mpf('0.8')),
         ('racah', 2, 1, (mp.mpf(1), mp.mpf('1.2')), mp.mpf('0.3')), ('racah', 1, 0, (mp.mpf(1), mp.mpf(2)), mp.mpf('-0.2')),
         ('racah', 1, 0, (mp.mpf(1), mp.mpf(2)), mp.mpf('0.1'))]
Ns = [6400, 25600]
for kind, m_, n_, ab, x in CASES:
    sl_min = None
    for N in Ns:
        ch, qs, tau, ks, A, c, rho = prepare(kind, m_, n_, ab, x, N)
        h = ch.h; qm0 = mp.sqrt(ch.edge_sigma())
        qe = qs + (qm0 - qs)/2                         # end of the window (a fixed distance from turn and hard edge)
        beta0, kappa, qm, u, Th, D3, D4, D5 = constants(ch, qs, tau, x, qe)
        qk = lambda k: (mp.mpf(k) + mp.mpf(1)/2)*h
        kk = [k for k in ks if qk(k)**2 - D4*h >= qs*qs + N**(-mp.mpf(2)/3)]
        k1 = kk[0] - 1                                  # last rank below the first admissible one
        ke = max(k for k in ks if qk(k) <= qe)
        win = [k for k in kk if k <= ke]
        Af = np.array([float(A[k]) for k in ks + [ks[-1] + 1]]); cf = np.array([float(c[k]) for k in ks])
        idx = {k: i for i, k in enumerate(ks)}
        worst_margin = np.inf; worst_ratio = -np.inf
        for jj in win[::max(1, len(win)//40)]:
            y = float(mp.exp(-Th(qk(jj)**2 - D4*h)))
            i0 = idx[jj]
            marg = cf[i0:] - Af[i0 + 1:len(ks) + 1]*y - Af[i0:len(ks)]/y     # c_i - F_i(y), i >= jj
            worst_margin = min(worst_margin, marg.min())
            worst_ratio = max(worst_ratio, float(rho[jj - 1])/y)
        check(worst_margin >= -1e-12, '(a) %s (%s,%s) x=%+.2f N=%d tau=%+d: F_i(y_j) <= c_i for all i >= j to the hard edge (D4=%.1f, min margin %.2e)'
              % (kind, m_, n_, float(x), N, tau, float(D4), worst_margin))
        check(worst_ratio <= 1 + 1e-12, '(a) %s (%s,%s) N=%d: rho_{j-1} <= y_j (max rho/y = %.4f)' % (kind, m_, n_, N, worst_ratio))
        lhs = mp.fsum(-mp.log(rho[j - 1]) for j in win)
        I = mp.quad(lambda q: Th(q*q), [qk(kk[0] - 1), qe])*N
        # window indices j = kk[0]..ke correspond to q from q_{k1} to q_{ke}
        rhs = N*mp.quad(lambda q: Th(q*q), [qk(k1), qk(ke)]) - D5
        check(lhs >= rhs, '(b) %s (%s,%s) N=%d: sum -log rho = %.3f >= N int Theta - D5 = %.3f (D5 = %.2f)' % (kind, m_, n_, N, float(lhs), float(rhs), float(D5)))
        # (D) evidence: relative O(h) tracking of the frozen rate
        dev = max(abs(mp.log(rho[j - 1]) + Th(qk(j)**2)) for j in win)
        sl_min = (N, float(dev)) if sl_min is None else sl_min + (N, float(dev))
    print('   (D) evidence: max_j |log rho_{j-1} + Theta(q_j^2)| at N = %d, %d: %.4f, %.4f (O(h) expected)' % (sl_min[0], sl_min[2], sl_min[1], sl_min[3]), flush=True)

print('\n(C) negative control: e = (1-s)^2, v = s (not hyperbolic beyond its turn): F_i(1) <= c_i, the hypothesis of the comparison, fails', flush=True)
x = mp.mpf('0.9'); bad = []
for N in (200, 400):
    ch = Chain('bad', 0, 0, N, None); qs, tau = ch.turn(x)
    ks = list(range(1, N))
    margin = [tau*(x - ch.B(k)) - ch.A(k) - ch.A(k + 1) for k in ks if (k + .5)/N > qs + 0.08]
    bad.append(min(margin))
    print('   N=%d: min over k > q*N + 0.08N of c_k - A_k - A_{k+1} = %.4f (F_k(1) <= c_k is hyperbolicity)' % (N, float(min(margin))), flush=True)
check(all(b < -1e-3 for b in bad), '(C) non-concave chain: c_k < A_k + A_{k+1} beyond the turn, so the comparison with a ratio y <= 1 is unavailable')
print('ALL CHECKS PASSED')
