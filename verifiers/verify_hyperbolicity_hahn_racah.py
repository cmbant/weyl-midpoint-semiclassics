#!/usr/bin/env python3
"""Exact hyperbolicity beyond the rank turn for the Hahn/3j and centered Racah/6j chains (prop:exact-hyperbolicity).
# log: verify_hyperbolicity_hahn_racah.txt

Statement: with c_k = tau (x - B_k), tau (x - B_k) >= A_k + A_{k+1} for every rank k <= k_max (hard edge A_{k_max+1} = 0) with
q_k >= q_* + C_0 N^-2, with the explicit constant C_0 = (rho_1 + rho_2)/(2 q_* kappa) of the proof, where
  sigma = q^2, e concave and decreasing on [0, sigma_max], b_0 = beta_0 sigma (beta_0 = 0 Hahn, 1/(4 alpha beta) Racah),
  Psi_tau = beta_0 sigma + tau e, kappa = (1 - tau x)/sigma_*, A_k = r_k e(k^2 h^2)/2, |r_k - 1| <= rho_1 h^2 (k >= q_* N/2),
  |B_k - beta_0 q_k^2| <= rho_2 h^2.
Checks:  (A) exact symbolic identities used (concavity of e, midpoint form of B_k, secant slopes);
(B) exact-coefficient scan: the intermediate inequality tau (x_edge(k) - Psi_tau(q_k^2)) <= (rho_1+rho_2) h^2 for all k >= q_* N/2, the slope
bound tau Psi_tau' <= -kappa, and hyperbolicity beyond q_* + C_0 N^-2 itself, Hahn and Racah, both turn types;
(C) negative control: a chain with non-concave Psi (e = (1-s)^2, v = s) is not hyperbolic beyond its turn.
Classification: (A) exact symbolic; (B), (C) finite-grid checks of the inequalities proved in the text (the proof is the argument, not the scan).
"""
import sys
import os
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


# ---------------- (A) exact symbolic ----------------
s, al, be = sp.symbols('s alpha beta', positive=True)
a_, b_ = 4*al**2, 4*be**2
e = sp.sqrt((1 - s/a_)*(1 - s/b_))
check(sp.simplify(sp.diff(e, s, 2)*4*e**3 + (1/a_ - 1/b_)**2) == 0,
      '(A) e\'\'(s) = -(1/a-1/b)^2/(4 e^3) <= 0 (a = 4 alpha^2, b = 4 beta^2): e is concave (Hahn: e\'\' = -1/(4 e^3))')
check(sp.simplify((sp.diff(e, s) + 1/(4*al*be)).subs(s, 0) + (al - be)**2/(8*al**2*be**2)) == 0,
      '(A) Psi_+\'(0) = e\'(0) + 1/(4 alpha beta) = -(alpha-beta)^2/(8 alpha^2 beta^2) <= 0: Psi_+ decreasing; Psi_-\' = beta_0 - e\' > 0')
kk, h, m, n = sp.symbols('k h m n', positive=True)
Bk = (2*(kk + R(1, 2))**2*h**2 + h**2*(R(1, 2) - m**2 - n**2))/(8*al*be)
check(sp.simplify(Bk - (((kk*h)**2 + ((kk + 1)*h)**2)/(8*al*be) - h**2*(m**2 + n**2)/(8*al*be))) == 0,
      '(A) Racah B_k (without the mn term) = (sigma_k + sigma_{k+1})/(8 alpha beta) - h^2 (m^2+n^2)/(8 alpha beta), sigma_k = k^2 h^2')
check(sp.simplify(((kk + R(1, 2))**2 + R(1, 4)) - (kk**2 + (kk + 1)**2)/2) == 0 and
      sp.simplify(((kk**2 + (kk + 1)**2)/2 - (kk + R(1, 2))**2) - R(1, 4)) == 0,
      '(A) (sigma_k + sigma_{k+1})/2 = q_k^2 + h^2/4')
# |r_k - 1| <= (a^2 + b^2 + 1/3)/k^2 (offsets a, b) for k > max(|a|,|b|): exhaustive check on a grid of offsets and ranks
bad = 0
for aa in [R(j, 2) for j in range(0, 9)]:
    for bb in [R(j, 2) for j in range(-8, 9)]:
        if (aa - bb).q != 1 and (2*aa) % 2 != (2*bb) % 2:
            continue
        for k in [max(abs(aa), abs(bb)) + 1 + i*R(1, 1) for i in range(0, 400)]:
            r = mp.sqrt((1 - mp.mpf(aa)**2/k**2)*(1 - mp.mpf(bb)**2/k**2)/(1 - 1/(4*mp.mpf(k)**2)))
            if abs(r - 1) > (mp.mpf(aa)**2 + mp.mpf(bb)**2 + mp.mpf(1)/3)/mp.mpf(k)**2:
                bad += 1
check(bad == 0, '(A) |r_k - 1| <= (a^2 + b^2 + 1/3)/k^2 for all k > max(|a|,|b|) (grid of offsets, k up to 400)')


# ---------------- (B) explicit constants ----------------
def constants(ch, qs, tau, x):
    sg = qs*qs
    beta0 = mp.mpf(0) if ch.kind == 'hahn' else 1/(4*ch.ab[0]*ch.ab[1])
    kappa = (1 - tau*x)/sg
    cr = ch.m**2 + ch.n**2 + mp.mpf(1)/3
    rho1 = 4*cr/sg
    rho2 = beta0*(1 + 2*(ch.m**2 + ch.n**2))/4 + 4*abs(ch.m*ch.n)/sg
    return beta0, kappa, rho1, rho2


def scan(ch, x):
    t = ch.turn(x)
    if t is None:
        return None
    qs, tau = t
    N, h = ch.N, ch.h
    beta0, kappa, rho1, rho2 = constants(ch, qs, tau, x)
    C0 = (rho1 + rho2)/(2*qs*kappa)
    kmax = N*mp.sqrt(ch.edge_sigma()) - 1
    k0 = int(mp.ceil(qs*N/2 - ch.kmin)) + 1
    ks = [ch.kmin + j for j in range(k0, int(mp.floor(kmax - ch.kmin + mp.mpf('1e-12'))) + 1)]
    assert abs(ks[-1] - kmax) < mp.mpf('1e-9')
    qk = lambda k: (mp.mpf(k) + mp.mpf(1)/2)*h
    ok1 = ok2 = True; ok3 = True; n3 = 0
    dev = mp.mpf(0)
    for k in ks:
        xe = ch.B(k) + tau*(ch.A(k) + ch.A(k + 1))
        d = tau*(xe - ch.Psi(qk(k)**2, tau))
        dev = max(dev, d/h**2)
        ok1 &= d <= (rho1 + rho2)*h**2 * (1 + mp.mpf('1e-20'))
        if qk(k) >= qs + C0*h**2:
            n3 += 1
            ok3 &= tau*(x - ch.B(k)) - ch.A(k) - ch.A(k + 1) >= 0
    # slope bound  tau Psi_tau'(sigma) <= -kappa  for sigma in [sigma_*, sigma_max)  (tau = -1: Psi_-' >= kappa)
    for j in range(0, 200):
        sg = qs*qs + (ch.edge_sigma() - qs*qs)*mp.mpf(j)/200
        dpsi = mp.diff(lambda y: ch.Psi(y, tau), sg)
        ok2 &= -tau*dpsi >= kappa*(1 - mp.mpf('1e-12'))
    return tau, qs, C0, kappa, ok1, ok2, ok3, n3, dev


print('\n(B) explicit-constant scan', flush=True)
CASES = [('hahn', 1, 0, None), ('hahn', 2, 1, None), ('hahn', R(3, 2), R(-1, 2), None), ('hahn', 0, 0, None),
         ('racah', 1, 0, (mp.mpf(1), mp.mpf(2))), ('racah', 2, 1, (mp.mpf(1), mp.mpf('1.2'))),
         ('racah', 0, 0, (mp.mpf(1), mp.mpf(1))), ('racah', R(3, 2), R(1, 2), (mp.mpf('1.1'), mp.mpf(1))),
         ('racah', 2, -1, (mp.mpf(2), mp.mpf('1.5')))]
XS = [mp.mpf(v)/20 for v in (-18, -14, -9, -4, 3, 6, 10, 14, 18)]
types_seen = set(); worst = mp.mpf(0); ncase = 0; nhyp = 0
for kind, m_, n_, ab in CASES:
    for x in XS:
        for N in (100, 400):
            ch = make_chain(kind, sp.N(m_, 30), sp.N(n_, 30), N, ab)
            res = scan(ch, x)
            if res is None:
                break
            tau, qs, C0, kappa, ok1, ok2, ok3, n3, dev = res
            types_seen.add((kind, tau)); worst = max(worst, dev); ncase += 1; nhyp += n3
            if not (ok1 and ok2 and ok3):
                print('  FAIL', kind, m_, n_, ab, float(x), N, ok1, ok2, ok3)
            assert ok1 and ok2 and ok3, (kind, m_, n_, ab, x, N, ok1, ok2, ok3)
check({('hahn', 1), ('hahn', -1), ('racah', 1), ('racah', -1)} <= types_seen, '(B) both turn types for both chains scanned')
check(True, '(B) %d (chain, x, N) cases: intermediate inequality, slope bound and hyperbolicity beyond q_* + C_0 N^-2 hold on all %d ranks tested; '
      'largest N^2 tau (x_edge - Psi_tau) = %.3f' % (ncase, nhyp, float(worst)))

# ---------------- (C) negative control ----------------
print('\n(C) negative control: e = (1-s)^2, v = s (Psi = (1-s)^2 + s not concave): hyperbolicity fails beyond the turn', flush=True)
x = mp.mpf('0.9'); fails = []
for N in (100, 200, 400, 800):
    ch = Chain('bad', 0, 0, N)
    qs, tau = ch.turn(x)
    kk_ = list(range(int(qs*N) + 3, N))
    bad = [k for k in kk_ if tau*(x - ch.B(k)) - ch.A(k) - ch.A(k + 1) < 0]
    fails.append(len(bad))
    print('  N=%d tau=%+d q*=%.4f ranks beyond turn+3 violating hyperbolicity: %d' % (N, tau, float(qs), len(bad)), flush=True)
check(all(f > 0 for f in fails), '(C) non-concave Psi: hyperbolicity beyond the turn FAILS (the proof needs concavity of e and affine b_0)')
print('ALL CHECKS PASSED')
