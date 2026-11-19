#!/usr/bin/env python3
"""Horizon figure: natural-gauge error of four approximations to an exact 3j chain, against rank.
# log: verify_horizon_figure.txt

Writes figures/horizon_error.pdf and prints every plotted number.

Case: Hahn/3j offsets (M, sigma) = (2, 1), N = 400 (j_- = 199, j_+ = 200), physical node x = 3/10
(mu = 60).  The exact Phi_L of eq:PhiDef is propagated by the exact rank recurrence eq:3jrank in the
natural gauge from the exact minimal-rank value (sympy wigner_3j), and is checked against sympy
wigner_3j at several ranks.  For each plotted rank L, with q = (L + 1/2)/N, the script prints
    sqrt(2L+1) |Phi_L - A_L|
for
    (d)  A = d^L_{M sigma}(arccos x)                       plain Wigner d (breaks at L ~ N^{2/3}),
    (U)  A = W_q d^L(Theta_q)                              leading pullback (eq:class-leading-pullback),
    (U2) A = W^[2]_q d^L(Theta^[2]_q)                      second-order pullback (eq:class-corrected-pullback),
    (U3) A = (1 - N^-2 J_2(q,x)) W^[2]_q d^L(Theta^[2]_q)  phase-exact pullback (thm:general-phase-exactness).
Asserted (numerical evidence, not a proof): the plain-d error exceeds 1e-1 beyond L = 2 N^{2/3} and is
below 1e-2 for L <= N^{2/3}/4; max over the plotted ranks of N^2 * err(U2) and N^3 * err(U3) is O(1)
(below 5); err(U) at fixed ratio q >= 0.3 is of order 1/N.
"""
import sys
from pathlib import Path

import mpmath as mp
import sympy as sp
from sympy.physics.wigner import wigner_3j
import sympy.physics.quantum.spin  # noqa: F401  (Rotation.d)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from p3_second_order_data import hahn, cab

mp.mp.dps = 30
R = sp.Rational
M, SG, N, MU = 2, 1, 400, 60
JM, JP = R(N - 1 - SG, 2), R(N - 1 + SG, 2)
X = mp.mpf(2*MU)/N
TH = mp.acos(X)
h = mp.mpf(1)/N
K0 = max(abs(M), abs(SG))
LMAX = int(0.65*N)          # turning-free corridor q <= 0.65 < q_* = sqrt(1 - x^2) = 0.954
OUT = Path(__file__).resolve().parents[1]/'paper'/'figures'/'horizon_error.pdf'


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


def phi_exact_sympy(L):
    w = wigner_3j(L, JM, JP, M, MU - R(M, 2), -MU - R(M, 2))
    return mp.mpf(sp.N((-1)**int(2*L + JP - MU - R(M, 2))*sp.sqrt(N)*w, 40))


def A3(k):                    # eq:AkNexact
    k = mp.mpf(k)
    return mp.sqrt((k*k - M*M)*(k*k - SG*SG))/(k*mp.sqrt(4*k*k - 1))*mp.sqrt(1 - k*k*h*h)


def B3(k):
    return mp.mpf(M*SG)/(k*(k + 1))


# exact chain in natural gauge
Y = {K0: mp.sqrt(2*K0 + 1)*phi_exact_sympy(K0)}
Y[K0 + 1] = (X - B3(K0))*Y[K0]/A3(K0 + 1)
for k in range(K0 + 1, LMAX + 1):
    Y[k + 1] = ((X - B3(k))*Y[k] - A3(k)*Y[k - 1])/A3(k + 1)
dev = max(abs(Y[L]/mp.sqrt(2*L + 1) - phi_exact_sympy(L)) for L in (K0 + 1, 10, 57, 200, LMAX))
check(dev < mp.mpf(10)**-20, 'recurrence values equal sympy wigner_3j at L = 3, 10, 57, 200, %d (max diff %.1e)' % (LMAX, float(dev)))


def dhat_all(T, Lmax):
    """sqrt(2k+1) d^k_{M sigma}(T), k = K0..Lmax, by the Wigner rank recurrence (eq:wigner-natural-jacobi)."""
    seed = sp.re(sp.N(sp.physics.quantum.spin.Rotation.d(K0, M, SG, sp.Float(str(T), 40)).doit(), 40))
    A0 = lambda k: mp.sqrt(mp.mpf((k*k - M*M)*(k*k - SG*SG)))/(k*mp.sqrt(mp.mpf(4*k*k - 1)))
    xc = mp.cos(T)
    out = {K0: mp.sqrt(2*K0 + 1)*mp.mpf(seed)}
    out[K0 + 1] = (xc - B3(K0))*out[K0]/A0(K0 + 1)
    for k in range(K0 + 1, Lmax):
        out[k + 1] = ((xc - B3(k))*out[k] - A0(k)*out[k - 1])/A0(k + 1)
    return out


# sympy Rotation.d convention vs the exact chain at h = 0: the chain's L0 value is d^L0 at arccos x
dplain = dhat_all(TH, LMAX)
check(abs(dplain[K0] - Y[K0]) < mp.mpf('1e-3'), 'minimal-rank seed agrees with d^{L0}_{M sigma}(arccos x) to O(N^-2)')

ch = hahn(M, SG)


def J2num(Q):
    T = ch.Th(Q, X); P = ch.p(Q, X)
    pqq = mp.diff(lambda r: ch.p(r, X), Q, 2)
    lDqq = mp.diff(lambda r: mp.log(ch.D(r, X)), Q, 2)
    return (cab(M, SG, T)*mp.cot(T)/Q**2 + mp.cot(P)*(ch.Sig(Q, X) + pqq/24) - lDqq/16)/2


Ls = sorted(set([K0 + 1, 4, 5, 6, 8] + [int(round(mp.mpf(10)**(mp.mpf(j)/16)*4)) for j in range(0, 40)]))
Ls = [L for L in Ls if 3 <= L <= LMAX]
rows = []
print('\n  L      q       err(d)      err(U)      err(U2)     err(U3)    N^2 err(U2)  N^3 err(U3)')
for L in Ls:
    Q = (mp.mpf(L) + mp.mpf(1)/2)/N
    T = ch.Th(Q, X)
    T2 = T + h*h/Q*ch.Gam2(Q, X)
    D = ch.D(Q, X)
    U = (mp.sin(T)**2/D)**mp.mpf('0.25')*dhat_all(T, L)[L]
    U2 = (mp.sin(T2)**2/D)**mp.mpf('0.25')*dhat_all(T2, L)[L]
    U3 = (1 - h*h*J2num(Q))*U2
    e = [abs(Y[L] - A) for A in (dplain[L], U, U2, U3)]
    rows.append((L, Q, e))
    print('%4d  %.4f  %.3e  %.3e  %.3e  %.3e   %8.4f     %8.4f' % (L, float(Q), *[float(v) for v in e],
                                                                     float(N**2*e[2]), float(N**3*e[3])), flush=True)

hor = mp.mpf(N)**(mp.mpf(2)/3)
check(all(r[2][0] > 0.1 for r in rows if r[0] >= 2*hor), 'plain d: natural-gauge error > 0.1 for L >= 2 N^{2/3} (%.0f)' % float(2*hor))
check(all(r[2][0] < 0.01 for r in rows if r[0] <= hor/4), 'plain d: natural-gauge error < 0.01 for L <= N^{2/3}/4')
m2 = max(N**2*r[2][2] for r in rows); m3 = max(N**3*r[2][3] for r in rows)
check(m2 < 5 and m3 < 5, 'second-order pullback O(N^-2) and phase-exact pullback O(N^-3) at every plotted rank (max N^2 err %.3f, N^3 err %.3f)' % (m2, m3))
mU = [N*r[2][1] for r in rows if r[1] >= 0.3]
check(0.01 < max(mU) < 5, 'leading pullback error O(1/N) at fixed ratio q >= 0.3 (max N err %.3f)' % max(mU))

# ---------------- figure ----------------
mpl.rcParams.update({'pdf.fonttype': 42, 'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8.5})
fig, ax = plt.subplots(figsize=(5.6, 3.6))
Lx = [r[0] for r in rows]
styles = [('#2a78d6', 'o', '-', r'Wigner $d^L(\theta)$'),
          ('#eb6834', 's', '--', r'leading pullback $W_q\,d^L(\Theta_q)$'),
          ('#1baf7a', '^', '-', r'second order $W^{[2]}_q\,d^L(\Theta^{[2]}_q)$'),
          ('#eda100', 'D', '-.', r'phase exact $(1-N^{-2}\mathsf{J}_2)\,W^{[2]}_q\,d^L(\Theta^{[2]}_q)$')]
for i, (c, mk, ls, lab) in enumerate(styles):
    ax.plot(Lx, [float(r[2][i]) for r in rows], color=c, marker=mk, ls=ls, lw=1.6, ms=3.5, label=lab)
ax.axvline(float(hor), color='0.55', lw=1, ls=':')
ax.text(float(hor)*1.05, 1e2, r'$L=N^{2/3}$', color='0.3', fontsize=8.5)
ax.axhline(1.0/N**2, color='0.75', lw=0.8)
ax.axhline(1.0/N**3, color='0.75', lw=0.8)
ax.text(150, 1.6/N**2, r'$N^{-2}$', color='0.3', fontsize=8)
ax.text(12, 1.6/N**3, r'$N^{-3}$', color='0.3', fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'rank $L$ ($N=400$, $x=0.3$, $(M,\sigma)=(2,1)$)')
ax.set_ylabel(r'$\sqrt{2L+1}\,|\Phi_L-\mathrm{approx.}|$')
ax.set_ylim(1e-11, 1e3)
ax.grid(True, which='major', color='0.92', lw=0.6)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(loc='upper left', frameon=False, fontsize=8)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print('wrote figures/%s' % OUT.name)
print('ALL CHECKS PASSED')
