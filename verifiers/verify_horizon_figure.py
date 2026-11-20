#!/usr/bin/env python3
"""Horizon figure: natural-gauge error of four approximations to an exact 3j chain, at every rank.
# log: verify_horizon_figure.txt

Writes figures/horizon_error.pdf and prints every plotted number.

Case: Hahn/3j offsets (M, sigma) = (2, 1), physical node x = 2 mu/N with mu = floor(0.15 N), at
N = 100, 200, 400, 800, 1600 (x = 3/10 exactly).  The exact Phi_L of eq:PhiDef is propagated by the
exact rank recurrence eq:3jrank in the natural gauge from the exact minimal-rank value (sympy
wigner_3j), and is checked against sympy wigner_3j at several ranks.  For every rank 3 <= L <= 0.65 N,
with q = (L + 1/2)/N, the script computes  sqrt(2L+1) |Phi_L - A_L|  for
    (d)  A = d^L_{M sigma}(arccos x)                       plain Wigner d (breaks at L ~ N^{2/3}),
    (U)  A = W_q d^L(Theta_q)                              leading pullback (eq:class-leading-pullback),
    (U2) A = W^[2]_q d^L(Theta^[2]_q)                      second-order pullback (eq:class-corrected-pullback),
    (U3) A = (1 - N^-2 J_2(q,x)) W^[2]_q d^L(Theta^[2]_q)  phase-exact pullback (thm:general-phase-exactness).
Theta_q = S(q)/q and Gamma_2 are accumulated rank by rank with 8-point Gauss-Legendre on each step
[q_{L-1}, q_L], starting from the helper's quadratures at the first rank (checked against the helper
at sample ranks).  Panels (a) and (b) show maximum errors in logarithmic rank bins (8 per decade,
L >= 6 in (a) and L >= 10 in (b), so that each bin holds several ranks) to capture the envelope of
the oscillatory point errors. Panel (c) uses all individual ranks.
The line in panel (a) marks the rank where the leading phase shift kappa^3 cot(theta)/(6 N^2),
kappa = L + 1/2, of eq:criticalphase-quant reaches one radian.

Asserted (numerical evidence, not a proof): plain d below 1e-2 for L <= N^{2/3}/4 and its binned
maximum above 1e-1 beyond 2 N^{2/3}, for every N; the binned plain-d error at L/N^{2/3} = 1/2, 1, 2
agrees across N to a factor 1.6 (the horizon scales as N^{2/3}); N^2 err(U2) and N^3 err(U3) below 5
at every rank and N; the worst error over all ranks has log-log slope in N within 0.3 of -1 (U),
-2 (U2) and -3 (U3) over N = 400..1600, while the plain-d worst error stays above 1.
"""
import logging
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from sympy.physics.wigner import wigner_3j
from sympy.physics.quantum.spin import Rotation
import matplotlib as mpl
mpl.use('Agg')
logging.getLogger('fontTools').setLevel(logging.ERROR)   # silence PDF font-subsetting timestamp notes
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator

from p3_second_order_data import hahn, cab

mp.mp.dps = 30
R = sp.Rational
M, SG = 2, 1
K0 = max(abs(M), abs(SG))
NS = [100, 200, 400, 800, 1600]
N_SCAN = 800
NODE = R(3, 10)
OUT = Path(__file__).resolve().parents[1]/'paper'/'figures'/'horizon_error.pdf'
ch = hahn(M, SG)
GLX = [mp.findroot(lambda t: mp.legendre(8, t), float(v)) for v in np.polynomial.legendre.leggauss(8)[0]]
GLW = [2/((1 - t*t)*mp.diff(lambda u: mp.legendre(8, u), t)**2) for t in GLX]   # nodes, weights to 30 digits


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


def gl(f, a, b):
    """8-point Gauss-Legendre on [a, b]; the steps are of length 1/N, far below any scale of f."""
    c, r = (a + b)/2, (b - a)/2
    return r*sum(w*f(c + r*t) for t, w in zip(GLX, GLW))


def wd(j, m, mq, b):
    """Wigner d^j_{m mq}(b), mpmath, sum formula (sign convention of sympy Rotation.d with m <-> mq)."""
    f = mp.factorial; c = mp.cos(b/2); s = mp.sin(b/2); tot = 0
    for k in range(max(0, m - mq), min(j + m, j - mq) + 1):
        tot += (-1)**(mq - m + k)*c**(2*j + m - mq - 2*k)*s**(mq - m + 2*k)/(f(j + m - k)*f(k)*f(mq - m + k)*f(j - mq - k))
    return mp.sqrt(f(j + m)*f(j - m)*f(j + mq)*f(j - mq))*tot


T0 = mp.mpf('0.7')
check(abs(wd(K0, SG, M, T0) - mp.mpf(sp.N(Rotation.d(K0, M, SG, sp.Float('0.7', 40)).doit(), 35))) < mp.mpf(10)**-25,
      'mpmath seed d^2_{1,2} equals sympy Rotation.d(2, 2, 1) (the convention of the exact chain)')


def B3(k):
    return mp.mpf(M*SG)/(k*(k + 1))


def A0(k):
    return mp.sqrt(mp.mpf((k*k - M*M)*(k*k - SG*SG)))/(k*mp.sqrt(mp.mpf(4*k*k - 1)))


def dhat(T, Lmax):
    """sqrt(2k+1) d^k_{M sigma}(T), k = K0..Lmax, by the Wigner rank recurrence (eq:wigner-natural-jacobi)."""
    xc = mp.cos(T)
    out = [mp.sqrt(2*K0 + 1)*wd(K0, SG, M, T)]
    out.append((xc - B3(K0))*out[0]/A0(K0 + 1))
    for k in range(K0 + 1, Lmax):
        out.append(((xc - B3(k))*out[-1] - A0(k)*out[-2])/A0(k + 1))
    return out                      # out[k - K0]


def run(N):
    MU = int(NODE*N/2); JM, JP = R(N - 1 - SG, 2), R(N - 1 + SG, 2)
    X = mp.mpf(2*MU)/N; h = mp.mpf(1)/N; LMAX = int(0.65*N)   # turning-free: q <= 0.65 < sqrt(1 - x^2)

    def phi(L):
        w = wigner_3j(L, JM, JP, M, MU - R(M, 2), -MU - R(M, 2))
        return mp.mpf(sp.N((-1)**int(2*L + JP - MU - R(M, 2))*sp.sqrt(N)*w, 40))

    def A3(k):                      # eq:AkNexact
        k = mp.mpf(k)
        return A0(k)*mp.sqrt(1 - k*k*h*h)
    Y = [mp.sqrt(2*K0 + 1)*phi(K0)]
    Y.append((X - B3(K0))*Y[0]/A3(K0 + 1))
    for k in range(K0 + 1, LMAX):
        Y.append(((X - B3(k))*Y[-1] - A3(k)*Y[-2])/A3(k + 1))
    if N <= 400:
        dev = max(abs(Y[L - K0]/mp.sqrt(2*L + 1) - phi(L)) for L in (K0 + 1, 10, N//7, N//2, LMAX))
        check(dev < mp.mpf(10)**-20, 'N=%d: recurrence values equal sympy wigner_3j at five ranks up to %d (max diff %.1e)' % (N, LMAX, float(dev)))
    dp = dhat(mp.acos(X), LMAX)

    def J2(Q, T):
        P = ch.p(Q, X)
        pqq = mp.diff(lambda r: ch.p(r, X), Q, 2)
        lDqq = mp.diff(lambda r: mp.log(ch.D(r, X)), Q, 2)
        return (cab(M, SG, T)*mp.cot(T)/Q**2 + mp.cot(P)*(ch.Sig(Q, X) + pqq/24) - lDqq/16)/2

    def rho2(u, S):                 # p3_second_order_data.Chain.rho2 with S(u) supplied
        T = S/u; Tq = (ch.p(u, X) - T)/u
        from p3_second_order_data import dcab
        return ch.Sig(u, X) - (dcab(M, SG, T)*Tq/u - cab(M, SG, T)/u**2)

    p = lambda u: ch.p(u, X)
    Ls = list(range(3, LMAX + 1))
    Qp = (mp.mpf(Ls[0]) + mp.mpf(1)/2)/N
    S, G = ch.S(Qp, X), ch.Gam2(Qp, X)
    err = {k: [] for k in ('d', 'U', 'U2', 'U3')}
    for L in Ls:
        Q = (mp.mpf(L) + mp.mpf(1)/2)/N
        if L > Ls[0]:
            G += gl(lambda u: rho2(u, S + gl(p, Qp, u)), Qp, Q)
            S += gl(p, Qp, Q)
            Qp = Q
        T = S/Q; T2 = T + h*h/Q*G; D = ch.D(Q, X)
        if L in (Ls[0] + 7, LMAX//2, LMAX):
            dv = max(abs(T - ch.Th(Q, X)), abs(G - ch.Gam2(Q, X)))
            check(dv < mp.mpf(10)**-18, 'N=%d, L=%d: accumulated Theta_q and Gamma_2 equal the helper quadratures (diff %.1e)' % (N, L, float(dv)))
        U = (mp.sin(T)**2/D)**mp.mpf('0.25')*dhat(T, L)[L - K0]
        U2 = (mp.sin(T2)**2/D)**mp.mpf('0.25')*dhat(T2, L)[L - K0]
        U3 = (1 - h*h*J2(Q, T))*U2
        for k, A in zip(('d', 'U', 'U2', 'U3'), (dp[L - K0], U, U2, U3)):
            err[k].append(float(abs(Y[L - K0] - A)))
    return np.array(Ls, float), {k: np.array(v) for k, v in err.items()}


def env(L, e, nb=8):
    """Maximum of e over logarithmic rank bins (nb per decade), at the bins' geometric centres."""
    k = np.floor(np.log10(L)*nb)
    xs, ys = [], []
    for b in np.unique(k):
        m = k == b
        xs.append(np.sqrt(L[m].min()*L[m].max())); ys.append(e[m].max())
    return np.array(xs), np.array(ys)


data = {N: run(N) for N in NS}
NAMES = ('d', 'U', 'U2', 'U3')

print('\nPanel (c): maximum error over all ranks 3 <= L <= 0.65N')
print('     N        d          U          U2         U3       N^2 max U2   N^3 max U3')
sup = {k: np.array([data[N][1][k].max() for N in NS]) for k in NAMES}
for i, N in enumerate(NS):
    print('%6d  %.3e  %.3e  %.3e  %.3e   %8.4f     %8.4f' % (N, *[sup[k][i] for k in NAMES], N**2*sup['U2'][i], N**3*sup['U3'][i]))

print('\nPanel (a): N=%d, binned maximum error (8 bins per decade, L >= 6)' % N_SCAN)
scan_ranks, scan_errors = data[N_SCAN]
mA = scan_ranks >= 6                # below L = 6 a bin holds a single rank, so its maximum is a point value
envA = {k: env(scan_ranks[mA], scan_errors[k][mA]) for k in NAMES}
print('   L_c        d          U          U2         U3')
for j, x in enumerate(envA['d'][0]):
    print('%7.1f  %.3e  %.3e  %.3e  %.3e' % (x, *[envA[k][1][j] for k in NAMES]))

print('\nPanel (b): binned maximum plain-d error against s = L/N^{2/3} (L >= 10, s <= 4)')
envB = {}
for N in NS:
    L, e = data[N]
    m = (L >= 10) & (L <= 4*N**(2/3))
    x, y = env(L[m], e['d'][m])
    envB[N] = (x/N**(2/3), y)
    print('N=%d: ' % N + '  '.join('%.3f:%.2e' % (s, v) for s, v in zip(*envB[N])))

for N in NS:
    L, e = data[N]; hz = N**(2/3)
    check(e['d'][L <= hz/4].max() < 1e-2, 'N=%d: plain d error < 1e-2 for L <= N^{2/3}/4' % N)
    x, y = env(L, e['d'])
    check(y[x >= 2*hz].min() > 0.1, 'N=%d: binned plain d error > 1e-1 beyond 2 N^{2/3}' % N)
    check((N**2*e['U2']).max() < 5 and (N**3*e['U3']).max() < 5, 'N=%d: N^2 err(U2) and N^3 err(U3) < 5 at every rank' % N)
for s0 in (0.5, 1.0, 2.0):
    Nin = [N for N in NS if envB[N][0][0] <= s0 <= envB[N][0][-1]]       # no extrapolation
    v = [np.interp(np.log(s0), np.log(envB[N][0]), np.log(envB[N][1])) for N in Nin]
    check(len(Nin) >= 4 and np.exp(max(v) - min(v)) < 1.6, 'horizon collapse: binned plain-d error at L = %.1f N^{2/3} agrees across N = %s to factor %.2f'
          % (s0, ','.join(map(str, Nin)), np.exp(max(v) - min(v))))
lN = np.log(np.array(NS[2:], float))
for k, p in (('U', 1), ('U2', 2), ('U3', 3)):
    sl = np.polyfit(lN, np.log(sup[k][2:]), 1)[0]
    check(abs(sl + p) < 0.3, 'worst-case %s error slope %.2f over N = 400..1600 (expected -%d)' % (k, sl, p))
check(sup['d'].min() > 1, 'plain d worst-case error stays O(1) (min over N %.2f)' % sup['d'].min())

# ---------------- figure ----------------
mpl.rcParams.update({
    'pdf.fonttype': 42, 'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'mathtext.fontset': 'cm',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65,
    'axes.axisbelow': True, 'xtick.major.size': 3, 'ytick.major.size': 3,
})
styles = {
    'd': dict(color='#0072B2', marker='o', label=r'Wigner $d^L(\theta)$'),
    'U': dict(color='#BD5200', marker='s', label='Leading pullback'),
    'U2': dict(color='#008363', marker='^', label='Second-order pullback'),
    'U3': dict(color='#8B4AA5', marker='D', label='Phase-exact pullback'),
}
fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.5, 2.7))
fig.subplots_adjust(left=0.113, right=0.98, bottom=0.18, top=0.79, wspace=0.55)


def log_axes(ax, xlim, xticks, ylim, yexponents):
    ax.set(xscale='log', yscale='log', xlim=xlim, ylim=ylim)
    ax.set_xticks(xticks, labels=[str(x) for x in xticks])
    ax.yaxis.set_major_locator(FixedLocator(10.0**np.array(yexponents)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis='y', color='0.90', linewidth=0.5)


x = float(NODE)
cot_theta = x/np.sqrt(1 - x*x)
one_radian_rank = (6*N_SCAN**2/abs(cot_theta))**(1/3) - 0.5
a.axvline(one_radian_rank, color='0.5', lw=0.8, zorder=1)
handles = []
for k in NAMES:
    line, = a.plot(*envA[k], **styles[k], lw=1.35, ms=2.8)
    handles.append(line)
a.annotate('1 rad', (one_radian_rank, 4e-13), xytext=(-4, 0), textcoords='offset points',
           ha='right', va='center', color='0.28')
log_axes(a, (5, 560), [10, 30, 100, 300], (1e-13, 8), [0, -3, -6, -9, -12])
a.set(xlabel=r'Rank $L$', ylabel=r'Binned max $E_L$')
a.set_title(r'(a) Error by rank, $N=%d$' % N_SCAN, loc='left')

for i, (N, color, marker) in enumerate(zip(NS, ['#719BB9', '#5488AC', '#3977A2', '#226794', '#004975'], 'os^Dv')):
    b.plot(*envB[N], color=color, marker=marker, ms=3, markerfacecolor='white',
           markeredgewidth=0.75, markevery=(i % 3, 3), lw=0.9, label=str(N))
s_guide = np.array([0.13, 0.57])
b.plot(s_guide, 0.014*s_guide**3, color='0.35', ls=':', lw=1.15)
b.text(0.65, 0.014*s_guide[-1]**3, r'$s^3$', va='center', color='0.28')
log_axes(b, (0.07, 4.5), [0.1, 0.3, 1, 3], (1.8e-5, 6), [0, -1, -2, -3, -4])
b.set(xlabel=r'$s=L/N^{2/3}$', ylabel=r'Binned max $E_L$')
b.set_title('(b) Wigner horizon scaling', loc='left')
b.legend(title='$N$', loc='upper left', frameon=False, fontsize=7.5,
         borderaxespad=0.3, handlelength=1.4, handletextpad=0.45, labelspacing=0.3)

for k in NAMES:
    c.plot(NS, sup[k], **styles[k], lw=1.35, ms=3.8)
label_index = -2
for k, power in zip(NAMES[1:], (1, 2, 3)):
    c.annotate(r'$\propto N^{-%d}$' % power, (NS[label_index], sup[k][label_index]),
               xytext=(1, 4), textcoords='offset points', color=styles[k]['color'])
log_axes(c, (85, 1900), NS, (1e-10, 8), [0, -2, -4, -6, -8, -10])
c.set(xlabel='$N$', ylabel=r'Max $E_L$ over all ranks')
c.set_title('(c) Worst-case error', loc='left')

fig.legend(handles=handles, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.99),
           frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.5)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print('Panel (a): leading phase shift reaches one radian at L = %.1f' % one_radian_rank)
print('wrote figures/%s' % OUT.name)
print('ALL CHECKS PASSED')
