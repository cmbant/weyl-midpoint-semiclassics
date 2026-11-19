#!/usr/bin/env python3
"""Second-order rank-direction Airy connection (thm:general-rank-airy-second).
# log: verify_rank_airy_second_order.txt

thm:general-rank-airy-second (proved modulo GBVA Thm 3.8, hypotheses verified)
states, for a chain whose physical solution is polynomially bounded beyond the
turn (Hahn: 3j orthogonality), uniformly for |t| <= A0, t = N^(2/3) zeta_tau(q):
   Y_L / pref = Ai(t) + N^(-4/3) varpi_tau(x) Ai'(t) + O(N^-2),
   varpi_tau(x) = lim_{q -> q_*-} [ tau*Gt(q,x)/sqrt(-zeta_tau) - 5/(48 zeta_tau^2) ],
   Gt = Gamma_2 + c_ab(Theta_q)/q        (eq:general-airy-varpi).
Numerical checks on the exact Hahn/3j rank sequence at a fixed physical node
(both turn types; integer and half-integer channels), N = 500..4000:
  (1) the bracket error max_{|t|<=2} |E| decays like N^(-4/3) (fitted exponent);
  (2) E N^(4/3) / Ai'(t) is t-independent and equals varpi_tau(x) computed
      independently from Gamma_2 (relative difference O(N^-2/3));
  (3) E - N^(-4/3) varpi Ai'(t) = O(N^-2); negative control: with -varpi the
      remainder is only O(N^-4/3).
"""
import sys
import mpmath as mp
import sympy as sp
from sympy.physics.wigner import wigner_3j
from p3_second_order_data import hahn, cab

mp.mp.dps = 30
A0 = 2
R = sp.Rational


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


def slope(Ns, errs):
    lx = [mp.log(n) for n in Ns]; ly = [mp.log(abs(e)) for e in errs]
    mx = sum(lx)/len(lx); my = sum(ly)/len(ly)
    return float(sum((a - mx)*(b - my) for a, b in zip(lx, ly))/sum((a - mx)**2 for a in lx))


def zeta_rank(q, x):
    """eq:general-zeta for the Hahn chain, tau = sgn x (real branch)."""
    tau = 1 if x > 0 else -1
    qs = mp.sqrt(1 - x*x)
    u = lambda s_: tau*x/mp.sqrt(1 - s_*s_)
    if q <= qs:
        return -(mp.mpf(3)/2*mp.re(mp.quad(lambda s_: mp.acos(u(s_)), [q, qs])))**(mp.mpf(2)/3)
    return (mp.mpf(3)/2*mp.re(mp.quad(lambda s_: mp.acosh(u(s_)), [qs, q])))**(mp.mpf(2)/3)


def varpi(ch, x):
    """eq:general-airy-varpi: quadratic extrapolation in delta of the bracket."""
    tau = 1 if x > 0 else -1
    qs = mp.sqrt(1 - x*x)
    ds = [mp.mpf('0.04'), mp.mpf('0.02'), mp.mpf('0.01')]
    w = []
    for d in ds:
        Q = qs - d
        z = zeta_rank(Q, x)
        Gt = ch.Gam2(Q, x) + cab(ch.a, ch.b, ch.Th(Q, x))/Q
        w.append(tau*Gt/mp.sqrt(-z) - mp.mpf(5)/(48*z*z))
    M = mp.matrix([[1, d, d*d] for d in ds])
    return mp.lu_solve(M, mp.matrix(w))[0]


def phi_seed(N, L, mu, M, sig):
    """eq:PhiDef evaluated exactly."""
    jm, jp = (N - 1 - sig)/2, (N - 1 + sig)/2
    mm, mpl = mu - R(M)/2, -mu - R(M)/2
    e = jp + mpl + 2*L
    val = wigner_3j(L, jm, jp, M, mm, mpl)
    return (-1)**int(e)*mp.sqrt(mp.mpf(sp.N(N, 40)))*mp.mpf(sp.N(val, 40))


def errors(M, sig, N, x0, edge):
    """Bracket errors E(t) at a fixed physical node, eq:upperHahnAiry/eq:lowerHahnAiry."""
    Nn = mp.mpf(sp.N(N, 40)); h = 1/Nn
    pp, pm = abs(M - sig), abs(M + sig)
    if edge == 'upper':
        nu = int(round(float((N - 1 - pp)/2 - x0*N/2))); mu = (N - 1 - pp)/2 - nu
    else:
        nu = int(round(float(x0*N/2 + (N - 1 - pm)/2))); mu = nu - (N - 1 - pm)/2
    x = mp.mpf(sp.N(2*mu/N, 40))
    kmin = max(abs(M), abs(sig))
    Mf, sf, k0 = (mp.mpf(sp.N(v, 30)) for v in (M, sig, kmin))
    A = lambda k: mp.sqrt((k*k - Mf*Mf)*(k*k - sf*sf))/(k*mp.sqrt(4*k*k - 1))*mp.sqrt(1 - k*k*h*h)
    B = lambda k: Mf*sf/(k*(k + 1)) if k > 0 else mp.mpf(0)
    Y = {0: mp.sqrt(2*k0 + 1)*phi_seed(N, kmin, mu, M, sig)}
    Y[1] = mp.sqrt(3)*x/mp.sqrt(1 - h*h) if kmin == 0 else (x - B(k0))*Y[0]/A(k0 + 1)
    qs = mp.sqrt(1 - x*x)
    jmax = int(float(qs*Nn - k0) + 3*float(Nn)**(1/3.))
    for j in range(1, jmax):
        k = k0 + j
        Y[j + 1] = ((x - B(k))*Y[j] - A(k)*Y[j - 1])/A(k + 1)
    eps = (-1)**int(max(sig - M, 0))
    out = []
    for j in range(1, jmax + 1):
        k = k0 + j; q = (k + mp.mpf(1)/2)/Nn
        if abs(q - qs) > 3*Nn**(-mp.mpf(2)/3):
            continue
        z = zeta_rank(q, x); t = Nn**(mp.mpf(2)/3)*z
        if abs(t) > A0:
            continue
        D = 1 - x*x - q*q
        Kt = nu + pp if edge == 'upper' else kmin + j - nu + (pp - pm)/2
        pref = eps*(-1)**int(Kt)*mp.sqrt(2/q)*Nn**(-mp.mpf(1)/3)*abs(z/D)**mp.mpf('0.25')
        out.append((t, mp.re(Y[j]/mp.sqrt(2*k + 1)/pref) - mp.airyai(t)))
    return x, out


for M, sig, edge, x0, NL in [(0, 0, 'upper', R(3, 5), [500, 1000, 2000, 4000]),
                             (1, 0, 'lower', R(-3, 5), [500, 1000, 2000, 4000]),
                             (2, 1, 'upper', R(3, 5), [500, 1000, 2000, 4000]),
                             (2, 1, 'lower', R(-3, 5), [500, 1000, 2000, 4000]),
                             (R(3, 2), R(-1, 2), 'lower', R(-1, 2), [R(1001, 2), R(2001, 2), R(4001, 2), R(8001, 2)])]:
    ch = hahn(M, sig)
    mx, mr, mrc, fits, Nf = [], [], [], [], []
    for N in NL:
        x, out = errors(M, sig, N, x0, edge)
        w = varpi(ch, x)
        Nn = mp.mpf(sp.N(N, 30)); s43 = Nn**(-mp.mpf(4)/3)
        mx.append(max(abs(e) for _, e in out))
        mr.append(max(abs(e - s43*w*mp.airyai(t, 1)) for t, e in out))
        mrc.append(max(abs(e + s43*w*mp.airyai(t, 1)) for t, e in out))
        num = sum(e*mp.airyai(t, 1) for t, e in out); den = sum(mp.airyai(t, 1)**2 for t, _ in out)
        fits.append(num/den/s43); Nf.append(Nn)
        print('  (%s,%s) %s N=%s x=%.6f pts=%d max|E|=%.3e N^(4/3)max|E|=%.4f fitted varpi=% .5f formula varpi=% .5f |E-N^-4/3 varpi Ai\'|=%.2e'
              % (M, sig, edge, N, float(x), len(out), float(mx[-1]), float(mx[-1]/s43), float(fits[-1]), float(w), float(mr[-1])),
              flush=True)
    s, sr, src = slope(Nf, mx), slope(Nf, mr), slope(Nf, mrc)
    rel = abs(fits[-1] - w)/abs(w)
    print('  exponents: bracket %.2f, after Ai\' term %.2f, control (-varpi) %.2f; rel. diff fitted/formula varpi %.2e' % (s, sr, src, rel))
    check(-1.5 < s < -1.2, '(%s,%s) %s: bracket error O(N^-4/3) (exponent %.2f)' % (M, sig, edge, s))
    check(rel < 0.03, '(%s,%s) %s: E N^4/3 / Ai\'(t) -> varpi_tau(x) of eq:general-airy-varpi (rel. diff %.1e)' % (M, sig, edge, rel))
    check(sr < -1.75 and src > -1.5, '(%s,%s) %s: remainder after the Ai\' term O(N^-2); control with -varpi O(N^-4/3)' % (M, sig, edge))
print('ALL CHECKS PASSED')
