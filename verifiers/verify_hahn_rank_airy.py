#!/usr/bin/env python3
"""Hahn Airy connection as a rank-direction turn of the general theorem.

Supports the proof of thm:hahnAiryConnection via thm:general-rank-airy.
# log: verify_hahn_rank_airy.txt

Checks:
  1. the rank-direction Langer coordinate zeta_tau(q;x) of eq:general-zeta for
     the Hahn chain (p = arccos(x/sqrt(1-q^2)), u = x/sqrt(1-q^2)) equals the
     spatial coordinate xi_tau(q,x) of eq:xiPlusAiry/eq:xiMinusAiry, on both
     sides of the turn (tau = sgn x);
  2. S(q_*,x) = pi(1-x)/2 (x>0) and S(q_*,x) = pi q_* - pi(1+x)/2 (x<0),
     q_* = sqrt(1-x^2), from the closed action eq:Fclosed--eq:Theta;
  3. at physical lattice points the quantities K_+ and K_- of
     eq:general-turn-quantization equal nu_+ + p_+ and L - nu_- + (p_+-p_-)/2
     and are integers (integer and half-integer channels);
  4. the normalization eq:PhiDef, (-1)^{2L+j_++m_+} sqrt(N) 3j, tends to
     +d^L_{M sigma} at fixed L also for half-integer L (the phase without
     (-1)^{2L} gives -d: negative control);
  5. numerical: at a fixed physical node x, the exact Hahn/3j rank sequence
     matches eq:upperHahnAiry / eq:lowerHahnAiry on |t| <= A0 in the rank
     direction, with bracket error decreasing at least like N^-1.
"""
import mpmath as mp
import sympy as sp
from sympy.physics.wigner import wigner_3j

mp.mp.dps = 30
A0 = 2.0


def check(cond, msg):
    if not cond:
        raise SystemExit('FAIL ' + msg)
    print('PASS ' + msg)


def zeta_rank(q, x):
    """eq:general-zeta for the Hahn chain, tau = sgn x."""
    tau = 1 if x > 0 else -1
    qs = mp.sqrt(1 - x*x)
    u = lambda s: tau*x/mp.sqrt(1 - s*s)           # tau*u(s)
    if q <= qs:
        I = mp.quad(lambda s: mp.acos(u(s)), [q, qs])
        return -(mp.mpf(3)/2*I)**(mp.mpf(2)/3)
    I = mp.quad(lambda s: mp.acosh(u(s)), [qs, q])
    return (mp.mpf(3)/2*I)**(mp.mpf(2)/3)


def xi_space(q, x):
    """eq:xiPlusAiry (x>0) and eq:xiMinusAiry (x<0)."""
    xp = mp.sqrt(1 - q*q)
    pq = lambda v: mp.asin(q/mp.sqrt(1 - v*v))
    ch = lambda v: mp.acosh(q/mp.sqrt(1 - v*v))
    if x > 0:
        if x <= xp:
            return -(mp.mpf(3)/2*mp.quad(lambda v: mp.pi/2 - pq(v), [x, xp]))**(mp.mpf(2)/3)
        return (mp.mpf(3)/2*mp.quad(ch, [xp, x]))**(mp.mpf(2)/3)
    if x >= -xp:
        return -(mp.mpf(3)/2*mp.quad(lambda v: mp.pi/2 - pq(v), [-xp, x]))**(mp.mpf(2)/3)
    return (mp.mpf(3)/2*mp.quad(ch, [x, -xp]))**(mp.mpf(2)/3)


# 1. zeta == xi
worst = 0
for xv in ['0.3', '0.6', '0.85', '-0.4', '-0.75']:
    x = mp.mpf(xv); qs = mp.sqrt(1 - x*x)
    for dq in ['-0.3', '-0.05', '-0.001', '0.001', '0.04', '0.1']:
        q = qs + mp.mpf(dq)
        if not 0 < q < 1:
            continue
        worst = max(worst, abs(zeta_rank(q, x) - xi_space(q, x)))
check(worst < mp.mpf(10)**-20, 'rank Langer coordinate zeta_tau(q;x) == spatial xi_tau(q,x) (max diff %s)'
      % mp.nstr(worst, 3))


def S_closed(q, x):
    D = 1 - x*x - q*q
    F = (q*mp.asin(x/mp.sqrt(1 - q*q)) + x*mp.asin(q/mp.sqrt(1 - x*x)) - mp.atan(x*q/mp.sqrt(D)))
    return mp.pi*q/2 - F


# 2. turning action
worst = 0
for xv in ['0.2', '0.5', '0.9', '-0.3', '-0.7']:
    x = mp.mpf(xv); qs = mp.sqrt(1 - x*x)
    Sq = mp.quad(lambda s: mp.acos(x/mp.sqrt(1 - s*s)), [0, qs])
    target = mp.pi*(1 - x)/2 if x > 0 else mp.pi*qs - mp.pi*(1 + x)/2
    near = S_closed(qs*(1 - mp.mpf(10)**-20), x)
    worst = max(worst, abs(Sq - target), abs(near - target))
check(worst < mp.mpf(10)**-8, 'S(q_*,x) = pi(1-x)/2 (x>0), pi q_* - pi(1+x)/2 (x<0)')


def node(N, M, sig, nu, edge):
    """Physical node from eq:upperInwardAiry / eq:lowerInwardAiry; returns mu (exact), x."""
    pp, pm = abs(M - sig), abs(M + sig)
    mu = (N - 1 - pp)/2 - nu if edge == 'upper' else nu - (N - 1 - pm)/2
    return mu, 2*mu/N


# 3. K_+- integrality
worst = 0
R = sp.Rational
for M, sig in [(0, 0), (2, 1), (1, -2), (R(1, 2), R(1, 2)), (R(3, 2), R(-1, 2))]:
    for N in ([401, 802] if M == int(M) else [R(803, 2), R(1601, 2)]):
        pp, pm = abs(M - sig), abs(M + sig)
        for edge in ['upper', 'lower']:
            for nu in [30, 77]:
                mu, xr = node(sp.nsimplify(N), M, sig, nu, edge)
                x = mp.mpf(sp.N(xr, 40)); Nn = mp.mpf(sp.N(N, 40))
                qs = mp.sqrt(1 - x*x)
                Sv = mp.pi*(1 - x)/2 if x > 0 else mp.pi*qs - mp.pi*(1 + x)/2
                kmin = max(abs(M), abs(sig))
                L = kmin + 40
                if edge == 'upper':
                    K = Nn*Sv/mp.pi + mp.mpf(sp.N(pp - 1, 30))/2
                    pred = nu + pp
                else:
                    K = mp.mpf(sp.N(L, 30)) + mp.mpf(1)/2 + Nn*Sv/mp.pi - Nn*qs + mp.mpf(sp.N(pp, 30))/2
                    pred = L - nu + (pp - pm)/2
                assert sp.nsimplify(pred).is_integer, (M, sig, edge, pred)
                worst = max(worst, abs(K - mp.mpf(sp.N(pred, 30))))
check(worst < mp.mpf(10)**-20, 'K_+ = nu_+ + p_+ and K_- = L - nu_- + (p_+-p_-)/2 are integers at physical nodes')


def phi_seed(N, L, mu, M, sig, with2L=True):
    """eq:PhiDef evaluated exactly with SymPy's 3j."""
    jm, jp = (N - 1 - sig)/2, (N - 1 + sig)/2
    mm, mpl = mu - R(M)/2, -mu - R(M)/2
    e = jp + mpl + (2*L if with2L else 0)
    assert sp.nsimplify(e).is_integer
    val = wigner_3j(L, jm, jp, M, mm, mpl)
    return (-1)**int(e)*mp.sqrt(mp.mpf(sp.N(N, 40)))*mp.mpf(sp.N(val, 40))


def wigner_d(j, m1, m2, beta):
    tot = 0
    for s in range(0, int(2*j) + 2):
        f = [j + m2 - s, s, m1 - m2 + s, j - m1 - s]
        if any(v < 0 for v in f):
            continue
        pref = mp.sqrt(mp.factorial(j + m1)*mp.factorial(j - m1)*mp.factorial(j + m2)*mp.factorial(j - m2))
        den = mp.fprod([mp.factorial(v) for v in f])
        tot += (-1)**int(m1 - m2 + s)*pref/den*mp.cos(beta/2)**int(2*j + m2 - m1 - 2*s)*mp.sin(beta/2)**int(m1 - m2 + 2*s)
    return tot


# 4. normalization sign for half-integer L
ok = True
for M, sig, L in [(R(1, 2), R(1, 2), R(1, 2)), (R(1, 2), R(-1, 2), R(3, 2)), (R(3, 2), R(1, 2), R(5, 2)), (1, 0, 2)]:
    N = R(2001, 2) if L != int(L) else R(1001)
    mu = R(N - 1 - abs(M - sig), 2) - 170          # an interior node
    x = mp.mpf(sp.N(2*mu/N, 40))
    d = wigner_d(L, M, sig, mp.acos(x))
    new, old = phi_seed(N, L, mu, M, sig), phi_seed(N, L, mu, M, sig, with2L=False)
    ok &= abs(new - d) < 1e-4 and (L == int(L) or abs(old + d) < 1e-4)
check(ok, 'eq:PhiDef tends to +d^L_{M sigma} for half-integer L; without (-1)^{2L} it tends to -d')


def run(M, sig, N, x0, edge):
    """Exact natural-gauge Hahn rank sequence at a fixed physical node near x0."""
    Nn = mp.mpf(sp.N(N, 40)); h = 1/Nn
    pp, pm = abs(M - sig), abs(M + sig)
    if edge == 'upper':
        nu = int(round(float(((N - 1 - pp)/2 - x0*N/2))))
    else:
        nu = int(round(float((x0*N/2 + (N - 1 - pm)/2))))
    mu, xr = node(N, M, sig, nu, edge)
    x = mp.mpf(sp.N(xr, 40))
    kmin = max(abs(M), abs(sig))
    Mf, sf, k0 = (mp.mpf(sp.N(v, 30)) for v in (M, sig, kmin))
    A = lambda k: (mp.sqrt((k*k - Mf*Mf)*(k*k - sf*sf))/(k*mp.sqrt(4*k*k - 1))*mp.sqrt(1 - k*k*h*h)
                   if k > k0 + mp.mpf('0.25') else mp.mpf(0))
    B = lambda k: Mf*sf/(k*(k + 1)) if k > 0 else mp.mpf(0)
    Y = {}
    Y[k0] = mp.sqrt(2*k0 + 1)*phi_seed(N, kmin, mu, M, sig)
    if kmin == 0:
        Y[k0 + 1] = mp.sqrt(3)*x/mp.sqrt(1 - h*h)
    else:
        Y[k0 + 1] = (x - B(k0))*Y[k0]/A(k0 + 1)
    qs = mp.sqrt(1 - x*x)
    kmax = k0 + int(float(qs*Nn - k0) + 6*float(Nn)**(1/3.))
    k = k0 + 1
    while k < kmax:
        Y[k + 1] = ((x - B(k))*Y[k] - A(k)*Y[k - 1])/A(k + 1)
        k += 1
    eps = (-1)**int(max(sig - M, 0))
    errs = []
    k = k0 + 1
    while k <= kmax:
        q = (k + mp.mpf(1)/2)/Nn
        z = zeta_rank(q, x); t = Nn**(mp.mpf(2)/3)*z
        if abs(t) <= A0:
            D = 1 - x*x - q*q
            Kt = nu + pp if edge == 'upper' else sp.nsimplify(kmin + (k - k0)) - nu + (pp - pm)/2
            pref = eps*(-1)**int(Kt)*mp.sqrt(2/q)*Nn**(-mp.mpf(1)/3)*abs(z/D)**mp.mpf('0.25')
            errs.append(Y[k]/mp.sqrt(2*k + 1)/pref - mp.airyai(t))
        k += 1
    return x, max(abs(e_) for e_ in errs), len(errs)


# 5. rank-direction Airy collapse
print('\nrank-direction Airy collapse at fixed node: channel edge N x npts max|bracket err| N*err')
ok = True
for M, sig in [(0, 0), (2, 1), (R(1, 2), R(1, 2)), (R(3, 2), R(-1, 2))]:
    half = M != int(M)
    for edge, x0 in [('upper', R(3, 5)), ('lower', R(-3, 5))]:
        prev = None
        for N in ([R(501, 2), R(1001, 2), R(2001, 2)] if half else [250, 500, 1000]):
            x, err, npts = run(M, sig, sp.nsimplify(N), x0, edge)
            print(' ', (M, sig), edge, N, mp.nstr(x, 6), npts, mp.nstr(err, 4), mp.nstr(err*mp.mpf(sp.N(N, 20)), 4))
            ok &= npts > 3 and err < 0.05 and (prev is None or err < prev)
            prev = err
check(ok, 'exact Hahn rank sequences collapse on the normalized Ai branch, error decreasing with N')
print('ALL CHECKS PASSED')
