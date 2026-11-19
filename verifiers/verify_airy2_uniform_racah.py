#!/usr/bin/env python3
"""Closed Racah varpi, its invariant (Olver) form, and the uniform second-order Airy statement.
# log: verify_airy2_uniform_racah.txt

thm:general-rank-airy-second(ii): bracket (1+O(N^-2))Ai(t) + N^(-4/3) B_tau(q) Ai'(t) + O(N^-3 M/E) on the
turning neighbourhood, B_tau = q1/g, B_tau(q_*) = varpi_tau; lem:airy2-olver; eq:racah-varpi, lem:racah-varpi.
Checks ((A) exact computer algebra; (B)-(D) numerical):
 (A) exact symbolic.  (A1) the local formula eq:general-airy-varpi-local equals the Olver form
     varpi = 3(5F1F3-3F2^2)/(35 F1^(8/3)) + F1^(-2/3)[F1^2/240 + G],  F = arccosh(tau u)^2 = -(p-p_*)^2,
     G = (a^2+b^2-1/4)/q_*^2 - 2(ab/q_*^2+v1)/(x-b0(q_*)) + (1/4)(log e(q^2))''(q_*).
     (A2) centered Racah data eq:racah-class-data: u1 = sin^2(th)/(rho_* eps^2) and
     varpi^Rac = 2^(1/3)[140 eps^3((m^2+n^2)x-2mn) + P(x,s_*)]/(280 (rho_* sin th)^(4/3)|eps|^(8/3)),
     P = -50x^4+15x^2-9 + 6 s_* x(25x^2-3) + s_*^2(21-149x^2) + 36 s_*^3 x + 4 s_*^4,
     s_* = rho_*^2/(4 alpha beta) (called vs below), eps = x - s_*, tau = sgn eps; generic (alpha, beta, x),
     modulo rho_*^2.  (A3) the term F1^2/240 of the Olver form: T(p) = cosh p gives F1^2(4 t2^2 - 2 t3) =
     F1^2/240 (ODE: 0), and dropping it shifts varpi by -F1^(4/3)/240 (Hahn (2,1), x = 3/5: -0.0304).
 (B) the limit eq:general-airy-varpi (quadrature of Sigma_2, extrapolated) equals the closed Racah form to
     1e-6: three physical 6j families (integer offsets, both tau; half-integer offsets (1/2,-1/2), tau = -1)
     and, as chain data only (not 6j symbols), two half-integer configurations with tau = +1 and -1.
 (C) Olver coefficient B_tau(q) = (q1/g)(q) on the fixed neighbourhood: psi(q) = -g [h^2]A[g,0](q)/zeta'(q)
     (Richardson in h on the exact natural-gauge coefficients), B = int_0^1 psi(zeta y^2) dy;
     B(q_*) = closed varpi, and B = tau*Gt/sqrt(-zeta) - 5/(48 zeta^2) at oscillatory points.
 (D) exact 6j on the whole neighbourhood q in [q_*-1/4, q_*+1/5] (every rank of the window; both sides
     of the turn), error relative to Olver's envelope M/E: Ai alone and Ai + N^(-4/3) varpi Ai' ~ N^-1;
     Ai + N^(-4/3) B_tau(q) Ai' ~ N^-2 (uniform statement).  Families: (m,n) = (1,0), (2,1) (integer N) and the
     physical half-integer family (1/2,-1/2), (alpha,beta,gamma) = (1,5/3,2), N = (4k-1)/4.
     The O(N^-3 M/E) remainder is tested by a joint fit: after the B_tau term the residual is fitted by
     N^-2 a(q) Ai + c_N Ai (a a degree-6 polynomial in q common to all N, c_N one constant per N); the fit
     error decays with slope close to -3 in N.
"""
import sys, os
import numpy as np
import mpmath as mp
import sympy as sp
from sympy.physics.wigner import wigner_6j
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_second_order_data import racah, cab

R = sp.Rational
mp.mp.dps = 30


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


def slope(Ns, errs):
    lx = [mp.log(n) for n in Ns]; ly = [mp.log(e) for e in errs]
    mx, my = sum(lx)/len(lx), sum(ly)/len(ly)
    return float(sum((a - mx)*(b - my) for a, b in zip(lx, ly))/sum((a - mx)**2 for a in lx))


# ---------------- (A) exact symbolic ----------------
qs, e0, u1 = sp.symbols('q_s e0 u1', positive=True)
e1, e2, u2, u3, v1, a, b, tau = sp.symbols('e1 e2 u2 u3 v1 a b tau', real=True)


def varpi_local(qs_, u1_, u2_, u3_, e0_, e1_, e2_, v1_, a_, b_, tau_):   # eq:general-airy-varpi-local
    return 2**R(1, 3)/(280*qs_**2*u1_**R(2, 3))*(35*(4*a_**2 + 4*b_**2 - 1) - 280*tau_*a_*b_/e0_ + qs_**2*(
        4*u1_**2 - 8*u2_ + 60*u3_/u1_ - 36*u2_**2/u1_**2 + 35*(2*e0_*e2_ - e1_**2)/e0_**2 - 280*tau_*v1_/e0_))


d = sp.Symbol('delta')
Fs = sp.expand(sp.series(sp.acosh(1 + u1*d + u2*d**2 + u3*d**3)**2, d, 0, 4).removeO())
F1, F2, F3 = [Fs.coeff(d, k) for k in (1, 2, 3)]
G = (a**2 + b**2 - R(1, 4))/qs**2 - 2*tau*(a*b/qs**2 + v1)/e0 + (2*e0*e2 - e1**2)/(4*e0**2)   # x - b0(q_*) = tau e0
olver = R(3, 35)*(5*F1*F3 - 3*F2**2)/F1**R(8, 3) + F1**R(-2, 3)*(F1**2/240 + G)
dA1 = sp.simplify((olver - varpi_local(qs, u1, u2, u3, e0, e1, e2, v1, a, b, tau)).subs(tau**2, 1))
dA1 = sp.simplify(sp.expand(dA1).subs({tau**2: 1, tau**3: tau}))
check(dA1 == 0, '(A1) local formula = Olver form psi_F(0) + F1^(-2/3)[F1^2/240 + G] (exact)')

m, n = sp.symbols('m n', real=True)
al, be, x, r = sp.symbols('alpha beta x r', positive=True)   # r = rho_*; only r^2 survives
r2 = 4*al**2*be**2*(1 - x**2)/(al**2 + be**2 - 2*al*be*x)    # rho_*^2 = (2 alpha beta sin(theta)/gamma)^2
Es = lambda s: (1 - s/(4*al**2))*(1 - s/(4*be**2))           # e(s)^2
K = (al**2 + be**2 - 2*al*be*x)/(4*al**2*be**2)               # D_6 = sin^2(theta) - K s   (eq:D6linear)
check(sp.cancel(Es(r2) - (x - r2/(4*al*be))**2 - (1 - x**2 - K*r2)) == 0,
      '(A2) D_6(rho_*) = 0 and D_6 = sin^2(theta) - gamma^2 rho^2/(4 alpha^2 beta^2) (turn and eq:D6linear)')
sv = sp.Symbol('s')
E0_, E1_, E2_ = Es(r2), sp.diff(Es(sv), sv).subs(sv, r2), sp.Rational(1, 16)/(al**2*be**2)
c1 = E1_/E0_; c2 = c1**2 - E2_/E0_                             # w = -D/E = K Dl/E(s_*+Dl), Dl = 2 r d + d^2
w1, w2, w3 = K/E0_*2*r, K/E0_*(1 - 4*r**2*c1), K/E0_*(-4*r*c1 + 8*r**3*c2)
U1, U2, U3 = w1/2, w2/2 - w1**2/8, w3/2 - w1*w2/4 + w1**3/16    # tau u = sqrt(1 + w)
vs = r2/(4*al*be); eps = x - vs
Eq, Eqq = 2*r*E1_, 2*E1_ + 4*r**2*E2_*2                         # d/dq, d^2/dq^2 of E(q^2) at r
curv = (Eqq/E0_ - (Eq/E0_)**2)/2                                # (2 e0 e2 - e1^2)/e0^2 = (log e)''
def rr(ex):
    """Numerator of a rational function of r, reduced modulo r^2 = rho_*^2: even and odd parts."""
    num = sp.Poly(sp.numer(sp.together(ex)), r)
    ev = sum(c*r2**(k//2) for (k,), c in num.terms() if k % 2 == 0)
    od = sum(c*r2**(k//2) for (k,), c in num.terms() if k % 2 == 1)
    return (sp.cancel(ev), sp.cancel(od))
check(rr(E0_ - eps**2) == (0, 0) and rr(U1*r - (1 - x**2)/eps**2) == (0, 0),
      '(A2) e(rho_*^2) = |eps| and u1 = sin^2(theta)/(rho_* eps^2) exactly')
v1R = (R(1, 2) - m**2 - n**2)/(8*al*be)
brk = (35*(4*m**2 + 4*n**2 - 1) - 280*m*n/eps + r**2*(4*U1**2 - 8*U2 + 60*U3/U1 - 36*U2**2/U1**2
       + 35*curv - 280*v1R/eps))                                # tau/e0 = 1/eps
Pfun = lambda X, S: -50*X**4 + 15*X**2 - 9 + 6*S*X*(25*X**2 - 3) + S**2*(21 - 149*X**2) + 36*S**3*X + 4*S**4
closed_brk = (140*eps**3*((m**2 + n**2)*x - 2*m*n) + Pfun(x, vs))/eps**4
check(rr(brk - closed_brk) == (0, 0),
      '(A2) Racah: 280 q_*^2 u1^(2/3) varpi/2^(1/3) = [140 eps^3((m^2+n^2)x-2mn) + P(x,vs)]/eps^4 (exact; gives the closed form)')
xs = sp.Symbol('x_s')
check(sp.expand(Pfun(xs, 0) - (-50*xs**4 + 15*xs**2 - 9)) == 0,
      '(A2) vs = 0 gives the Hahn numerator of eq:hahn-varpi')


# (A3) the F1^2/240 term of the Olver form: symbol T(p) = cosh p versus the ODE symbol
t2s, t3s = sp.symbols('t2 t3')
Tc = sp.series(sp.cosh(sp.Symbol('p')), sp.Symbol('p'), 0, 8).removeO()
t2c, t3c = Tc.coeff(sp.Symbol('p'), 4), Tc.coeff(sp.Symbol('p'), 6)
check((4*t2s**2 - 2*t3s).subs({t2s: t2c, t3s: t3c}) == R(1, 240) and (4*t2s**2 - 2*t3s).subs({t2s: 0, t3s: 0}) == 0,
      '(A3) F1^2(4 t2^2 - 2 t3) = F1^2/240 for cosh (t2 = 1/24, t3 = 1/720) and 0 for the ODE symbol')
Fh = sp.series(sp.acosh(1 + sp.Rational(4, 5)/sp.Rational(9, 25)*d)**2, d, 0, 2).removeO().coeff(d, 1)  # Hahn: u1 = q_*/x^2
check(Fh == R(40, 9), '(A3) Hahn x = 3/5: u1 = q_*/x^2 = 20/9, F1 = 2 u1 = 40/9')
drop = float(Fh**R(4, 3)/240)
check(abs(drop - 0.0304) < 5e-5, '(A3) dropping F1^2/240 changes varpi by -F1^(4/3)/240 = -%.4f (Hahn x = 3/5; the offsets only enter G)' % drop)


def varpi_racah(m_, n_, al_, be_, ga_):
    al_, be_, ga_ = (mp.mpf(sp.N(v, 40)) for v in (al_, be_, ga_))
    m_, n_ = mp.mpf(sp.N(m_, 40)), mp.mpf(sp.N(n_, 40))
    X = (al_**2 + be_**2 - ga_**2)/(2*al_*be_); s2 = 1 - X*X
    r = 2*al_*be_*mp.sqrt(s2)/ga_; sg = r*r/(4*al_*be_); ep = X - sg
    P = -50*X**4 + 15*X**2 - 9 + 6*sg*X*(25*X**2 - 3) + sg**2*(21 - 149*X**2) + 36*sg**3*X + 4*sg**4
    return mp.cbrt(2)*(140*ep**3*((m_**2 + n_**2)*X - 2*m_*n_) + P)/(280*(r*mp.sqrt(s2))**(mp.mpf(4)/3)*abs(ep)**(mp.mpf(8)/3))


# ---------------- numerical machinery ----------------
class RT:
    """Centered Racah chain at fixed (m, n, alpha, beta, gamma)."""
    def __init__(s, m_, n_, abg):
        s.abg = abg
        s.m, s.n = mp.mpf(sp.N(m_, 40)), mp.mpf(sp.N(n_, 40))
        s.al, s.be, s.ga = (mp.mpf(sp.N(v, 40)) for v in abg)
        s.x = (s.al**2 + s.be**2 - s.ga**2)/(2*s.al*s.be)
        s.qs = 2*s.al*s.be*mp.sqrt(1 - s.x**2)/s.ga
        s.tau = int(mp.sign(s.x - s.qs**2/(4*s.al*s.be)))
        s.varpi = varpi_racah(m_, n_, *abg)

    def e(s, q): return mp.sqrt((1 - q*q/(4*s.al**2))*(1 - q*q/(4*s.be**2)))
    def tu(s, q): return s.tau*(s.x - q*q/(4*s.al*s.be))/s.e(q)
    def D(s, q): return s.e(q)**2 - (s.x - q*q/(4*s.al*s.be))**2

    def zeta(s, q):
        if q < s.qs:
            return -(mp.mpf(3)/2*mp.quad(lambda r: mp.acos(min(s.tu(r), 1)), [q, s.qs]))**(mp.mpf(2)/3)
        return (mp.mpf(3)/2*mp.quad(lambda r: mp.acosh(max(s.tu(r), 1)), [s.qs, q]))**(mp.mpf(2)/3)

    def g(s, q, z): return mp.sqrt(2)*abs(z/s.D(q))**mp.mpf('0.25')

    def a(s, S, h2):         # exact natural-gauge hopping a(S; h^2) = A_{S/h}
        return mp.sqrt((S*S - s.m**2*h2)*(S*S - s.n**2*h2))/(S*mp.sqrt(4*S*S - h2))*s.e(S)

    def bd(s, q, h2):        # exact diagonal b(q; h^2) = B_k at q = q_k
        return s.m*s.n*h2/(q*q - h2/4) + (2*q*q + h2*(mp.mpf(1)/2 - s.m**2 - s.n**2))/(8*s.al*s.be)


def Cser(sv, z, h, K=70):    # C(s; zeta, h): Y'' = (zeta + h s) Y, C(0) = 1, C'(0) = 0
    c = [mp.mpf(1), mp.mpf(0)]
    for k in range(K):
        c.append((z*c[k] + (h*c[k - 1] if k else 0))/((k + 2)*(k + 1)))
    return mp.polyval(c[::-1], sv)


def psi_disc(T, q, hs=(mp.mpf('0.004'), mp.mpf('0.002'), mp.mpf('0.001'))):
    zq = T.zeta(q); gq = T.g(q, zq); vals = []
    for h in hs:                                  # A[g,0](q; h), Lemma airy2-shift-parity
        A = -T.tau*(T.x - T.bd(q, h*h))*gq
        for sg in (1, -1):
            zz = T.zeta(q + sg*h)
            A += T.a(q + sg*h/2, h*h)*T.g(q + sg*h, zz)*Cser((zz - zq)/h, zq, h)
        vals.append(A/h**2)
    r = [(4*vals[i + 1] - vals[i])/3 for i in range(2)]
    A2 = (16*r[1] - r[0])/15                      # [h^2]A[g,0](q)
    zp = (mp.acos(T.tu(q))/mp.sqrt(-zq)) if zq < 0 else (mp.acosh(T.tu(q))/mp.sqrt(zq))
    return -gq*A2/zp, zq


def cheb(f_vals, lo, hi):
    """Chebyshev interpolant (values at first-kind nodes) and its derivative, Clenshaw evaluation."""
    nc = len(f_vals)
    c = [2*mp.fsum(f_vals[j]*mp.cos(mp.pi*k*(j + mp.mpf(1)/2)/nc) for j in range(nc))/nc for k in range(nc)]
    c[0] /= 2
    dc = [mp.mpf(0)]*(nc + 1)
    for k in range(nc - 1, 0, -1):
        dc[k - 1] = dc[k + 1] + 2*k*c[k]
    dc[0] /= 2
    def clen(cs, q):
        y = (2*q - lo - hi)/(hi - lo); b1 = b2 = mp.mpf(0)
        for ck in cs[:0:-1]:
            b1, b2 = 2*y*b1 - b2 + ck, b1
        return y*b1 - b2 + cs[0]
    return (lambda q: clen(c, q)), (lambda q: 2/(hi - lo)*clen(dc[:nc], q)), abs(c[-1])


class OlverB:
    """B_tau(q): psi and phi = zeta/(q - q_*) on Chebyshev nodes; B = phi^(-1/2) int_0^1 psi zeta' phi^(-1/2) dw."""
    def __init__(s, T, half, up=None, nc=28):
        lo, hi = T.qs - half, T.qs + (half if up is None else up); s.T = T
        nodes = [(lo + hi)/2 + (hi - lo)/2*mp.cos(mp.pi*(j + mp.mpf(1)/2)/nc) for j in range(nc)]
        ps, ph = [], []
        for q in nodes:
            p_, z = psi_disc(T, q); ps.append(p_); ph.append(z/(q - T.qs))
        s.psi, _, s.tail_psi = cheb(ps, lo, hi)
        s.phi, s.dphi, s.tail_phi = cheb(ph, lo, hi)
        s.gl = mp.calculus.quadrature.GaussLegendre(mp.mp).get_nodes(0, 1, 4, mp.mp.prec)   # 24 Gauss nodes on [0,1]

    def zeta(s, q): return (q - s.T.qs)*s.phi(q)

    def B(s, q):
        qs_ = s.T.qs; tot = mp.mpf(0)
        for wv, wt in s.gl:
            qq = qs_ + (q - qs_)*wv*wv
            tot += wt*s.psi(qq)*(s.phi(qq) + (qq - qs_)*s.dphi(qq))/mp.sqrt(s.phi(qq))
        return tot/mp.sqrt(s.phi(q))


def gamma_tilde_B(T, ch, q):
    """tau*Gt(q)/sqrt(-zeta) - 5/(48 zeta^2), Gt = Gamma_2 + c_ab(Theta_q)/q (bulk second-order phase)."""
    Gt = ch.Gam2(q, T.x) + cab(T.m, T.n, ch.Th(q, T.x))/q
    z = T.zeta(q)
    return T.tau*Gt/mp.sqrt(-z) - mp.mpf(5)/(48*z*z)


def limit_B(T, ch):
    """eq:general-airy-varpi: Gt by Sigma_2 quadrature towards q_*, Neville extrapolation to delta = 0."""
    qs_ = T.qs; q0 = qs_/2
    Gt = ch.Gam2(q0, T.x) + cab(T.m, T.n, ch.Th(q0, T.x))/q0
    ds = [qs_/16/mp.mpf(2)**j for j in range(7)]; vals = []; prev = q0
    for dd in ds:
        Q = qs_ - dd
        Gt += mp.quad(lambda r: ch.Sig(r, T.x), [prev, (prev + Q)/2, Q]); prev = Q
        z = T.zeta(Q); vals.append(T.tau*Gt/mp.sqrt(-z) - mp.mpf(5)/(48*z*z))
    tab = vals[:]
    for j in range(1, len(ds)):
        for i in range(len(ds) - 1, j - 1, -1):
            tab[i] = (ds[i - j]*tab[i] - ds[i]*tab[i - 1])/(ds[i - j] - ds[i])
    return tab[-1], abs(tab[-1] - tab[-2])


# ---------------- (B) limit formula vs closed Racah form ----------------
print('\n(B) limit eq:general-airy-varpi vs closed Racah form', flush=True)
# (m, n, (alpha, beta, gamma), physical): physical = the family is a family of exact 6j symbols (labels A = N alpha, ...
# with integer N, or N in Z/4 for the half-integer family); the two other half-integer chains are chain data only.
CFG = [(1, 0, (1, 2, R(6, 5)), True), (2, -1, (1, 2, R(6, 5)), True), (R(1, 2), R(-1, 2), (1, R(5, 3), 2), True),
       (R(1, 2), R(-1, 2), (1, 3, R(201, 80)), False), (2, 1, (1, R(6, 5), R(3, 2)), True), (0, 0, (1, 1, R(13, 10)), True),
       (R(3, 2), R(1, 2), (R(11, 10), 1, R(9, 10)), False)]
TOL_B = mp.mpf('1e-6')
kinds = set()
for m_, n_, abg, phys in CFG:
    T = RT(m_, n_, abg); ch = racah(m_, n_, abg[0], abg[1])
    lim, est = limit_B(T, ch)
    print('  (m,n)=(%s,%s) (al,be,ga)=%s q*=%.6f tau=%+d %s limit=% .12f (extrap %.0e) closed=% .12f'
          % (m_, n_, abg, float(T.qs), T.tau, 'physical 6j' if phys else 'chain data only', float(lim), float(est),
             float(T.varpi)), flush=True)
    check(abs(lim - T.varpi) < TOL_B, '(B) limit = closed Racah varpi (diff %.1e < 1e-6)' % float(abs(lim - T.varpi)))
    kinds.add((T.tau, sp.Rational(m_).q == 2, phys))
check(kinds >= {(1, False, True), (-1, False, True), (-1, True, True), (1, True, False), (-1, True, False)},
      '(B) both tau with integer offsets (6j), tau = -1 with half-integer offsets (6j), both tau with half-integer chain data only')


# ---------------- (C), (D) Olver coefficient and uniform exact 6j ----------------
def env(t):                                       # Olver's M/E for Ai
    return mp.sqrt(2)*mp.airyai(t) if t >= mp.mpf('-0.36605') else mp.sqrt(mp.airyai(t)**2 + mp.airybi(t)**2)


def exact_6j(m_, n_, abg, N, qhi, dps=90, spot=True):
    """Natural-gauge exact 6j values (sympy seed at k_min, exact recurrence eq:6jrec at high precision).  N rational."""
    with mp.workdps(dps):
        Nm = mp.mpf(N.p)/N.q
        Ar, Br, Cr = (R(v)*N for v in abg)
        a6, b6, c6 = Ar - R(n_ + 1, 2), Br - R(m_ + 1, 2), Cr - R(1, 2)
        def Psi(f):
            ph = a6 + b6 + c6 + f + R(m_)
            assert ph.q == 1, 'not a 6j label set'
            return (-1)**int(ph)*2*sp.sqrt(Ar*Br)*wigner_6j(a6, b6, c6, b6 + m_, a6 + n_, f)
        kmin = max(abs(R(m_)), abs(R(n_)))
        Am, Bm = mp.mpf(sp.N(Ar, dps)), mp.mpf(sp.N(Br, dps)); X = (Am**2 + Bm**2 - mp.mpf(sp.N(Cr, dps))**2)/(2*Am*Bm)
        mf, nf, k0 = (mp.mpf(sp.N(v, dps)) for v in (R(m_), R(n_), kmin))
        eta = lambda k: mp.sqrt((1 - k*k/(4*Am*Am))*(1 - k*k/(4*Bm*Bm)))
        A = lambda k: mp.sqrt((k*k - mf*mf)*(k*k - nf*nf))/(k*mp.sqrt(4*k*k - 1))*eta(k)
        B = lambda k: (2*k*(k + 1) + 1 - mf*mf - nf*nf)/(8*Am*Bm) + (mf*nf/(k*(k + 1)) if k > 0 else 0)
        Y = [mp.sqrt(2*k0 + 1)*mp.mpf(sp.N(Psi(kmin), dps))]
        Y.append(mp.sqrt(3)*mp.mpf(sp.N(Psi(1), dps)) if kmin == 0 else (X - B(k0))*Y[0]/A(k0 + 1))
        for j in range(1, int(qhi*Nm - k0) + 1):
            k = k0 + j; Y.append(((X - B(k))*Y[j] - A(k)*Y[j - 1])/A(k + 1))
        jt = len(Y) - 3                           # spot check against sympy (last forbidden-side rank)
        dev = abs(Y[jt]/mp.sqrt(2*(k0 + jt) + 1) - mp.mpf(sp.N(Psi(kmin + jt), dps))) if spot else mp.mpf(0)
    return Y, kmin, Ar, Br, Cr, dev


print('\n(C)/(D) Olver coefficient B_tau(q) and uniform exact 6j on q in [q*-1/4, q*+1/5], every rank of the window',
      flush=True)
FAMS = [(1, 0, (1, 2, R(6, 5)), [R(40), R(80), R(160), R(320)], mp.mpf('0.3')),
        (2, 1, (1, R(6, 5), R(3, 2)), [R(40), R(80), R(160), R(320)], mp.mpf('0.25')),
        (R(1, 2), R(-1, 2), (1, R(5, 3), 2), [R(159, 4), R(321, 4), R(639, 4), R(1281, 4)], mp.mpf('0.3'), mp.mpf('0.25'))]
for m_, n_, abg, Ns, half, *up in FAMS:
    T = RT(m_, n_, abg); OB = OlverB(T, half, *up); ch = racah(m_, n_, abg[0], abg[1])
    b0 = OB.B(T.qs + mp.mpf('1e-20'))
    check(abs(b0 - T.varpi) < 1e-8 and OB.tail_psi < 1e-7,
          '(C) (%s,%s) tau=%+d: B(q*) = psi(0) = closed varpi (diff %.1e, Chebyshev tail %.0e)'
          % (m_, n_, T.tau, float(abs(b0 - T.varpi)), float(OB.tail_psi)))
    dg = max(abs(OB.B(T.qs - dq) - gamma_tilde_B(T, ch, T.qs - dq)) for dq in (mp.mpf('0.25'), mp.mpf('0.1')))
    check(dg < 1e-8, '(C) (%s,%s): B = tau*Gt/sqrt(-zeta) - 5/(48 zeta^2) on the oscillatory side (max diff %.1e)'
          % (m_, n_, float(dg)))
    lo, hi = T.qs - mp.mpf(1)/4, T.qs + mp.mpf(1)/5
    E0, Ec, E1 = [], [], []; fitdata = []
    for N in Ns:
        Nm = mp.mpf(N.p)/N.q
        Y, kmin, Ar, Br, Cr, dev = exact_6j(m_, n_, abg, N, hi, spot=(N <= 81))
        if N <= 81:
            check(dev < 1e-25, '(D) N=%s: recurrence = sympy wigner_6j at the last rank (diff %.0e)' % (N, float(dev)))
        p = abs(R(n_) - R(m_)); sgn = (-1)**int(max(R(n_) - R(m_), 0)); h = 1/Nm
        e0_ = ec_ = e1_ = mp.mpf(0); cnt = 0; rows = []
        for j in range(1, len(Y)):
            q = (mp.mpf(sp.N(kmin, 30)) + j + mp.mpf(1)/2)/Nm
            if q < lo or q > hi:
                continue
            f = kmin + j; z = OB.zeta(q); t = Nm**(mp.mpf(2)/3)*z
            K = Cr - abs(Ar - Br) + (p - 1)/2 if T.tau == 1 else f + R(1, 2) - Ar - Br + Cr + p/2
            assert K.q == 1, 'Airy integer K_tau must be an integer'
            pref = sgn*(-1)**int(K)*mp.sqrt(2/q)*Nm**(-mp.mpf(1)/3)*abs(z/T.D(q))**mp.mpf('0.25')
            br = Y[j]/mp.sqrt(2*f + 1)/pref; Ev = env(t); ai, aip = mp.airyai(t), mp.airyai(t, 1)
            Bq = OB.B(q); res = br - ai - h**(mp.mpf(4)/3)*Bq*aip
            e0_ = max(e0_, abs(br - ai)/Ev)
            ec_ = max(ec_, abs(br - ai - h**(mp.mpf(4)/3)*T.varpi*aip)/Ev)
            e1_ = max(e1_, abs(res)/Ev); cnt += 1
            rows.append((float((q - T.qs)/half), float(ai/Ev), float(res/Ev)))
        E0.append(e0_); Ec.append(ec_); E1.append(e1_); fitdata.append((float(Nm), rows))
        print('  (%s,%s) tau=%+d N=%-6s pts=%3d  max |err|/(M/E): Ai %.3e | Ai+N^-4/3 varpi Ai\' %.3e | Ai+N^-4/3 B(q) Ai\' %.3e'
              % (m_, n_, T.tau, N, cnt, float(e0_), float(ec_), float(e1_)), flush=True)
    Nf = [float(mp.mpf(N.p)/N.q) for N in Ns]
    s0, sc, s1 = slope(Nf, E0), slope(Nf, Ec), slope(Nf, E1)
    print('  exponents: Ai %.2f, constant varpi %.2f, B(q) %.2f' % (s0, sc, s1), flush=True)
    check(s1 < -1.8 and sc > -1.2 and s0 > -1.2,
          '(D) (%s,%s) tau=%+d: uniform relative-envelope error N^-2 with B_tau(q); N^-1 without it' % (m_, n_, T.tau))
    # O(N^-3 M/E) remainder: residual after the B_tau term = N^-2 a(q) Ai/E + c_N Ai/E + O(N^-3), a common to all N
    deg = 6; cols = deg + len(Ns); X_, y_, w_, idx_ = [], [], [], []
    for i, (Nv, rows) in enumerate(fitdata):
        for yq, aiE, rE in rows:
            X_.append([np.polynomial.chebyshev.chebval(yq, np.eye(deg + 1)[l])*aiE/Nv**2 for l in range(1, deg + 1)]
                      + [aiE if ii == i else 0. for ii in range(len(Ns))])
            y_.append(rE); w_.append(Nv**3); idx_.append(i)
    X_, y_, w_, idx_ = np.array(X_), np.array(y_), np.array(w_), np.array(idx_)
    co = np.linalg.lstsq(X_*w_[:, None], y_*w_, rcond=None)[0]
    fe = [float(np.max(np.abs((y_ - X_@co)[idx_ == i]))) for i in range(len(Ns))]
    s3 = float(np.polyfit(np.log(Nf), np.log(fe), 1)[0])
    print('  fit N^-2 a(q) Ai + c_N Ai (common degree-%d a(q), %d constants c_N) to the B_tau residual: max error %s; slope %.2f; '
          'error x N^3: %s' % (deg, len(Ns), ', '.join('%.2e' % e for e in fe), s3,
                               ', '.join('%.2f' % (e*Nv**3) for e, Nv in zip(fe, Nf))), flush=True)
    check(s3 < -2.6, '(D) (%s,%s) tau=%+d: remainder after the fit is O(N^-3 M/E) (log-slope %.2f <= -2.6)'
          % (m_, n_, T.tau, s3))

print('ALL CHECKS PASSED')
