#!/usr/bin/env python3
"""Closed local formula for the second-order Airy coefficient varpi_tau.
# log: verify_airy2_varpi_closed.txt

thm:general-rank-airy-second (article) has bracket Ai(t) + N^(-4/3) varpi_tau Ai'(t)
+ O(N^-2), with varpi_tau = (q_1/g)(q_*) (appendix proof).  Evaluating the
transport equation (W q_1^2)'/q_1 = -[h^2] A[g,0] of thm:airy2-formal at q_*,
where W(q_*) = 0, gives q_1(q_*) = -[h^2]A[g,0](q_*)/W'(q_*), i.e. the local
formula eq:general-airy-varpi-local in the Taylor data at the turn
    tau u(q_*+d) = 1 + u1 d + u2 d^2 + u3 d^3 + ...,
    e_pi((q_*+d)^2) = e0 + e1 d + e2 d^2 + ...,   v1 = d_t v_pi(q_*^2, 0):
  varpi_tau = 2^(1/3)/(280 q_*^2 u1^(2/3)) [35(4a^2+4b^2-1) - 280 tau a b/e0
      + q_*^2(4u1^2 - 8u2 + 60u3/u1 - 36u2^2/u1^2 + 35(2e0e2-e1^2)/e0^2 - 280 tau v1/e0)].
Checks:
  (A) exact symbolic: [h^0]A[g,0] = [h^1]A[g,0] = 0 at q_*, the ODE evaluation equals
      the displayed formula, and for Hahn (e = sqrt(1-s), v = 0, tau = sgn x) it reduces to
      varpi^Hahn = 2^(1/3)(140(M^2+s^2)x^4 - 280 M s x^3 - 50x^4 + 15x^2 - 9)/(280|x|^(8/3)(1-x^2)^(4/3)).
  (B) numerical: the limit eq:general-airy-varpi (quadrature of Sigma_2, extrapolated to
      q -> q_*) equals the local formula: Hahn channels for both tau (integer and
      half-integer), centered Racah configurations for both tau.
  (C) exact 3j / 6j brackets in |t| <= 2: error of Ai alone ~ N^(-4/3); after the
      N^(-4/3) varpi Ai' term with the local formula ~ N^(-2); control with -varpi ~ N^(-4/3).
      6j: exact sympy seed at the lowest rank, exact recurrence eq:6jrec (spot-checked
      against sympy wigner_6j).
"""
import sys
import mpmath as mp
import sympy as sp
from sympy.physics.wigner import wigner_3j, wigner_6j
from p3_second_order_data import Chain, cab

mp.mp.dps = 30
R = sp.Rational


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


def slope(Ns, errs):
    lx = [mp.log(n) for n in Ns]; ly = [mp.log(abs(e)) for e in errs]
    mx = sum(lx)/len(lx); my = sum(ly)/len(ly)
    return float(sum((a - mx)*(b - my) for a, b in zip(lx, ly))/sum((a - mx)**2 for a in lx))


# ---------------- (A) symbolic derivation at the turn ----------------
d, h = sp.symbols('delta h')
qs, e0, u1 = sp.symbols('q_s e0 u1', positive=True)
e1, e2, e3, u2, u3, u4, v1, a, b, tau = sp.symbols('e1 e2 e3 u2 u3 u4 v1 a b tau', real=True)
w = u1*d + u2*d**2 + u3*d**3 + u4*d**4
zs = sp.symbols('z1:6')
zeta = sum(zs[i]*d**(i + 1) for i in range(5))
zp = sp.diff(zeta, d)
eik = sp.expand(sp.series(sum(zeta**k*zp**(2*k)/sp.factorial(2*k) for k in range(4)) - 1 - w, d, 0, 6).removeO())
sol = {zs[0]: (2*u1)**R(1, 3)}                       # cosh(zeta^(1/2) zeta') = tau u
for k in range(2, 6):
    sol[zs[k - 1]] = sp.solve(sp.expand(eik.coeff(d, k).subs(sol)), zs[k - 1])[0]
zeta_s = zeta.subs(sol)
zp_s = sp.diff(zeta_s, d)
e_d = e0 + e1*d + e2*d**2 + e3*d**3
V = sp.expand(sp.series(e_d/2*sum(zeta_s**k*zp_s**(2*k + 1)/sp.factorial(2*k + 1) for k in range(4)), d, 0, 4).removeO())
g = sp.series(V**R(-1, 2), d, 0, 4).removeO()                     # g = |V|^(-1/2)
ca = R(1, 8) - (a**2 + b**2)/2
Cser = lambda s_: 1 + h*s_**3/6 + h**2*s_**6/180                   # C(s; 0, h) at zeta(q_*) = 0
Aexpr = -(e0 - tau*h**2*(a*b/qs**2 + v1))*g.subs(d, 0)             # -tau(x - b(q_*; h^2)) g
for sg in (1, -1):
    Aexpr += e_d.subs(d, sg*h/2)/2*(1 + h**2*ca/(qs + sg*h/2)**2)*g.subs(d, sg*h)*Cser(sp.expand(zeta_s.subs(d, sg*h)/h))
Aexpr = sp.expand(sp.series(Aexpr, h, 0, 3).removeO())
A0, A1, A2 = [sp.simplify(Aexpr.coeff(h, k)) for k in range(3)]
check(A0 == 0 and A1 == 0, '(A) [h^0]A[g,0] = [h^1]A[g,0] = 0 at q_* (eikonal and parity)')
Wp = e0/2*sol[zs[0]]**2                                            # W'(q_*) = zeta'(q_*) V(q_*)
varpi_ode = -A2/(Wp*g.subs(d, 0))


def varpi_closed(qs_, u1_, u2_, u3_, e0_, e1_, e2_, v1_, a_, b_, tau_):
    return 2**R(1, 3)/(280*qs_**2*u1_**R(2, 3))*(35*(4*a_**2 + 4*b_**2 - 1) - 280*tau_*a_*b_/e0_ + qs_**2*(
        4*u1_**2 - 8*u2_ + 60*u3_/u1_ - 36*u2_**2/u1_**2 + 35*(2*e0_*e2_ - e1_**2)/e0_**2 - 280*tau_*v1_/e0_))


diffA = sp.simplify(varpi_ode - varpi_closed(qs, u1, u2, u3, e0, e1, e2, v1, a, b, tau))
check(diffA == 0, '(A) ODE of thm:airy2-formal at q_* gives the local formula eq:general-airy-varpi-local exactly')
X, M_, S_, qq = sp.symbols('X M sigma q', real=True)
for tv in (1, -1):                                                  # Hahn: x = tv*X, X = |x| > 0
    Xp = sp.Symbol('Xp', positive=True)
    qsH = sp.sqrt(1 - Xp**2)
    tu = sp.series(Xp/sp.sqrt(1 - (qsH + d)**2), d, 0, 4).removeO()   # tau u = |x|/sqrt(1-q^2)
    eH = sp.series(sp.sqrt(1 - (qsH + d)**2), d, 0, 3).removeO()
    vH = varpi_closed(qsH, tu.coeff(d, 1), tu.coeff(d, 2), tu.coeff(d, 3), eH.coeff(d, 0), eH.coeff(d, 1),
                      eH.coeff(d, 2), 0, M_, S_, tv)
    xx = tv*Xp
    hahn_closed = 2**R(1, 3)*(140*(M_**2 + S_**2)*xx**4 - 280*M_*S_*xx**3 - 50*xx**4 + 15*xx**2 - 9)/(
        280*Xp**R(8, 3)*(1 - Xp**2)**R(4, 3))
    dH = [sp.N((vH - hahn_closed).subs({Xp: xv, M_: mv, S_: sv}), 30)
          for xv, mv, sv in [(R(3, 5), 2, 1), (R(1, 3), R(3, 2), R(-1, 2)), (R(7, 9), 0, 0), (R(1, 2), 5, -3)]]
    check(sp.simplify(sp.radsimp(vH - hahn_closed)) == 0 and max(abs(v) for v in dH) < 1e-25,
          '(A) Hahn specialization, tau=%+d: closed form varpi^Hahn_(sgn x)(x) exactly (max diff at test points %.1e)'
          % (tv, float(max(abs(v) for v in dH))))


# ---------------- (B) limit eq:general-airy-varpi vs local formula ----------------
class Turn:
    """Chain (e(s), v(s,t)) at fixed x with a simple rank turn."""
    def __init__(self, e_expr, v_expr, aa, bb, x):
        s, t = sp.symbols('s t', real=True)
        self.ch = Chain(e_expr.subs(sp.Symbol('s', real=True), s), v_expr, aa, bb)
        self.a, self.b, self.x = aa, bb, mp.mpf(sp.N(x, 40))
        q = sp.Symbol('q', real=True)
        ef = sp.lambdify(q, e_expr.subs(s, q**2), 'mpmath')
        bf = sp.lambdify(q, v_expr.subs({s: q**2, t: 0}), 'mpmath')
        self.vt = sp.lambdify(q, sp.diff(v_expr, t).subs({s: q**2, t: 0}), 'mpmath')
        self.e, self.b0 = ef, bf
        X_ = self.x
        D = lambda Q: ef(Q)**2 - (X_ - bf(Q))**2
        self.D = D
        grid = [mp.mpf(j)/400 for j in range(1, 1600)]
        k = next(j for j in range(len(grid) - 1) if D(grid[j]) > 0 and D(grid[j + 1]) <= 0)
        self.qs = mp.findroot(D, (grid[k], grid[k + 1]), solver='anderson')
        self.tau = 1 if (X_ - bf(self.qs))/ef(self.qs) > 0 else -1
        self.u = lambda Q: (X_ - bf(Q))/ef(Q)

    def zeta(self, Q):
        tq = self.tau
        if Q <= self.qs:
            f = lambda r: mp.acos(tq*self.u(r))                      # p or pi - p
            return -(mp.mpf(3)/2*mp.re(mp.quad(f, [Q, self.qs])))**(mp.mpf(2)/3)
        return (mp.mpf(3)/2*mp.re(mp.quad(lambda r: mp.acosh(tq*self.u(r)), [self.qs, Q])))**(mp.mpf(2)/3)

    def varpi_limit(self):
        """eq:general-airy-varpi; Gt' = Sigma_2, integrated on geometric segments, Neville to delta = 0."""
        X_, ch, qs_ = self.x, self.ch, self.qs
        q0 = qs_/2
        G0 = ch.Gam2(q0, X_) + cab(self.a, self.b, ch.Th(q0, X_))/q0
        ds = [qs_/16/mp.mpf(2)**j for j in range(7)]
        vals, Gt, prev = [], G0, q0
        for dd in ds:
            Q = qs_ - dd
            Gt += mp.quad(lambda r: ch.Sig(r, X_), [prev, (prev + Q)/2, Q]); prev = Q
            z = self.zeta(Q)
            vals.append(self.tau*Gt/mp.sqrt(-z) - mp.mpf(5)/(48*z*z))
        tab = vals[:]
        for j in range(1, len(ds)):
            for i in range(len(ds) - 1, j - 1, -1):
                tab[i] = (ds[i - j]*tab[i] - ds[i]*tab[i - 1])/(ds[i - j] - ds[i])
        return tab[-1], abs(tab[-1] - tab[-2])

    def varpi_local(self):
        tq, qs_ = self.tau, self.qs
        tu = mp.taylor(lambda Q: tq*self.u(Q), qs_, 3)
        ej = mp.taylor(self.e, qs_, 2)
        v = varpi_closed(qs_, tu[1], tu[2], tu[3], ej[0], ej[1], ej[2], self.vt(qs_),
                         mp.mpf(sp.N(self.a, 30)), mp.mpf(sp.N(self.b, 30)), tq)
        return mp.mpf(sp.N(v, 30)) if isinstance(v, sp.Basic) else v


s_, t_ = sp.symbols('s t', real=True)


def hahn_turn(M, sg, x):
    return Turn(sp.sqrt(1 - s_), sp.Integer(0)*t_, M, sg, x)


def racah_turn(m, n, al, be, ga):
    al, be, ga = (sp.nsimplify(v) for v in (al, be, ga))
    return Turn(sp.sqrt(1 - s_/(4*al**2))*sp.sqrt(1 - s_/(4*be**2)),
                s_/(4*al*be) + (R(1, 2) - sp.nsimplify(m)**2 - sp.nsimplify(n)**2)*t_/(8*al*be),
                m, n, (al**2 + be**2 - ga**2)/(2*al*be))


def hahn_varpi(M, sg, x):
    x = mp.mpf(x); M, sg = mp.mpf(M), mp.mpf(sg)
    return mp.cbrt(2)*(140*(M*M + sg*sg)*x**4 - 280*M*sg*x**3 - 50*x**4 + 15*x*x - 9)/(
        280*abs(x)**(mp.mpf(8)/3)*(1 - x*x)**(mp.mpf(4)/3))


print('\n(B) limit eq:general-airy-varpi vs local formula', flush=True)
taus = set()
for lab, T, closed in [('Hahn (2,1) x=3/5', hahn_turn(2, 1, R(3, 5)), hahn_varpi(2, 1, mp.mpf(3)/5)),
                       ('Hahn (2,1) x=-3/5', hahn_turn(2, 1, R(-3, 5)), hahn_varpi(2, 1, -mp.mpf(3)/5)),
                       ('Hahn (3/2,-1/2) x=-1/2', hahn_turn(R(3, 2), R(-1, 2), R(-1, 2)), hahn_varpi(1.5, -0.5, -0.5)),
                       ('Hahn (0,0) x=7/10', hahn_turn(0, 0, R(7, 10)), hahn_varpi(0, 0, mp.mpf(7)/10)),
                       ('Racah (2,1) (1,6/5,3/2)', racah_turn(2, 1, 1, R(6, 5), R(3, 2)), None),
                       ('Racah (0,0) (1,1,13/10)', racah_turn(0, 0, 1, 1, R(13, 10)), None),
                       ('Racah (3/2,1/2) (11/10,1,9/10)', racah_turn(R(3, 2), R(1, 2), R(11, 10), 1, R(9, 10)), None),
                       ('Racah (1,0) (1,2,6/5)', racah_turn(1, 0, 1, 2, R(6, 5)), None),
                       ('Racah (2,-1) (1,2,6/5)', racah_turn(2, -1, 1, 2, R(6, 5)), None)]:
    lim, est = T.varpi_limit(); loc = T.varpi_local()
    msg = '  %-32s q*=%.6f tau=%+d  limit=% .12f (extrap. %.0e)  local=% .12f' % (lab, float(T.qs), T.tau, float(lim), float(est), float(loc))
    if closed is not None:
        msg += '  Hahn closed=% .12f' % float(closed)
    print(msg, flush=True)
    taus.add((lab.split()[0], T.tau))
    check(abs(lim - loc) < 1e-7*max(1, abs(loc)) and (closed is None or abs(closed - loc) < 1e-20),
          '(B) %s: limit eq:general-airy-varpi = local formula (diff %.1e)' % (lab, float(abs(lim - loc))))
check(taus >= {('Hahn', 1), ('Hahn', -1), ('Racah', 1), ('Racah', -1)}, '(B) both turn types covered for Hahn and Racah')


# ---------------- (C) exact 3j / 6j brackets ----------------
A0W = 2


def report(lab, Ns, rows):
    """rows: per N a list of (t, bracket - Ai(t)); varpi fixed per N in vps."""
    E0, E1, Ec = [], [], []
    for N, (vp, out) in zip(Ns, rows):
        s43 = mp.mpf(N)**(-mp.mpf(4)/3)
        E0.append(max(abs(e) for _, e in out))
        E1.append(max(abs(e - s43*vp*mp.airyai(t, 1)) for t, e in out))
        Ec.append(max(abs(e + s43*vp*mp.airyai(t, 1)) for t, e in out))
        print('  %-28s N=%-7s varpi=% .6f pts=%2d  max|br-Ai|=%.3e  max|br-Ai-N^-4/3 varpi Ai\'|=%.3e'
              % (lab, N, float(vp), len(out), float(E0[-1]), float(E1[-1])), flush=True)
    s0, s1, sc = slope(Ns, E0), slope(Ns, E1), slope(Ns, Ec)
    print('  %s exponents: Ai alone %.2f, with varpi Ai\' %.2f, control -varpi %.2f' % (lab, s0, s1, sc), flush=True)
    check(-1.6 < s0 < -1.1 and s1 < -1.75 and sc > -1.6,
          '(C) %s: bracket error N^-4/3 -> N^-2 with the local varpi; control -varpi stays N^-4/3' % lab)


def hahn_rows(M, sig, N, x0, edge):
    """Exact 3j (sympy seed at the lowest rank, exact rank recurrence) at a physical node."""
    Nn = mp.mpf(sp.N(N, 40)); h = 1/Nn
    pp, pm = abs(M - sig), abs(M + sig)
    if edge == 'upper':
        nu = int(round(float((N - 1 - pp)/2 - x0*N/2))); mu = (N - 1 - pp)/2 - nu
    else:
        nu = int(round(float(x0*N/2 + (N - 1 - pm)/2))); mu = nu - (N - 1 - pm)/2
    x = mp.mpf(sp.N(2*mu/N, 40)); kmin = max(abs(M), abs(sig))
    Mf, sf, k0 = (mp.mpf(sp.N(v, 30)) for v in (M, sig, kmin))
    jm, jp = (N - 1 - sig)/2, (N - 1 + sig)/2
    w3 = wigner_3j(kmin, jm, jp, M, mu - R(M)/2, -mu - R(M)/2)
    A = lambda k: mp.sqrt((k*k - Mf*Mf)*(k*k - sf*sf))/(k*mp.sqrt(4*k*k - 1))*mp.sqrt(1 - k*k*h*h)
    B = lambda k: Mf*sf/(k*(k + 1)) if k > 0 else mp.mpf(0)
    Y = {0: mp.sqrt(2*k0 + 1)*(-1)**int(jp - mu - R(M)/2 + 2*kmin)*mp.sqrt(Nn)*mp.mpf(sp.N(w3, 40))}
    Y[1] = mp.sqrt(3)*x/mp.sqrt(1 - h*h) if kmin == 0 else (x - B(k0))*Y[0]/A(k0 + 1)
    tau = 1 if x > 0 else -1; qs_ = mp.sqrt(1 - x*x)
    u = lambda r: tau*x/mp.sqrt(1 - r*r)
    jmax = int(float(qs_*Nn - k0) + 3*float(Nn)**(1/3.))
    for j in range(1, jmax):
        k = k0 + j
        Y[j + 1] = ((x - B(k))*Y[j] - A(k)*Y[j - 1])/A(k + 1)
    eps = (-1)**int(max(sig - M, 0)); out = []
    for j in range(1, jmax + 1):
        k = k0 + j; q = (k + mp.mpf(1)/2)/Nn
        if abs(q - qs_) > 3*Nn**(-mp.mpf(2)/3):
            continue
        if q <= qs_:
            z = -(mp.mpf(3)/2*mp.quad(lambda r: mp.acos(u(r)), [q, qs_]))**(mp.mpf(2)/3)
        else:
            z = (mp.mpf(3)/2*mp.quad(lambda r: mp.acosh(u(r)), [qs_, q]))**(mp.mpf(2)/3)
        t = Nn**(mp.mpf(2)/3)*mp.re(z)
        if abs(t) > A0W:
            continue
        Kt = nu + pp if edge == 'upper' else kmin + j - nu + (pp - pm)/2
        pref = eps*(-1)**int(Kt)*mp.sqrt(2/q)*Nn**(-mp.mpf(1)/3)*abs(z/(1 - x*x - q*q))**mp.mpf('0.25')
        out.append((t, mp.re(Y[j]/mp.sqrt(2*k + 1)/pref) - mp.airyai(t)))
    return hahn_varpi(M, sig, x), out


def racah_rows(m, n, al, be, ga, N, T, spot=False):
    """Exact 6j: sympy seed at f = k_min, exact recurrence eq:6jrec in the natural gauge."""
    Ar, Br, Cr = R(al)*N, R(be)*N, R(ga)*N
    a6, b6, c6 = Ar - R(n + 1, 2), Br - R(m + 1, 2), Cr - R(1, 2)
    Psi = lambda f: (-1)**int(a6 + b6 + c6 + f + m)*2*sp.sqrt(Ar*Br)*wigner_6j(a6, b6, c6, b6 + m, a6 + n, f)
    kmin = max(abs(R(m)), abs(R(n)))
    Am, Bm, x = mp.mpf(sp.N(Ar, 40)), mp.mpf(sp.N(Br, 40)), T.x
    mf, nf, k0 = mp.mpf(sp.N(m, 30)), mp.mpf(sp.N(n, 30)), mp.mpf(sp.N(kmin, 30))
    eta = lambda k: mp.sqrt((1 - k*k/(4*Am*Am))*(1 - k*k/(4*Bm*Bm)))
    A = lambda k: mp.sqrt((k*k - mf*mf)*(k*k - nf*nf))/(k*mp.sqrt(4*k*k - 1))*eta(k)
    B = lambda k: (2*k*(k + 1) + 1 - mf*mf - nf*nf)/(8*Am*Bm) + (mf*nf/(k*(k + 1)) if k > 0 else 0)
    Y = {0: mp.sqrt(2*k0 + 1)*mp.mpf(sp.N(Psi(kmin), 40))}
    Y[1] = mp.sqrt(3)*mp.mpf(sp.N(Psi(1), 40)) if kmin == 0 else (x - B(k0))*Y[0]/A(k0 + 1)
    Nn = mp.mpf(N); jmax = int(float(T.qs*Nn - k0) + 3*float(Nn)**(1/3.))
    for j in range(1, jmax):
        k = k0 + j
        Y[j + 1] = ((x - B(k))*Y[j] - A(k)*Y[j - 1])/A(k + 1)
    if spot:                                     # recurrence values against sympy wigner_6j
        js = [jmax//2, jmax - 2]
        dev = max(abs(Y[j]/mp.sqrt(2*(k0 + j) + 1) - mp.mpf(sp.N(Psi(kmin + j), 40))) for j in js)
        check(dev < 1e-20, '(C) 6j recurrence values equal sympy wigner_6j at N=%s (max diff %.1e)' % (N, float(dev)))
    p = abs(R(n) - R(m)); eps = (-1)**int(max(R(n) - R(m), 0)); out = []
    for j in range(1, jmax + 1):
        f = kmin + j; rho = (k0 + j + mp.mpf(1)/2)/Nn
        if abs(rho - T.qs) > 3*Nn**(-mp.mpf(2)/3):
            continue
        z = T.zeta(rho); t = Nn**(mp.mpf(2)/3)*z
        if abs(t) > A0W:
            continue
        K = Cr - abs(Ar - Br) + (p - 1)/2 if T.tau == 1 else f + R(1, 2) - Ar - Br + Cr + p/2
        pref = eps*(-1)**int(K)*mp.sqrt(2/rho)*Nn**(-mp.mpf(1)/3)*abs(z/T.D(rho))**mp.mpf('0.25')
        out.append((t, Y[j]/mp.sqrt(2*f + 1)/pref - mp.airyai(t)))
    return T.varpi_local(), out


print('\n(C) exact 3j/6j brackets in |t| <= 2', flush=True)
for M, sig, edge, x0, NL in [(2, 1, 'upper', R(3, 5), [500, 1000, 2000, 4000]),
                             (R(3, 2), R(-1, 2), 'lower', R(-1, 2), [R(1001, 2), R(2001, 2), R(4001, 2), R(8001, 2)])]:
    report('3j (%s,%s) %s' % (M, sig, edge), [mp.mpf(sp.N(N, 30)) for N in NL],
           [hahn_rows(M, sig, N, x0, edge) for N in NL])
for m, n, abg, NL in [(2, 1, (1, R(6, 5), R(3, 2)), [40, 80, 160, 320]),
                      (1, 0, (1, 2, R(6, 5)), [40, 80, 160, 320])]:
    T = racah_turn(m, n, *abg)
    report('6j (%s,%s) tau=%+d' % (m, n, T.tau), NL,
           [racah_rows(m, n, *abg, N, T, spot=(N == NL[0])) for N in NL])

print('ALL CHECKS PASSED')
