#!/usr/bin/env python3
"""Amplitude of the uniform Airy law: a_N = N^-2 [delta_fl(x) - J(q)/2], J = [h^2] of the Airy-form flux (prop:airy2-amplitude, lem:airy2-flux).
# log: verify_airy2_amplitude_flux.txt

For the Airy-form branch u = (g + h^2 p_2)(Ai + i Bi)(h^-2/3 zeta) + h^(4/3) q_1 (Ai' + i Bi')(h^-2/3 zeta)  (g = sqrt(2)|zeta/D|^(1/4), q_1 = g B_tau)
put F(q;h) = a(q;h^2)(A_- B_+ - B_- A_+) (A_pm, B_pm of lem:airy2-shift-parity at the links q pm h/2).  Then F = 1 + h^2 (2 p_2/g + J(q)) + O(h^3),
J independent of p_2, and matching the flux 1/2 + h^2 delta_fl of the physical branch (lem:pullback-flux, thm:general-phase-exactness) gives
a_N(q) = N^-2 [delta_fl - J(q)/2] + O(N^-3).  At the turn
  J(q_*) = B_zeta(0) + zeta'^2 varpi/6 + 13 zeta'^6/10080 - zeta'^2 zeta''/12 - zeta'''/(12 zeta') + zeta''^2/(8 zeta'^2)
           + a_2/a_0 - (log a_0)''/8,           a_2/a_0 = (1/8 - (a^2+b^2)/2)/q_*^2,
and for the Hahn chain (x in (-1,1), x != 0, offsets (M,sigma), P = M sigma, S2 = M^2 + sigma^2)
  J^H(q_*) = [ -2 S2/5 + (5x^2-1) P/(5x) + (125x^4-35x^2+14)/(900x^4) ] / (1-x^2)^2.
Checks: (A0) numerical: the exact flux identity with the C,S conventions of lem:airy2-shift-parity, Im(conj(chi) chi') = 1/pi for chi = Ai + i Bi, the
sign of the carrier (chi ~ exp(-i(xi - pi/4)): phase increasing with q for tau = +1; its conjugate for tau = -1 after the alternating gauge);
(A) exact symbolic, generic jets: F0 = 1, F1 = 0, J(q_*) (the formula above);
(A2) exact symbolic: the Taylor coefficients psi_tau(q_*) (= varpi_tau, checked against eq:general-airy-varpi-local for v = 0) and psi_tau'(q_*) of Olver's
transport function from [h^2] A[g,0], hence B_tau(q_*) and B_tau'(q_*) = psi_tau'(q_*)/3 in the jets of the chain, and the closed form J^H(q_*) by
substituting the Hahn jets (no fit);
(B) numerical: J(q_*) formulas against the numerically evaluated flux (Racah both turn types, Hahn both signs of x);
(C) exact 6j / 3j: uniform error relative to M/E is O(N^-3) with a_N, O(N^-2) without, and O(N^-2) with a wrong constant delta != 0 (so delta_fl = 0 is
needed, and holds for Hahn/Racah).  Classification: (A), (A2) exact symbolic; (A0), (B), (C) numerical evidence.
"""
import sys
import mpmath as mp
import sympy as sp
from p3_airy2_numerics import (RT, HT, OlverB, cheb, flux_F2, exact_6j, exact_3j, env, slope, R)

mp.mp.dps = 30


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


# ---------------- (A0) numerical: flux identity, C/S conventions, carrier ----------------
print('(A0) flux identity and carrier', flush=True)
from p3_airy2_numerics import airy_cs
mp.mp.dps = 30
hh = mp.mpf('0.013')
zf = lambda q: (q - mp.mpf('0.5'))*(1 + mp.mpf('0.3')*q)                    # a monotone Langer coordinate
Pf = lambda q: 1 + mp.mpf('0.2')*q**2; Qf = lambda q: hh*(mp.mpf('0.7') - q)    # real P, Q = h * (smooth)
af = lambda q: mp.mpf('0.5') + mp.mpf('0.1')*q
chi = lambda X: (mp.airyai(X) + 1j*mp.airybi(X), mp.airyai(X, 1) + 1j*mp.airybi(X, 1))


def psic(q):
    c0, c1 = chi(hh**(-mp.mpf(2)/3)*zf(q))
    return Pf(q)*c0 + hh**(mp.mpf(1)/3)*Qf(q)*c1


q0 = mp.mpf('0.31')
Jdirect = mp.pi*af(q0)*mp.im(mp.conj(psic(q0 - hh/2))*psic(q0 + hh/2))
v = {}
for sg in (1, -1):
    qq = q0 + sg*hh/2
    C, S, Cp, Sp_ = airy_cs((zf(qq) - zf(q0))/hh, zf(q0), hh)
    v[sg] = (Pf(qq)*C + Qf(qq)*Cp, Pf(qq)*S + Qf(qq)*Sp_)
Fq = af(q0)*(v[-1][0]*v[1][1] - v[-1][1]*v[1][0])
check(abs(Jdirect - hh**(mp.mpf(1)/3)*Fq) < 1e-20,
      '(A0) pi a Im(conj(psi^c(q-h/2)) psi^c(q+h/2)) = h^(1/3) a (A_- B_+ - B_- A_+) exactly, with C,S as in lem:airy2-shift-parity (diff %.1e)'
      % float(abs(Jdirect - hh**(mp.mpf(1)/3)*Fq)))
c0, c1 = chi(mp.mpf('-0.8'))
check(abs(mp.im(mp.conj(c0)*c1) - 1/mp.pi) < 1e-25, "(A0) Im(conj(chi) chi') = Ai Bi' - Bi Ai' = 1/pi for chi = Ai + i Bi")
yv = mp.mpf(40); xi = 2*yv**mp.mpf('1.5')/3
ch_ = (mp.airyai(-yv) + 1j*mp.airybi(-yv))*mp.exp(1j*(xi - mp.pi/4))
check(abs(mp.im(ch_)) < 1e-3*abs(ch_) and mp.re(ch_) > 0,
      '(A0) chi(-y) ~ pi^(-1/2) y^(-1/4) exp(-i(xi - pi/4)): phase increases with q (xi decreases): the + Darboux branch for tau = +1')
Jpsi = lambda a_, b_: mp.im(mp.conj(a_)*b_)
pa, pb = psic(q0 - hh/2), psic(q0 + hh/2)
check(abs(Jpsi(mp.conj(pa)*(-1)**3, mp.conj(pb)*(-1)**4) - Jpsi(pa, pb)) < 1e-25,
      '(A0) tau = -1: u_k = (-1)^k conj(psi_k) has the same flux as psi: the physical + branch is the conjugate carrier in the alternating gauge')


# ---------------- (A) exact symbolic: Airy-form flux at the turn ----------------
h, s, t = sp.symbols('h s t')
z1, z2, z3, A0, A1, A2c, a2, B0, B1 = sp.symbols('z1 z2 z3 A0 A1 A2c a2 B0 B1', real=True)


def tr(e, n=2):
    p = sp.Poly(sp.expand(e), h)
    return sum(c*h**m for (m,), c in p.terms() if m <= n)


def mul(a, b): return tr(sp.expand(a*b))


def cs_series(c0, c1, K=30):                       # Y'' = h s Y  (zeta = 0), Y(0)=c0, Y'(0)=c1
    c = [sp.Integer(c0), sp.Integer(c1)]
    for k in range(K):
        c.append(sp.Rational(1, (k + 2)*(k + 1))*(h*c[k - 1] if k >= 1 else 0))
    return tr(sum(ci*s**i for i, ci in enumerate(c)))


def ev(poly_s, sv):
    out = 0
    for (m,), c in sp.Poly(poly_s, s).terms():
        pw = sp.Integer(1)
        for _ in range(m): pw = mul(pw, sv)
        out += c*pw
    return tr(sp.expand(out))


Cf, Sf = cs_series(1, 0), cs_series(0, 1)
zeta_t = z1*t + z2*t**2/2 + z3*t**3/6
a0_t = A0 + A1*t + A2c*t**2/2
S0 = sum(zeta_t**j*sp.diff(zeta_t, t)**(2*j + 1)/sp.factorial(2*j + 1) for j in range(3))
gser = sp.series((a0_t*S0)**sp.Rational(-1, 2), t, 0, 3).removeO()
g0, g1, g2 = gser.coeff(t, 0), gser.coeff(t, 1), 2*gser.coeff(t, 2)
res = {}
for sg in (1, -1):
    sv = sg*z1/2 + z2*h/8 + sg*z3*h**2/48
    P = g0 + sg*h*g1/2 + h**2*g2/8
    Q = h*(g0*B0 + sg*h*(g1*B0 + g0*B1)/2)
    Cv, Sv = ev(Cf, sv), ev(Sf, sv); Cpv, Spv = ev(sp.diff(Cf, s), sv), ev(sp.diff(Sf, s), sv)
    res[sg] = (tr(mul(P, Cv) + mul(Q, Cpv)), tr(mul(P, Sv) + mul(Q, Spv)))
Fq = sp.Poly(mul(mul(res[-1][0], res[1][1]) - mul(res[-1][1], res[1][0]), A0 + h**2*a2), h)
Jq_sym = Fq.coeff_monomial(h**2)
Jq_claim = (B0*z1**2/6 + B1/z1 + sp.Rational(13, 10080)*z1**6 - z1**2*z2/12 - z3/(12*z1) + z2**2/(8*z1**2)
            + a2/A0 - (A2c/A0 - (A1/A0)**2)/8)
check(sp.simplify(Fq.coeff_monomial(1) - 1) == 0 and sp.simplify(Fq.coeff_monomial(h)) == 0,
      '(A) F(q_*;h) = 1 + 0*h + ...: normalization V g^2 = 1 and parity (exact, generic jets)')
check(sp.simplify(Jq_sym - Jq_claim) == 0, '(A) J(q_*) = B_zeta + zeta\'^2 B/6 + 13 zeta\'^6/10080 - ... (exact symbolic, generic jets of the class)')


# ---------------- (A2) exact symbolic: psi_tau(q_*), psi_tau'(q_*), and the Hahn closed form ----------------
def A2_symbolic():
    """[h^2] A[g,0](q_* + d) to first order in d (generic jets), psi = -g [h^2]A/zeta'."""
    d, hs_, ds, s_ = sp.symbols('d h dd s')
    zz = sp.symbols('zz1:5', positive=True); Aa = sp.symbols('Aa0:4', positive=True)
    T0, T1, Bb0, Bb1 = sp.symbols('T0 T1 Bb0 Bb1')

    def trn(e_):
        p = sp.Poly(sp.expand(e_), d, hs_)
        return sum(c*d**i*hs_**j for (i, j), c in p.terms() if i <= 1 and j <= 2)

    def mulN(*a):
        r = sp.Integer(1)
        for x_ in a:
            r = trn(r*x_)
        return r
    zeta = lambda t_: sum(zz[k]*t_**(k + 1)/sp.factorial(k + 1) for k in range(4))
    a0f = lambda t_: sum(Aa[k]*t_**k/sp.factorial(k) for k in range(4))
    zt = zeta(ds); zp = sp.diff(zt, ds)
    S0 = sum(zt**j*zp**(2*j + 1)/sp.factorial(2*j + 1) for j in range(5))
    Z0 = sum(zt**j*zp**(2*j)/sp.factorial(2*j) for j in range(5))
    gser = sp.series((a0f(ds)*S0)**sp.Rational(-1, 2), ds, 0, 4).removeO()
    g_t = lambda t_: sum(gser.coeff(ds, k)*t_**k for k in range(4))
    Zs = sp.Symbol('Zs')
    c = [sp.Integer(1), sp.Integer(0)]
    for k in range(14):
        c.append(sp.expand((Zs*c[k] + (hs_*c[k - 1] if k >= 1 else 0))/((k + 2)*(k + 1))))
    Cf = 0
    for i_, ci in enumerate(c):
        for (a_, b_), co in sp.Poly(ci, Zs, hs_).terms():
            if a_ <= 1 and b_ <= 2:
                Cf += co*Zs**a_*hs_**b_*s_**i_
    zq = zeta(d); tot = 0
    for sg in (1, -1):
        sv = trn(sp.expand(sp.cancel(sp.expand(zeta(d + sg*hs_) - zq)/hs_)))
        Cv = 0
        for (m_,), co in sp.Poly(Cf, s_).terms():
            pw = sp.Integer(1)
            for _ in range(m_):
                pw = trn(pw*sv)
            Cv += trn(co.subs(Zs, zq)*pw)
        tot += mulN(trn(a0f(d + sg*hs_/2)), trn(g_t(d + sg*hs_)), trn(Cv))
    pt = sp.Poly(tot, d, hs_)
    h2 = sum(co*d**i for (i, j), co in pt.terms() if j == 2)
    Z0d = trn(sum(sp.Poly(Z0, ds).coeff_monomial(ds**k)*d**k for k in range(2)))
    chk = [sp.simplify(sum(co*d**i for (i, j), co in pt.terms() if j == jj) - cc) for jj, cc in
           ((0, trn(2*mulN(a0f(d), g_t(d), Z0d))), (1, 0))]
    A2tot = sp.expand(h2 + mulN(2, trn(T0 + T1*d), g_t(d), Z0d) + mulN(trn(Bb0 + Bb1*d), g_t(d)))
    zpd = zp.subs(ds, d)
    psi = sp.series(-g_t(d)*A2tot/zpd, d, 0, 2).removeO()
    return dict(zz=zz, Aa=Aa, T=(T0, T1), Bb=(Bb0, Bb1), psi0=sp.simplify(psi.coeff(d, 0)), psi1=sp.simplify(psi.coeff(d, 1)),
                Zs=sp.expand(sp.series(Z0, ds, 0, 5).removeO()), ds=ds, h0h1=chk)


print('\n(A2) symbolic Olver transport function', flush=True)
SY = A2_symbolic()
z1_, z2_, z3_, z4_ = SY['zz']; A0_, A1_, A2_, A3_ = SY['Aa']; T0_, T1_ = SY['T']; Bb0_, Bb1_ = SY['Bb']
check(all(c_ == 0 for c_ in SY['h0h1']),
      '(A2) [h^0] A[g,0] = 2 a_0 g tau u (the eikonal) and [h^1] A[g,0] = 0 at first order in q - q_* (exact, generic jets)')
u1s, u2s, u3s = [SY['Zs'].coeff(SY['ds'], k) for k in (1, 2, 3)]
qs_, M_s, sg_s, tau_s = sp.symbols('q_s M sigma tau')
cab_ = sp.Rational(1, 8) - (M_s**2 + sg_s**2)/2
e0_, e1_, e2_ = 2*A0_, 2*A1_, A2_
varloc = 2**sp.Rational(1, 3)/(280*qs_**2*u1s**sp.Rational(2, 3))*(35*(4*M_s**2 + 4*sg_s**2 - 1) - 280*tau_s*M_s*sg_s/e0_ + qs_**2*(
    4*u1s**2 - 8*u2s + 60*u3s/u1s - 36*u2s**2/u1s**2 + 35*(2*e0_*e2_ - e1_**2)/e0_**2))
sub0 = {T0_: A0_*cab_/qs_**2, Bb0_: tau_s*M_s*sg_s/qs_**2}
vals = {z1_: R(13, 10), z2_: R(7, 10), z3_: R(-3, 10), z4_: R(1, 5), A0_: R(3, 10), A1_: R(-2, 5), A2_: R(1, 3), A3_: R(1, 7),
        qs_: R(4, 5), M_s: 2, sg_s: 1}
okv = True
for tv in (1, -1):
    vals[tau_s] = tv
    okv &= abs(sp.N(SY['psi0'].subs(sub0).subs(vals) - varloc.subs(vals), 30)) < 1e-25
check(okv, '(A2) psi_tau(q_*) from [h^2] A[g,0] equals the local formula eq:general-airy-varpi-local for v = 0 (independent derivation; both turn types)')
# B_tau(q_*) = psi(q_*), B_tau'(q_*) = psi'(q_*)/3 (transport B = int_0^1 psi(zeta y^2) dy); J(q_*) in the jets; then the Hahn jets, x = 2w/(1+w^2) > 0
Jgen = (SY['psi0']*z1_**2/6 + SY['psi1']/(3*z1_) + R(13, 10080)*z1_**6 - z1_**2*z2_/12 - z3_/(12*z1_) + z2_**2/(8*z1_**2)
        + T0_/A0_ - (A2_/A0_ - (A1_/A0_)**2)/8)
w, Pp, S2s, cc = sp.symbols('w P S2 c', positive=True)
xw = 2*w/(1 + w**2); qsw = (1 - w**2)/(1 + w**2)
qq_, EE = sp.symbols('qq EE')
Dop = lambda f: sp.diff(f, qq_) + sp.diff(f, EE)*(-qq_/EE)           # e = sqrt(1-q^2): e' = -q/e


def jets(f, nn):
    out = [f]
    for _ in range(nn):
        out.append(sp.together(Dop(out[-1])))
    return [sp.cancel(o.subs({qq_: qsw, EE: xw})) for o in out]


ej = jets(EE, 3); uj = jets(xw/EE, 4)
a2j = jets((EE/2)*(R(1, 8) - S2s/2)/qq_**2, 1); b2j = jets(Pp/qq_**2, 1)     # a_2 = a_0 (1/8 - S2/2)/q^2; tau b_2 = tau M sigma/q^2, tau = +1
subH = {A0_: ej[0]/2, A1_: ej[1]/2, A2_: ej[2]/2, A3_: ej[3]/2, T0_: a2j[0], T1_: a2j[1], Bb0_: b2j[0], Bb1_: b2j[1]}
U = [uj[k]/sp.factorial(k) for k in range(5)]
sol = {}
for k, zk in ((2, z2_), (3, z3_), (4, z4_)):
    sol[zk] = sp.cancel(sp.solve(sp.expand(SY['Zs'].coeff(SY['ds'], k).subs(sol) - U[k]), zk)[0])
U1 = sp.cancel(U[1])
JH = sp.cancel(sp.together(Jgen.subs(subH).subs(sol).subs(z1_, cc)))
num, den = sp.fraction(JH)
modp = sp.Poly(cc**3 - 2*U1, cc)                                       # z_1^3 = 2 u_1 (eikonal)
nr, dr = sp.rem(sp.Poly(sp.expand(num), cc), modp), sp.rem(sp.Poly(sp.expand(den), cc), modp)
check(nr.degree() <= 0 and dr.degree() <= 0, '(A2) after z_1^3 = 2 u_1 the Hahn J(q_*) contains no cube root: a rational function of w (exact)')
xs_ = sp.Symbol('x')
claimH = (-2*S2s/5 + (5*xs_**2 - 1)*Pp/(5*xs_) + (125*xs_**4 - 35*xs_**2 + 14)/(900*xs_**4))/(1 - xs_**2)**2
check(sp.cancel(nr.as_expr()/dr.as_expr() - claimH.subs(xs_, xw)) == 0,
      '(A2) J^H(q_*) = [-2 S2/5 + (5x^2-1) P/(5x) + (125x^4-35x^2+14)/(900x^4)]/(1-x^2)^2 for x > 0: exact derivation from the general formula and the Hahn jets '
      '(no fit); tau = -1 enters only through tau M sigma and |x|')


# ---------------- (B) J(q_*) vs the numerical flux ----------------
def Jturn_jets(T, OB):
    q0 = T.qs
    z1_, z2_, z3_ = (mp.diff(OB.zeta, q0, k) for k in (1, 2, 3))
    a0f = lambda q: T.e(q)/2
    A0_, A1_, A2_ = a0f(q0), mp.diff(a0f, q0, 1), mp.diff(a0f, q0, 2)
    a2_over = (mp.mpf(1)/8 - (T.m**2 + T.n**2)/2)/q0**2
    B0_, B1_ = OB.B(q0), mp.diff(OB.B, q0, 1)
    return (B0_*z1_**2/6 + B1_/z1_ + mp.mpf(13)/10080*z1_**6 - z1_**2*z2_/12 - z3_/(12*z1_) + z2_**2/(8*z1_**2)
            + a2_over - (A2_/A0_ - (A1_/A0_)**2)/8)


def J_hahn(M_, S_, x):
    x, M_, S_ = mp.mpf(x), mp.mpf(M_), mp.mpf(S_)
    return (-2*(M_**2 + S_**2)/5 + (5*x**2 - 1)*M_*S_/(5*x) + (125*x**4 - 35*x**2 + 14)/(900*x**4))/(1 - x**2)**2


print('\n(B) J(q_*): closed forms vs the numerical flux (Richardson in h)', flush=True)
for m_, n_, abg in [(1, 0, (1, 2, R(6, 5))), (2, 1, (1, R(6, 5), R(3, 2)))]:
    T = RT(m_, n_, abg); OB = OlverB(T, mp.mpf('0.3'))
    jn, jc = flux_F2(T, OB, T.qs), Jturn_jets(T, OB)
    print('  Racah (%s,%s) %s tau=%+d: J(q*) numeric % .10f  jets formula % .10f' % (m_, n_, abg, T.tau, float(jn), float(jc)), flush=True)
    check(abs(jn - jc) < 1e-8, '(B) Racah (%s,%s) tau=%+d: jets formula = numerical flux (diff %.1e)' % (m_, n_, T.tau, float(abs(jn - jc))))
for M_, S_, x in [(2, 1, R(3, 5)), (R(3, 2), R(-1, 2), R(7, 10)), (2, 1, R(-3, 5)), (0, 0, R(3, 5)), (3, -2, R(4, 5))]:
    T = HT(M_, S_, x); OB = OlverB(T, min(mp.mpf('0.15'), (1 - T.qs)*mp.mpf('0.3')), nc=30)
    jn, jc = flux_F2(T, OB, T.qs), J_hahn(M_, S_, x)
    print('  Hahn (%s,%s) x=%s tau=%+d: J(q*) numeric % .10f  closed form % .10f' % (M_, S_, x, T.tau, float(jn), float(jc)), flush=True)
    check(abs(jn - jc) < 1e-6*max(1, abs(jc)), '(B) Hahn (%s,%s) x=%s: closed form J^H(q*) = numerical flux (diff %.1e)' % (M_, S_, x, float(abs(jn - jc))))


# ---------------- (C) exact 6j / 3j ----------------
def uniform_run(T, OB, make_chain, Ns, lo_off, hi_off, nc=16):
    lo, hi = T.qs - lo_off, T.qs + hi_off
    nodes = [(lo + hi)/2 + (hi - lo)/2*mp.cos(mp.pi*(j + mp.mpf(1)/2)/nc) for j in range(nc)]
    Jq, _, tail = cheb([flux_F2(T, OB, q) for q in nodes], lo, hi)
    E = {k: [] for k in ('B', 'a', 'a+', 'a-')}
    for N in Ns:
        Y, kmin, p, sgn = make_chain(N, hi)
        h = mp.mpf(1)/N; pts = []
        for j in range(1, len(Y)):
            q = (mp.mpf(sp.N(kmin, 30)) + j + mp.mpf(1)/2)/N
            if q < lo or q > hi or j % max(1, N//60): continue
            z = OB.zeta(q); tt = mp.mpf(N)**(mp.mpf(2)/3)*z
            alt = (-1)**j if T.tau == -1 else 1
            br0 = Y[j]/mp.sqrt(2*(kmin + j) + 1)/(mp.sqrt(2/q)*mp.mpf(N)**(-mp.mpf(1)/3)*abs(z/T.D(q))**mp.mpf('0.25'))*alt
            pts.append((q, tt, br0))
        s0 = 1 if sum(b*mp.airyai(tt) for q, tt, b in pts) > 0 else -1
        e = {k: mp.mpf(0) for k in E}
        for q, tt, b in pts:
            Ev = env(tt); ai, aip = mp.airyai(tt), mp.airyai(tt, 1)
            base = s0*b - ai - h**(mp.mpf(4)/3)*OB.B(q)*aip
            a = -h**2*Jq(q)/2
            e['B'] = max(e['B'], abs(base)/Ev); e['a'] = max(e['a'], abs(base - a*ai)/Ev)
            e['a+'] = max(e['a+'], abs(base - (a + mp.mpf('0.2')*h**2)*ai)/Ev); e['a-'] = max(e['a-'], abs(base - (a - mp.mpf('0.2')*h**2)*ai)/Ev)
        for k in E: E[k].append(e[k])
        print('    N=%-4d pts=%3d max err/(M/E): B only %.2e | with a_N %.2e | a_N + 0.2/N^2 %.2e | a_N - 0.2/N^2 %.2e'
              % (N, len(pts), float(e['B']), float(e['a']), float(e['a+']), float(e['a-'])), flush=True)
    return {k: slope(Ns, E[k]) for k in E}, tail


def chain6(m_, n_, abg):
    def f(N, hi):
        Y, kmin, (Ar, Br, Cr), dev = exact_6j(m_, n_, abg, N, hi)
        assert dev < 1e-20
        return Y, kmin, 0, 1
    return f


def chain3(M_, S_, xr):
    def f(N, hi):
        X, Y, kmin, dev = exact_3j(M_, S_, xr, N, hi)
        assert dev < 1e-20
        return Y, kmin, 0, 1
    return f


print('\n(C) exact chains: uniform bracket error with a_N = -N^-2 J(q)/2 (delta = 0 for Hahn and Racah)', flush=True)
Ns = [40, 80, 160, 320]
for lab, T, mk, lo_off, hi_off, half in [
        ('Racah (1,0) (1,2,6/5) tau=+1', RT(1, 0, (1, 2, R(6, 5))), chain6(1, 0, (1, 2, R(6, 5))), mp.mpf(1)/4, mp.mpf(1)/5, mp.mpf('0.3')),
        ('Racah (2,1) (1,6/5,3/2) tau=-1', RT(2, 1, (1, R(6, 5), R(3, 2))), chain6(2, 1, (1, R(6, 5), R(3, 2))), mp.mpf(1)/4, mp.mpf(1)/5, mp.mpf('0.3')),
        ('Hahn (2,1) x~3/5 tau=+1', HT(2, 1, R(3, 5)), chain3(2, 1, R(3, 5)), mp.mpf('0.12'), mp.mpf('0.1'), mp.mpf('0.14')),
        ('Hahn (1,0) x~-1/2 tau=-1', HT(1, 0, R(-1, 2)), chain3(1, 0, R(-1, 2)), mp.mpf('0.1'), mp.mpf('0.08'), mp.mpf('0.1'))]:
    OB = OlverB(T, half)
    sl, tail = uniform_run(T, OB, mk, Ns, lo_off, hi_off)
    print('  %s: exponents  B only %.2f | with a_N %.2f | with a_N +- 0.2/N^2 %.2f, %.2f' % (lab, sl['B'], sl['a'], sl['a+'], sl['a-']), flush=True)
    check(sl['a'] < -2.7 and sl['B'] > -2.3 and sl['a+'] > -2.3 and sl['a-'] > -2.3,
          '(C) %s: relative-envelope error N^-3 with a_N; N^-2 without it or with a wrong constant' % lab)
print('ALL CHECKS PASSED')
