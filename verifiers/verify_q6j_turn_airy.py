"""verify_q6j_turn_airy.py -- turning action (Schlaefli-type quantization), turn type, and the second-order Airy bracket for the q-6j.
A  turn quantization: int_0^{q*} p = (pi/2)(Z-|X-Y|) [tau=+1], int p - pi q* = -(pi/2)(X+Y-Z) [tau=-1]; d/dZ proof ingredients; turn type rule.
B  Airy bracket Ai + N^{-4/3} varpi Ai' against the exact q-6j for tau=-1 and tau=+1, controls without eta / without varpi.
C  flat limit c->0 of varpi against the main paper's closed Racah formula.
D  uniform second-order Airy law with the flux-matched amplitude (delta_fl = -c^2/12) on exact q-6j symbols, O(N^-3 M/E), controls without delta_fl.  Classes: A exact-symbolic ingredients + numerics (25 digits); B,C,D numerical evidence."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from mpmath import mp, mpf, sinh, cosh, tanh, coth, asinh, acosh, acos, sqrt, quad, pi, airyai, sin, cos
import random, sympy as sp
from p3_q6j_lib import *
from p3_q6j_lib import q, x as xs, c as cs, X as Xs, Y as Ys, m as ms, n as ns, e as e_sym, b0 as b0_sym, cab, eta as eta_sym, vt as vt_sym, fp, fD, fe, fb0, feta, fvt
ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ')+name, flush=True)
mp.dps = 25
# ---- A. turn quantization -------------------------------------------------------------------------------------
random.seed(3); worst = 0; cnt = {1: 0, -1: 0}; ruleerr = 0
def data(X, Y, Z):
    x = (cosh(X)*cosh(Y)-cosh(Z))/(sinh(X)*sinh(Y)); G = sinh(X)**2*sinh(Y)**2*(1-x**2); Ks = asinh(sqrt(G)/sinh(Z))
    return x, Ks
for _ in range(60):
    X = mpf(random.uniform(0.3, 2.2)); Y = mpf(random.uniform(0.3, 2.2)); Z = mpf(random.uniform(abs(X-Y)+0.05, X+Y-0.05))
    x, Ks = data(X, Y, Z); T = lambda K: tanh(K)**2
    p = lambda K: acos(max(-1, min(1, (x-T(K)*coth(X)*coth(Y))/sqrt((1-T(K)*coth(X)**2)*(1-T(K)*coth(Y)**2)))))
    tau = 1 if x-T(Ks)*coth(X)*coth(Y) > 0 else -1
    rule = 1 if (cosh(X)-cosh(Y)*cosh(Z))*(cosh(Y)-cosh(X)*cosh(Z)) < 0 else -1
    ruleerr += (rule != tau)
    I = quad(p, [0, Ks]); pred = (pi/2)*(Z-abs(X-Y)) if tau == 1 else pi*Ks-(pi/2)*(X+Y-Z)
    worst = max(worst, abs(I-pred)); cnt[tau] += 1
chk('turn quantization at 60 random hyperbolic triangles (tau=+1: %d, tau=-1: %d), max error %s' % (cnt[1], cnt[-1], mp.nstr(worst, 3)), worst < mpf(10)**-18 and min(cnt.values()) > 5)
chk('turn type rule tau = -sgn[(coshX-coshY coshZ)(coshY-coshX coshZ)], mismatches %d' % ruleerr, ruleerr == 0)
# negative control: the flat formula (Z-|X-Y|)*pi/2 is NOT the hyperbolic value after a deformation of the wrong sign
X, Y, Z = mpf('1.2'), mpf('1.32'), mpf('1.8'); x, Ks = data(X, Y, Z)
p = lambda K: acos((x-tanh(K)**2*coth(X)*coth(Y))/sqrt((1-tanh(K)**2*coth(X)**2)*(1-tanh(K)**2*coth(Y)**2)))
chk('negative control: int p dK - pi q* differs from pi*(Z-|X-Y|)/2 - pi q* by %s' % mp.nstr(abs(quad(p, [0, Ks])-(pi*Ks-(pi/2)*(Z-abs(X-Y)))), 4), abs(quad(p, [0, Ks])-(pi*Ks-(pi/2)*(Z-abs(X-Y)))) > 0.1)
# proof ingredient: d/dZ of S_* equals pi/2 (tau=+1) -- finite difference at a tau=+1 triangle
def Sstar(X, Y, Z):
    x, Ks = data(X, Y, Z)
    pp = lambda K: acos(max(-1, min(1, (x-tanh(K)**2*coth(X)*coth(Y))/sqrt((1-tanh(K)**2*coth(X)**2)*(1-tanh(K)**2*coth(Y)**2)))))
    return quad(pp, [0, Ks]), pp(Ks*(1-mpf(10)**-12))
X, Y, Z = mpf('1.6'), mpf('1.0'), mpf('1.0'); dz = mpf(10)**-6
d = (Sstar(X, Y, Z+dz)[0]-Sstar(X, Y, Z-dz)[0])/(2*dz)
chk('dS*/dZ = pi/2 at a tau=+1 triangle: %s' % mp.nstr(d, 10), abs(d-pi/2) < mpf(10)**-8)
# integrand identity d_Z p = mu/sqrt(D)
x, Ks = data(X, Y, Z); K = Ks/2
pf = lambda Zv: acos((data(X, Y, Zv)[0]-tanh(K)**2*coth(X)*coth(Y))/sqrt((1-tanh(K)**2*coth(X)**2)*(1-tanh(K)**2*coth(Y)**2)))
mu = sinh(Z)/(sinh(X)*sinh(Y)); D = (1-x**2)/cosh(K)**2-mu**2*sinh(K)**2/cosh(K)**2
chk('d_Z p = mu/sqrt(D): %s vs %s' % (mp.nstr((pf(Z+dz)-pf(Z-dz))/(2*dz), 10), mp.nstr(mu/sqrt(D), 10)), abs((pf(Z+dz)-pf(Z-dz))/(2*dz)-mu/sqrt(D)) < mpf(10)**-8)

# ---- B. Airy bracket ------------------------------------------------------------------------------------------
mp.dps = 40
def admissible_N(alpha, beta, gamma, m, n, lo, hi, want):
    out = [N for N in range(lo, hi) if (N*alpha-Fraction(n+1, 2)).denominator == 1 and (N*beta-Fraction(m+1, 2)).denominator == 1 and (N*gamma-Fraction(1, 2)).denominator == 1]
    return [out[int(i*(len(out)-1)/(want-1))] for i in range(want)] if len(out) >= want else out
def mpq(fr): return mpf(fr.numerator)/fr.denominator
args = (q, xs, cs, Xs, Ys, ms, ns)
u_sym = None
def varpi_data(c, X, Y, x, m, n, qs, tau, use_eta=True, use_vt=True):
    """varpi from the Weyl-hyperbolic local form with exact (symbolic) Taylor jets."""
    u = tau*(xs-b0_sym)/e_sym
    jets = lambda f, N: [sp.diff(f, q, j)/sp.factorial(j) for j in range(N+1)]
    fj = sp.lambdify(args, jets(e_sym, 2)+jets(u, 3)+[cab/q**2+(eta_sym if use_eta else 0), ms*ns/q**2+(vt_sym if use_vt else 0)], 'mpmath')
    e0, e1, e2, u0, u1, u2, u3, tha, b2 = fj(qs, x, c, X, Y, m, n)
    br = (4*u1**2-8*u2+60*u3/u1-36*u2**2/u1**2)/280+(2*e0*e2-e1**2)/(8*e0**2)-tha-tau*b2/e0
    return mpf(2)**(mpf(1)/3)*u1**(-mpf(2)/3)*br
def airy_test(name, c, al, be, ga, m, n, Ns, tau_expected):
    X, Y, Z = 2*c*mpq(al), 2*c*mpq(be), 2*c*mpq(ga)
    xx = (cosh(X)*cosh(Y)-cosh(Z))/(sinh(X)*sinh(Y)); G = sinh(X)**2*sinh(Y)**2*(1-xx**2); qs = asinh(sqrt(G)/sinh(Z))/c
    par = (xx, c, X, Y, m, n); u_f = lambda s_: (xx-fb0(s_, *par))/fe(s_, *par)
    tau = 1 if u_f(qs) > 0 else -1
    chk(name+': turn type tau=%d (expected %d), q*=%s' % (tau, tau_expected, mp.nstr(qs, 8)), tau == tau_expected)
    vp = varpi_data(c, X, Y, xx, m, n, qs, tau); vn = varpi_data(c, X, Y, xx, m, n, qs, tau, use_eta=False)
    print('   varpi = %s (without eta term: %s)' % (mp.nstr(vp, 10), mp.nstr(vn, 10)))
    def zeta(s_):
        if tau == 1: f1 = lambda t: fp(t, *par); f2 = lambda t: acosh(u_f(t))
        else: f1 = lambda t: pi-fp(t, *par); f2 = lambda t: acosh(-u_f(t))
        if s_ <= qs: return -(mpf(3)/2*quad(f1, [s_, qs]))**(mpf(2)/3)
        return (mpf(3)/2*quad(f2, [qs, s_]))**(mpf(2)/3)
    res = {0: [], 1: [], 2: []}
    for N in Ns:
        a = int(N*al-Fraction(n+1, 2)); b = int(N*be-Fraction(m+1, 2)); cc = int(N*ga-Fraction(1, 2))
        F = Fam(a, b, cc, m, n, c/N); f0 = int(qs*N-mpf(1)/2)
        mp.dps = max(60, 2*N); Yc = phi_chain(F, f0+3); mp.dps = 40
        worst = [0, 0, 0]
        for f in (f0-1, f0, f0+1, f0+2):
            qq = (f+mpf(1)/2)/N; Yp = (-1)**int(m-n)*Yc[f]/sqrt(2*f+1)       # Psi = (-1)^(m-n) Phi, class Y_k
            z = zeta(qq); t = N**(mpf(2)/3)*z
            pref = sqrt(2/qq)*N**(-mpf(1)/3)*abs(z/fD(qq, *par))**mpf('0.25')
            kap = f+mpf(1)/2
            Kt = (F.C-abs(F.A-F.B))+mpf(abs(m-n)-1)/2 if tau == 1 else kap-(F.A+F.B-F.C)+mpf(abs(m-n))/2
            assert abs(Kt-round(Kt)) < mpf(10)**-20, Kt          # K_tau is an integer: lattice quantization
            R = Yp/((-1)**int(round(Kt))*pref); A0, A1 = airyai(t), airyai(t, derivative=1); r0 = R-A0
            n43 = N**(mpf(4)/3)
            worst[0] = max(worst[0], abs(r0)*n43)                                  # no varpi: N^-4/3
            worst[1] = max(worst[1], abs(r0-vp*A1/n43)*N**2)                       # with varpi: N^-2
            worst[2] = max(worst[2], abs(r0-vn*A1/n43)*n43)                        # varpi without eta: still N^-4/3
        for i in range(3): res[i].append(worst[i])
        print('   N=%4d  |R-Ai|N^(4/3)=%s  |R-Ai-N^(-4/3)varpi Ai\'|N^2=%s  control(no eta)*N^(4/3)=%s' % (N, mp.nstr(worst[0], 4), mp.nstr(worst[1], 4), mp.nstr(worst[2], 4)))
    k = len(Ns)//2
    chk(name+': bracket error O(N^-2) with varpi (max ratio late/early %s)' % mp.nstr(max(res[1][k:])/max(res[1][:k]), 3), max(res[1][k:]) < 2*max(res[1][:k]))
    chk(name+': control without varpi term: error does not decay faster than N^-4/3 (late/early N^2-scaled growth %s)' % mp.nstr((max(res[0][k:])*Ns[-1]**(mpf(2)/3))/(max(res[0][:k])*Ns[k]**(mpf(2)/3)), 3), max(res[0][k:]) > 0.5*max(res[0][:k]) and max(res[0][k:]) > 10*max(res[1][k:])/Ns[-1]**(mpf(2)/3))
    chk(name+': control without eta in varpi leaves an N^-4/3 residual (%s vs %s)' % (mp.nstr(max(res[2][k:]), 4), mp.nstr(max(res[1][k:])/Ns[-1]**(mpf(2)/3), 4)), max(res[2][k:]) > 5*max(res[1][k:])/Ns[-1]**(mpf(2)/3))
    return vp
vp1 = airy_test('B1 tau=-1, (m,n)=(2,1), c=.6, triangle (1,1.1,1.5)', mpf('0.6'), Fraction(1), Fraction(11, 10), Fraction(3, 2), 2, 1, admissible_N(Fraction(1), Fraction(11, 10), Fraction(3, 2), 2, 1, 30, 420, 5), -1)
vp2 = airy_test('B2 tau=+1, (m,n)=(3,3), c=.5, triangle (2,1,1.3)', mpf('0.5'), Fraction(2), Fraction(1), Fraction(13, 10), 3, 3, admissible_N(Fraction(2), Fraction(1), Fraction(13, 10), 3, 3, 30, 420, 5), 1)

# ---- C. flat limit of varpi (the main paper closed Racah formula) ---------------------------------------------------------------
def racah_varpi(al, be, ga, m, n):
    x = (al**2+be**2-ga**2)/(2*al*be); s = sqrt(1-x**2); rs = 2*al*be*s/ga; ss = rs**2/(4*al*be); eps = x-ss
    P = -50*x**4+15*x**2-9+6*ss*x*(25*x**2-3)+ss**2*(21-149*x**2)+36*ss**3*x+4*ss**4
    return mpf(2)**(mpf(1)/3)*(140*eps**3*((m*m+n*n)*x-2*m*n)+P)/(280*(rs*s)**mpf(4/3 if False else 1)**0*(rs*s)**(mpf(4)/3)*abs(eps)**(mpf(8)/3)) if False else \
        mpf(2)**(mpf(1)/3)*(140*eps**3*((m*m+n*n)*x-2*m*n)+P)/(280*(rs*s)**(mpf(4)/3)*abs(eps)**(mpf(8)/3))
al, be, ga = mpf(1), mpf('1.1'), mpf('1.5'); m, n = 2, 1
vals = []
for cc in (mpf('0.02'), mpf('0.01')):
    X, Y, Z = 2*cc*al, 2*cc*be, 2*cc*ga; xx = (cosh(X)*cosh(Y)-cosh(Z))/(sinh(X)*sinh(Y)); G = sinh(X)**2*sinh(Y)**2*(1-xx**2); qs = asinh(sqrt(G)/sinh(Z))/cc
    par = (xx, cc, X, Y, m, n); tau = 1 if (xx-fb0(qs, *par))/fe(qs, *par) > 0 else -1
    vals.append(varpi_data(cc, X, Y, xx, m, n, qs, tau))
vr = racah_varpi(al, be, ga, m, n)
chk('flat limit: varpi_q(c=.02)=%s, varpi_q(c=.01)=%s -> Racah varpi=%s (error halves/quarters: O(c^2))' % (mp.nstr(vals[0], 8), mp.nstr(vals[1], 8), mp.nstr(vr, 8)),
    abs(vals[1]-vr) < abs(vals[0]-vr)/3 and abs(vals[1]-vr) < 1e-3*abs(vr))

# ---- D. uniform second-order Airy law with the flux-matched amplitude (prop:airy2-amplitude, cor:q6j-consequences(v)) -------------------------
# bracket = (1 + N^-2 (delta_fl - j(q)/2)) Ai + N^-4/3 B_tau(q) Ai',  delta_fl = -c^2/12 (curvature c), error measured against M/E on a window around q*.
# 'plain' = Ai + N^-4/3 varpi Ai' (the local bracket, no amplitude, B = varpi); 'full' = the uniform law; controls drop delta_fl, or the whole N^-2 amplitude.   [numerical evidence]
import p3_airy2_numerics as a2lib
class QT(a2lib._Chain):
    """one-small q-6j chain at curvature c: exact A(S/h;h), B(q/h-1/2;h) of eq:q6j-AkBk and class data e, b0 (Olver transport via a2lib.OlverB)."""
    def __init__(s, m_, n_, c_, al, be, ga):
        s.m, s.n, s.cs = mpf(m_), mpf(n_), mpf(c_); s.al, s.be, s.ga = mpq(al), mpq(be), mpq(ga)
        s.X, s.Y, s.Z = 2*s.cs*s.al, 2*s.cs*s.be, 2*s.cs*s.ga
        s.x = (cosh(s.X)*cosh(s.Y)-cosh(s.Z))/(sinh(s.X)*sinh(s.Y)); G = sinh(s.X)**2*sinh(s.Y)**2*(1-s.x**2)
        s.qs = asinh(sqrt(G)/sinh(s.Z))/s.cs; s.tau = 1 if s.x-s.b0(s.qs) > 0 else -1
    def e(s, q_): t2 = tanh(s.cs*q_)**2; return sqrt((1-t2*coth(s.X)**2)*(1-t2*coth(s.Y)**2))
    def b0(s, q_): return tanh(s.cs*q_)**2*coth(s.X)*coth(s.Y)
    def tu(s, q_): return s.tau*(s.x-s.b0(q_))/s.e(q_)
    def D(s, q_): return s.e(q_)**2-(s.x-s.b0(q_))**2
    def a(s, S, h2):
        H = s.cs*sqrt(h2); shk = sinh(s.cs*S); shn, shm = sinh(H*s.n), sinh(H*s.m)
        num = 4*(shk**2-shn**2)*(shk**2-shm**2)*(1-shk**2/sinh(s.X)**2)*(1-shk**2/sinh(s.Y)**2)
        return sqrt(num/(sinh(2*s.cs*S)**2*(sinh(2*s.cs*S)**2-sinh(H)**2)))
    def bd(s, q_, h2):
        H = s.cs*sqrt(h2); U = s.cs*q_
        return coth(s.X)*coth(s.Y)*(1-2*cosh(H*s.m)*cosh(H*s.n)/(cosh(2*U)+cosh(H)))+2*sinh(H*s.m)*sinh(H*s.n)/(cosh(2*U)-cosh(H))
for name, c_, al, be, ga, m_, n_, tau_exp, Nt in [('D1 tau=-1, (m,n)=(2,1), c=.6, triangle (1,1.1,1.5)', '0.6', Fraction(1), Fraction(11, 10), Fraction(3, 2), 2, 1, -1, (40, 130, 320, 800)),
                                                    ('D2 tau=+1, (m,n)=(3,3), c=.5, triangle (2,1,1.3)', '0.5', Fraction(2), Fraction(1), Fraction(13, 10), 3, 3, 1, (40, 130, 320, 800))]:
    mp.dps = 45; T = QT(m_, n_, c_, al, be, ga); chk(name+': turn type tau=%d' % T.tau, T.tau == tau_exp)
    hard = 2*min(T.al, T.be); half = min(mpf('0.18'), mpf('0.45')*(hard-T.qs)); OB = a2lib.OlverB(T, half, nc=24); varpi0 = OB.B(T.qs); dfl = -T.cs**2/12
    adm = [N for N in range(30, 900) if (N*al-Fraction(n_+1, 2)).denominator == 1 and (N*be-Fraction(m_+1, 2)).denominator == 1 and (N*ga-Fraction(1, 2)).denominator == 1]
    Ns = [min(adm, key=lambda v: abs(v-t)) for t in Nt]; E = {'plain': [], 'full': [], 'no_dfl': [], 'no_amp': []}
    for N in Ns:
        a_i, b_i, c_i = int(N*al-Fraction(n_+1, 2)), int(N*be-Fraction(m_+1, 2)), int(N*ga-Fraction(1, 2))
        dq_hi = min(mpf('0.08'), mpf('0.20')*(hard-T.qs)); dq_lo = min(mpf('0.10'), half*mpf('0.6'))
        with mp.workdps(max(65, 2*N)):
            F = Fam(a_i, b_i, c_i, m_, n_, T.cs/N); Yc = phi_chain(F, int((T.qs+dq_hi)*N)+2)
        f_lo, f_hi = int((T.qs-dq_lo)*N), int((T.qs+dq_hi)*N); w = {k: mpf(0) for k in E}
        for f in range(f_lo, f_hi+1, max(1, (f_hi-f_lo)//7)):
            qq = (f+mpf('0.5'))/N; Psi = (-1)**int(m_-n_)*Yc[f]/sqrt(2*f+1); zq = OB.zeta(qq); t = N**(mpf(2)/3)*zq
            pref = sqrt(2/qq)*N**(-mpf(1)/3)*abs(zq/T.D(qq))**mpf('0.25')
            Kt = (F.C-abs(F.A-F.B))+mpf(abs(m_-n_)-1)/2 if T.tau == 1 else qq*N-(F.A+F.B-F.C)+mpf(abs(m_-n_))/2
            R = Psi/(((-1)**int(round(Kt)))*pref); A0, A1 = airyai(t), airyai(t, derivative=1); Bq = OB.B(qq); jq = a2lib.flux_F2(T, OB, qq); ev = a2lib.env(t); n43 = N**(-mpf(4)/3)
            w['plain'] = max(w['plain'], abs(R-(A0+n43*varpi0*A1))/ev)
            w['full'] = max(w['full'], abs(R-((1+(dfl-jq/2)/N**2)*A0+n43*Bq*A1))/ev)
            w['no_dfl'] = max(w['no_dfl'], abs(R-((1-jq/2/N**2)*A0+n43*Bq*A1))/ev)
            w['no_amp'] = max(w['no_amp'], abs(R-(A0+n43*Bq*A1))/ev)
        for k in E: E[k].append(w[k])
        print('   N=%4d  plain(varpi) %s  full(B_tau, amplitude) %s  no delta_fl %s  no N^-2 amplitude %s' % (N, *(mp.nstr(w[k], 3) for k in ('plain', 'full', 'no_dfl', 'no_amp'))), flush=True)
    sl = {k: a2lib.slope(Ns, E[k]) for k in E}
    print('   slopes in N: plain %.2f  full %.2f  no delta_fl %.2f  no amplitude %.2f' % (sl['plain'], sl['full'], sl['no_dfl'], sl['no_amp']))
    chk(name+': uniform law with delta_fl=-c^2/12 is O(N^-3 M/E): slope %.2f < -2.85' % sl['full'], sl['full'] < -2.85)
    chk(name+': negative control without delta_fl is only O(N^-2): slope %.2f' % sl['no_dfl'], -2.4 < sl['no_dfl'] < -1.7)
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
