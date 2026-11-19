#!/usr/bin/env python3
"""Closed amplitude coefficient alpha of the phase-exact pullback (thm:general-phase-exactness).
# log: verify_phase_exactness_alpha.txt

Main text, lem:pullback-flux: for the complex Darboux branch u of the corrected pullback the
flux  J(s) = pi a(s;h^2) Im( conj(u(s-h/2)) u(s+h/2) )  satisfies the exact identity
    pi Im( conj(u) L_h u )(q) = -[ J(q+h/2) - J(q-h/2) ],
is even in h, and J = 1/2 + h^2 J_2 + O(h^4) with (eq:phase-exactness-J2)
    2 J_2(s,x) = c_ab(Theta_s) cot(Theta_s)/s^2 + cot p (Sigma_2 + p_ss/24) - (log D)_ss/16.
Hence d_q alpha = -d_q J_2 and alpha(q,x) = c(x) + J_2(0,x) - J_2(q,x), c(x) the fixed-rank datum.
Checks:
  (A) exact symbolic: J_2 has no pole at s = 0, and J_2(0,x) is the closed rational function of
      (epsilon_1, beta_1, beta_2) in eq:phase-exactness-J20 (only these three jets enter);
      the Wigner-chain normalization A_2 = -c_ab'/2 (trig identity 1/8-(a^2+b^2)/2 - c' = c cot).
  (B) exact symbolic: with the fixed-rank N^-2 coefficients of Paper II (thm:3jNLO for Hahn,
      thm:6jclosedcorrection for centered Racah; recalled), c(x) = -J_2(0,x) for all offsets
      (and all (alpha,beta) for Racah); so alpha = -J_2 for both chains.
  (C) numerical: amplitude-phase Darboux branch with A_2 = -c'/2: (J(s;h) - 1/2)/h^2 -> J_2(s);
      the Darboux amplitude correction A_2 = -c'/2 against Wigner d at large kappa.
  (D) exact 3j / 6j at fixed ratio: sqrt(2L+1)|Y_L - (1 - h^2 J_2) U^[2]_L| = O(N^-3), while
      without the factor the error is O(N^-2).  Hahn (2,1), (3/2,-1/2); Racah (2,1), (1,0).
"""
import sys
import mpmath as mp
import sympy as sp
from sympy.physics.wigner import wigner_3j, wigner_6j
from p3_second_order_data import hahn, racah, cab, dcab

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


# ---------------- (A) J_2 at q = 0 ----------------
q, x, d, th = sp.symbols('q x d theta', real=True)
a, b, e1, e2, b1, b2, b20, b11 = sp.symbols('a b epsilon1 epsilon2 beta1 beta2 beta20 beta11', real=True)
s = 1 - x**2
e = 1 - e1*q**2 + e2*q**4; b0 = b1*q**2 + b20*q**4; vt = b2 + b11*q**2
cosp = sp.expand(sp.series((x - b0)/e, q, 0, 6).removeO())
p = sp.expand(sp.series(sp.series(sp.acos(x + d), d, 0, 4).removeO().subs(d, cosp - x), q, 0, 6).removeO())
D = sp.expand(sp.series(e**2 - (x - b0)**2, q, 0, 6).removeO())
a0 = e/2; sinp = sp.sqrt(D)/e; g = -sp.log(D)/4
ca = R(1, 8) - (a**2 + b**2)/2
gq, gqq = sp.diff(g, q), sp.diff(g, q, 2); pq, pqq = sp.diff(p, q), sp.diff(p, q, 2)
a0q, a0qq = sp.diff(a0, q), sp.diff(a0, q, 2)
Sig = (a0*(gq**2 + gqq)*cosp - a0*gq*pq*sinp - a0/4*pq**2*cosp - a0/3*pqq*sinp
       + 2*a0/q**2*ca*cosp + a0q*gq*cosp - a0q/2*pq*sinp + a0qq/4*cosp + a*b/q**2 + vt)/(2*a0*sinp)
SS = sp.expand(sp.series(Sig*q**2, q, 0, 4).removeO())
sig_m2, sig_0 = SS.coeff(q, 0), SS.coeff(q, 2)
cab_s = lambda T: ((a**2 + b**2 - R(1, 4))*sp.cos(T) - 2*a*b)/(2*sp.sin(T))
p2 = sp.expand(p).coeff(q, 2)                                   # p = theta + p2 q^2 + O(q^4)
check(sp.simplify(sig_m2 + cab_s(sp.acos(x))) == 0, '(A) Sigma_2 = -c_ab(theta)/q^2 + O(1) (prop:general-second-eikonal-pole)')
F = cab_s(th)*sp.cos(th)/sp.sin(th)
# 2J_2 = F(Theta)/q^2 + cot p (Sigma_2 + p_qq/24) - (log D)_qq/16, Theta = theta + p2 q^2/3 + ...
pole = sp.simplify(F.subs(th, sp.acos(x)) + x/sp.sqrt(s)*sig_m2)
J20 = sp.simplify(R(1, 2)*(sp.diff(F, th).subs(th, sp.acos(x))*p2/3 + x/sp.sqrt(s)*sig_0 - p2/s*sig_m2
                           + x/sp.sqrt(s)*p2/12 - 2*D.coeff(q, 2)/D.coeff(q, 0)/16))
J20_closed = -(e1*((8*(a**2 + b**2) - 5)*x**2 + 4*a*b*x*(x**2 - 5) - 3) + 8*b1*((1 - a**2 - b**2)*x + a*b*(1 + x**2))
               - 12*b2*x*(1 - x**2))/(24*(1 - x**2)**2)
check(pole == 0, '(A) the q^-2 terms of J_2 cancel, so J_2 extends to q = 0')
check(sp.simplify(J20 - J20_closed) == 0 and not (J20.free_symbols & {e2, b20, b11}),
      '(A) J_2(0,x) = eq:phase-exactness-J20 (depends on epsilon_1, beta_1, beta_2 only)')
cp = sp.diff(cab_s(th), th)
check(sp.simplify(R(1, 8) - (a**2 + b**2)/2 - cp - cab_s(th)*sp.cos(th)/sp.sin(th)) == 0,
      '(A) identity 1/8 - (a^2+b^2)/2 - c_ab\' = c_ab cot, so c_a + 2 A_2 = c_ab cot for A_2 = -c_ab\'/2')

# ---------------- (B) fixed-rank datum c(x) = -J_2(0,x) for Hahn and Racah ----------------
M, sg, lam = sp.symbols('M sigma lambda', real=True)
al, be, m, n = sp.symbols('alpha beta m n', positive=True)
rho20 = (e1*(9*x - x**3 - 8*(a**2 + b**2)*x + 4*a*b*x**2 + 12*a*b) + b1*(8*(a**2 + b**2) - 16*a*b*x - 2*x**2 - 6)
         + 12*b2*s)/(12*s**R(3, 2))                                # eq:general-rho20
chi = x*e1 - b1
Q2_dx = (lam + R(1, 4))*chi/3 - sp.sqrt(s)*rho20                   # Q_{2,k} of prop:general-fixed-rank-contact,
Q2_0 = (lam + R(1, 4))*((e1 - x*b1)/2 - x*chi/6)/s + x*rho20/(2*sp.sqrt(s))   # kappa^2 = lambda + 1/4
hd = {a: M, b: sg, e1: R(1, 2), b1: 0, b2: 0}
A2 = 2*lam*(x**3 - x) - 4*x*(M**2 + sg**2) + 2*M*sg*(x**2 + 3) + 4*x  # Paper II thm:3jNLO, J = lambda
B2 = lam*(x**2 - 3) + 2*M*sg*x
check(sp.simplify(A2/(12*(x**2 - 1)) - Q2_dx.subs(hd)) == 0, '(B) Hahn: the d\' coefficients of y_k and u_k agree')
cH = sp.simplify(B2/(12*(x**2 - 1)) - Q2_0.subs(hd))
check(sp.simplify(cH + J20_closed.subs(hd)) == 0, '(B) Hahn, all (M,sigma): c(x) = -J_2(0,x), c = %s' % sp.factor(cH))
rr = (al**-2 + be**-2)/8                                           # Paper II thm:6jclosedcorrection
c0 = (1 - m**2 - n**2)/(8*al*be) + rr*m*n; c1 = 1/(4*al*be) - rr*x
Pf = -c0 + c1*(2 - 2*(m**2 + n**2) + 4*m*n*x - lam*s)/(3*s)
Qf = rr/2*lam + c1*(m*n - lam*x)/(3*s)
rd = {a: m, b: n, e1: (al**-2 + be**-2)/8, b1: 1/(4*al*be), b2: (R(1, 2) - m**2 - n**2)/(8*al*be)}
check(sp.simplify(Pf - Q2_dx.subs(rd)) == 0, '(B) Racah: the d\' coefficients of y_k and u_k agree')
cR = sp.simplify(Qf - Q2_0.subs(rd))
check(sp.simplify(cR + J20_closed.subs(rd)) == 0, '(B) Racah, all (m,n,alpha,beta): c(x) = -J_2(0,x)')


# ---------------- numerical J_2(q,x) and corrected pullback ----------------
def J2num(ch, Q, X):
    T = ch.Th(Q, X); P = ch.p(Q, X)
    pqq_ = mp.diff(lambda r: ch.p(r, X), Q, 2)
    lDqq = mp.diff(lambda r: mp.log(ch.D(r, X)), Q, 2)
    return (cab(ch.a, ch.b, T)*mp.cot(T)/Q**2 + mp.cot(P)*(ch.Sig(Q, X) + pqq_/24) - lDqq/16)/2


def dhat_at(aa, bb, L, T):
    """sqrt(2L+1) d^L_ab(T): Wigner rank recurrence from the exact lowest-rank value."""
    km = max(abs(aa), abs(bb)); X = mp.cos(T)
    seed = sp.re(sp.N(sp.physics.quantum.spin.Rotation.d(km, aa, bb, sp.Float(str(T), 40)).doit(), 40))
    af, bf = mp.mpf(sp.N(aa, 30)), mp.mpf(sp.N(bb, 30)); k0 = mp.mpf(sp.N(km, 30))
    A = lambda k: mp.sqrt((k*k - af*af)*(k*k - bf*bf))/(k*mp.sqrt(4*k*k - 1))
    B = lambda k: af*bf/(k*(k + 1)) if k > 0 else 0
    Yp, Yc = mp.sqrt(2*k0 + 1)*mp.mpf(seed), None
    Yc = (X - B(k0))*Yp/A(k0 + 1)
    if L == km:
        return Yp
    for j in range(1, int(L - km)):
        k = k0 + j
        Yp, Yc = Yc, ((X - B(k))*Yc - A(k)*Yp)/A(k + 1)
    return Yc


# ---------------- (C) flux expansion and the Darboux amplitude ----------------
import sympy.physics.quantum.spin  # noqa: E402
ch = hahn(2, 1); X0, S0 = mp.mpf('0.3'), mp.mpf('0.45')


def flux(hh):
    def u(r):
        T = ch.Th(r, X0); T2 = T + hh*hh/r*ch.Gam2(r, X0)
        amp = (1 - hh*hh/r**2*dcab(2, 1, T2)/2)/(mp.sqrt(mp.pi)*ch.D(r, X0)**mp.mpf('0.25'))
        return amp*mp.expj(r/hh*T2 + hh/r*cab(2, 1, T2))
    aa = ch.e(S0)/2*mp.sqrt((1 - 4*hh*hh/S0**2)*(1 - hh*hh/S0**2)/(1 - hh*hh/(4*S0**2)))
    return mp.pi*aa*mp.im(mp.conj(u(S0 - hh/2))*u(S0 + hh/2))


J2s = J2num(ch, S0, X0)
dev = [abs((flux(hh) - mp.mpf(1)/2)/hh**2 - J2s) for hh in (mp.mpf(1)/100, mp.mpf(1)/200)]
print('  Hahn (2,1) x=0.3 s=0.45: J_2 = %.10f; |(J-1/2)/h^2 - J_2| at h=1/100, 1/200: %.2e, %.2e'
      % (float(J2s), float(dev[0]), float(dev[1])), flush=True)
check(dev[1] < 1e-2 and 3.5 < dev[0]/dev[1] < 4.5, '(C) flux of the amplitude-phase branch: J = 1/2 + h^2 J_2 + O(h^4) (difference quotient error O(h^2))')
T0 = mp.mpf('1.1'); errs = []
for kap in (mp.mpf(80.5), mp.mpf(160.5)):
    dk = dhat_at(2, 1, int(kap - mp.mpf(1)/2), T0)*mp.sqrt(mp.pi*mp.sin(T0)/4)
    ph = kap*T0 + mp.pi/2 - mp.pi/4 + cab(2, 1, T0)/kap
    errs.append((abs(dk - (1 - dcab(2, 1, T0)/(2*kap**2))*mp.cos(ph)), abs(dk - mp.cos(ph))))
print('  Darboux at theta=1.1, kappa=80.5,160.5: with A_2 %.2e, %.2e; without %.2e, %.2e'
      % (float(errs[0][0]), float(errs[1][0]), float(errs[0][1]), float(errs[1][1])), flush=True)
check(errs[0][0]/errs[1][0] > 6 and errs[0][1]/errs[1][1] < 5, '(C) Darboux amplitude 1 - c_ab\'/(2 kappa^2): residual O(kappa^-3) vs O(kappa^-2)')


# ---------------- (D) exact 3j / 6j ----------------
def compare(lab, chn, aa, bb, qt, Ns, exact):
    E2, E3 = [], []
    km = max(abs(R(aa)), abs(R(bb)))
    for N in Ns:
        Nn = mp.mpf(sp.N(N, 30)); h = 1/Nn
        Lc = km + int(qt*Nn - km); e2m = e3m = mp.mpf(0)
        X, Yf = exact(N)
        for L in [Lc + j for j in range(-3, 4)]:
            Q = (mp.mpf(sp.N(L, 30)) + mp.mpf(1)/2)/Nn
            T2 = chn.Th(Q, X) + h*h/Q*chn.Gam2(Q, X)
            U = (mp.sin(T2)**2/chn.D(Q, X))**mp.mpf('0.25')*dhat_at(aa, bb, L, T2)
            Y = Yf(L)
            e2m = max(e2m, abs(Y - U)); e3m = max(e3m, abs(Y - (1 - h*h*J2num(chn, Q, X))*U))
        E2.append(e2m); E3.append(e3m)
        print('  %-30s N=%-6s max|Y-U2|=%.3e  max|Y-(1-h^2 J_2)U2|=%.3e' % (lab, N, float(e2m), float(e3m)), flush=True)
    Nf = [mp.mpf(sp.N(N, 30)) for N in Ns]
    s2, s3 = slope(Nf, E2), slope(Nf, E3)
    print('  %s exponents: U2 %.2f, (1 - h^2 J_2) U2 %.2f' % (lab, s2, s3), flush=True)
    check(-2.4 < s2 < -1.6 and s3 < -2.7, '(D) %s: natural-gauge error O(N^-2) -> O(N^-3) with alpha = -J_2' % lab)


def hahn_exact(M_, S_, xr):
    """eq:PhiDef at the physical node 2 mu/N nearest to xr, natural gauge."""
    M_, S_ = R(M_), R(S_)
    def f(N):
        jm, jp = (N - 1 - S_)/2, (N - 1 + S_)/2
        off = (jm + M_/2) % 1                                        # mu - M/2 - jm integer
        mu = off + int(round(float(xr*N/2 - off)))
        X = mp.mpf(sp.N(2*mu/N, 40))
        def Y(L):
            w3 = wigner_3j(L, jm, jp, M_, mu - M_/2, -mu - M_/2)
            return mp.sqrt(2*L + 1)*mp.mpf(sp.N((-1)**int(2*L + jp - mu - M_/2)*sp.sqrt(N)*w3, 40))
        return X, Y
    return f


def racah_exact(m_, n_, al_, be_, ga_):
    """eq:6jPhi, natural gauge; x fixed by the triangle."""
    def f(N):
        Ar, Br, Cr = R(al_)*N, R(be_)*N, R(ga_)*N
        a6, b6, c6 = Ar - R(n_ + 1, 2), Br - R(m_ + 1, 2), Cr - R(1, 2)
        X = mp.mpf(sp.N((Ar**2 + Br**2 - Cr**2)/(2*Ar*Br), 40))
        def Y(L):
            W = wigner_6j(a6, b6, c6, b6 + m_, a6 + n_, L)
            return mp.sqrt(2*L + 1)*mp.mpf(sp.N((-1)**int(a6 + b6 + c6 + L + m_)*2*sp.sqrt(Ar*Br)*W, 40))
        return X, Y
    return f


print('\n(D) exact 3j/6j at fixed ratio', flush=True)
compare('3j (2,1) x=3/10 q~0.45', hahn(2, 1), 2, 1, mp.mpf('0.45'), [80, 160, 320, 640], hahn_exact(2, 1, R(3, 10)))
compare('3j (3/2,-1/2) x~1/5 q~0.5', hahn(R(3, 2), R(-1, 2)), R(3, 2), R(-1, 2), mp.mpf('0.5'),
        [R(161, 2), R(321, 2), R(641, 2), R(1281, 2)], hahn_exact(R(3, 2), R(-1, 2), R(1, 5)))
compare('6j (2,1) (1,6/5,3/2) rho~0.9', racah(2, 1, 1, mp.mpf(6)/5), 2, 1, mp.mpf('0.9'), [80, 160, 320, 640],
        racah_exact(2, 1, 1, R(6, 5), R(3, 2)))
compare('6j (1,0) (1,2,6/5) rho~0.8', racah(1, 0, 1, 2), 1, 0, mp.mpf('0.8'), [80, 160, 320],
        racah_exact(1, 0, 1, 2, R(6, 5)))
print('ALL CHECKS PASSED')
