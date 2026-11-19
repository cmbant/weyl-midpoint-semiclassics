"""verify_q6j_second_order.py -- second-order pullback and phase-exact pullback of the one-small q-6j (widened class (E1')).
Parts: A exact symbolic fixed-rank contact identity with the lambda_1 shift; B class data (rho_2(0,x), J_2(0,x), fixed-rank datum c(x))
vs the exact chain; C error scaling vs exact q-6j: first order N^-1, second order N^-2, phase-exact N^-3 (natural gauge), with negative controls.
D exact-symbolic proof of the fixed-rank datum c(x) = -J_2(0,x) - c^2/12 (symbolic m, n).
Classes: A, D exact-symbolic; B,C numerical evidence (40-digit exact q-6j by Jacobi recurrence from the closed seed; recurrence = KR sum, see verify_q6j_class_chain.py)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from p3_q6j_lib import *
import sympy as sp, time
ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ')+name, flush=True)

# ---- A. contact identity (exact) ------------------------------------------------------------------------------
res = [contact_residual(a, b, k) for (a, b, k) in [(2, 1, 3), (2, 1, 4), (1, 0, 2), (3, -1, 5), (2, 2, 4), (sp.Rational(3, 2), sp.Rational(1, 2), sp.Rational(7, 2))]]
chk('contact identity with lambda_1 (rho_20 -> rho_20 + lam1 x/sqrt(1-x^2), source + lam1 x d_k): residuals %s' % res, all(r == 0 for r in res))
# negative control: drop the lambda_1 x shift in rho_20 only
r_bad = contact_residual(2, 1, 3, shift=False)
chk('negative control: without the lam1*x term in rho_20 the contact residual is %s != 0' % sp.simplify(r_bad), sp.simplify(r_bad) != 0)

# ---- helpers: families with exact fixed Weyl lengths -------------------------------------------------------------
def admissible_N(alpha, beta, gamma, m, n, lo, hi, want):
    """N with A=N*alpha in Z+(n+1)/2, B in Z+(m+1)/2, C in Z+1/2 (alpha.. Fractions)"""
    out = []
    for N in range(lo, hi):
        A, B, C = N*alpha, N*beta, N*gamma
        if (A-Fraction(n+1, 2)).denominator == 1 and (B-Fraction(m+1, 2)).denominator == 1 and (C-Fraction(1, 2)).denominator == 1:
            out.append(N)
    return [out[int(i*(len(out)-1)/(want-1))] for i in range(want)] if len(out) >= want else out
def setup(c, al, be, ga, m, n):
    X, Y, Z = 2*c*mpf(al.numerator)/al.denominator, 2*c*mpf(be.numerator)/be.denominator, 2*c*mpf(ga.numerator)/ga.denominator
    xx = (cosh(X)*cosh(Y)-cosh(Z))/(sinh(X)*sinh(Y)); G = sinh(X)**2*sinh(Y)**2*(1-xx**2)
    qs = asinh(sqrt(G)/sinh(Z))/c
    return X, Y, Z, xx, qs, Cls(c, X, Y, m, n, xx, ng=34)
def exact_chain(c, al, be, ga, m, n, N, f):
    A, B, C = N*al, N*be, N*ga
    a = int(A-Fraction(n+1, 2)); b = int(B-Fraction(m+1, 2)); cc = int(C-Fraction(1, 2))
    mp.dps = max(50, 2*N); F = Fam(a, b, cc, m, n, c/N); Yc = phi_chain(F, f)[f]; mp.dps = 40
    return Yc
def errors(c, al, be, ga, m, n, fracs, Ns, alpha_mode):
    X, Y, Z, xx, qs, C = setup(c, al, be, ga, m, n)
    out = []
    for N in Ns:
        e1 = e2 = e3 = e2n = e3n = 0
        for fr in fracs:
            f = int(fr*qs*N); qq = (f+mpf(1)/2)/N; h = mpf(1)/N
            Yc = exact_chain(c, al, be, ga, m, n, N, f)
            Th = C.Theta(qq); Th2 = C.Theta2(qq, h); Th2n = C.Theta2(qq, h, False)
            U1 = C.W(qq, Th)*dhat_chain(m, n, Th, f)[f]; U2 = C.W(qq, Th2)*dhat_chain(m, n, Th2, f)[f]
            U2n = C.W(qq, Th2n)*dhat_chain(m, n, Th2n, f)[f]
            alpha = -C.J2(qq)-c**2/12
            alpha_bad1 = -C.J2(qq)                      # control: missing -c^2/12 normalisation
            e1 = max(e1, abs(Yc-U1)); e2 = max(e2, abs(Yc-U2)); e2n = max(e2n, abs(Yc-U2n))
            e3 = max(e3, abs(Yc-(1+h**2*alpha)*U2)); e3n = max(e3n, abs(Yc-(1+h**2*alpha_bad1)*U2))
        out.append((N, e1, e2, e3, e2n, e3n))
    return out
def trend(rows, col, p):   # (max of e*N^p over the last half)/(max over the first half)
    v = [r[col]*mpf(r[0])**p for r in rows]; k = len(v)//2
    return max(v[k:])/max(v[:k])

# ---- C. scaling against exact q-6j --------------------------------------------------------------------------------
configs = [('S1 (m,n)=(2,1) c=.6 triangle (1,1.1,1.5)', mpf('0.6'), Fraction(1), Fraction(11, 10), Fraction(3, 2), 2, 1, Fraction(5, 1)),
           ('S2 (m,n)=(1,0) c=.7 triangle (1.5,.8,1.1)', mpf('0.7'), Fraction(3, 2), Fraction(4, 5), Fraction(11, 10), 1, 0, Fraction(5, 1)),
           ('S3 (m,n)=(3,-1) c=.5 triangle (1.2,1,1.7)', mpf('0.5'), Fraction(6, 5), Fraction(1), Fraction(17, 10), 3, -1, Fraction(5, 1))]
for name, c, al, be, ga, m, n, _ in configs:
    Ns = admissible_N(al, be, ga, m, n, 20, 260, 7)
    rows = errors(c, al, be, ga, m, n, [mpf('0.35'), mpf('0.5'), mpf('0.65')], Ns, None)
    print(name, 'N =', Ns, flush=True)
    for r in rows: print('   N=%3d  e1*N=%s  e2*N^2=%s  e3*N^3=%s | control e2noEta*N=%s e3noShift*N^3=%s' % (r[0], mp.nstr(r[1]*r[0], 4), mp.nstr(r[2]*r[0]**2, 4), mp.nstr(r[3]*r[0]**3, 4), mp.nstr(r[4]*r[0], 4), mp.nstr(r[5]*r[0]**3, 4)))
    chk(name+': first order O(N^-1) (growth of e1*N %s)' % mp.nstr(trend(rows, 1, 1), 3), trend(rows, 1, 1) < 2)
    chk(name+': second order O(N^-2) (growth of e2*N^2 %s)' % mp.nstr(trend(rows, 2, 2), 3), trend(rows, 2, 2) < 2)
    chk(name+': phase-exact O(N^-3) (growth of e3*N^3 %s)' % mp.nstr(trend(rows, 3, 3), 3), trend(rows, 3, 3) < 2)
    Rr = [r[4]/r[2] for r in rows]; kk = len(Rr)//2; gr = (sum(Rr[kk:])/len(Rr[kk:]))/(sum(Rr[:kk])/kk)
    chk(name+': control without eta in a_2: error/second-order error grows ~N (late/early mean ratio %s, last %s)' % (mp.nstr(gr, 3), mp.nstr(Rr[-1], 3)), gr > 1.2 and Rr[-1] > 2)
    chk(name+': control without -c^2/12: phase-exact fails (e3*N^3 grows by %s)' % mp.nstr(trend(rows, 5, 3), 3), trend(rows, 5, 3) > 2.5)

# ---- B. class data at fixed rank (analytic formulas vs the exact chain) -------------------------------------------
mp.dps = 60
c = mpf('0.6'); al, be, ga = Fraction(1), Fraction(11, 10), Fraction(3, 2); m, n = 2, 1
X, Y, Z, xx, qs, C = setup(c, al, be, ga, m, n); a, b = m, n
cX, cY = coth(X), coth(Y)
e1 = c**2*(cX**2+cY**2)/2; b1 = c**2*cX*cY; b2 = c**2*((mpf(1)/2-m*m-n*n)*cX*cY/2-mpf(m*n)/3); lam1 = c**2*(m*m+n*n-1)/6
rho20 = lambda x_: (e1*(9*x_-x_**3-8*(a*a+b*b)*x_+4*a*b*x_**2+12*a*b)+b1*(8*(a*a+b*b)-16*a*b*x_-2*x_**2-6)+12*(b2+lam1*x_)*(1-x_*x_))/(12*(1-x_*x_)**mpf('1.5'))
J20 = -(e1*((8*(a*a+b*b)-5)*xx**2+4*a*b*xx*(xx**2-5)-3)+8*b1*((1-a*a-b*b)*xx+a*b*(1+xx**2))-12*b2*xx*(1-xx*xx))/(24*(1-xx*xx)**2)+lam1/(2*(1-xx*xx))
q0 = mpf('1e-6')
chk('rho_2(0,x) = main-paper formula + lam1 x/sqrt(1-x^2): %s vs %s' % (mp.nstr(C.rho2(q0), 12), mp.nstr(rho20(xx), 12)), abs(C.rho2(q0)-rho20(xx)) < mpf(10)**-9)
chk('negative control: main-paper rho_20 without the lam1 term differs by %s' % mp.nstr(lam1*xx/sqrt(1-xx**2), 4), abs(C.rho2(q0)-(rho20(xx)-lam1*xx/sqrt(1-xx**2))) > mpf('0.01'))
chk('J_2(0,x) = main-paper formula + lam1/(2(1-x^2)): %s vs %s' % (mp.nstr(C.J2(q0), 12), mp.nstr(J20, 12)), abs(C.J2(q0)-J20) < mpf(10)**-8)
def Yk(k, t):
    Hh = c*sqrt(t); return phi_chain(FamC(X, Y, Z, m, n, Hh), k)[k]
def Uk(k, t):
    h = sqrt(t); qq = (k+mpf(1)/2)*h; Th = C.Theta2(qq, h); return C.W(qq, Th)*dhat_chain(m, n, Th, k)[k]
dh = dhat_chain(m, n, acos(xx), m+2)
def tcoef(fun, k):
    ts = [mpf(10)**-6*j for j in (1, 2, 4)]; vals = [(fun(k, t)-dh[k])/t for t in ts]
    r1 = [2*vals[0]-vals[1], 2*vals[1]-vals[2]]; return (4*r1[0]-r1[1])/3
cs = [(tcoef(Yk, k)-tcoef(Uk, k))/dh[k] for k in (m, m+1, m+2)]
chk('fixed-rank datum c(x) is k-independent: %s' % [mp.nstr(v, 10) for v in cs], max(cs)-min(cs) < mpf(10)**-8)
chk('c(x) = -J_2(0,x) - c^2/12: %s vs %s' % (mp.nstr(cs[1], 10), mp.nstr(-J20-c**2/12, 10)), abs(cs[1]-(-J20-c**2/12)) < mpf(10)**-8)

# ---- D. EXACT-SYMBOLIC proof of c(x) = -J_2(0,x) - c^2/12 from the closed seed at k = m (c = 1; general c by scaling h -> c h) -------------------
# c(x) := [t] log (Yhat_m / U_m), t = h^2 at fixed Weyl lengths.  Yhat_m^2 = [2m+1][2A][2B] S_m is a finite product of q-numbers, so [t] log Yhat_m is the explicit sum Q/2 below
# (log sinh(hz)/sinh h = log z + (z^2-1) h^2/6 + O(h^4), z = integer or half-integer shift of A, B, C; hyperbolic Weyl lengths enter through 1/sinh^2 of
# the four triangle sides L1..L4 = 2l(A+B-C), ...);  U_m = W d_m(Theta^[2]) has [t] log U_m = uhat from rho_{2,0} (Lemma pullback) and d_m = closed Wigner seed.
# In exponential variables u = e^{lA}, v = e^{lB}, w = e^{lC} the difference c + J_2(0,x) + 1/12 is a rational function of (u, v, w) with symbolic m, n; its numerator is 0.
print('D. exact symbolic fixed-rank datum (symbolic m, n)', flush=True)
t0 = time.time()
u, v, w = sp.symbols('u v w', positive=True); ms_, ns_ = sp.symbols('m n')
chh = lambda z: (z+1/z)/2; shh = lambda z: (z-1/z)/2
XX, YY, ZZ = u**2, v**2, w**2
xq = (chh(XX)*chh(YY)-chh(ZZ))/(shh(XX)*shh(YY)); sq = 1-xq**2
cXq, cYq = chh(XX)/shh(XX), chh(YY)/shh(YY)
E1 = (cXq**2+cYq**2)/2; B1 = cXq*cYq; LAM1 = (ms_**2+ns_**2-1)/6; B2 = (sp.Rational(1, 2)-ms_**2-ns_**2)*cXq*cYq/2-ms_*ns_/3
CHI = xq*E1-B1
R3 = E1*(9*xq-xq**3-8*(ms_**2+ns_**2)*xq+4*ms_*ns_*xq**2+12*ms_*ns_)+B1*(8*(ms_**2+ns_**2)-16*ms_*ns_*xq-2*xq**2-6)+12*(B2+LAM1*xq)*sq
KAP = ms_+sp.Rational(1, 2); dlogd = -(ms_*xq-ns_)/sq
UHAT = (KAP**2*CHI/3-R3/(12*sq))*dlogd+KAP**2*((E1-xq*B1)/2-xq*CHI/6)/sq+xq*R3/(24*sq**2)          # [t] log U_m
J20q = -(E1*((8*(ms_**2+ns_**2)-5)*xq**2+4*ms_*ns_*xq*(xq**2-5)-3)+8*B1*((1-ms_**2-ns_**2)*xq+ms_*ns_*(1+xq**2))-12*B2*xq*sq)/(24*sq**2)+LAM1/(2*sq)
n1, n2, L3 = ms_+ns_, ms_-ns_, 2*ms_+1
tri = lambda N_: N_*(N_**2-1)/12; fix = lambda N_: N_*(N_+1)*(2*N_+1)/36-N_/6              # sum_{j<=N} (j^2-1)/6
L1, L2, L3x, L4 = u*v/w, v*w/u, u*w/v, u*v*w
Qs = fix(L3)-fix(n1)-fix(n2)-sp.Rational(1, 2)*(tri(n1)*(1/shh(L1)**2+1/shh(L4)**2)+tri(n2)*(1/shh(L2)**2+1/shh(L3x)**2)-tri(L3)*(1/shh(XX)**2+1/shh(YY)**2))
cq = Qs/2-UHAT
num = sp.expand(sp.numer(sp.together(cq+J20q+sp.Rational(1, 12))))
chk('c(x) + J_2(0,x) + 1/12 = 0 identically in (u,v,w) with symbolic (m,n): the numerator, a polynomial in (u,v,w,m,n), expands to 0 (%.0f s)' % (time.time()-t0), num == 0)
chk('negative control: c + J_2(0,x) is NOT identically 0, it equals the nonzero constant (exact value at a rational point (u,v,w,m,n)) %s' %
    sp.nsimplify((cq+J20q).subs({u: sp.Rational(13, 10), v: sp.Rational(17, 10), w: sp.Rational(23, 10), ms_: 2, ns_: 1})), (cq+J20q).subs({u: sp.Rational(13, 10), v: sp.Rational(17, 10), w: sp.Rational(23, 10), ms_: 2, ns_: 1}) != 0)
# link to the numerics of part B: the symbolic c at the part-B triangle equals the chain's measured fixed-rank datum divided by c^2
cnum = sp.N(cq.subs({u: sp.exp(sp.Float(str(X), 40)/2), v: sp.exp(sp.Float(str(Y), 40)/2), w: sp.exp(sp.Float(str(Z), 40)/2), ms_: 2, ns_: 1}), 15)
chk('symbolic c (c=1 units) at the part-B triangle = measured chain datum / c^2: %s vs %s' % (cnum, mp.nstr(cs[1]/c**2, 10)), abs(mpf(str(cnum))-cs[1]/c**2) < mpf(10)**-6)
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
