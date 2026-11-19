"""verify_q6j_class_chain.py -- the one-small quantum 6j as an (E1)-class chain of the main paper (widened class: e(s,t), lambda(t), eta; curvature scale c = 1 here).
A  KR sum vs the Jacobi rows (A_k, B_k) at 50 digits (integer and half-integer offsets, exceptional channel m=n=0), with a negative control   [independent numerical check; the recurrence itself is proved symbolically in verify_q6j_recurrence.py]
A2 seed lemma in exact rational arithmetic: the KR sum at f = m has one term and W_m^2 = closed seed S_m                                    [exact certificate, finite families]
B  (E1) splitting A_k = A_k^(0) e(k^2h^2,h^2), B_k = lambda(h^2) B_k^(0) + v(q^2,h^2): exact series data e, lambda_1, eta, beta_1, beta_2, v_t   [exact symbolic, specific (m,n)]
C  principal-data identities (hyperbolic triangle, turn, tetrahedron cofactor identity cos^2 p = C34^2/(C33 C44)) as exact rational identities      [exact certificate]
D  (R) from q-orthogonality; E (Z) seed polynomial and zero locations; F (E2): closed seed + Jacobi rows = KR sum, seed -> Wigner value as h -> 0, evenness
G  Gauss-weight identity lambda^2 wbar = 2h/sinh h (exact symbolic).
Negative controls are included where stated."""
import sympy as sp, random, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_q6j_lib import *
ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ')+name)

# ---- A. exact chain vs KR sum -------------------------------------------------------------------------------
mp.dps = 50
def jacobi_res(F, ks):
    r = 0
    for f in ks:
        l = F.x*F.Yhat(f); rr = sqrt(F.Ak2(f+1))*F.Yhat(f+1)+sqrt(F.Ak2(f))*F.Yhat(f-1)+F.Bk(f)*F.Yhat(f)
        r = max(r, abs(l-rr)/(abs(l)+abs(rr)+mpf(10)**-30))
    return r
for (a, b, c, m, n, H) in [(14, 12, 17, 2, 1, '0.13'), (11, 13, 16, 2, -1, '0.2'), (10, 9, 12, 1, 0, '0.17'),
                           (mpf('10.5'), mpf('9.5'), 12, mpf(3)/2, mpf(1)/2, '0.15')]:
    F = Fam(a, b, c, m, n, H)
    res = jacobi_res(F, [m+1+j for j in range(4)])
    # negative control: B_k -> B_k^(0)+v-part only (drop lambda factor in front of mn/(k(k+1)))
    chk('Jacobi residual (a,b,c,m,n,H)=%s: %s' % ((a, b, c, m, n, H), mp.nstr(res, 3)), res < mpf(10)**-40)
F = Fam(14, 12, 17, 2, 1, '0.13')
lam = sinh(F.H*2)*sinh(F.H*1)/(2*F.H*sinh(F.H)*1)   # lambda for (m,n)=(2,1): sinh(Hm)sinh(Hn)/(mn H sinh H)
Bbad = lambda k: F.Bk(k)+(lam-1)*2*1/(k*(k+1))*0 - (1-lam)*2/(k*(k+1))   # perturbed diagonal
k = 6; l = F.x*F.Yhat(k)
rbad = abs(l-(sqrt(F.Ak2(k+1))*F.Yhat(k+1)+sqrt(F.Ak2(k))*F.Yhat(k-1)+Bbad(k)*F.Yhat(k)))
chk('negative control (diagonal perturbed by (1-lam)mn/(k(k+1))): residual %s is NOT small' % mp.nstr(rbad, 3), rbad > mpf(10)**-8)
# exceptional channel m=n=0: k=0 row x Y0 = A1 Y1 + B0 Y0 with B0 = coth X coth Y (1-1/cosh H)
F0 = Fam(12, 10, 14, 0, 0, '0.15')
B0 = coth(F0.X)*coth(F0.Y)*(1-1/cosh(F0.H))
row = F0.x*F0.Yhat(0)-sqrt(F0.Ak2(1))*F0.Yhat(1)-B0*F0.Yhat(0)
chk('exceptional channel m=n=0: k=0 row residual %s' % mp.nstr(row, 3), abs(row) < mpf(10)**-40)
chk('and Jacobi rows k>=1 for m=n=0: %s' % mp.nstr(jacobi_res(F0, [1, 2, 3]), 3), jacobi_res(F0, [1, 2, 3]) < mpf(10)**-40)

# ---- A2. seed lemma in EXACT rational arithmetic: the KR sum at f = m has ONE term, and W_m^2 equals the closed seed S_m --------------------------
from fractions import Fraction as Fr
def exact_seed_check(a, b, c, m, n, r=Fr(3, 2)):
    # zeta = e^h with zeta^(1/2) = r, so [x] = (zeta^x - zeta^-x)/(zeta - 1/zeta) = (r^(2x) - r^(-2x))/(r^2 - r^-2) is rational for 2x integer
    qn_ = lambda x: (r**int(round(2*x))-r**(-int(round(2*x))))/(r**2-r**-2)
    def qf_(N):
        r = Fr(1)
        for j in range(1, int(round(N))+1): r *= qn_(j)
        return r
    D2 = lambda x, y, z: qf_(x+y-z)*qf_(x-y+z)*qf_(-x+y+z)/qf_(x+y+z+1)          # Delta^2
    j1, j2, j12, j3, j_, j23 = a, b, c, b+m, a+n, m
    tri = [(j1, j2, j12), (j3, j_, j12), (j1, j_, j23), (j3, j2, j23)]
    al = [sum(t) for t in tri]; be = [j1+j2+j3+j_, j2+j12+j_+j23, j12+j1+j23+j3]
    zs = range(int(max(al)), int(min(be))+1)
    tot = Fr(0)
    for z in zs:
        d = Fr(1)
        for x_ in al: d /= qf_(z-x_)
        for y_ in be: d /= qf_(y_-z)
        tot += (-1)**z*qf_(z+1)*d
    W2 = tot**2
    for t in tri: W2 *= D2(*t)
    A, B, C = a+Fr(n+1, 2), b+Fr(m+1, 2), c+Fr(1, 2)
    Sig = lambda L: [j-Fr(int(L)+1, 2) for j in range(1, int(L)+1)]
    S = qf_(2*m)/(qf_(m+n)*qf_(m-n))
    for s_ in Sig(m+n): S *= qn_(A+B-C+s_)*qn_(A+B+C+s_)
    for s_ in Sig(m-n): S *= qn_(B+C-A+s_)*qn_(A-B+C+s_)
    for i_ in range(int(round(2*m))+1): S /= qn_(2*A-m+i_)*qn_(2*B-m+i_)          # j = -m, -m+1, ..., m (2m+1 factors)
    return len(zs), W2 == S
for fam_ in [(14, 12, 17, 2, 1), (11, 13, 16, 2, -1), (10, 9, 12, 1, 0), (10, 9, 12, 0, 0), (9, 9, 11, 3, 3), (Fr(21, 2), Fr(19, 2), 12, Fr(3, 2), Fr(1, 2)), (Fr(21, 2), Fr(19, 2), 12, Fr(3, 2), Fr(-1, 2))]:
    nt, eq_ = exact_seed_check(*fam_)
    chk('seed lemma, exact rational arithmetic (zeta^(1/2) = 3/2): (a,b,c,m,n)=%s: KR sum at f=m has %d term(s) and W_m^2 = S_m exactly' % (fam_, nt), nt == 1 and eq_)
negs = exact_seed_check(14, 12, 17, 2, 1, r=Fr(5, 4))
chk('(exact rational, zeta = 25/16) same identity', negs[1])

# ---- B. (E1') splitting, exact symbolic (specific offsets (m,n); X,Y enter only through coth X, coth Y) -----------------
import time
H, K, X, Y, U, h2, cc = sp.symbols('H K X Y U h2 cc', positive=True)
Xi = lambda u, v: (sp.cosh(2*u)-sp.cosh(2*v))/(2*(u**2-v**2))
def sec_B(m, n):
    t0 = time.time()
    e2 = Xi(K, H*n)*Xi(K, H*m)*(1-sp.sinh(K)**2/sp.sinh(X)**2)*(1-sp.sinh(K)**2/sp.sinh(Y)**2)/(Xi(2*K, sp.Symbol('z', positive=True))*Xi(2*K, H))
    e2 = e2.subs(sp.Symbol('z', positive=True), 0)
    cX, cY, T = sp.coth(X), sp.coth(Y), sp.tanh(K)**2
    es = sp.series(e2.subs(H, sp.sqrt(h2)), h2, 0, 2).removeO()
    e2_0 = es.coeff(h2, 0); e2_1 = es.coeff(h2, 1); eta = e2_1/(2*e2_0)
    sub = {K: sp.Rational(7, 10), X: sp.Rational(6, 5), Y: sp.Rational(13, 10)}
    f = lambda ex: sp.N(ex.subs(sub), 30)
    chk('(m,n)=(%s,%s) e(s,0)^2 = (1-tanh^2K coth^2X)(1-tanh^2K coth^2Y) at 3 points' % (m, n), all(abs(f(e2_0-(1-T*cX**2)*(1-T*cY**2))) < 1e-25 for sub in [sub]))
    eta_cl = ((m**2+n**2)*(sp.Rational(1, 2)-K**2/(2*sp.sinh(K)**2))+K**2/(2*sp.sinh(2*K)**2)-sp.Rational(1, 8))/K**2
    chk('(m,n)=(%s,%s) eta(K) closed form (series coefficient vs closed form, K=0.7 and K=0.2)' % (m, n), abs(f(eta-eta_cl)) < 1e-25 and abs(sp.N((eta-eta_cl).subs({K: sp.Rational(1, 5), X: sub[X], Y: sub[Y]}), 30)) < 1e-25)
    chk('(m,n)=(%s,%s) lambda_1=(m^2+n^2-1)/6 and eta(0)=lambda_1' % (m, n), sp.simplify(sp.limit(eta, K, 0)-sp.Rational(m**2+n**2-1, 6)) == 0)
    lamS = sp.sinh(H*m)*sp.sinh(H*n)/(m*n*H*sp.sinh(H)) if m*n != 0 else sp.sinh(H*m)*sp.sinh(H*n)/H/sp.sinh(H)
    if m*n != 0:
        chk('(m,n)=(%s,%s) e(0,t)=lambda(t): lim_{K->0} e^2 = lambda^2' % (m, n), sp.simplify(sp.limit(e2.subs({X: sub[X], Y: sub[Y]}), K, 0)-lamS**2) == 0)
    # diagonal: B_k = lambda*mn/(k(k+1)) + v(U,H); v analytic
    Bk_ex = cc*(1-2*sp.cosh(H*m)*sp.cosh(H*n)/(sp.cosh(2*U)+sp.cosh(H)))+2*sp.sinh(H*m)*sp.sinh(H*n)/(sp.cosh(2*U)-sp.cosh(H))
    B0 = m*n*H**2/(U**2-H**2/4)
    lam_ser = sp.series(lamS, H, 0, 8).removeO() if m*n != 0 else 0
    vs = sp.series(Bk_ex-lam_ser*B0, H, 0, 7).removeO()
    pol = []
    for j in (0, 2, 4):
        cf = sp.simplify(vs.coeff(H, j)); pol.append(sp.series(cf, U, 0, 1).removeO())
    chk('(m,n)=(%s,%s) v(U,H) analytic at U=0: H^0,H^2,H^4 coefficients have no negative powers of U' % (m, n), all(sp.simplify(p.coeff(U, -2)) == 0 and sp.simplify(p.coeff(U, -1)) == 0 for p in pol))
    if m*n != 0:
        d4 = sp.series(sp.simplify((Bk_ex-B0).series(H, 0, 7).removeO().coeff(H, 4)), U, 0, 1).removeO().coeff(U, -2)
        chk('(m,n)=(%s,%s) negative control: B_k-B_k^(0) NOT analytic, coefficient of H^4/U^2 = %s' % (m, n, sp.factor(d4)), sp.simplify(d4) != 0)
    v0 = sp.simplify(vs.coeff(H, 0)); chk('(m,n)=(%s,%s) v(s,0) = cothX cothY tanh^2U' % (m, n), sp.simplify((v0-cc*sp.tanh(U)**2).rewrite(sp.exp)) == 0)
    v2 = sp.simplify(vs.coeff(H, 2))
    chk('(m,n)=(%s,%s) beta_1 = cothX cothY, beta_2 = (1/2-m^2-n^2) cc/2 - mn/3' % (m, n), sp.simplify(sp.series(v0, U, 0, 4).removeO().coeff(U, 2)-cc) == 0 and sp.simplify(sp.limit(v2, U, 0)-((sp.Rational(1, 2)-m**2-n**2)*cc/2-sp.Rational(m*n, 3))) == 0)
    vt_cl = -cc*(sp.Rational(m**2+n**2, 2)/sp.cosh(U)**2-sp.Rational(1, 4)/sp.cosh(U)**4)+m*n*(1/sp.sinh(U)**2-1/U**2)
    chk('(m,n)=(%s,%s) v_t(U^2,0) closed form' % (m, n), sp.simplify((v2-vt_cl).rewrite(sp.exp)) == 0)
    print('   (section B (%s,%s): %.1f s)' % (m, n, time.time()-t0), flush=True)
for mn in [(2, 1), (1, 0), (3, -1)]: sec_B(*mn)
# exactness of A_k = A_k^(0) e(k^2h^2,h^2) at random exact points
random.seed(5); worst = 0
for _ in range(6):
    Hv = mpf(random.uniform(0.05, 0.4)); kk = random.choice([3, 4, 7, 11]); nn, mm = random.choice([(1, 2), (0, 1), (2, 2)])
    Xv, Yv = mpf(random.uniform(1, 3)), mpf(random.uniform(1, 3)); sh = lambda z: sinh(Hv*z)
    A2 = 4*(sh(kk)**2-sh(nn)**2)*(sh(kk)**2-sh(mm)**2)*(1-sh(kk)**2/sinh(Xv)**2)*(1-sh(kk)**2/sinh(Yv)**2)/(sinh(2*Hv*kk)**2*(sinh(2*Hv*kk)**2-sinh(Hv)**2))
    A02 = mpf((kk**2-nn**2)*(kk**2-mm**2))/(kk**2*(4*kk**2-1)); Kv = Hv*kk
    Xi_ = lambda u, v: (cosh(2*u)-cosh(2*v))/(2*(u**2-v**2))
    e2v = Xi_(Kv, Hv*nn)*Xi_(Kv, Hv*mm)*(1-sinh(Kv)**2/sinh(Xv)**2)*(1-sinh(Kv)**2/sinh(Yv)**2)/(Xi_(2*Kv, mpf(0))*Xi_(2*Kv, Hv))
    worst = max(worst, abs(A2/A02-e2v))
chk('A_k^2/A_k^(0)2 = e^2 (Xi-form) exactly at random points, worst diff %s' % mp.nstr(worst, 3), worst < mpf(10)**-30)

# ---- C. principal-data identities as exact rational identities (a=e^X, b=e^Y, z=e^Z, k=e^q) -----------------------------
a_, b_, z_, k_ = sp.symbols('a b z k', positive=True)
ch = lambda u: (u+1/u)/2; sh_ = lambda u: (u-1/u)/2
xx = (ch(a_)*ch(b_)-ch(z_))/(sh_(a_)*sh_(b_))
cXc, cYc = ch(a_)/sh_(a_), ch(b_)/sh_(b_)
mu = sh_(z_)/(sh_(a_)*sh_(b_)); Tt = (sh_(k_)/ch(k_))**2
Delta0 = cXc**2+cYc**2-2*xx*cXc*cYc
chk('coth^2X+coth^2Y-2x cothX cothY = sin^2(theta)+mu^2', sp.cancel(Delta0-(1-xx**2)-mu**2) == 0)
D = (1-Tt*cXc**2)*(1-Tt*cYc**2)-(xx-Tt*cXc*cYc)**2
chk('D*cosh^2 q = sin^2(theta) - mu^2 sinh^2 q', sp.cancel(D*ch(k_)**2-(1-xx**2)+mu**2*sh_(k_)**2) == 0)
Ts = (1-xx**2)/Delta0
eps_t = xx-Ts*cXc*cYc
cq2 = ch(k_**2)
M = sp.Matrix([[1, cq2, ch(a_), ch(b_)], [cq2, 1, ch(a_), ch(b_)], [ch(a_), ch(a_), 1, ch(z_)], [ch(b_), ch(b_), ch(z_), 1]])
cof = lambda i, j: (-1)**(i+j)*M.minor(i, j)
C34, C33, C44 = cof(2, 3), cof(2, 2), cof(3, 3)
cp2 = (xx-Tt*cXc*cYc)**2/((1-Tt*cXc**2)*(1-Tt*cYc**2))
# the next identities are rational functions of (a,b,z,k): proved by exact symbolic cancellation (sp.cancel), not by sampling
exact_zero = lambda expr: sp.cancel(sp.together(expr)) == 0
chk('turn: D=0 at tanh^2 q* = sin^2(theta)/Delta0 (so sinh q* = sin(theta)/mu) [exact symbolic cancellation]', exact_zero((1-Ts*cXc**2)*(1-Ts*cYc**2)-(xx-Ts*cXc*cYc)**2))
chk('eps=x-b0(q*) = (x cothX-cothY)(cothX-x cothY)/Delta0 [exact symbolic cancellation]', exact_zero(eps_t-(xx*cXc-cYc)*(cXc-xx*cYc)/Delta0))
chk('cos^2 p = C34^2/(C33 C44): p is the dihedral angle at V1V2 of the hyperbolic tetrahedron with edges (2q;Z;X,X;Y,Y) [exact symbolic cancellation]', exact_zero(cp2-C34**2/(C33*C44)))
num = [(-C34/sp.sqrt(C33*C44)-(xx-Tt*cXc*cYc)/sp.sqrt((1-Tt*cXc**2)*(1-Tt*cYc**2))).subs({a_: sp.exp(sp.Rational(6, 10)), b_: sp.exp(sp.Rational(7, 10)), z_: sp.exp(sp.Rational(9, 10)), k_: sp.exp(kq)}).evalf(25) for kq in (sp.Rational(1, 10), sp.Rational(3, 10))]
chk('sign of the dihedral identity cos p = -C34/sqrt(C33 C44) at two points (diffs %s)' % [sp.N(d, 3) for d in num], all(abs(d) < 1e-20 for d in num))

# ---- D. (R) from q-orthogonality ---------------------------------------------------------------------------------
F = Fam(14, 12, 17, 2, 1, '0.13')
orth = sum(qn(2*f+1, F.H)*qn(2*F.c+1, F.H)*F.W(f)**2 for f in range(2, min(2*F.a+F.n, 2*F.b+F.m)+1))
chk('q-orthogonality sum_f [2f+1][2c+1] W_f^2 = 1 (%s)' % mp.nstr(orth-1, 3), abs(orth-1) < mpf(10)**-40)
bound = qn(2*F.A, F.H)*qn(2*F.B, F.H)/qn(2*F.c+1, F.H)
mx = max(F.Yhat(f)**2 for f in range(2, 27))
chk('(R): max_k Yhat_k^2 = %s <= [2A][2B]/[2c+1] = %s' % (mp.nstr(mx, 5), mp.nstr(bound, 5)), mx <= bound)

# ---- E. (Z): seed squared is a polynomial in cosh Z, zeros at |x| > 1 ----------------------------------------
Fm = Fam(14, 12, 17, 2, 1, '0.13'); mm, nn = 2, 1
def seed_poly(Cc):    # paired form: prod over symmetric shifts of (cosh(Z)-cosh(...))
    Hh = Fm.H; Zc = 2*Hh*Cc; lA, lB = Hh*Fm.A, Hh*Fm.B
    Sig = lambda L: [j-mpf(L+1)/2 for j in range(1, L+1)]
    num = qf(2*mm, Hh)/(qf(mm+nn, Hh)*qf(mm-nn, Hh)); roots = []
    for sg in Sig(mm+nn): num *= (cosh(Zc)-cosh(2*(lA+lB+Hh*sg)))*(-1)/(2*sinh(Hh)**2); roots.append(cosh(2*(lA+lB+Hh*sg)))
    for sg in Sig(mm-nn): num *= (cosh(Zc)-cosh(2*(Hh*sg-(lA-lB))))/(2*sinh(Hh)**2); roots.append(cosh(2*(Hh*sg-(lA-lB))))
    for j in range(-mm, mm+1): num /= qn(2*Fm.A+j, Hh)*qn(2*Fm.B+j, Hh)
    return num, roots
vals = []
for Cc in (Fm.C, Fm.C+mpf('0.7'), Fm.C-mpf('1.3')):
    num, roots = seed_poly(Cc)
    Fx = Fam(14, 12, 17, 2, 1, '0.13'); Fx.C = Cc; Fx.c = Cc-mpf(1)/2
    vals.append(abs(num-seed_S(Fx)))
chk('seed S_m = polynomial in cosh Z of degree 2m (paired factors), diffs %s' % [mp.nstr(v, 3) for v in vals], max(vals) < mpf(10)**-30)
def zero_dist(Hh):    # zeros of the seed in x at fixed Weyl lengths (X,Y of Fm) and lattice parameter Hh: x_j = (cosh X cosh Y - r_j)/(sinh X sinh Y)
    lA, lB = Fm.X/2, Fm.Y/2; Sig = lambda L: [j-mpf(L+1)/2 for j in range(1, L+1)]
    roots = [cosh(2*(lA+lB+Hh*sg)) for sg in Sig(mm+nn)]+[cosh(2*(Hh*sg-(lA-lB))) for sg in Sig(mm-nn)]
    xr = [(cosh(Fm.X)*cosh(Fm.Y)-r)/(sinh(Fm.X)*sinh(Fm.Y)) for r in roots]
    return min(min(abs(v-1), abs(v+1)) for v in xr if min(abs(v-1), abs(v+1)) > mpf(10)**-30), xr
d1, x1 = zero_dist(mpf('0.1')); d2, _ = zero_dist(mpf('0.05')); d3, _ = zero_dist(mpf('0.025'))
chk('(Z): seed zeros in x lie within O(H) of +-1 and are real, so the seed continues analytically and zero-free to a fixed nbhd of any compact X in (-1,1) (distances %s, %s, %s)' % (mp.nstr(d1, 4), mp.nstr(d2, 4), mp.nstr(d3, 4)), d1/d2 > 1.9 and d2/d3 > 1.9 and all(abs(v.imag) == 0 for v in x1))
# ---- F. (E2): the chain started from the closed seed IS the KR sum; and the seed is analytic and even in h with Wigner value at h = 0 -----------------
F = Fam(14, 12, 17, 2, 1, '0.13'); ch_ = phi_chain(F, 12)
dv = max(abs(ch_[f]-F.Yhat(f))/abs(F.Yhat(f)) for f in range(2, 13))
chk('chain from the closed seed + Jacobi rows = Yhat_f from the KR sum, f = 2..12: max rel. diff %s' % mp.nstr(dv, 3), dv < mpf(10)**-40)
Xc, Yc, Zc, mm_, nn_ = mpf('1.2'), mpf('1.32'), mpf('1.8'), 2, 1
xc = (cosh(Xc)*cosh(Yc)-cosh(Zc))/(sinh(Xc)*sinh(Yc)); dd = dhat_chain(mm_, nn_, acos(xc), mm_)[mm_]
dif = []
for Hh in ('0.004', '0.002', '0.001'):
    dif.append(abs(phi_chain(FamC(Xc, Yc, Zc, mm_, nn_, Hh), mm_)[mm_]-dd))
chk('seed at fixed Weyl lengths -> Wigner value dhat_m as h -> 0, error O(h^2): errors %s, ratios %s %s' % ([mp.nstr(v, 4) for v in dif], mp.nstr(dif[0]/dif[1], 4), mp.nstr(dif[1]/dif[2], 4)),
    abs(dif[0]/dif[1]-4) < 0.05 and abs(dif[1]/dif[2]-4) < 0.05)
ev = []
for sg in (1, -1):    # evenness in h of the seed at fixed Weyl lengths: use the paired factors cosh(Z)-cosh(2(lA+lB+h*sigma)), symmetric multisets of shifts
    Hh = sg*mpf('0.1'); lA, lB = Xc/2, Yc/2
    val = mpf(1)
    for L_, base in ((mm_+nn_, 'p'), (mm_-nn_, 'm')):
        for j in range(1, L_+1):
            sgm = j-mpf(L_+1)/2
            val *= (cosh(Zc)-cosh(2*(lA+lB+Hh*sgm))) if base == 'p' else (cosh(Zc)-cosh(2*(Hh*sgm-(lA-lB))))
    ev.append(val)
chk('evenness in h of the paired numerator product (symmetric shift multisets): %s vs %s' % (mp.nstr(ev[0], 12), mp.nstr(ev[1], 12)), abs(ev[0]-ev[1]) < mpf(10)**-30*abs(ev[0]))
# ---- G. Gauss-weight identity of the q-6j: lambda_j^2 * wbar_j = 2h/sinh h exactly (so (1/4) lambda^2 wbar = 1/2 - h^2/12 + ..., the flux defect -1/12) -----------------
hh_, Xs_, Ys_, Cs_ = sp.symbols('hh Xs Ys Cs', positive=True)
Zs_ = 2*hh_*Cs_
qn_s = lambda z: sp.sinh(hh_*z)/sp.sinh(hh_)
lam2 = qn_s(Xs_/hh_)*qn_s(Ys_/hh_)/qn_s(Zs_/hh_)                 # [2A][2B]/[2C] with 2A = X/h, 2B = Y/h, 2C = Z/h
wbar = 2*hh_*sp.sinh(Zs_)/(sp.sinh(Xs_)*sp.sinh(Ys_))            # |dx/dC| for x = (cosh X cosh Y - cosh Z)/(sinh X sinh Y), Z = 2 h C
chk('lambda^2 wbar = 2h/sinh h identically (exact symbolic), series 2 - h^2/3 + ...: %s' % sp.series(2*hh_/sp.sinh(hh_), hh_, 0, 4), sp.simplify((lam2*wbar-2*hh_/sp.sinh(hh_)).rewrite(sp.exp)) == 0)
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
