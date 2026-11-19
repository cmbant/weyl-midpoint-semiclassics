"""verify_q6j_recurrence.py -- exact-symbolic proof of the three-term recurrence of the one-small quantum 6j symbol (Proposition q6j-class(a)).
The Kirillov-Reshetikhin sum S_f = sum_z (-1)^z [z+1]! / (prod_i [z-alpha_i]! prod_i [beta_i-z]!) (alpha_3 = 2a+n+f, alpha_4 = 2b+m+f,
beta_2 = a+b+c+n+f, beta_3 = a+b+c+m+f; alpha_1, alpha_2 = alpha_1+m+n, beta_1 fixed) satisfies, for every integer f >= 1,
    (x - B_f) S_f = rho_+(f) S_{f+1} + rho_-(f) S_{f-1}                                                              (R3)
with explicit rho_+, rho_-.  Proof by q-Zeilberger creative telescoping.  With r(z) the reference term of the paper (appendix), the termwise identity
    rho_- t(z,f-1) + rho_0 t(z,f) + rho_+ t(z,f+1) = G(z+1) - G(z),   G(z) = r(z) [z-al1][z-al2][z-al3+1][z-al4+1] Pol(q^z),  rho_0 = -(x-B_f),
reduces to ONE identity of rational functions in (q, A=q^a, B=q^b, C=q^c, F=q^f, M=q^m, N=q^n, w=q^z)  [part A, exact symbolic, all symbols free],
with Pol(w) = pm2 w^-2 + p0 + p2 w^2 (coefficients rational in the other symbols, solved by linear algebra over Q(q,A,B,C,F,M,N) and embedded below).
Parts:
 A  the Zeilberger identity E(w) = 0 as a rational function in all eight symbols; negative controls (wrong diagonal, wrong rho_-, perturbed p0) give E != 0.
 B  gauge: rho_+^2 = A_{f+1}^2 [2f+3] P(f+1)^2 / ([2f+1] P(f)^2) and rho_-^2 = A_f^2 [2f-1] P(f-1)^2/([2f+1] P(f)^2) as exact rational identities (P(f)^2 = Delta(a,a+n,f)^2 Delta(b+m,b,f)^2;
    its f-shift ratios are checked against the q-factorials in exact rational arithmetic), so that (R3) is the Jacobi row of the paper in the gauge sqrt[2f+1] P(f) (-1)^f S_f.
 C  the row f = 0 of the exceptional channel m = n = 0 (where the certificate has the pole s_f = 0): (x - B_0) S_0 = rho_+(0) S_1 exactly, symbolic in (q, A, B, C).
 D  exact rational arithmetic (zeta^(1/2) rational) on integer and half-integer families: (R3) at every f from m-1 to f_max+2, S_f = 0 outside m <= f <= f_max = min(2a+n, 2b+m), signs of rho_+-, so that the rows
    at f = m (A_m = 0) and f = f_max (A_{f_max+1} = 0) are exactly the boundary rows of the Jacobi matrix.
Class: exact-symbolic certificate (A, B, C), exact rational arithmetic (D)."""
import sympy as sp, time, sys
from sympy import QQ
from sympy.polys.fields import field
from fractions import Fraction as Fr
ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ')+name, flush=True)

# ------------------------------------------------------------------------------------------------------------------------------
# certificate coefficients Pol(w) = pm2 w^-2 + p0 + p2 w^2 (numerators / denominators in the symbols q,A,B,C,F,M,N)
P_pm2_NUM = ' '.join([
    '2*A**2*B**2*F**6*M*N*q**3*(A**4*B**4*C**2*F**4*M**2*N**2*q**8 - A**4*B**4*C**2*M**2*N**2*q**4 -',
    'A**4*C**2*F**6*N**2*q**6 + A**4*C**2*F**2*N**2*q**2 + A**2*B**2*F**6*M**2*q**4 + A**2*B**2*F**6*N**2*q**4',
    '- A**2*B**2*F**4*M**2*N**2*q**2 - A**2*B**2*F**4*q**2 - A**2*B**2*F**2*M**2*q**4 -',
    'A**2*B**2*F**2*N**2*q**4 + A**2*B**2*M**2*N**2*q**2 + A**2*B**2*q**2 - B**4*C**2*F**6*M**2*q**6 +',
    'B**4*C**2*F**2*M**2*q**2 + C**2*F**4*q**4 - C**2)'
])
P_pm2_DEN = ' '.join([
    '(F - 1)*(F + 1)*(F**2 + 1)*(F*q - 1)*(F*q + 1)*(F**2*q - 1)*(F**2*q + 1)*(F**2*q**2 + 1)*(A**2*N*q -',
    '1)*(A**2*N*q + 1)*(B**2*M*q - 1)*(B**2*M*q + 1)'
])
P_p0_NUM = ' '.join([
    '-2*F**2*q*(A**4*B**4*C**2*F**6*M**4*N**2*q**8 + A**4*B**4*C**2*F**6*M**2*N**4*q**8 -',
    'A**4*B**4*C**2*F**2*M**4*N**2*q**4 - A**4*B**4*C**2*F**2*M**2*N**4*q**4 - A**4*C**2*F**6*M**2*N**4*q**4 -',
    'A**4*C**2*F**6*N**2*q**4 - A**4*C**2*F**4*M**2*N**2*q**6 + A**4*C**2*F**4*M**2*N**2*q**2 -',
    'A**4*C**2*F**4*N**4*q**6 + A**4*C**2*F**4*N**4*q**2 + A**4*C**2*F**2*M**2*N**4*q**4 +',
    'A**4*C**2*F**2*N**2*q**4 - A**2*B**2*C**4*F**8*M**2*N**2*q**8 + A**2*B**2*C**4*F**4*M**2*N**2*q**8 +',
    'A**2*B**2*C**4*F**4*M**2*N**2*q**4 - A**2*B**2*C**4*M**2*N**2*q**4 + A**2*B**2*F**8*M**2*N**2*q**4 -',
    'A**2*B**2*F**4*M**2*N**2*q**4 - A**2*B**2*F**4*M**2*N**2 + A**2*B**2*M**2*N**2 -',
    'B**4*C**2*F**6*M**4*N**2*q**4 - B**4*C**2*F**6*M**2*q**4 - B**4*C**2*F**4*M**4*q**6 +',
    'B**4*C**2*F**4*M**4*q**2 - B**4*C**2*F**4*M**2*N**2*q**6 + B**4*C**2*F**4*M**2*N**2*q**2 +',
    'B**4*C**2*F**2*M**4*N**2*q**4 + B**4*C**2*F**2*M**2*q**4 + C**2*F**6*M**2*q**4 + C**2*F**6*N**2*q**4 -',
    'C**2*F**2*M**2 - C**2*F**2*N**2)'
])
P_p0_DEN = ' '.join([
    'C**2*M*N*(F - 1)*(F + 1)*(F**2 + 1)*(F*q - 1)*(F*q + 1)*(F**2*q - 1)*(F**2*q + 1)*(F**2*q**2 +',
    '1)*(A**2*N*q - 1)*(A**2*N*q + 1)*(B**2*M*q - 1)*(B**2*M*q + 1)'
])
P_p2_NUM = ' '.join([
    '2*A**4*B**4*F**6*M**2*N**2*q**8 - 2*A**4*B**4*F**2*M**2*N**2*q**4 - 2*A**4*F**4*N**2*q**6 +',
    '2*A**4*N**2*q**2 - 2*A**2*B**2*C**2*F**6*M**2*N**2*q**6 - 2*A**2*B**2*C**2*F**6*q**6 +',
    '2*A**2*B**2*C**2*F**4*M**2*q**4 + 2*A**2*B**2*C**2*F**4*N**2*q**4 + 2*A**2*B**2*C**2*F**2*M**2*N**2*q**6',
    '+ 2*A**2*B**2*C**2*F**2*q**6 - 2*A**2*B**2*C**2*M**2*q**4 - 2*A**2*B**2*C**2*N**2*q**4 -',
    '2*B**4*F**4*M**2*q**6 + 2*B**4*M**2*q**2 + 2*F**6*q**4 - 2*F**2'
])
P_p2_DEN = ' '.join([
    'A**2*B**2*C**2*M*N*q*(F - 1)*(F + 1)*(F**2 + 1)*(F*q - 1)*(F*q + 1)*(F**2*q - 1)*(F**2*q + 1)*(F**2*q**2',
    '+ 1)*(A**2*N*q - 1)*(A**2*N*q + 1)*(B**2*M*q - 1)*(B**2*M*q + 1)'
])

t0 = time.time()
K, q, A, B, C, F, M, N, w = field("q,A,B,C,F,M,N,w", QQ)
syms = dict(zip("q A B C F M N w".split(), sp.symbols("q A B C F M N w")))
fe = lambda s: K.from_expr(sp.sympify(s, locals=syms))
pm2 = fe(P_pm2_NUM)/fe(P_pm2_DEN); p0 = fe(P_p0_NUM)/fe(P_p0_DEN); p2 = fe(P_p2_NUM)/fe(P_p2_DEN)
qn = lambda u: (u-1/u)/(q-1/q)                       # [k] = (q^k-q^-k)/(q-1/q) with u = q^k
chh = lambda u: (u+1/u)/2; shh = lambda u: (u-1/u)/2  # cosh, sinh of log u
al1 = A*B*C; al2 = al1*M*N; al3 = A**2*N*F; al4 = B**2*M*F
be1 = A**2*B**2*M*N; be2 = A*B*C*N*F; be3 = A*B*C*M*F
AX = A**2*N*q; BX = B**2*M*q; CX = C**2*q               # q^(2A), q^(2B), q^(2C) with 2A = 2a+n+1, 2B = 2b+m+1, 2C = 2c+1
x = (chh(AX)*chh(BX)-chh(CX))/(shh(AX)*shh(BX))
Bf = chh(AX)/shh(AX)*chh(BX)/shh(BX)*(1-2*chh(M)*chh(N)/(chh(F**2*q)+chh(q)))+shh(M)*shh(N)/(shh(F)*shh(F*q))   # B_f of the paper
rho0 = -(x-Bf)
rhop = -2*shh(F*q/N)*shh(F*q*N)*shh(F*q/M)*shh(F*q*M)/(shh(AX)*shh(BX)*shh(F**2*q**2)*shh(F**2*q))               # rho_+(f)
rhom = -2*shh(AX/F)*shh(AX*F)*shh(BX/F)*shh(BX*F)/(shh(AX)*shh(BX)*shh(F**2)*shh(F**2*q))                        # rho_-(f)
f1 = qn(w*q/al3)*qn(w/al3)*qn(w*q/al4)*qn(w/al4)                  # t(z,f+1)/r(z)
f2 = qn(w*q/al3)*qn(w*q/al4)*qn(be2*q/w)*qn(be3*q/w)              # t(z,f)/r(z)
f3 = qn(be2*q/w)*qn(be2/w)*qn(be3*q/w)*qn(be3/w)                  # t(z,f-1)/r(z)
g1 = -qn(w*q**2)*qn(be1/w)*qn(be2*q/w)*qn(be3*q/w)                # G(z+1)/(r(z) Pol(q w)) up to sign: -[z+2][beta1-z][beta2+1-z][beta3+1-z]
g2 = -qn(w/al1)*qn(w/al2)*qn(w*q/al3)*qn(w*q/al4)                 # -[z-al1][z-al2][z-al3+1][z-al4+1]
Pol = lambda wv: pm2/wv**2+p0+p2*wv**2
def E(rm=rhom, r0=rho0, pp0=p0): 
    P_ = lambda wv: pm2/wv**2+pp0+p2*wv**2
    return rm*f3+r0*f2+rhop*f1-(g1*P_(w*q)+g2*P_(w))
print('A. Zeilberger identity (all eight symbols free)', flush=True)
E0 = E()
chk('A1 E(w) = rho_- f3 + rho_0 f2 + rho_+ f1 - (g1 Pol(qw) + g2 Pol(w)) = 0 identically in (q,A,B,C,F,M,N,w)  (%.1f s)' % (time.time()-t0), E0 == 0)
chk('A2 negative control: diagonal without the s_m s_n/(s_f s_{f+1}) term gives E != 0', E(r0=-(x-(Bf-shh(M)*shh(N)/(shh(F)*shh(F*q))))) != 0)
chk('A3 negative control: rho_- with the factor s_{2B+f} -> s_{2B-f-1} gives E != 0', E(rm=rhom*shh(BX*F)/shh(BX/(F*q))) != 0)
chk('A4 negative control: p0 -> p0 + 1 gives E != 0', E(pp0=p0+1) != 0)

print('B. gauge identities', flush=True)
# A_k^2 of the paper in the symbols: s_z = sinh(z h) -> shh(q^z)
def Ak2(k, kn):        # k = q^k (symbol), kn = q^(k) ; A_k^2 = 4 s_{k-n}s_{k+n}s_{k-m}s_{k+m}(1-s_k^2/sinh^2X)(1-s_k^2/sinh^2Y)/(s_2k^2 s_{2k+1} s_{2k-1})
    sk = shh(kn)
    return (4*shh(kn/N)*shh(kn*N)*shh(kn/M)*shh(kn*M)*(1-sk**2/shh(AX)**2)*(1-sk**2/shh(BX)**2)/(shh(kn**2)**2*shh(kn**2*q)*shh(kn**2/q)))
br = lambda u: qn(u)   # [k]
# P(f)^2 = [2a+n-f]![2b+m-f]![f-n]![f+n]![f-m]![f+m]! / ([2a+n+f+1]![2b+m+f+1]!); ratios under f -> f+1 and f -> f-1 as products of brackets
Rp = br(F*q/N)*br(F*q*N)*br(F*q/M)*br(F*q*M)/(br(AX/(F*q))*br(BX/(F*q))*br(AX*F*q)*br(BX*F*q))
Rm = br(AX/F)*br(BX/F)*br(AX*F)*br(BX*F)/(br(F/N)*br(F*N)*br(F/M)*br(F*M))                                      # P(f-1)^2/P(f)^2
chk('B1 rho_+^2 = A_{f+1}^2 ([2f+3]/[2f+1]) P(f+1)^2/P(f)^2', rhop**2 == Ak2(None, F*q)*br(F**2*q**3)/br(F**2*q)*Rp)
chk('B2 rho_-^2 = A_f^2 ([2f-1]/[2f+1]) P(f-1)^2/P(f)^2', rhom**2 == Ak2(None, F)*br(F**2/q)/br(F**2*q)*Rm)
# exact check of the factorial shift ratios Rp, Rm against q-factorials (rational zeta^(1/2) = r)
r_ = Fr(6, 5)
def qnum(k):
    e = int(2*Fr(k)); assert 2*Fr(k) == e
    return (r_**e-r_**(-e))/(r_**2-r_**-2)
def qfac(n_):
    n_ = int(n_); assert n_ >= 0
    p = Fr(1)
    for j in range(1, n_+1): p *= qnum(j)
    return p
def P2(a, b, m, n, f):
    return qfac(2*a+n-f)*qfac(2*b+m-f)*qfac(f-n)*qfac(f+n)*qfac(f-m)*qfac(f+m)/(qfac(2*a+n+f+1)*qfac(2*b+m+f+1))
worst = 0
for (a, b, m, n, f) in [(14, 12, 2, 1, 5), (11, 13, 2, -1, 7), (10, 9, 1, 0, 3), (Fr(21, 2), Fr(19, 2), Fr(3, 2), Fr(1, 2), Fr(7, 2))]:
    def ev(expr):   # evaluate a field element at q = r^2 etc.
        vals = {'q': r_**2, 'A': r_**int(2*a), 'B': r_**int(2*b), 'C': Fr(1), 'F': r_**int(2*f), 'M': r_**int(2*m), 'N': r_**int(2*n), 'w': Fr(1)}
        ex = expr.as_expr(); sub = {syms[k_]: sp.Rational(v_.numerator, v_.denominator) for k_, v_ in vals.items()}
        v = sp.nsimplify(ex.subs(sub)); return Fr(int(v.p), int(v.q))
    worst += (ev(Rp) != P2(a, b, m, n, f+1)/P2(a, b, m, n, f)) + (ev(Rm) != P2(a, b, m, n, f-1)/P2(a, b, m, n, f))
chk('B3 exact rational arithmetic: shift ratios Rp, Rm equal the q-factorial ratios of P(f)^2 (4 families, mismatches %d)' % worst, worst == 0)

print('C. row f = 0 of the exceptional channel m = n = 0', flush=True)
K4, q4, A4, B4, C4 = field("q,A,B,C", QQ)
qn4 = lambda u: (u-1/u)/(q4-1/q4); ch4 = lambda u: (u+1/u)/2; sh4 = lambda u: (u-1/u)/2
AX4, BX4, CX4 = A4**2*q4, B4**2*q4, C4**2*q4
x4 = (ch4(AX4)*ch4(BX4)-ch4(CX4))/(sh4(AX4)*sh4(BX4))
B04 = ch4(AX4)/sh4(AX4)*ch4(BX4)/sh4(BX4)*(1-1/ch4(q4))                  # B_0 = cothX cothY (1-1/cosh h)
rho_p0 = -2*sh4(q4)**3/(sh4(AX4)*sh4(BX4)*sh4(q4**2))                      # rho_+(0): F=1, M=N=1 in rho_+
S1_over_S0 = qn4(B4*C4/A4)*qn4(A4*C4/B4)-qn4(A4*B4*C4*q4**2)*qn4(A4*B4/C4)    # [b+c-a][a+c-b]-[a+b+c+2][a+b-c]
chk('C1 (x - B_0) S_0 = rho_+(0) S_1 (S_1/S_0 = [b+c-a][a+c-b]-[a+b+c+2][a+b-c]), exact symbolic in (q,A,B,C)', x4-B04 == rho_p0*S1_over_S0)
chk('C2 negative control: dropping the second term of S_1 breaks the row', x4-B04 != rho_p0*(qn4(B4*C4/A4)*qn4(A4*C4/B4)))

print('D. exact rational arithmetic on families', flush=True)
r = Fr(6, 5)
def qn_(k):
    e = int(2*Fr(k)); assert 2*Fr(k) == e
    return (r**e-r**(-e))/(r**2-r**-2)
def qf_(n_):
    n_ = int(n_); p = Fr(1)
    for j in range(1, n_+1): p *= qn_(j)
    return p
def invqf(n_): return Fr(0) if n_ < 0 else 1/qf_(n_)
def S_sum(a, b, c, m, n, f):
    al = [a+b+c, a+b+c+m+n, 2*a+n+f, 2*b+m+f]; be = [2*a+2*b+m+n, a+b+c+n+f, a+b+c+m+f]
    lo, hi = int(min(al)), int(max(be)); tot = Fr(0)
    for z in range(lo-3, hi+4):
        t = (1 if z % 2 == 0 else -1)*(qf_(z+1) if z+1 >= 0 else 0)
        if t == 0: continue
        for u in al: t *= invqf(z-u)
        for v in be: t *= invqf(v-z)
        tot += t
    return tot
ch_ = lambda k: (r**int(2*k)+r**(-int(2*k)))/2; sh_ = lambda k: (r**int(2*k)-r**(-int(2*k)))/2
def rows_exact(a, b, c, m, n):
    A2, B2, C2 = 2*a+n+1, 2*b+m+1, 2*c+1
    x_ = (ch_(A2)*ch_(B2)-ch_(C2))/(sh_(A2)*sh_(B2))
    fmax = min(2*a+n, 2*b+m); good = True; nrows = 0; signs = True
    fs = [m-1+j for j in range(0, int(fmax-m)+4)]
    S = {f: S_sum(a, b, c, m, n, f) for f in fs}
    S[fs[0]-1] = S_sum(a, b, c, m, n, fs[0]-1)   # S_{m-2}: not used except at f = m-1 = 0
    for f in fs[:-1]:
        if f < m: good &= (S[f] == 0); continue
        if f > fmax: good &= (S[f] == 0); continue
        Bk_ = ch_(A2)/sh_(A2)*ch_(B2)/sh_(B2)*(1-2*ch_(m)*ch_(n)/(ch_(2*f+1)+ch_(1)))+(sh_(m)*sh_(n)/(sh_(f)*sh_(f+1)) if f > 0 else 0)
        rp = -2*sh_(f+1-n)*sh_(f+1+n)*sh_(f+1-m)*sh_(f+1+m)/(sh_(A2)*sh_(B2)*sh_(2*f+2)*sh_(2*f+1))
        rm = -2*sh_(A2-f)*sh_(A2+f)*sh_(B2-f)*sh_(B2+f)/(sh_(A2)*sh_(B2)*sh_(2*f)*sh_(2*f+1)) if f > 0 else Fr(0)
        if f == 0: Bk_ = ch_(A2)/sh_(A2)*ch_(B2)/sh_(B2)*(1-2/(2*ch_(1)))
        lhs = (x_-Bk_)*S[f]; rhs = rp*S[f+1]+(rm*S[f-1] if f > 0 else 0)
        good &= (lhs == rhs); nrows += 1; signs &= (rp < 0 and (f == 0 or rm < 0))
    good &= (S[fs[-1]] == 0)
    return good, nrows, signs, S[fmax] != 0, S[m] != 0
for fam in [(14, 12, 17, 2, 1), (11, 13, 16, 2, -1), (10, 9, 12, 1, 0), (9, 9, 11, 3, 3), (10, 9, 12, 0, 0), (12, 9, 11, 1, 1), (Fr(21, 2), Fr(19, 2), 12, Fr(3, 2), Fr(1, 2)), (Fr(21, 2), Fr(19, 2), 12, Fr(3, 2), Fr(-1, 2)), (8, 13, 14, 2, 0), (15, 6, 18, 1, -1)]:
    g_, nr, sg, top_nonzero, bot_nonzero = rows_exact(*fam)
    chk('D (a,b,c,m,n)=%s: (R3) exactly at all %d rows f=%s..f_max, S_f=0 outside [m,f_max], rho_+- < 0, S_m, S_f_max != 0: %s' % (fam, nr, fam[3], (g_, sg, top_nonzero, bot_nonzero)), g_ and sg and top_nonzero and bot_nonzero)
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
