#!/usr/bin/env python3
"""Automatic second-order fixed-rank contact for even centered chains.

Supports prop:general-fixed-rank-contact of the article (which replaced the
former hypothesis (E3) and the two chain lemmas
lem:3j-fixed-rank-second-contact / lem:6j-fixed-rank-second-contact).
# log: verify_general_fixed_rank_contact.txt

Checks (exact SymPy unless stated):
  1. From the generic second density Sigma_2 and the Darboux subtraction,
     rho_2(0,x) is the closed linear form in (eps1, beta1, beta2) printed in
     the article, the q^-2 and q^-1 parts cancel, and the higher Taylor
     coefficients eps2, beta20, beta11, beta02 do not enter.
  2. The O(t) coefficients of Theta^[2] and W^[2] at fixed rank give the
     operator Q_{2,k} printed in the article.
  3. The ladder identities and the natural-gauge Wigner recurrence hold for
     explicit Wigner d-functions (integer and half-integer channels).
  4. Generic-k proof: reducing to the basis (d_k, d_{k-1}) with the ladder
     identities, the contact identity holds identically in
     (a, b, k, x, eps1, beta1, beta2).
  5. Direct check of the contact identity with explicit d-functions in many
     channels (integer and half-integer, k = k_min included) and in the
     exceptional channel a = b = k = 0 with B_0 := 0.
  6. Negative controls: a wrong rho_2(0,x) or a wrong source fails.
  7. Specializations reproduce the former Hahn and Racah lemma formulas, and
     (numerical, carried over from verify_racah_fixed_rank_contact.py) the
     Racah closed rho_20 matches a small-rho evaluation of the full density.
"""
import math
import sympy as sp
from sympy import Rational as Q
from scipy.integrate import quad

q = sp.Symbol('q', positive=True)
X = sp.Symbol('x')
e1, e2, b1, b2, b20, b11, b02 = sp.symbols('eps1 eps2 beta1 beta2 beta20 beta11 beta02')
a, b = sp.symbols('a b')

S = 1 - X**2
RS = sp.Symbol('rs', positive=True)      # stands for sqrt(1-x^2)
CHI = X*e1 - b1
RHO_TILDE = (e1*(9*X - X**3 - 8*(a**2 + b**2)*X + 4*a*b*X**2 + 12*a*b)
             + b1*(8*(a**2 + b**2) - 16*a*b*X - 2*X**2 - 6) + 12*b2*S)
RHO20 = RHO_TILDE/(12*RS**3)   # the article's closed form, rs = sqrt(1-x^2)


def check(cond, msg):
    if not cond:
        raise SystemExit('FAIL ' + msg)
    print('PASS ' + msg)


# ---------------------------------------------------------------- 1. rho_20
# Truncated power series in q with coefficients in (x, rs = sqrt(1-x^2), T = theta).
T = sp.Symbol('T')
NQ = 7


def tr(expr, n=NQ):
    expr = sp.expand(expr)
    return sum(expr.coeff(q, j)*q**j for j in range(0, n))


def ser_fun(z, coeffs):
    """sum_j coeffs[j] z^j truncated in q (z = O(q^2))."""
    out, zp = 0, 1
    for c_ in coeffs:
        out = tr(out + c_*zp); zp = tr(zp*z)
    return out


def reduce_rs(expr):
    """Rationalize: numerator modulo rs^2 = 1 - x^2."""
    num_, den_ = sp.together(sp.expand(expr)).as_numer_denom()
    return sp.expand(sp.rem(sp.Poly(sp.expand(num_), RS), sp.Poly(RS**2 - S, RS)).as_expr())


ev = 1 - e1*q**2 + e2*q**4                       # e(q^2)
a0 = ev/2
b0 = b1*q**2 + b20*q**4                          # v(q^2, 0)
qa2 = a0*(Q(1, 8) - (a**2 + b**2)/2)             # q^2 a_2
qb2 = a*b + (b2 + b11*q**2)*q**2                 # q^2 b_2,  b_2 = ab/q^2 + d_t v(q^2,0)
inv_e = ser_fun(e1*q**2 - e2*q**4, [1, 1, 1, 1])  # 1/e
cosp = tr((X - b0)*inv_e)
dlt = tr(cosp - X)
p = tr(T - dlt/RS - X*dlt**2/(2*RS**3) - (1 + 2*X**2)*dlt**3/(6*RS**5))
z = tr((2*X*dlt + dlt**2)/RS**2)                 # sin p = rs sqrt(1-z)
sinp = tr(RS*ser_fun(z, [1, -Q(1, 2), -Q(1, 8), -Q(1, 16)]))
inv_sinp = tr(ser_fun(z, [1, Q(1, 2), Q(3, 8), Q(5, 16)])/RS)
log_e = ser_fun(e1*q**2 - e2*q**4, [0, -1, -Q(1, 2), -Q(1, 3)])
g = tr(-(log_e + ser_fun(z, [0, -1, -Q(1, 2), -Q(1, 3)])/2)/2)   # -1/2 log(a0 sin p) + const
gq, pq = sp.diff(g, q), sp.diff(p, q)
gqq, pqq = sp.diff(gq, q), sp.diff(pq, q)
a0q = sp.diff(a0, q); a0qq = sp.diff(a0q, q)
q2num = tr(q**2*(a0*(gq**2 + gqq)*cosp - a0*gq*pq*sinp - a0/4*pq**2*cosp - a0/3*pqq*sinp
                 + a0q*gq*cosp - a0q/2*pq*sinp + a0qq/4*cosp) + 2*qa2*cosp + qb2)
q2Sigma = tr(q2num*inv_e*inv_sinp)               # q^2 Sigma_2  (1/(2 a0) = 1/e)
Theta = sp.expand(sp.integrate(p, (q, 0, q))/q)  # (1/q) int_0^q p
Dl = tr(Theta - T)
cosD = ser_fun(Dl, [1, 0, -Q(1, 2), 0]); sinD = ser_fun(Dl, [0, 1, 0, -Q(1, 6)])
cosTh = tr(X*cosD - RS*sinD)
yv = tr((RS*cosD + X*sinD)/RS - 1)               # sin Theta = rs (1 + y)
inv_sinTh = tr(ser_fun(yv, [1, -1, 1, -1])/RS)
G = tr(((a**2 + b**2 - Q(1, 4))*cosTh - 2*a*b)*inv_sinTh/2)   # c_ab(Theta_q)
q2rho = tr(q2Sigma - (sp.diff(G, q)*q - G))      # q^2 [Sigma_2 - d/dq(c_ab(Theta)/q)]
pole2 = reduce_rs(q2rho.coeff(q, 0)); pole1 = reduce_rs(q2rho.coeff(q, 1))
rho0 = q2rho.coeff(q, 2)
check(pole2 == 0 and pole1 == 0, 'q^-2 and q^-1 parts of rho_2 cancel (Darboux renormalization)')
check(reduce_rs(rho0 - RHO20) == 0, 'rho_2(0,x) equals the closed form linear in (eps1,beta1,beta2)')
check(all(sp.diff(rho0, s_) == 0 for s_ in (e2, b20, b11, b02)),
      'rho_2(0,x) is independent of eps2, beta20, beta11, beta02')

# ------------------------------------------------- 2. Q_{2,k} from Theta, W
kap = sp.Symbol('kappa', positive=True)
Theta2 = Theta.coeff(q, 2)
check(reduce_rs(Theta2 + CHI/(3*RS)) == 0, 'Theta_q = theta - chi q^2/(3 sin theta) + O(q^4)')
sin2Th = tr((RS*(1 + yv))**2)
Dq = tr(RS**2 + (ev**2 - 1) - (X - b0)**2 + X**2)        # D = e^2 - (x-b0)^2, D(0)=rs^2
invD = tr(ser_fun(tr(1 - Dq/RS**2), [1, 1, 1, 1])/RS**2)
W4 = tr(sin2Th*invD)                                      # W_q^4 = 1 + 4 W2 q^2 + ...
W2coef = W4.coeff(q, 2)/4
W2_art = ((e1 - X*b1)/2 - X*CHI/6)/S
check(reduce_rs(W4.coeff(q, 0) - 1) == 0 and reduce_rs(W2coef - W2_art) == 0,
      'W_q expansion of eq:class-small-q')
# fixed rank: q^2 = kappa^2 t and Gamma_2/q = rho20 + O(q^2)
dTheta = kap**2*Theta2 + rho0                      # t-coefficient of Theta^[2]
dW = kap**2*W2coef + X/(2*RS)*rho0                 # t-coefficient of [sin^2 Theta^[2]/D]^{1/4}
deriv_art = kap**2*CHI/3 - RS*RHO20                # coefficient of d/dx (d_theta = -rs d_x)
amp_art = kap**2*W2_art + X*RHO20/(2*RS)
check(reduce_rs(-RS*dTheta - deriv_art) == 0 and reduce_rs(dW - amp_art) == 0,
      'Q_{2,k} = (kappa^2 chi/3 - sqrt(s) rho20) d_x + kappa^2 W2 + x rho20/(2 sqrt s)')

# -------------------------------------- explicit Wigner d (rational in w)
w = sp.Symbol('w', positive=True)          # w = tan(theta/4)
C = (1 - w**2)/(1 + w**2); Sn = 2*w/(1 + w**2)
XW = sp.cancel(C**2 - Sn**2); SQ = sp.cancel(2*Sn*C)    # x, sqrt(1-x^2)
DXDW = sp.cancel(sp.diff(XW, w))


def wigner_d(j, m1, m2):
    """d^j_{m1 m2}(theta) by Wigner's formula, rational in w."""
    j, m1, m2 = map(sp.nsimplify, (j, m1, m2))
    tot = 0
    for s_ in range(0, int(2*j) + 2):
        f = [j + m2 - s_, s_, m1 - m2 + s_, j - m1 - s_]
        if any(v < 0 for v in f):
            continue
        pref = sp.sqrt(sp.factorial(j + m1)*sp.factorial(j - m1)*sp.factorial(j + m2)*sp.factorial(j - m2))
        den = sp.prod([sp.factorial(v) for v in f])
        tot += (-1)**int(m1 - m2 + s_)*pref/den*C**int(2*j + m2 - m1 - 2*s_)*Sn**int(m1 - m2 + 2*s_)
    return sp.cancel(tot)


def dx(f):
    return sp.cancel(sp.diff(f, w)/DXDW)


def A0(k, aa, bb):
    if k <= max(abs(aa), abs(bb)):
        return 0
    return sp.sqrt((k**2 - aa**2)*(k**2 - bb**2))/(k*sp.sqrt(4*k**2 - 1))


def B0(k, aa, bb):
    return 0 if k == 0 else Q(aa*bb)/(k*(k + 1))


def is_zero(expr):
    return sp.cancel(sp.together(expr)) == 0


def subs_w(expr, aa, bb):
    return expr.subs({a: aa, b: bb}).subs(X, XW)


def Qop(j, f, aa, bb, rho_t=RHO_TILDE):
    kj = j + Q(1, 2)
    rt = subs_w(rho_t, aa, bb)
    sw = sp.cancel(1 - XW**2)
    deriv = kj**2*subs_w(CHI, aa, bb)/3 - rt/(12*sw)
    amp = kj**2*subs_w(W2_art, aa, bb) + XW*rt/(24*sw**2)
    return deriv*dx(f) + amp*f


def contact_residual(aa, bb, k, rho_t=RHO_TILDE, src_scale=1):
    kmin = max(abs(aa), abs(bb))
    js = [j for j in (k - 1, k, k + 1) if j >= kmin]
    d = {j: wigner_d(j, aa, bb) for j in js}
    nat = lambda j: sp.sqrt(2*j + 1)
    dh = {j: nat(j)*d[j] for j in js}
    uh = {j: nat(j)*Qop(j, d[j], aa, bb, rho_t) for j in js}
    zero = lambda m: m.get(k - 1, 0)
    lhs = XW*uh[k] - A0(k + 1, aa, bb)*uh[k + 1] - A0(k, aa, bb)*zero(uh) - B0(k, aa, bb)*uh[k]
    rhs = (-src_scale*e1*((k + 1)**2*A0(k + 1, aa, bb)*dh[k + 1] + k**2*A0(k, aa, bb)*zero(dh))
           + (b1*(k + Q(1, 2))**2 + b2)*dh[k])
    return lhs - rhs, d, dh


# ---------------------------------- 3. ladders and recurrence (explicit)
chans = [(0, 0), (1, 0), (2, 1), (1, -2), (3, 3), (Q(1, 2), Q(1, 2)), (Q(3, 2), Q(1, 2)),
         (Q(1, 2), -Q(5, 2)), (Q(-3, 2), Q(3, 2))]
ladder_ok = True
for aa, bb in chans:
    kmin = max(abs(aa), abs(bb))
    for k in [kmin + 1, kmin + 2]:
        dk, dkm, dkp = wigner_d(k, aa, bb), wigner_d(k - 1, aa, bb), wigner_d(k + 1, aa, bb)
        R = sp.sqrt((k**2 - aa**2)*(k**2 - bb**2))
        sw = sp.cancel(1 - XW**2)
        ladder_ok &= is_zero(sw*dx(dk) - (-k*XW + Q(aa*bb)/k)*dk - R/k*dkm)
        ladder_ok &= is_zero(sw*dx(dkm) - (k*XW - Q(aa*bb)/k)*dkm + R/k*dk)
        nat = lambda j: sp.sqrt(2*j + 1)
        ladder_ok &= is_zero(XW*nat(k)*dk - A0(k + 1, aa, bb)*nat(k + 1)*dkp
                       - A0(k, aa, bb)*nat(k - 1)*dkm - B0(k, aa, bb)*nat(k)*dk)
check(ladder_ok, 'ladder identities (1-x^2)d_k\' = (-kx+ab/k)d_k + (R_k/k)d_{k-1}, its partner, and '
      'eq:wigner-natural-jacobi on %d integer/half-integer channels' % len(chans))

# ------------------------------------------- 4. generic-k symbolic proof
k, R = sp.symbols('k R', positive=True)
P2 = (k**2 - a**2)*(k**2 - b**2)
D0, D1 = 0, 1          # basis: d_k, d_{k-1}


def deriv(v):
    """v = (alpha, beta) meaning alpha d_k + beta d_{k-1}."""
    al, be = v
    return (sp.diff(al, X) + (al*(-k*X + a*b/k) - be*R/k)/S,
            sp.diff(be, X) + (al*R/k + be*(k*X - a*b/k))/S)


def Qgen(j, v):
    kj = j + Q(1, 2)
    deriv_c = kj**2*CHI/3 - RHO_TILDE/(12*S)
    amp_c = kj**2*W2_art + X*RHO_TILDE/(24*S**2)
    dv = deriv(v)
    return tuple(deriv_c*dv[i] + amp_c*v[i] for i in range(2))


v_k = (1, 0); v_km = (0, 1)
# alpha_{k+1} d_{k+1}, plain gauge: alpha_{k+1} = R_{k+1}/((k+1)(2k+1))
ad_kp = (((2*k + 1)*(k*(k + 1)*X - a*b))/(k*(k + 1)*(2*k + 1)), -(k + 1)*R/(k*(k + 1)*(2*k + 1)))
gam = R/(k*(2*k + 1))                        # A_k sqrt(2k-1)/sqrt(2k+1)
u_k, u_kp, u_km = Qgen(k, v_k), Qgen(k + 1, ad_kp), Qgen(k - 1, v_km)
Bk = a*b/(k*(k + 1))
res = []
for i in range(2):
    lhs = X*u_k[i] - u_kp[i] - gam*u_km[i] - Bk*u_k[i]
    rhs = -e1*((k + 1)**2*ad_kp[i] + k**2*gam*v_km[i]) + (b1*(k + Q(1, 2))**2 + b2)*v_k[i]
    r_ = sp.expand(sp.together(lhs - rhs).as_numer_denom()[0])
    r_ = sp.rem(sp.Poly(r_, R), sp.Poly(R**2 - P2, R)).as_expr()
    res.append(sp.expand(r_))
check(res[0] == 0 and res[1] == 0,
      'contact identity holds identically in (a,b,k,x,eps1,beta1,beta2) on the basis (d_k,d_{k-1})')

# --------------------------------------------- 5. explicit channel checks
count, contact_ok = 0, True
for aa, bb in chans:
    kmin = max(abs(aa), abs(bb))
    ks = [kmin, kmin + 1, kmin + 3] if kmin > 0 else [1, 2, 4]
    for kk in ks:
        r_, _, _ = contact_residual(aa, bb, kk)
        contact_ok &= is_zero(r_)
        count += 1
r_, d, dh = contact_residual(0, 0, 0)
check(is_zero(d[0] - 1) and is_zero(d[1] - XW) and is_zero(r_),
      'exceptional channel a=b=k=0 (d_0=1, d_1=x, B_0:=0): contact identity holds')
check(contact_ok and count == 27, 'contact identity on %d explicit (channel, rank) pairs incl. k=k_min and half-integer offsets' % count)

# --------------------------------------------------- 6. negative controls
bad_rho = RHO_TILDE - 6*b2*S
r_bad1, _, _ = contact_residual(2, 1, 4, rho_t=bad_rho)
r_bad2, _, _ = contact_residual(Q(3, 2), Q(1, 2), Q(5, 2), src_scale=2)
check(not is_zero(r_bad1) and not is_zero(r_bad2), 'negative controls (wrong rho20 / wrong source) fail')

# ------------------------------------------------- 7. specializations
M, sig, m, n, al_, be_ = sp.symbols('M sigma m n alpha beta', positive=True)
hahn = RHO_TILDE.subs({e1: Q(1, 2), b1: 0, b2: 0, a: M, b: sig})/12
hahn_old = (9*X - X**3 - 8*X*(M**2 + sig**2) + 4*M*sig*X**2 + 12*M*sig)/24
Eg, Pg = al_**-2 + be_**-2, 1/(al_*be_)
qmn = Q(1, 8) - (m**2 + n**2)/2
racah = RHO_TILDE.subs({e1: Eg/8, b1: Pg/4, b2: (Q(1, 2) - m**2 - n**2)*Pg/8, a: m, b: n})/12
racah_old = (16*Eg*qmn*X + 4*Eg*m*n*X**2 + 12*Eg*m*n - Eg*X**3 + 7*Eg*X - 24*Pg*qmn*X**2
             - 8*Pg*qmn - 32*Pg*m*n*X - 7*Pg*X**2 - 5*Pg)/96
delta6 = (2*k*(k + 1) + 1 - m**2 - n**2)/(8*al_*be_)
src6 = (b1*(k + Q(1, 2))**2 + b2).subs({b1: Pg/4, b2: (Q(1, 2) - m**2 - n**2)*Pg/8})
check(sp.simplify(hahn - hahn_old) == 0 and sp.simplify(racah - racah_old) == 0
      and sp.simplify(src6 - delta6) == 0,
      'specializations reproduce the former Hahn/Racah rho20 and the Racah source delta_k^(6)')
# class data: Hahn e=sqrt(1-s), v=0; Racah e, v of eq:racah-class-data
s_, t_ = sp.symbols('s t')
eH = sp.series(sp.sqrt(1 - s_), s_, 0, 2).removeO()
eR = sp.series(sp.sqrt((1 - s_/(4*al_**2))*(1 - s_/(4*be_**2))), s_, 0, 2).removeO()
vR = s_/(4*al_*be_) + (Q(1, 2) - m**2 - n**2)*t_/(8*al_*be_)
check(sp.simplify(eH - (1 - s_/2)) == 0 and sp.simplify(eR - (1 - Eg/8*s_)) == 0
      and sp.simplify(vR - (Pg/4*s_ + (Q(1, 2) - m**2 - n**2)*Pg/8*t_)) == 0,
      'class data: Hahn (eps1,beta1,beta2)=(1/2,0,0); Racah (E/8, P/4, (1/2-m^2-n^2)P/8)')


def rho2_numeric(alpha, beta, xv, mm, nn, r):
    """Full Racah second density minus Darboux derivative at small rho (floats)."""
    A_ = 1 - r*r/(4*alpha*alpha); B_ = 1 - r*r/(4*beta*beta)
    eta = math.sqrt(A_*B_); a0v = eta/2
    ell = -r/(4*alpha*alpha*A_) - r/(4*beta*beta*B_)
    ellp = (-1/(4*alpha**2*A_) - r*r/(8*alpha**4*A_*A_) - 1/(4*beta**2*B_) - r*r/(8*beta**4*B_*B_))
    etap = eta*ell; etapp = eta*(ell*ell + ellp)
    numv = xv - r*r/(4*alpha*beta); nump = -r/(2*alpha*beta); numpp = -1/(2*alpha*beta)
    u = numv/eta; up = nump/eta - u*ell
    upp = numpp/eta - (nump/eta)*ell - up*ell - u*ellp
    sinw = math.sqrt(1 - u*u)
    omp = -up/sinw; ompp = -upp/sinw - u*up*up/sinw**3
    Dv = eta*eta - numv*numv; Dp = 2*eta*etap - 2*numv*nump
    Dpp = 2*(etap*etap + eta*etapp) - 2*(nump*nump + numv*numpp)
    gp = -Dp/(4*Dv); gpp = -Dpp/(4*Dv) + Dp*Dp/(4*Dv*Dv)
    qq = 1/8 - (mm*mm + nn*nn)/2
    a2v = a0v*qq/(r*r); b2v = (.5 - mm*mm - nn*nn)/(8*alpha*beta) + mm*nn/(r*r)
    Rv = (a0v*gp*gp*u - a0v*gp*omp*sinw + a0v*gpp*u - a0v*omp*omp*u/4 - a0v*ompp*sinw/3
          + 2*a2v*u + (etap/2)*gp*u - (etap/2)*omp*sinw/2 + (etapp/2)*u/4 + b2v)
    Sig = Rv/(2*a0v*sinw)
    om = lambda tt: math.acos((xv - tt*tt/(4*alpha*beta))/math.sqrt((1 - tt*tt/(4*alpha**2))*(1 - tt*tt/(4*beta**2))))
    Th = quad(om, 0, r, epsabs=1e-14, epsrel=1e-14, limit=100)[0]/r
    Thp = (math.acos(u) - Th)/r
    AA = mm*mm + nn*nn - .25
    c = (AA*math.cos(Th) - 2*mm*nn)/(2*math.sin(Th))
    dc = (-AA + 2*mm*nn*math.cos(Th))/(2*math.sin(Th)**2)
    return Sig - (dc*Thp/r - c/(r*r))


worst = 0.0
for case in [(1.0, .25, .4, 0, 0), (1.0, .6, .2, 1, 0), (1.2, .7, -.3, 1, -1), (.9, .5, .5, 2, 1),
             (1.1, .8, .3, 1.5, .5)]:
    al, be, xv, mm, nn = case
    closed = float(racah.subs({al_: al, be_: be, m: mm, n: nn, X: xv})/(1 - xv*xv)**1.5)
    worst = max(worst, abs(closed - rho2_numeric(al, be, xv, mm, nn, 1e-3)))
check(worst < 6e-6, 'numerical: Racah closed rho20 matches full density at rho=1e-3 (max diff %.1e)' % worst)
print('ALL CHECKS PASSED')
