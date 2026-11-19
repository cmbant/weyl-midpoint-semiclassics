"""p3_q6j_lib.py -- shared helper of the q-6j verifiers (widened class (E1') of the main paper, one-small quantum 6j).
Contents: (1) exact Kirillov-Reshetikhin q-6j sum, closed seed, Jacobi chain; (2) sympy class data of the one-small chain (e, b0, p, D, a_2, b_2, eta, v_t) and lambdified
versions; (3) pullback data S, Gamma_2, Theta^[2], W, J_2 by Gauss-Legendre; (4) the exact fixed-rank contact residual with the lambda_1 shift.
Not a verifier (no pass/fail output); imported by verify_q6j_class_chain.py, verify_q6j_second_order.py, verify_q6j_turn_airy.py."""
from mpmath import mp, mpf, sinh, cosh, sqrt, tanh, coth, acos, asinh, quad, pi, diff, atanh
mp.dps = 50

def qn(x, H): return sinh(H*x)/sinh(H)
def qf(n, H):
    r = mpf(1)
    for k in range(1, int(n)+1): r *= qn(k, H)
    return r
def inv_qf(n, H):
    n = int(round(n)); return mpf(0) if n < 0 else 1/qf(n, H)
def Delta(x, y, z, H):
    return sqrt(qf(round(x+y-z), H)*qf(round(x-y+z), H)*qf(round(-x+y+z), H)/qf(round(x+y+z+1), H))
def sixj(j1, j2, j12, j3, j, j23, H):
    tri = [(j1, j2, j12), (j3, j, j12), (j1, j, j23), (j3, j2, j23)]
    al = [sum(t) for t in tri]
    be = [j1+j2+j3+j, j2+j12+j+j23, j12+j1+j23+j3]
    pref = 1
    for t in tri: pref *= Delta(*t, H)
    s = 0
    for z in range(int(max(al)), int(min(be))+1):
        d = 1
        for a in al: d *= inv_qf(z-a, H)
        for b in be: d *= inv_qf(b-z, H)
        s += (-1)**z*qf(z+1, H)*d
    return pref*s

class Fam:
    """one-small family {a b c; b+m a+n f}_q, Weyl lengths l=H*A etc.; X=2lA, Y=2lB, Z=2lC."""
    def __init__(s, a, b, c, m, n, H):
        s.a, s.b, s.c, s.m, s.n, s.H = a, b, c, m, n, mpf(H)
        s.A = a+mpf(n+1)/2; s.B = b+mpf(m+1)/2; s.C = c+mpf(1)/2
        s.X = 2*s.H*s.A; s.Y = 2*s.H*s.B; s.Z = 2*s.H*s.C
        s.x = (cosh(s.X)*cosh(s.Y)-cosh(s.Z))/(sinh(s.X)*sinh(s.Y))
        s.xi0 = a+b+c+n
    def W(s, f): return sixj(s.a, s.b, s.c, s.b+s.m, s.a+s.n, f, s.H)
    def Phi(s, f):  # (-1)^xi sqrt([2A][2B]) W_f
        return (-1)**int(round(s.xi0+f))*sqrt(qn(2*s.A, s.H)*qn(2*s.B, s.H))*s.W(f)
    def Yhat(s, f): return sqrt(qn(2*f+1, s.H))*s.Phi(f)
    def Ak2(s, k):
        H, n, m = s.H, s.n, s.m
        sh = lambda z: sinh(H*z)
        return 4*(sh(k)**2-sh(n)**2)*(sh(k)**2-sh(m)**2)*(1-sh(k)**2/sinh(s.X)**2)*(1-sh(k)**2/sinh(s.Y)**2) / (sinh(2*H*k)**2*(sinh(2*H*k)**2-sinh(H)**2))
    def Bk(s, k):
        H, n, m = s.H, s.n, s.m
        U = H*(k+mpf(1)/2)
        return coth(s.X)*coth(s.Y)*(1-2*cosh(H*m)*cosh(H*n)/(cosh(2*U)+cosh(H))) \
            + sinh(H*m)*sinh(H*n)/(sinh(H*k)*sinh(H*(k+1)))

def seed_S(F):
    """S_m of Lemma seed (closed form) -> Phi_m = sqrt([2A][2B] S_m)"""
    H, m, n, A, B, C = F.H, F.m, F.n, F.A, F.B, F.C
    Sig = lambda L: [j-mpf(int(round(L))+1)/2 for j in range(1, int(round(L))+1)]
    num = qf(2*m, H)/(qf(m+n, H)*qf(m-n, H))
    for s in Sig(m+n): num *= qn(A+B-C+s, H)*qn(A+B+C+s, H)
    for s in Sig(m-n): num *= qn(B+C-A+s, H)*qn(A-B+C+s, H)
    for i_ in range(int(round(2*m))+1): num /= qn(2*A-m+i_, H)*qn(2*B-m+i_, H)      # j = -m,...,m: 2m+1 factors
    return num

def phi_chain(F, fmax):
    """forward recurrence for Yhat_f, f=m..fmax from the closed seed (Phi_f = Yhat/sqrt[2f+1])"""
    m = F.m; Y = {m-1: mpf(0)}
    Y[m] = sqrt(qn(2*m+1, F.H))*sqrt(qn(2*F.A, F.H)*qn(2*F.B, F.H)*seed_S(F))
    for f in range(m, fmax):
        Y[f+1] = ((F.x-F.Bk(f))*Y[f]-sqrt(F.Ak2(f))*Y[f-1])/sqrt(F.Ak2(f+1))
    return Y

def dhat_chain(mm, nn, th, kmax, s0=1):
    """Wigner natural-gauge d-hat_k(theta), k=mm..kmax, offsets (mm,nn), seed d^m_{nm}-type (positive)."""
    from mpmath import binomial, cos, sin
    x = cos(th); d = {mm-1: mpf(0)}
    d[mm] = s0*sqrt(2*mm+1)*sqrt(binomial(2*mm, mm+nn))*cos(th/2)**(mm+nn)*sin(th/2)**(mm-nn)
    for k in range(mm, kmax):
        A1 = sqrt(mpf(((k+1)**2-mm**2)*((k+1)**2-nn**2)))/((k+1)*sqrt(mpf(4*(k+1)**2-1)))
        A0 = sqrt(mpf((k**2-mm**2)*(k**2-nn**2)))/(k*sqrt(mpf(4*k**2-1))) if k > 0 else mpf(0)
        B0 = mpf(mm*nn)/(k*(k+1)) if k > 0 else mpf(0)
        d[k+1] = ((x-B0)*d[k]-A0*d[k-1])/A1
    return d

class FamC(Fam):
    """continuous-label continuation at fixed Weyl lengths X,Y,Z (lengths 2l) and H."""
    def __init__(s, X, Y, Z, m, n, H):
        s.m, s.n, s.H = m, n, mpf(H); s.X, s.Y, s.Z = mpf(X), mpf(Y), mpf(Z)
        s.A = s.X/(2*s.H); s.B = s.Y/(2*s.H); s.C = s.Z/(2*s.H)
        s.x = (cosh(s.X)*cosh(s.Y)-cosh(s.Z))/(sinh(s.X)*sinh(s.Y))

# ---------------------------------------------------------------- class data
import sympy as sp
q = sp.Symbol('q', positive=True)
x, c, X, Y, m, n = sp.symbols('x c X Y m n', real=True)

K = c*q
cX, cY = sp.coth(X), sp.coth(Y)
T = sp.tanh(K)**2
e = sp.sqrt((1-T*cX**2)*(1-T*cY**2))
b0 = T*cX*cY
a0 = e/2
cosp = (x-b0)/e
p = sp.acos(cosp)
D = e**2-(x-b0)**2
g = -sp.log(a0*sp.sin(p))/2
cab = sp.Rational(1, 8)-(m**2+n**2)/2
eta = ((m**2+n**2)*(sp.sinh(K)**2-K**2)/(2*sp.sinh(K)**2)+(K**2/sp.sinh(2*K)**2-sp.Rational(1, 4))/2)/q**2
# v_t(q^2,0): v = B - lam*mn/(k(k+1)); compute via series (done in sym_v)
h2 = sp.Symbol('h2', positive=True)
def vt_expr():
    H = sp.sqrt(h2)*c; U = K
    term1 = cX*cY*(1-2*sp.cosh(H*m)*sp.cosh(H*n)/(sp.cosh(2*U)+sp.cosh(H)))
    d1 = sp.series(term1, h2, 0, 2).removeO().coeff(h2, 1)
    d2 = c**2*m*n*(1/sp.sinh(K)**2-1/K**2)
    return d1+d2
vt = vt_expr()

def Sigma2_expr(use_eta=True):
    a2 = a0*(cab/q**2+(eta if use_eta else 0))
    b2 = m*n/q**2+vt
    pq, pqq = sp.diff(p, q), sp.diff(p, q, 2)
    gq, gqq = sp.diff(g, q), sp.diff(g, q, 2)
    a0q, a0qq = sp.diff(a0, q), sp.diff(a0, q, 2)
    sp_, cp_ = sp.sin(p), sp.cos(p)
    br = (a0*(gq**2+gqq)*cp_-a0*gq*pq*sp_-a0/4*pq**2*cp_-a0/3*pqq*sp_
          +2*a2*cp_+a0q*gq*cp_-a0q/2*pq*sp_+a0qq/4*cp_+b2)
    return br/(2*a0*sp_), pqq
S2e, pqq = Sigma2_expr(True)
S2e0, _ = Sigma2_expr(False)
args = (q, x, c, X, Y, m, n)
mod = 'mpmath'
fS2 = sp.lambdify(args, S2e, mod); fS2_0 = sp.lambdify(args, S2e0, mod)
fp = sp.lambdify(args, p, mod); fD = sp.lambdify(args, D, mod)
fpqq = sp.lambdify(args, pqq, mod)
fD_qq = sp.lambdify(args, sp.diff(sp.log(D), q, 2), mod)
fe = sp.lambdify(args, e, mod); fb0 = sp.lambdify(args, b0, mod)
feta = sp.lambdify(args, eta, mod); fvt = sp.lambdify(args, vt, mod)

# ---------------------------------------------------------------- pullback data
import mpmath
from mpmath import cot, cos, sin

_GL = {}
def GL(nn):
    if nn not in _GL:
        # compute via mpmath legendre roots (Newton)
        xs = []; ws = []
        for i in range(1, nn+1):
            xi = mpmath.cos(pi*(i-mpf(1)/4)/(nn+mpf(1)/2))
            for _ in range(100):
                p0 = mpmath.legendre(nn, xi); dp = nn*(xi*p0-mpmath.legendre(nn-1, xi))/(xi**2-1)
                dx = p0/dp; xi -= dx
                if abs(dx) < mpf(10)**(-mp.dps+3): break
            dp = nn*(xi*mpmath.legendre(nn, xi)-mpmath.legendre(nn-1, xi))/(xi**2-1)
            xs.append((1+xi)/2); ws.append(1/((1-xi**2)*dp**2))   # weights for [0,1]: 2/((1-x^2)dp^2)/2
        _GL[nn] = (xs, ws)
    return _GL[nn]

class Cls:
    """second-order class data for given (c,X,Y,m,n,x)."""
    def __init__(s, c, X, Y, m, n, x, ng=40):
        s.par = (x, c, X, Y, m, n); s.ng = ng; s.m, s.n = m, n; s.x = x
    def a(s, qq): return (qq,)+s.par
    def p(s, qq): return fp(*s.a(qq))
    def S(s, qq):
        xs, ws = GL(s.ng); return qq*sum(w*s.p(qq*t) for t, w in zip(xs, ws))
    def Theta(s, qq): return s.S(qq)/qq
    def frakc(s, th):
        m, n = s.m, s.n
        return ((m*m+n*n-mpf(1)/4)*cos(th)-2*m*n)/(2*sin(th))
    def dfrakc(s, th):
        return mpmath.diff(s.frakc, th)
    def Sigma2(s, qq, eta=True): return (fS2 if eta else fS2_0)(*s.a(qq))
    def rho2(s, qq, eta=True):
        Th = s.Theta(qq); Thp = (s.p(qq)-Th)/qq
        return s.Sigma2(qq, eta)-(s.dfrakc(Th)*Thp/qq-s.frakc(Th)/qq**2)
    def Gamma2(s, qq, eta=True):
        xs, ws = GL(s.ng); return qq*sum(w*s.rho2(qq*t, eta) for t, w in zip(xs, ws))
    def D(s, qq): return fD(*s.a(qq))
    def W(s, qq, Th): return (sin(Th)**2/s.D(qq))**mpf('0.25')
    def Theta2(s, qq, h, eta=True): return s.Theta(qq)+h**2*s.Gamma2(qq, eta)/qq
    def J2(s, qq, eta=True):
        Th = s.Theta(qq); pp = s.p(qq)
        t = s.frakc(Th)*cot(Th)/qq**2+cot(pp)*(s.Sigma2(qq, eta)+fpqq(*s.a(qq))/24)-fD_qq(*s.a(qq))/16
        if eta: t += feta(*s.a(qq))
        return t/2

# ---------------------------------------------------------------- exact contact residual
e1, b1, b2, l1 = sp.symbols('eps1 beta1 beta2 lam1')
dk, dm = sp.symbols('dk dm')   # d_k, d_{k-1} (non-hat Wigner functions)
def contact_residual(a,b,k,shift=True):
    a,b,k=map(sp.Rational,(a,b,k)); s=1-x**2
    R=lambda j: sp.sqrt((j*j-a*a)*(j*j-b*b))
    A0=lambda j: R(j)/(j*sp.sqrt(4*j*j-1)) if j>0 else 0   # natural gauge hopping
    B0=lambda j: a*b/(j*(j+1)) if j>0 else 0
    kap=k+sp.Rational(1,2); chi=x*e1-b1
    # rho20 (with lambda1 shift):
    rho20=(e1*(9*x-x**3-8*(a*a+b*b)*x+4*a*b*x**2+12*a*b)+b1*(8*(a*a+b*b)-16*a*b*x-2*x**2-6)+12*(b2+(l1*x if shift else 0))*(1-x**2))/(12*s**sp.Rational(3,2))
    def dnext(j,dj,djm):    # d_{j+1} via natural-gauge Wigner recurrence on hats: x dh_j = A_{j+1}dh_{j+1}+A_j dh_{j-1}+B_j dh_j
        r=lambda i: sp.sqrt(2*i+1)
        dhn=((x-B0(j))*r(j)*dj-A0(j)*r(j-1)*djm)/A0(j+1)
        return dhn/r(j+1)
    # ladder: (1-x^2) d_j' = (-j x + ab/j) d_j + R_j/j d_{j-1}
    def dprime(j,dj,djm): return ((-j*x+a*b/j)*dj+R(j)/j*djm)/s
    def Q(j,dj,djm):
        # t-coefficient operator Q_{2,j} d_j (non-hat), kappa_j=j+1/2
        kj=j+sp.Rational(1,2)
        return (kj**2*chi/3-sp.sqrt(s)*rho20)*dprime(j,dj,djm)+(kj**2*((e1-x*b1)/2-x*chi/6)/s+x*rho20/(2*sp.sqrt(s)))*dj
    r=lambda i: sp.sqrt(2*i+1)
    d_kp=dnext(k,dk,dm)
    # d_{k-1} derivative needs d_{k-2}: use ladder on d_{k-1}: (1-x^2) d_{k-1}' = (k x - ab/k) d_{k-1} - R_k/k d_k
    dprime_km=((k*x-a*b/k)*dm-R(k)/k*dk)/s
    kmj=k-sp.Rational(1,2)
    Qkm=(kmj**2*chi/3-sp.sqrt(s)*rho20)*dprime_km+(kmj**2*((e1-x*b1)/2-x*chi/6)/s+x*rho20/(2*sp.sqrt(s)))*dm
    Qk=Q(k,dk,dm)
    # d_{k+1}' via ladder at k+1: (1-x^2) d_{k+1}'=(-(k+1)x+ab/(k+1)) d_{k+1}+R_{k+1}/(k+1) d_k
    dprime_kp=((-(k+1)*x+a*b/(k+1))*d_kp+R(k+1)/(k+1)*dk)/s
    kpj=k+sp.Rational(3,2)
    Qkp=(kpj**2*chi/3-sp.sqrt(s)*rho20)*dprime_kp+(kpj**2*((e1-x*b1)/2-x*chi/6)/s+x*rho20/(2*sp.sqrt(s)))*d_kp
    # hats
    uh=lambda j,Qj: r(j)*Qj
    lhs=x*uh(k,Qk)-A0(k+1)*uh(k+1,Qkp)-(A0(k)*uh(k-1,Qkm) if k>0 else 0)-B0(k)*uh(k,Qk)
    dhat=lambda j,dj: r(j)*dj
    rhs=-e1*((k+1)**2*A0(k+1)*dhat(k+1,d_kp)+(k**2*A0(k)*dhat(k-1,dm) if k>0 else 0))+(b1*kap**2+b2+l1*x)*dhat(k,dk)
    return sp.simplify(sp.expand(lhs-rhs))
