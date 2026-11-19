#!/usr/bin/env python3
"""The q-Hahn chain is the one-small quantum 6j chain at Y = infinity; hyperbolicity and forbidden decay there
(prop:exact-hyperbolicity, prop:forbidden-decay, rem:qhahn-limit).
# log: verify_qhahn_q6j_limit.txt

Setting.  Base p = exp(-2/N) (deformation q = e^{1/N}), q-Hahn parameters alpha = p^{a'}, beta = p^{b'}, size N_H (KLS 14.6.3:
-(1 - p^{-x_H}) Q_n = A_n Q_{n+1} - (A_n + C_n) Q_n + C_n Q_{n-1}).  Orthonormal Jacobi matrix: diagonal -(A_n + C_n), hopping
sqrt(A_{n-1} C_n).  Quantum 6j side: eq:q6j-AkBk with coth Y = 1 and 1/sinh Y = 0, offsets m = (a'+b')/2, n = (b'-a')/2,
rank k = n_H + m, hard edge k_max = N_H + m, X = (k_max + 1)/N.

 (0) EXACT SYMBOLIC identity (SymPy, rational functions of s = e^{1/N}, A = s^{a'/2}, B = s^{b'/2}, so a', b' are arbitrary reals):
     J_qHahn = c_0 I + c_1 J_6j(Y=inf) entrywise up to the gauge sign of the hopping, with
     c_1 = -e^{(K-a')/N} sinh X, c_0 = e^{(K-a')/N} cosh X - 1, K = k_max; i.e. p^{-x_H} = e^{(K-a')/N}(cosh X - x sinh X).
     Checked once with s^n and s^{N_H} as further free symbols (every row of every size; the boundary rows n = 0, N_H follow
     by specialization, C_0 = 0 and A_{N_H} = 0 appearing on both sides), and row by row at N_H = 2..6; negative control:
     dropping the prefactor e^{(K-a')/N} must fail.
 (A)-(D) as in verify_hyperbolicity_q6j.py, on exact q-Hahn chains (30 digits, coefficients from the KLS formulas through the
     map of (0)) at physical nodes x, with sigma_2 = tanh^2 Y = 1, beta_0 = coth X and the explicit C_0, D of the proof:
     (B) hyperbolicity beyond q_* + C_0 h^2, (C) forbidden decay bound, (D) negative control on the oscillatory side.
Classification: (0) exact-symbolic identity; (B)-(D) finite checks of inequalities proved in the text.
"""
import sys
import sympy as sp
from mpmath import mp, mpf, sinh, cosh, tanh, coth, sqrt, acosh, exp, findroot

mp.dps = 30


# ---------- (0) exact symbolic identity ----------
def symbolic_identity(NH, wrong=False):
    s, A, B = sp.symbols('s A B', positive=True)
    al, be = A**-4, B**-4                      # p^{a'} = s^{-2a'} = A^{-4}
    pp = lambda e: s**(-2*e)                   # p^e for integer e
    Aq = lambda n: (1-pp(n-NH))*(1-al*pp(n+1))*(1-al*be*pp(n+1))/((1-al*be*pp(2*n+1))*(1-al*be*pp(2*n+2)))
    Cq = lambda n: -al*pp(n-NH)*(1-pp(n))*(1-al*be*pp(n+NH+1))*(1-be*pp(n))/((1-al*be*pp(2*n))*(1-al*be*pp(2*n+1)))
    sh = lambda e: (e - 1/e)/2                 # sinh from e^{z/N}
    ch = lambda e: (e + 1/e)/2
    eM, eNn = A*B, B/A                         # e^{m/N}, e^{n/N}
    eX = s**(NH+1)*A*B                         # e^{X} = e^{(K+1)/N}
    shX, chX, cothX = sh(eX), ch(eX), ch(eX)/sh(eX)
    pref = s**NH*B/A                           # e^{(K-a')/N}
    c1, c0 = -pref*shX, pref*chX - 1
    if wrong:                                  # negative control: the map without the e^{(K-a')/N} prefactor
        c1, c0 = -shX, chX - 1
    bad = 0
    for n in range(NH+1):
        ek = s**n*A*B                          # e^{k/N}, k = n + m
        Bk = cothX*(1 - 2*ch(eM)*ch(eNn)/(ch(ek**2*s) + ch(s))) + sh(eM)*sh(eNn)/(sh(ek)*sh(ek*s))
        diag = -(Aq(n) + (Cq(n) if n > 0 else 0))
        if sp.cancel(sp.together(diag - c0 - c1*Bk)) != 0:
            bad += 1
        if n > 0:
            A2 = 4*sh(ek/eNn)*sh(ek*eNn)*sh(ek/eM)*sh(ek*eM)*(1 - sh(ek)**2/shX**2)/(sh(ek**2)**2*sh(ek**2*s)*sh(ek**2/s))
            if sp.cancel(sp.together(Aq(n-1)*Cq(n) - c1**2*A2)) != 0:
                bad += 1
    return bad


def symbolic_general():
    """Same identity with u = s^n and V = s^{N_H} as further free symbols: one rational identity for every row n and size N_H."""
    s, A, B, u, V = sp.symbols('s A B u V', positive=True)
    al, be = A**-4, B**-4
    sh = lambda e: (e - 1/e)/2
    ch = lambda e: (e + 1/e)/2

    def Aq(U):                                  # A_n with s^n = U
        return ((1-U**-2*V**2)*(1-al*U**-2*s**-2)*(1-al*be*U**-2*s**-2)
                / ((1-al*be*U**-4*s**-2)*(1-al*be*U**-4*s**-4)))

    def Cq(U):
        return (-al*U**-2*V**2*(1-U**-2)*(1-al*be*U**-2*V**-2*s**-2)*(1-be*U**-2)
                / ((1-al*be*U**-4)*(1-al*be*U**-4*s**-2)))
    eM, eNn, eX, pref = A*B, B/A, V*s*A*B, V*B/A
    shX, chX = sh(eX), ch(eX)
    c1, c0 = -pref*shX, pref*chX - 1
    ek = u*A*B
    Bk = chX/shX*(1 - 2*ch(eM)*ch(eNn)/(ch(ek**2*s) + ch(s))) + sh(eM)*sh(eNn)/(sh(ek)*sh(ek*s))
    A2 = 4*sh(ek/eNn)*sh(ek*eNn)*sh(ek/eM)*sh(ek*eM)*(1 - sh(ek)**2/shX**2)/(sh(ek**2)**2*sh(ek**2*s)*sh(ek**2/s))
    d1 = sp.cancel(sp.together(-(Aq(u) + Cq(u)) - c0 - c1*Bk))
    d2 = sp.cancel(sp.together(Aq(u/s)*Cq(u) - c1**2*A2))
    return d1, d2


# ---------- (A)-(D) on exact q-Hahn chains ----------
def chain(N, NH, ap, bp):
    p = exp(mpf(-2)/N)
    al, be = p**ap, p**bp
    Aq = lambda n: (1-p**(n-NH))*(1-al*p**(n+1))*(1-al*be*p**(n+1))/((1-al*be*p**(2*n+1))*(1-al*be*p**(2*n+2)))
    Cq = lambda n: -al*p**(n-NH)*(1-p**n)*(1-al*be*p**(n+NH+1))*(1-be*p**n)/((1-al*be*p**(2*n))*(1-al*be*p**(2*n+1)))
    m, nn = mpf(ap+bp)/2, mpf(bp-ap)/2
    K = NH + m
    X = (K+1)/mpf(N)
    pref = exp((K-ap)/mpf(N))
    c1, c0 = -pref*sinh(X), pref*cosh(X) - 1
    kmin = m
    Bk = {kmin+n: (-(Aq(n) + (Cq(n) if n > 0 else 0)) - c0)/c1 for n in range(NH+1)}
    Ak = {kmin+n: sqrt(Aq(n-1)*Cq(n))/abs(c1) for n in range(1, NH+1)}
    Ak[K+1] = mpf(0)
    xs = [(cosh(X) - exp(-(K-ap)/mpf(N))*p**(-xh))/sinh(X) for xh in range(NH+1)]
    return dict(N=N, H=mpf(1)/N, X=X, m=m, n=nn, K=K, kmin=kmin, Ak=Ak, Bk=Bk, xs=xs)


sp1 = lambda q: 2*tanh(q)/cosh(q)**2


def run(c, x):
    H, X, m, n, K, kmin, A, Bk = c['H'], c['X'], c['m'], c['n'], c['K'], c['kmin'], c['Ak'], c['Bk']
    sig1, sig2, b0 = tanh(X)**2, mpf(1), coth(X)
    et = lambda S: sqrt(max((1-S/sig1)*(1-S/sig2), mpf(0)))
    Dq = lambda q: et(tanh(q)**2)**2 - (x - b0*tanh(q)**2)**2
    qs = findroot(Dq, (mpf('1e-6'), X*(1-mpf('1e-9'))), solver='anderson')
    if not (0 < qs < X - 4*H) or qs < 4*H:
        return None
    ss = tanh(qs)**2
    tau = 1 if x - b0*ss > 0 else -1
    kap = (1-tau*x)/ss
    rho1 = 4*(m*m+n*n+mpf(1)/3)/qs**2
    rho2 = b0*(mpf(1)/8+m*m+n*n)+4*abs(m*n)/qs**2
    C0 = (kap/12+rho1+rho2)/(kap*min(sp1(qs), sp1(X)))
    ks = sorted(Bk)
    dA = max(abs(A[k] - sqrt((1-sinh(H*m)**2/sinh(H*k)**2)*(1-sinh(H*n)**2/sinh(H*k)**2)/(1-sinh(H)**2/sinh(2*H*k)**2))
                 * et(tanh(k*H)**2)/2) for k in ks if k > kmin)
    dB = max(abs(Bk[k] - b0*(1-cosh(H*m)*cosh(H*n)/(cosh(H*k)*cosh(H*(k+1))))
                 - (sinh(H*m)*sinh(H*n)/(sinh(H*k)*sinh(H*(k+1))) if m*n != 0 else 0)) for k in ks if k > kmin)
    gap = lambda k: tau*(x-Bk[k]) - A.get(k, mpf(0)) - A[k+1]
    beyond = [k for k in ks if k > kmin and (k+mpf(1)/2)*H >= qs + C0*H**2]
    viol = sum(1 for k in beyond if gap(k) < 0)
    osc = sum(1 for k in ks if k > kmin and (k+mpf(1)/2)*H <= qs - 5*H and gap(k) < 0)
    d0 = (X-qs)/2
    Lam = tau*(x-b0*tanh(qs+d0)**2)/et(tanh(qs+d0)**2)
    D3 = b0/2+rho2+rho1*Lam
    D = 2*(qs+d0)*(mpf(1)/2+D3/kap)/min(sp1(qs), sp1(qs+d0))
    Theta = lambda q: acosh(max(tau*(x-b0*tanh(q)**2)/et(tanh(q)**2), mpf(1))) if q > qs else mpf(0)
    rho = {K: mpf(0)}
    for k in reversed(ks[1:]):
        rho[k-1] = A[k]/(tau*(x-Bk[k]) - A[k+1]*rho[k])
    vdec = cdec = 0
    for j in beyond:
        qj = (j+mpf(1)/2)*H
        if qj > qs + d0 or j-1 not in rho or (j-1+mpf(1)/2)*H < qs + C0*H**2:
            continue
        v = qj**2 - D*H
        bound = exp(-Theta(sqrt(v))) if v > 0 else mpf(1)
        cdec += 1
        vdec += not (0 < rho[j-1] <= bound)
    return dict(tau=tau, dA=dA, dB=dB, cnt=len(beyond), viol=viol, cdec=cdec, vdec=vdec, osc=osc, qs=qs)


def main():
    ok = True
    d1, d2 = symbolic_general()
    ok &= d1 == 0 and d2 == 0
    print(('PASS ' if d1 == 0 and d2 == 0 else 'FAIL ') + '(0) exact symbolic identity for all rows n and sizes N_H '
          '(s^n, s^{N_H} free symbols): diagonal residual ' + str(d1) + ', squared-hopping residual ' + str(d2), flush=True)
    for NH in range(2, 7):
        bad = symbolic_identity(NH)
        ok &= bad == 0
        print(('PASS ' if bad == 0 else 'FAIL ') + f'(0) exact symbolic q-Hahn = q6j(Y=inf) identity, N_H={NH}: {bad} failing entries', flush=True)
    badw = symbolic_identity(4, wrong=True)
    ok &= badw > 0
    print(('PASS ' if badw > 0 else 'FAIL ') + f'(0) negative control: affine map without the prefactor fails in {badw} entries (N_H=4)', flush=True)
    taus, nfam, negfail = set(), 0, 0
    for (N, NH, ap, bp) in [(200, 150, 0, 0), (200, 150, 1, 1), (200, 150, 2, 0), (300, 120, 0, 2), (300, 200, 1, 3)]:
        c = chain(N, NH, ap, bp)
        for xh in (NH//10, NH//4, NH//2, (3*NH)//4, NH - NH//10):
            r = run(c, c['xs'][xh])
            if r is None:
                continue
            nfam += 1
            taus.add(r['tau'])
            good = r['dA'] < mpf('1e-24') and r['dB'] < mpf('1e-24') and r['cnt'] > 0 and r['viol'] == 0 and r['vdec'] == 0
            ok &= good
            negfail += r['osc'] > 0
            print(('PASS ' if good else 'FAIL ') + f"N,N_H,a',b'={N},{NH},{ap},{bp} node x_H={xh} tau={r['tau']:+d} q*={float(r['qs']):.3f} "
                  f"closed-form dev A,B={float(r['dA']):.0e},{float(r['dB']):.0e} hyperbolic {r['cnt']-r['viol']}/{r['cnt']} "
                  f"decay {r['cdec']-r['vdec']}/{r['cdec']}", flush=True)
    print(f'turn types covered: {sorted(taus)}; node families checked: {nfam}')
    print(('PASS ' if negfail == nfam else 'FAIL ') + f'(D) negative control: hyperbolicity fails on the oscillatory side in {negfail} of {nfam}')
    sys.exit(0 if ok and negfail == nfam and len(taus) == 2 and nfam >= 10 else 1)


main()
