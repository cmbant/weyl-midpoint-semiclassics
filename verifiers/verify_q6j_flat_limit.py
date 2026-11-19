"""verify_q6j_flat_limit.py -- Corollary q6j-consequences(vi): the flat limit (curvature scale c -> 0, Weyl labels alpha,beta,gamma fixed) of ALL class data of the one-small
quantum 6j chain against the centered Racah/6j values of the paper (eq:racah-class-data, D_6, rho_*, Lemma racahFlatTurnAction): hyperbolic triangle -> Euclidean triangle (x, sin theta),
e(q), b0(q), eps_1, beta_1, beta_2, chi, lambda_1, eta(q), D(q,x), the turn q_* and type tau, the turning action, and delta_fl = -c^2/12.  Each difference is O(c^2): halving c divides it by 4.
Class: numerical evidence (30 digits, c = 0.02, 0.01, 0.005) for the O(c^2) rate of exact formulas; the limits themselves are exact (series in c)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpmath import mp, mpf, sinh, cosh, tanh, coth, sqrt, acos, asinh, quad, pi
from p3_q6j_lib import fe, fb0, fD, fp, feta, fvt
mp.dps = 30
ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ')+name, flush=True)
def flat(al, be, ga, m, n, qq):
    x = (al**2+be**2-ga**2)/(2*al*be); s = sqrt(1-x**2); rs = 2*al*be*s/ga
    e6 = sqrt((1-qq**2/(4*al**2))*(1-qq**2/(4*be**2))); b06 = qq**2/(4*al*be)
    eps1 = (al**-2+be**-2)/8; b1 = 1/(4*al*be); b2 = (mpf(1)/2-m*m-n*n)/(8*al*be)
    chi = x*eps1-b1; D6 = e6**2-(x-b06)**2
    tau = 1 if x-rs**2/(4*al*be) > 0 else -1
    S = pi*(ga-abs(al-be)) if tau == 1 else pi*rs-pi*(al+be-ga)
    return dict(x=x, rs=rs, e=e6, b0=b06, eps1=eps1, b1=b1, b2=b2, chi=chi, D=D6, tau=tau, S=S)
def hyp(c, al, be, ga, m, n, qq):
    X, Y, Z = 2*c*al, 2*c*be, 2*c*ga
    x = (cosh(X)*cosh(Y)-cosh(Z))/(sinh(X)*sinh(Y)); G = sinh(X)**2*sinh(Y)**2*(1-x**2); Ks = asinh(sqrt(G)/sinh(Z))
    cX, cY = coth(X), coth(Y)
    par = (x, c, X, Y, m, n)
    tau = 1 if x-fb0(Ks/c, *par) > 0 else -1
    pq = lambda r: fp(r, *par)
    I = quad(pq, [0, Ks/c])
    return dict(x=x, rs=Ks/c, e=fe(qq, *par), b0=fb0(qq, *par), eps1=c**2*(cX**2+cY**2)/2, b1=c**2*cX*cY,
                b2=c**2*((mpf(1)/2-m*m-n*n)*cX*cY/2-mpf(m*n)/3), lam1=c**2*(m*m+n*n-1)/6, eta=feta(qq, *par),
                chi=x*c**2*(cX**2+cY**2)/2-c**2*cX*cY, D=fD(qq, *par), tau=tau, S=(I if tau == 1 else I-pi*Ks/c), Ks=Ks)
m, n = 2, 1
for name, (al, be, ga) in [('triangle (1,1.1,1.5)', (mpf(1), mpf('1.1'), mpf('1.5'))), ('triangle (2,1,1.3)', (mpf(2), mpf(1), mpf('1.3')))]:
    qq = mpf('0.6')
    fl = flat(al, be, ga, m, n, qq); errs = {}
    for c in (mpf('0.02'), mpf('0.01'), mpf('0.005')):
        h = hyp(c, al, be, ga, m, n, qq)
        # the hyperbolic data live in the variable q with K = c q; for D and the action the flat limits are D6(rho), pi*(gamma-|alpha-beta|) etc.
        Sfl = fl['S'] if fl['tau'] == 1 else fl['S']-pi*fl['rs']
        errs[c] = dict(x=abs(h['x']-fl['x']), rs=abs(h['rs']-fl['rs']), e=abs(h['e']-fl['e']), b0=abs(h['b0']-fl['b0']), eps1=abs(h['eps1']-fl['eps1']),
                       b1=abs(h['b1']-fl['b1']), b2=abs(h['b2']-fl['b2']), chi=abs(h['chi']-fl['chi']), D=abs(h['D']-fl['D']), lam1=abs(h['lam1']), eta=abs(h['eta']),
                       S=abs(h['S']-Sfl), tau=(h['tau'] != fl['tau']))
    cs = sorted(errs, reverse=True)
    for key in ['x', 'rs', 'e', 'b0', 'eps1', 'b1', 'b2', 'chi', 'D', 'lam1', 'eta']:
        r1 = errs[cs[0]][key]/errs[cs[1]][key]; r2 = errs[cs[1]][key]/errs[cs[2]][key]
        chk('%s: %s -> flat value, errors %s, %s, %s (ratios %s, %s: O(c^2))' % (name, key, mp.nstr(errs[cs[0]][key], 3), mp.nstr(errs[cs[1]][key], 3), mp.nstr(errs[cs[2]][key], 3), mp.nstr(r1, 4), mp.nstr(r2, 4)),
            abs(r1-4) < 0.15 and abs(r2-4) < 0.15)
    chk('%s: turning action int_0^{q*} p dq (- pi q* for tau=-1) equals the Racah value pi(gamma-|alpha-beta|) (resp. -pi(alpha+beta-gamma)) EXACTLY at every scale (Lemma q6j-turn-action is scale invariant): errors %s' % (name, [mp.nstr(errs[c]['S'], 2) for c in cs]), all(errs[c]['S'] < mpf(10)**-24 for c in cs))
    chk('%s: turn type tau = %d agrees with the flat type at all three scales' % (name, fl['tau']), not any(errs[c]['tau'] for c in cs))
chk('flux defect delta_fl = -c^2/12 -> 0 (c=0.005: %s)' % mp.nstr(-mpf('0.005')**2/12, 4), abs(mpf('0.005')**2/12) < 3e-6)
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
