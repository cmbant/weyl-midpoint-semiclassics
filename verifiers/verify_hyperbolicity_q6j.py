#!/usr/bin/env python3
"""Exact hyperbolicity and forbidden-side decay for the one-small quantum 6j chain (prop:exact-hyperbolicity, prop:forbidden-decay).
# log: verify_hyperbolicity_q6j.txt

Companion of verify_hyperbolicity_hahn_racah.py / verify_forbidden_side_decay.py for the quantum chain, with sigma(q) = tanh^2 q,
sigma_1 = tanh^2 X, sigma_2 = tanh^2 Y, beta_0 = coth X coth Y, A_k = r_k e~(sigma_k)/2, e~(s) = sqrt((1 - s/sigma_1)(1 - s/sigma_2)).
Checks on exact quantum 6j chains (p3_q6j_lib, 30 digits), for each family:
 (A) the closed forms A_k = r_k e~(sigma_k)/2 and B_k = beta_0 (1 - ch(m)ch(n)/(ch(k)ch(k+1))) + sh(m)sh(n)/(sh(k)sh(k+1)) (steps of size h);
 (B) hyperbolicity tau(x - B_k) >= A_k + A_{k+1} for every k with q_k >= q_* + C_0 h^2, explicit C_0 of the proof
     (rho_1 = 4(m^2+n^2+1/3)/q_*^2, rho_2 = beta_0(1/8+m^2+n^2) + 4|mn|/q_*^2);
 (C) the decay bound rho_{j-1} <= exp(-Theta(sqrt((q_j^2 - D h)_+))) of prop:forbidden-decay(b), explicit D of the proof, on the backward
     continued fraction from the hard edge;
 (D) negative control: the same inequality must FAIL for ranks on the oscillatory side q_k <= q_* - 5 h (it is a forbidden-side statement).
Classification: (A) exact-precision identity checks; (B)-(D) finite checks of inequalities proved in the text (the proof is the argument).
Families cover both turn types tau = +-1 and several offsets in the fundamental chamber m >= |n| (tau = +1 families need N >= 600).
"""
import sys
import os
from mpmath import mp, mpf, sinh, cosh, tanh, coth, sqrt, asinh, acosh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3_q6j_lib as L  # noqa: E402

mp.dps = 30
sp1 = lambda q: 2*tanh(q)/cosh(q)**2  # sigma'(q)


def data(N, a, b, c, m, n):
    H = mpf(1)/N
    F = L.Fam(a, b, c, m, n, H)
    X, Y, Z, x = F.X, F.Y, F.Z, F.x
    mu = sinh(Z)/(sinh(X)*sinh(Y))
    qs = asinh(mp.sin(mp.acos(x))/mu)
    sig1, sig2, b0 = tanh(X)**2, tanh(Y)**2, coth(X)*coth(Y)
    ss = tanh(qs)**2
    tau = 1 if x - b0*ss > 0 else -1
    qm = min(X, Y)
    if qs >= qm:
        return None
    et = lambda S: sqrt(max((1-S/sig1)*(1-S/sig2), mpf(0)))
    kap = (1-tau*x)/ss
    fmax = int(min(2*a+n, 2*b+m))
    rho1 = 4*(m*m+n*n+mpf(1)/3)/qs**2
    rho2 = b0*(mpf(1)/8+m*m+n*n)+4*abs(m*n)/qs**2
    csig = min(sp1(qs), sp1(qm))
    C0 = (kap/12+rho1+rho2)/(kap*csig)
    return dict(H=H, F=F, x=x, qs=qs, qm=qm, tau=tau, b0=b0, et=et, kap=kap, fmax=fmax, rho1=rho1, rho2=rho2, C0=C0,
                sig1=sig1, sig2=sig2)


def Ak(d, k):
    return sqrt(d['F'].Ak2(k)) if k <= d['fmax'] else mpf(0)   # A_{k_max+1} = 0 (hard edge)


def run(N, a, b, c, m, n):
    d = data(N, a, b, c, m, n)
    if d is None:
        return None
    H, F, x, qs, tau, b0, et = d['H'], d['F'], d['x'], d['qs'], d['tau'], d['b0'], d['et']
    sh = lambda z: sinh(H*z)
    ch = lambda z: cosh(H*z)
    kmin = max(abs(m), abs(n))
    dA = dB = mpf(0)
    for k in range(kmin+1, d['fmax']+1):
        r2 = (1-sh(m)**2/sh(k)**2)*(1-sh(n)**2/sh(k)**2)/(1-sh(1)**2/sh(2*k)**2)
        dA = max(dA, abs(Ak(d, k)-sqrt(r2)*et(tanh(k*H)**2)/2))
        Bc = b0*(1-ch(m)*ch(n)/(ch(k)*ch(k+1)))+sh(m)*sh(n)/(sh(k)*sh(k+1))
        dB = max(dB, abs(F.Bk(k)-Bc))
    viol, cnt = 0, 0
    for k in range(kmin+1, d['fmax']+1):
        if (k+mpf(1)/2)*H >= qs + d['C0']*H**2:
            cnt += 1
            if tau*(x-F.Bk(k)) - Ak(d, k) - Ak(d, k+1) < 0:
                viol += 1
    # (D) oscillatory side: hyperbolicity must fail somewhere with q_k <= q_* - 5h
    osc = sum(1 for k in range(kmin+1, d['fmax']+1) if (k+mpf(1)/2)*H <= qs - 5*H
              and tau*(x-F.Bk(k)) - Ak(d, k) - Ak(d, k+1) < 0)
    # (C) decay bound
    d0 = (d['qm']-qs)/2
    Lam = tau*(x-b0*tanh(qs+d0)**2)/et(tanh(qs+d0)**2)
    cs = min(sp1(qs), sp1(qs+d0))
    D3 = b0/2+d['rho2']+d['rho1']*Lam
    D = 2*(qs+d0)*(mpf(1)/2+D3/d['kap'])/cs
    Theta = lambda q: acosh(max(tau*(x-b0*tanh(q)**2)/et(tanh(q)**2), mpf(1))) if q > qs else mpf(0)
    rho = {d['fmax']: mpf(0)}
    for k in range(d['fmax'], kmin, -1):
        rho[k-1] = Ak(d, k)/(tau*(x-F.Bk(k)) - Ak(d, k+1)*rho[k])
    kc = min(k for k in range(0, d['fmax']+1) if (k+mpf(1)/2)*H >= qs + d['C0']*H**2)
    vdec, cdec, worst = 0, 0, mpf(0)
    for j in range(kc+1, d['fmax']+1):
        qj = (j+mpf(1)/2)*H
        if qj > qs + d0:
            break
        v = qj**2 - D*H
        bound = mp.exp(-Theta(sqrt(v))) if v > 0 else mpf(1)
        cdec += 1
        if not (0 < rho[j-1] <= bound):
            vdec += 1
        worst = max(worst, rho[j-1]/bound)
    return dict(tau=tau, dA=dA, dB=dB, cnt=cnt, viol=viol, cdec=cdec, vdec=vdec, worst=worst, osc=osc)


FAMS = [(60, 29, 35, 44, 1, 1), (60, 29, 35, 50, 1, 1), (100, 48, 60, 70, 2, 1), (100, 48, 60, 75, 2, 0),
        (100, 48, 60, 60, 2, 2), (100, 48, 60, 80, 0, 0), (200, 98, 119, 150, 1, 1), (80, 39, 47, 60, 3, -1),
        (300, 148, 180, 230, 1, 1), (400, 199, 240, 300, 1, 0), (400, 190, 260, 330, 0, 0),
        (600, 290, 350, 102, 1, 1), (600, 290, 350, 102, 2, 0), (600, 290, 350, 102, 0, 0), (1200, 590, 700, 212, 1, 1)]


def main():
    ok, taus, ncheck, negfail = True, set(), 0, 0
    for fam in FAMS:
        r = run(*fam)
        if r is None:
            print('skip (no simple turn inside the chain):', fam)
            continue
        taus.add(r['tau'])
        ncheck += 1
        good = (r['dA'] < mpf('1e-25') and r['dB'] < mpf('1e-25') and r['cnt'] > 0 and r['cdec'] > 0
                and r['viol'] == 0 and r['vdec'] == 0)
        ok &= good
        print(('PASS ' if good else 'FAIL ') + f"N,a,b,c,m,n={fam} tau={r['tau']:+d} closed-form dev A,B={float(r['dA']):.0e},{float(r['dB']):.0e} "
              f"hyperbolic {r['cnt']-r['viol']}/{r['cnt']}  decay {r['cdec']-r['vdec']}/{r['cdec']} max rho/bound={float(r['worst']):.2f}", flush=True)
        negfail += r['osc'] > 0
    print(f'turn types covered: {sorted(taus)}; families checked: {ncheck}')
    print(('PASS ' if negfail == ncheck else 'FAIL ') + f'(D) negative control: hyperbolicity fails on the oscillatory side in {negfail} of {ncheck} families')
    sys.exit(0 if ok and negfail == ncheck and len(taus) == 2 else 1)


main()
