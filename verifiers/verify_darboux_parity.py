#!/usr/bin/env python3
"""Parity of the interior Wigner--Darboux expansion (lem:darboux-parity).
# log: verify_darboux_parity.txt

lem:darboux-parity (main text) proves, from the Bessel-type uniform Jacobi
expansion (DLMF 18.15.6) and the Hankel expansion of J_nu, that
  sqrt(pi sin(th)/2)*sqrt(2k+1) d^k_ab(th)/sqrt(2) = +-A cos(kappa th + phi0 + P)
with amplitude A = 1 + O(kappa^-2) even and phase P = c_ab(th)/kappa + O(kappa^-3)
odd in 1/kappa, phi0 = p_ab pi/2 - pi/4.  Numerical checks, kappa = k + 1/2,
k ~ 100, 200, 400, 800, integer and half-integer offsets:
  (1) max over th in [0.7, 2.4] of |F - (+-)cos(kappa th + phi0 + c_ab/kappa)| = O(kappa^-2)
      (an odd amplitude term would give kappa^-1); control: without c_ab/kappa it is O(kappa^-1);
  (2) at zeros th_j of d^k_ab in [0.8, 2.3],
      kappa th_j + phi0 + c_ab(th_j)/kappa - (nu+1/2) pi = O(kappa^-3)
      (an even phase term c_2/kappa^2 would give kappa^-2); control: without
      c_ab/kappa it is O(kappa^-1).
"""
import sys
import mpmath as mp

mp.mp.dps = 40
KS = [100, 200, 400, 800]


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


def slope(Ns, errs):
    lx = [mp.log(n) for n in Ns]; ly = [mp.log(abs(e)) for e in errs]
    mx = sum(lx)/len(lx); my = sum(ly)/len(ly)
    return float(sum((a - mx)*(b - my) for a, b in zip(lx, ly))/sum((a - mx)**2 for a in lx))


def wigner_d(j, m1, m2, beta):
    tot = 0
    for s in range(0, int(2*j) + 2):
        f = [j + m2 - s, s, m1 - m2 + s, j - m1 - s]
        if any(v < 0 for v in f):
            continue
        pref = mp.sqrt(mp.factorial(j + m1)*mp.factorial(j - m1)*mp.factorial(j + m2)*mp.factorial(j - m2))
        tot += ((-1)**int(m1 - m2 + s)*pref/mp.fprod([mp.factorial(v) for v in f])
                * mp.cos(beta/2)**int(2*j + m2 - m1 - 2*s)*mp.sin(beta/2)**int(m1 - m2 + 2*s))
    return tot


def dhat(a, b, K, th):
    """sqrt(2K+1) d^K_ab(th) from the exact seed and the Wigner rank recurrence."""
    km = max(abs(a), abs(b)); X = mp.cos(th)
    A = lambda k: mp.sqrt((k*k - a*a)*(k*k - b*b))/(k*mp.sqrt(4*k*k - 1))
    B = lambda k: a*b/(k*(k + 1)) if k > 0 else 0
    Yp, Yc = 0, mp.sqrt(2*km + 1)*wigner_d(km, a, b, th)
    k = km
    while k < K - mp.mpf('0.25'):
        Yp, Yc = Yc, ((X - B(k))*Yc - (A(k)*Yp if k > km else 0))/A(k + 1)
        k += 1
    return Yc


def cab(a, b, T):
    return ((a*a + b*b - mp.mpf(1)/4)*mp.cos(T) - 2*a*b)/(2*mp.sin(T))


ok_all = True
for (a, b) in [(0, 0), (2, 1), (1, -3), (mp.mpf(1)/2, mp.mpf(1)/2), (mp.mpf(3)/2, -mp.mpf(1)/2)]:
    a, b = mp.mpf(a), mp.mpf(b)
    frac = a - mp.floor(a)
    phi0 = abs(a - b)*mp.pi/2 - mp.pi/4
    amp, amp0, ph, ph0 = [], [], [], []
    for K0 in KS:
        K = K0 + frac; kap = K + mp.mpf(1)/2
        m2 = m1 = 0
        for i in range(12):
            th = mp.mpf('0.7') + i*(mp.mpf('1.7')/11)
            F = dhat(a, b, K, th)*mp.sqrt(mp.pi*mp.sin(th))/2
            G = mp.cos(kap*th + phi0 + cab(a, b, th)/kap); G0 = mp.cos(kap*th + phi0)
            m2 = max(m2, min(abs(F - G), abs(F + G))); m1 = max(m1, min(abs(F - G0), abs(F + G0)))
        amp.append(m2); amp0.append(m1)
        r2 = r1 = 0
        for th0 in [mp.mpf('0.8'), mp.mpf('1.5'), mp.mpf('2.3')]:
            # nearest zero of d^K to the predicted one
            nu = mp.nint((kap*th0 + phi0 - mp.pi/2)/mp.pi)
            guess = ((nu + mp.mpf(1)/2)*mp.pi - phi0)/kap
            tz = mp.findroot(lambda T: dhat(a, b, K, T), (guess - mp.pi/(3*kap), guess + mp.pi/(3*kap)), solver='anderson', tol=mp.mpf(10)**-30)
            v = kap*tz + phi0 - mp.pi/2
            r2 = max(r2, abs(v + cab(a, b, tz)/kap - mp.pi*mp.nint(v/mp.pi)))
            r1 = max(r1, abs(v - mp.pi*mp.nint(v/mp.pi)))
        ph.append(r2); ph0.append(r1)
        print('  (a,b)=(%s,%s) k=%s: amp err %.3e (k^2* %.3f) no-c %.3e | zero phase err %.3e (k^3* %.3f) no-c %.3e'
              % (a, b, K, float(m2), float(m2*kap**2), float(m1), float(r2), float(r2*kap**3), float(r1)), flush=True)
    sa, sa0, sp_, sp0 = slope(KS, amp), slope(KS, amp0), slope(KS, ph), slope(KS, ph0)
    print('  exponents: amplitude %.2f (control %.2f), zero phase %.2f (control %.2f)' % (sa, sa0, sp_, sp0))
    check(sa < -1.7 and sp_ < -2.7, '(a,b)=(%s,%s): amplitude even (O(kappa^-2)) and phase odd (zeros to O(kappa^-3))' % (a, b))
    check(sa0 > -1.3 and sp0 > -1.3, '(a,b)=(%s,%s): controls without c_ab/kappa are O(kappa^-1)' % (a, b))
print('ALL CHECKS PASSED')
