#!/usr/bin/env python3
"""Bulk zero rates of the second-order pullback (thm:general-bulk-zeros).
# log: verify_bulk_zero_rates.txt

Numerical evidence for the rates proved in thm:general-bulk-zeros and
thm:general-phase-exactness: for the exact polynomial P_{L,N}
(zeros of the physical solution) at fixed ratio q ~ q0,
  (1) zeros of d^L(Theta^[2]_q(x)) are within O(N^-4) of the exact zeros;
  (2) negative control: zeros of the leading pullback d^L(Theta_q(x)) are only
      O(N^-2) accurate;
  (3) the Bohr--Sommerfeld condition with Gamma~ = Gamma_2 + c_ab(Theta_q)/q,
        N S + h Gamma~ + p_ab pi/2 - pi/4 = (nu + 1/2) pi + O(N^-3),
      and negative controls: without the Darboux term c_ab(Theta_q)/q, and
      with N S alone, the residual is only O(N^-1).
Channels: Hahn (M,sigma) = (0,0) [exceptional start], (2,1); centered Racah
(m,n) = (1,0) with (alpha,beta) = (1, 5/4).  N = 100, 200, 400, 800.
Also checks the helper's rho_2(0,x) against the closed form eq:general-rho20.
"""
import sys
import mpmath as mp
from p3_second_order_data import hahn, racah, cab

mp.mp.dps = 30
NS_ = [100, 200, 400, 800]


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


def slope(Ns, errs):
    lx = [mp.log(n) for n in Ns]; ly = [mp.log(abs(e)) for e in errs]
    mx = sum(lx)/len(lx); my = sum(ly)/len(ly)
    return float(sum((a - mx)*(b - my) for a, b in zip(lx, ly))/sum((a - mx)**2 for a in lx))


def P_exact(ch, ev, L, X, h):
    """Exact chain value at rank L, seed 1 at k_min (zeros of P_{L,N})."""
    a, b = mp.mpf(ch.a), mp.mpf(ch.b)
    km = max(abs(a), abs(b))
    A0 = lambda k: mp.sqrt((k*k - a*a)*(k*k - b*b))/(k*mp.sqrt(4*k*k - 1))
    A = lambda k: A0(k)*ev['e']((k*h)**2)
    B = lambda k: (a*b/(k*(k + 1)) if k > 0 else 0) + ev['v'](((k + mp.mpf(1)/2)*h)**2, h*h)
    Yp, Yc = mp.mpf(1), (X - B(km))/A(km + 1)          # k_min row (A_{k_min}=0)
    for j in range(1, int(L - km)):              # ranks k = k_min + j (half-integer safe)
        k = km + j
        Yp, Yc = Yc, ((X - B(k))*Yc - A(k)*Yp)/A(k + 1)
    return Yc


def dhat(a, b, L, th):
    """sqrt(2L+1) d^L_ab(th) up to a positive factor (Wigner rank recurrence)."""
    a, b = mp.mpf(a), mp.mpf(b); X = mp.cos(th); km = max(abs(a), abs(b))
    A = lambda k: mp.sqrt((k*k - a*a)*(k*k - b*b))/(k*mp.sqrt(4*k*k - 1))
    B = lambda k: a*b/(k*(k + 1)) if k > 0 else 0
    Yp, Yc = mp.mpf(1), (X - B(km))/A(km + 1)
    for j in range(1, int(L - km)):              # ranks k = k_min + j (half-integer safe)
        k = km + j
        Yp, Yc = Yc, ((X - B(k))*Yc - A(k)*Yp)/A(k + 1)
    return Yc


def solve_theta(f, target, x0):
    """Solve f(x) = target by secant from x0 (f smooth, monotone)."""
    return mp.findroot(lambda X: f(X) - target, (x0, x0 + mp.mpf('1e-6')), tol=mp.mpf(10)**-28)


def run(name, ch, ev, q0, x0):
    print('\n%s: q0=%s x0=%s' % (name, q0, x0), flush=True)
    a, b = ch.a, ch.b
    phi0 = abs(mp.mpf(a) - mp.mpf(b))*mp.pi/2 - mp.pi/4
    E2, E1, B3, B2, B1 = [], [], [], [], []
    for N in NS_:
        h = mp.mpf(1)/N; L = int(round(q0*N - 0.5)); Q = (L + mp.mpf(1)/2)/N; kap = L + mp.mpf(1)/2
        dx = h/8; xs = [x0 + i*dx for i in range(-64, 65)]
        vals = [P_exact(ch, ev, L, X, h) for X in xs]
        br = [(xs[i], xs[i + 1]) for i in range(len(xs) - 1) if vals[i]*vals[i + 1] < 0]
        lo, hi = min(br, key=lambda c: abs(c[0] - x0))
        xe = mp.findroot(lambda X: P_exact(ch, ev, L, X, h), (lo, hi), solver='anderson', tol=mp.mpf(10)**-30)
        Th = lambda X: ch.Th(Q, X)
        Th2 = lambda X: ch.Th(Q, X) + h*h/Q*ch.Gam2(Q, X)
        # nearest zero of d^L in the angle, pulled back by Theta^[2] and by Theta
        t2 = mp.findroot(lambda T: dhat(a, b, L, T), Th2(xe), tol=mp.mpf(10)**-30)
        x2 = solve_theta(Th2, t2, xe); x1 = solve_theta(Th, t2, xe)   # same angular zero, leading pullback
        G = ch.Gam2(Q, xe); T = Th(xe)
        base = N*ch.S(Q, xe) + phi0 - mp.pi/2
        res = lambda v: abs(v - mp.pi*mp.nint(v/mp.pi))
        E2.append(xe - x2); E1.append(xe - x1)
        B3.append(res(base + h*G + h*cab(a, b, T)/Q)); B2.append(res(base + h*G)); B1.append(res(base))
        print('  N=%4d L=%4d x=%.8f  |x-x[2]|=%.3e N^4*=% .4f  |x-x[1]|=%.3e N^2*=% .4f  BS full=%.2e (N^3*=%.3f) no-Darboux=%.2e lead=%.2e'
              % (N, L, float(xe), float(abs(E2[-1])), float(N**4*E2[-1]), float(abs(E1[-1])), float(N**2*E1[-1]),
                 float(B3[-1]), float(N**3*B3[-1]), float(B2[-1]), float(B1[-1])), flush=True)
    s2, s1, s3, sB2, sB1 = (slope(NS_, v) for v in (E2, E1, B3, B2, B1))
    print('  fitted exponents: zeros[2] %.2f  zeros[1] %.2f  BS full %.2f  BS no-Darboux %.2f  BS lead %.2f' % (s2, s1, s3, sB2, sB1))
    check(s2 < -3.6, '%s: second-order pullback zeros O(N^-4) (exponent %.2f)' % (name, s2))
    check(-2.4 < s1 < -1.6, '%s: control, leading pullback zeros only O(N^-2) (exponent %.2f)' % (name, s1))
    check(s3 < -2.6, '%s: Bohr-Sommerfeld with h(Gamma_2 + c_ab/q) holds to O(N^-3) (exponent %.2f)' % (name, s3))
    check(sB2 > -1.4 and sB1 > -1.4, '%s: controls, BS without the Darboux term or without hGamma_2 only O(N^-1)' % name)


def ev_hahn():
    return {'e': lambda u: mp.sqrt(1 - u), 'v': lambda u, w: 0}


def ev_racah(m, n, al, be):
    return {'e': lambda u: mp.sqrt((1 - u/(4*al*al))*(1 - u/(4*be*be))),
            'v': lambda u, w: (2*u + w*(mp.mpf(1)/2 - m*m - n*n))/(8*al*be)}


# helper sanity: rho_2(0,x) against eq:general-rho20
worst = 0
for (ch, e1, b1, b2) in [(hahn(2, 1), mp.mpf(1)/2, 0, 0), (hahn(0, 0), mp.mpf(1)/2, 0, 0),
                         (racah(1, 0, 1, mp.mpf(5)/4), (1 + mp.mpf(16)/25)/8, mp.mpf(1)/5, (mp.mpf(1)/2 - 1)*mp.mpf(4)/5/8)]:
    a, b = ch.a, ch.b
    for X in [mp.mpf('0.2'), mp.mpf('0.55')]:
        sv = 1 - X*X
        closed = (e1*(9*X - X**3 - 8*(a*a + b*b)*X + 4*a*b*X*X + 12*a*b)
                  + b1*(8*(a*a + b*b) - 16*a*b*X - 2*X*X - 6) + 12*b2*sv)/(12*sv**mp.mpf(1.5))
        worst = max(worst, abs(ch.rho2_at0(X) - closed))
check(worst < 1e-8, 'helper rho_2(0,x) equals the closed form eq:general-rho20 (max diff %.1e)' % worst)

run('Hahn (M,sigma)=(2,1)', hahn(2, 1), ev_hahn(), mp.mpf('0.4'), mp.mpf('0.5'))
run('Hahn (M,sigma)=(0,0)', hahn(0, 0), ev_hahn(), mp.mpf('0.4'), mp.mpf('-0.3'))
run('Racah (m,n)=(1,0), (alpha,beta)=(1,5/4)', racah(1, 0, 1, mp.mpf(5)/4), ev_racah(1, 0, 1, mp.mpf(5)/4),
    mp.mpf('0.4'), mp.mpf('0.3'))
print('ALL CHECKS PASSED')
