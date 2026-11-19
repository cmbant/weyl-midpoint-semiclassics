#!/usr/bin/env python3
"""Hypothesis (R) of thm:general-rank-airy-second is not implied by (E1)-(E2), the turn and
the quantization (rem:airy2-R, part (a)).
# log: verify_airy2_R_counterexample.txt

The chain e(s) = sqrt(1-s), v(s,t) = beta2 t (beta2 = 0.4), with the Hahn/3j physical seed at the
node x = 1/2 (offsets (1,0)), satisfies (E1)-(E2); its principal data, simple turn q_* = sqrt(1-x^2)
and turning quantization are those of Hahn.  Its solution is the Hahn solution at the off-lattice
point x - beta2 N^-2, which is not recessive: the turning-layer bracket is
    Ai(t) + c_Bi Bi(t) + ...,   c_Bi ~ -pi beta2/(2N),
so the bracket error of thm:general-rank-airy is only O(N^-1) and (R) fails.  Control: beta2 = 0
(the 3j sequence itself) has no Bi component.  N = 100..800, |t| <= 2.  Numerical evidence.
"""
import sys
import mpmath as mp
import sympy as sp
from sympy.physics.wigner import wigner_3j

mp.mp.dps = 30
R = sp.Rational
A0W = 2


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


def slope(Ns, errs):
    lx = [mp.log(n) for n in Ns]; ly = [mp.log(abs(e)) for e in errs]
    mx = sum(lx)/len(lx); my = sum(ly)/len(ly)
    return float(sum((a - mx)*(b - my) for a, b in zip(lx, ly))/sum((a - mx)**2 for a in lx))


def r_counter(N, beta2, M=1):
    """Hahn seed at x = 1/2, recurrence with B_k + beta2 h^2 (v = beta2 t); bracket vs Ai, Bi fit."""
    Nn = mp.mpf(N); h = 1/Nn; mu = N//4; x = mp.mpf(2*mu)/Nn
    jm = jp = R(N - 1, 2)
    w3 = wigner_3j(M, jm, jp, M, mu - R(M, 2), -mu - R(M, 2))
    A = lambda k: mp.sqrt(mp.mpf(k*k - M*M)*k*k)/(k*mp.sqrt(4*mp.mpf(k)*k - 1))*mp.sqrt(1 - mp.mpf(k)**2*h*h)
    Y = {M: mp.sqrt(2*M + 1)*mp.sqrt(Nn)*mp.mpf(sp.N(w3, 40))}
    Y[M + 1] = (x - beta2*h*h)*Y[M]/A(M + 1)
    qs_ = mp.sqrt(1 - x*x); Lmax = min(int((qs_ + 2*Nn**(-mp.mpf(2)/3))*Nn) + 2, N - 2)
    for k in range(M + 1, Lmax):
        Y[k + 1] = ((x - beta2*h*h)*Y[k] - A(k)*Y[k - 1])/A(k + 1)
    pts = []
    for L in range(int((qs_ - 2*Nn**(-mp.mpf(2)/3))*Nn), Lmax):
        q = (L + mp.mpf(1)/2)/Nn
        if q <= qs_:
            z = -(mp.mpf(3)/2*mp.quad(lambda r: mp.acos(x/mp.sqrt(1 - r*r)), [q, qs_]))**(mp.mpf(2)/3)
        else:
            z = (mp.mpf(3)/2*mp.quad(lambda r: mp.acosh(x/mp.sqrt(1 - r*r)), [qs_, q]))**(mp.mpf(2)/3)
        t = Nn**(mp.mpf(2)/3)*mp.re(z)
        if abs(t) <= A0W:
            pts.append((t, Y[L]/(2*Nn**(mp.mpf(1)/6)*abs(z/(1 - x*x - q*q))**mp.mpf('0.25'))))
    sg = 1 if sum(v*mp.airyai(t) for t, v in pts) > 0 else -1
    err = max(abs(sg*v - mp.airyai(t)) for t, v in pts)
    cbi = sum((sg*v - mp.airyai(t))*mp.airybi(t) for t, v in pts)/sum(mp.airybi(t)**2 for t, _ in pts)
    return err, cbi*Nn


print('chain e = sqrt(1-s), v = beta2 t with the Hahn seed at x = 1/2 (beta2 = 0.4) and control beta2 = 0', flush=True)
NR = [100, 200, 400, 800]; b2 = mp.mpf('0.4')
res = {bb: [r_counter(N, bb) for N in NR] for bb in (b2, mp.mpf(0))}
for N, (e4, c4), (e0_, c0_) in zip(NR, res[b2], res[mp.mpf(0)]):
    print('  N=%4d  beta2=0.4: max|br-Ai|=%.3e  N*(Bi coefficient)=% .5f   beta2=0: max|br-Ai|=%.3e  N*(Bi coeff)=% .2e'
          % (N, float(e4), float(c4), float(e0_), float(c0_)), flush=True)
sb, s0 = slope(NR, [r[0] for r in res[b2]]), slope(NR, [r[0] for r in res[mp.mpf(0)]])
cN = res[b2][-1][1]
print('  exponents: beta2=0.4 %.2f, control %.2f;  -pi beta2/2 = %.5f' % (sb, s0, float(-mp.pi*b2/2)), flush=True)
check(-1.15 < sb < -0.85 and s0 < -1.2 and abs(cN + mp.pi*b2/2) < 0.02 and abs(res[mp.mpf(0)][-1][1]) < 0.01,
      '(R) fails for v = beta2 t: Bi coefficient ~ -pi beta2/(2N), bracket error O(N^-1); control has none')
print('ALL CHECKS PASSED')
