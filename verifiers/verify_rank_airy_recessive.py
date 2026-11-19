#!/usr/bin/env python3
"""Recessive Airy solution at a simple rank turn (lem:airy2-recessive).
# log: verify_rank_airy_recessive.txt

lem:airy2-recessive (appendix) constructs the exact solution f_1 of the
midpoint recurrence with f_1 = psi_1 = g Ai(h^(-2/3) zeta) at the last two
lattice points of the turning neighbourhood I and proves, by a final-value
Volterra sum with the positive comparison function psi_1,
    |f_1/psi_1 - 1| <= C h   at every lattice point with X = h^(-2/3) zeta >= -1.
Numerical checks (evidence only; the lemma is proved analytically):
  (0) source reading.  The envelope (-x/3)^(1/2) e^(i pi/6) H^(1)_(1/3)(2/3 (-x)^(3/2))
      of Geronimo 2009 (2.22) / GBVA Section 3 equals Ai - iBi for x < 0; for
      x > 0 it equals Ai + iBi (principal branch of -x) or -2Ai (branch -x = x e^(-i pi)),
      neither of which continues Ai - iBi as a solution of Airy's equation.  So the
      printed weight is Bi-sized on the forbidden side, and its recessive reading is
      not a smooth Airy solution across x = 0.
  (1) model chain a(q+h/2)f(q+h)+a(q-h/2)f(q-h)-2a(q)z(q)f(q)=0 with a variable
      hopping a and a simple turn of z at q_* = 1: max_J |f_1/psi_1 - 1| = O(h)
      (fitted exponent ~ -1), N = 250..4000.
  (2) negative control: the exact solution that agrees with psi_1 at the two
      FIRST (oscillatory-side) lattice points of I is O(h)-close to psi_1 there,
      yet max_J |f/psi_1 - 1| is exponentially large: closeness to g Ai on the
      oscillatory side and in the turning layer does not make a solution recessive.
"""
import sys
import mpmath as mp

mp.mp.dps = 30
QS, DELTA = mp.mpf(1), mp.mpf('0.3')


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        sys.exit(1)


def a(q):
    return mp.mpf('0.5') + mp.mpf('0.1') * q


def z(q):  # characteristic quotient, z(q_*) = 1, z' > 0: forbidden side q > q_*
    s = q - QS
    return 1 + s + mp.mpf('0.3') * s**2


def zeta(q):
    if q >= QS:
        I = mp.quad(lambda s: mp.acosh(z(s)), [QS, q])
        return (mp.mpf(3) / 2 * I) ** (mp.mpf(2) / 3)
    I = mp.quad(lambda s: mp.acos(z(s)), [q, QS])
    return -(mp.mpf(3) / 2 * I) ** (mp.mpf(2) / 3)


def gfac(q, zt):  # g = |V|^(-1/2), |V| = a sqrt|z^2-1| / |zeta|^(1/2)
    return (abs(zt) ** mp.mpf('0.5') / (a(q) * mp.sqrt(abs(z(q)**2 - 1)))) ** mp.mpf('0.5')


def run(N):
    h = mp.mpf(1) / N
    n = int(DELTA * N)
    qs = [QS + (k + mp.mpf('0.37')) * h for k in range(-n, n)]  # generic lattice offset
    X, psi = [], []
    for q in qs:
        zt = zeta(q)
        x = zt * h ** (-mp.mpf(2) / 3)
        X.append(x)
        psi.append(gfac(q, zt) * mp.airyai(x))
    m = len(qs)
    # (1) final-value solution, backward recursion (stable for the recessive branch)
    f = [mp.mpf(0)] * m
    f[m - 1], f[m - 2] = psi[m - 1], psi[m - 2]
    for k in range(m - 2, 0, -1):
        q = qs[k]
        f[k - 1] = (2 * a(q) * z(q) * f[k] - a(q + h / 2) * f[k + 1]) / a(q - h / 2)
    J = [k for k in range(m) if X[k] >= -1]
    sig = max(abs(f[k] / psi[k] - 1) for k in J)
    # (2) control: initial-value solution from the oscillatory end
    F = [mp.mpf(0)] * m
    F[0], F[1] = psi[0], psi[1]
    for k in range(1, m - 1):
        q = qs[k]
        F[k + 1] = (2 * a(q) * z(q) * F[k] - a(q - h / 2) * F[k - 1]) / a(q + h / 2)
    ctl = max(abs(F[k] / psi[k] - 1) for k in J)
    osc = max(abs(F[k] - psi[k]) for k in range(m) if X[k] <= 0)
    return sig, ctl, osc, max(X)


def main():
    print('(0) source reading of the Hankel envelope (Geronimo 2009 (2.22), GBVA Sec. 3)')
    third = mp.mpf(1) / 3
    def ut(x, br):
        w12 = mp.sqrt(-x) if x < 0 else mp.sqrt(x) * mp.expj(br / 2)
        w32 = (-x) ** 1.5 if x < 0 else x ** 1.5 * mp.expj(1.5 * br)
        return w12 / mp.sqrt(3) * mp.expj(mp.pi / 6) * mp.hankel1(third, mp.mpf(2) / 3 * w32)
    ok_neg = all(abs(ut(mp.mpf(x), 0) - (mp.airyai(x) - 1j * mp.airybi(x))) < 1e-20 for x in (-4, -1, '-0.1'))
    check(ok_neg, 'for x<0 the envelope is Ai(x) - i Bi(x)')
    for x in (mp.mpf('0.5'), mp.mpf(2), mp.mpf(5)):
        up, um = ut(x, mp.pi), ut(x, -mp.pi)
        ai, bi = mp.airyai(x), mp.airybi(x)
        print(f'  x={float(x):4.1f}  |u|(principal)/Ai={float(abs(up)/ai):.4e}  '
              f'u(-pi)/(-2Ai)={mp.nstr(um/(-2*ai), 8)}  |Ai-iBi|/Ai={float(abs(ai-1j*bi)/ai):.4e}')
        check(abs(up - (ai + 1j * bi)) < 1e-15 * abs(up) and abs(um + 2 * ai) < 1e-15 * abs(ai),
              f'x={float(x)}: principal branch gives Ai+iBi (Bi-sized), branch -pi gives -2Ai (recessive)')
    print('(1)-(2) model chain, turning neighbourhood [0.7, 1.3], lattice offset 0.37')
    Ns = [250, 500, 1000, 2000, 4000]
    sigs = []
    for N in Ns:
        sig, ctl, osc, Xmax = run(N)
        sigs.append(sig)
        print(f'  N={N:5d}  X_max={float(Xmax):7.2f}  max_J|f1/psi1-1|={float(sig):.3e}  '
              f'N*max={float(sig*N):.4f}  control: max_osc|F-psi1|={float(osc):.2e}  max_J|F/psi1-1|={mp.nstr(ctl, 3)}')
        check(ctl > 1e3, f'N={N}: control solution is not recessive')
    lx = [mp.log(n) for n in Ns]; ly = [mp.log(s) for s in sigs]
    mx = sum(lx) / len(lx); my = sum(ly) / len(ly)
    slope = float(sum((p - mx) * (r - my) for p, r in zip(lx, ly)) / sum((p - mx) ** 2 for p in lx))
    print(f'  fitted exponent of max_J|f1/psi1-1|: {slope:.3f}')
    check(abs(slope + 1) < 0.15, 'recessive relative error is O(h)')
    check(max(float(s * n) for s, n in zip(sigs, Ns)) < 10, 'N * max_J|f1/psi1-1| bounded')
    print('ALL PASS')


if __name__ == '__main__':
    main()
