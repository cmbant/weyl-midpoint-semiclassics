"""Helper (not a verifier): numerical machinery for the uniform second-order Airy law
(thm:general-rank-airy-second(ii)) on the centered Racah/6j and fixed-offset Hahn/3j chains, in mpmath.

  RT, HT       chain data (e, b_0, exact a(S;h^2), b(q;h^2)), the Airy coordinate zeta, g = sqrt(2)|zeta/D|^(1/4)
  psi_disc     [h^2]A[g,0](q) by Richardson in h (lem:airy2-shift-parity); OlverB: B_tau = q_1/g (lem:airy2-olver)
  flux_disc    the Airy-form flux F(q;h) = a(q;h^2) (A_- B_+ - B_- A_+) of the formal parametrix (P = g, Q = h g B)
  exact_6j/3j  natural-gauge exact chains (sympy seeds + exact recurrence at high precision)
  Chain, setup  exact natural-gauge coefficients A_k, B_k of the Hahn/3j and centered Racah/6j rank chains (eq:adiabaticJacobi,
                eq:racahAkexact), class data e(s), b_0, the hard edge e = 0 and the principal turn (q_*, tau)
Used by verify_airy2_amplitude_flux.py, verify_hyperbolicity_hahn_racah.py, verify_forbidden_side_decay.py.
"""
import mpmath as mp
import sympy as sp
from sympy.physics.wigner import wigner_6j, wigner_3j

R = sp.Rational
mp.mp.dps = 30


def slope(Ns, errs):
    lx = [mp.log(n) for n in Ns]; ly = [mp.log(e) for e in errs]
    mx, my = sum(lx)/len(lx), sum(ly)/len(ly)
    return float(sum((a - mx)*(b - my) for a, b in zip(lx, ly))/sum((a - mx)**2 for a in lx))


class _Chain:
    def zeta(s, q):
        if q < s.qs:
            return -(mp.mpf(3)/2*mp.quad(lambda r: mp.acos(min(s.tu(r), 1)), [q, s.qs]))**(mp.mpf(2)/3)
        return (mp.mpf(3)/2*mp.quad(lambda r: mp.acosh(max(s.tu(r), 1)), [s.qs, q]))**(mp.mpf(2)/3)

    def g(s, q, z): return mp.sqrt(2)*abs(z/s.D(q))**mp.mpf('0.25')


class RT(_Chain):
    """Centered Racah chain at fixed (m, n, alpha, beta, gamma)."""
    def __init__(s, m_, n_, abg):
        s.m, s.n = mp.mpf(sp.N(m_, 40)), mp.mpf(sp.N(n_, 40))
        s.al, s.be, s.ga = (mp.mpf(sp.N(v, 40)) for v in abg)
        s.x = (s.al**2 + s.be**2 - s.ga**2)/(2*s.al*s.be)
        s.qs = 2*s.al*s.be*mp.sqrt(1 - s.x**2)/s.ga
        s.tau = int(mp.sign(s.x - s.qs**2/(4*s.al*s.be)))

    def e(s, q): return mp.sqrt((1 - q*q/(4*s.al**2))*(1 - q*q/(4*s.be**2)))
    def tu(s, q): return s.tau*(s.x - q*q/(4*s.al*s.be))/s.e(q)
    def D(s, q): return s.e(q)**2 - (s.x - q*q/(4*s.al*s.be))**2

    def a(s, S, h2):
        return mp.sqrt((S*S - s.m**2*h2)*(S*S - s.n**2*h2))/(S*mp.sqrt(4*S*S - h2))*s.e(S)

    def bd(s, q, h2):
        return s.m*s.n*h2/(q*q - h2/4) + (2*q*q + h2*(mp.mpf(1)/2 - s.m**2 - s.n**2))/(8*s.al*s.be)


class HT(_Chain):
    """Fixed-offset Hahn chain at the node x (class data e = sqrt(1-s), v = 0)."""
    def __init__(s, M_, S_, x):
        s.m, s.n = mp.mpf(sp.N(M_, 40)), mp.mpf(sp.N(S_, 40)); s.x = mp.mpf(sp.N(x, 40))
        s.qs = mp.sqrt(1 - s.x**2); s.tau = 1 if s.x > 0 else -1

    def e(s, q): return mp.sqrt(1 - q*q)
    def tu(s, q): return abs(s.x)/s.e(q)
    def D(s, q): return 1 - s.x**2 - q*q

    def a(s, S, h2):
        return mp.sqrt((S*S - s.m**2*h2)*(S*S - s.n**2*h2))/(S*mp.sqrt(4*S*S - h2))*mp.sqrt(1 - S*S)

    def bd(s, q, h2): return s.m*s.n*h2/(q*q - h2/4)


HS = (mp.mpf('0.004'), mp.mpf('0.002'), mp.mpf('0.001'))


def airy_cs(sv, z, h, K=80):
    """C, S, C', S' at s = sv for Y'' = (z + h s) Y (C(0)=1, C'(0)=0, S(0)=0, S'(0)=1), power series."""
    out = []
    for c0, c1 in ((1, 0), (0, 1)):
        c = [mp.mpf(c0), mp.mpf(c1)]
        for k in range(K):
            c.append((z*c[k] + (h*c[k - 1] if k else 0))/((k + 2)*(k + 1)))
        out += [mp.polyval(c[::-1], sv), mp.polyval([k*c[k] for k in range(1, len(c))][::-1], sv)]
    return out[0], out[2], out[1], out[3]          # C, S, C', S'


def psi_disc(T, q, hs=HS):
    zq = T.zeta(q); gq = T.g(q, zq); vals = []
    for h in hs:                                   # A[g,0](q; h)
        A = -T.tau*(T.x - T.bd(q, h*h))*gq
        for sg in (1, -1):
            zz = T.zeta(q + sg*h)
            A += T.a(q + sg*h/2, h*h)*T.g(q + sg*h, zz)*airy_cs((zz - zq)/h, zq, h)[0]
        vals.append(A/h**2)
    r = [(4*vals[i + 1] - vals[i])/3 for i in range(2)]
    A2 = (16*r[1] - r[0])/15
    zp = (mp.acos(T.tu(q))/mp.sqrt(-zq)) if zq < 0 else (mp.acosh(T.tu(q))/mp.sqrt(zq))
    return -gq*A2/zp, zq


def cheb(f_vals, lo, hi):
    nc = len(f_vals)
    c = [2*mp.fsum(f_vals[j]*mp.cos(mp.pi*k*(j + mp.mpf(1)/2)/nc) for j in range(nc))/nc for k in range(nc)]
    c[0] /= 2
    dc = [mp.mpf(0)]*(nc + 1)
    for k in range(nc - 1, 0, -1):
        dc[k - 1] = dc[k + 1] + 2*k*c[k]
    dc[0] /= 2
    def clen(cs, q):
        y = (2*q - lo - hi)/(hi - lo); b1 = b2 = mp.mpf(0)
        for ck in cs[:0:-1]:
            b1, b2 = 2*y*b1 - b2 + ck, b1
        return y*b1 - b2 + cs[0]
    return (lambda q: clen(c, q)), (lambda q: 2/(hi - lo)*clen(dc[:nc], q)), abs(c[-1])


class OlverB:
    """B_tau(q) = q_1/g on [q* - half, q* + half] (lem:airy2-olver): psi, phi = zeta/(q-q*) on Chebyshev nodes."""
    def __init__(s, T, half, nc=28):
        lo, hi = T.qs - half, T.qs + half; s.T = T
        nodes = [(lo + hi)/2 + (hi - lo)/2*mp.cos(mp.pi*(j + mp.mpf(1)/2)/nc) for j in range(nc)]
        ps, ph = [], []
        for q in nodes:
            p_, z = psi_disc(T, q); ps.append(p_); ph.append(z/(q - T.qs))
        s.psi, _, s.tail_psi = cheb(ps, lo, hi)
        s.phi, s.dphi, s.tail_phi = cheb(ph, lo, hi)
        s.gl = mp.calculus.quadrature.GaussLegendre(mp.mp).get_nodes(0, 1, 4, mp.mp.prec)

    def zeta(s, q): return (q - s.T.qs)*s.phi(q)

    def B(s, q):
        qs_ = s.T.qs; tot = mp.mpf(0)
        for wv, wt in s.gl:
            qq = qs_ + (q - qs_)*wv*wv
            tot += wt*s.psi(qq)*(s.phi(qq) + (qq - qs_)*s.dphi(qq))/mp.sqrt(s.phi(qq))
        return tot/mp.sqrt(s.phi(q))


def flux_disc(T, OB, q, h):
    """F(q;h) = a(q;h^2)(A_- B_+ - B_- A_+) for P = g, Q = h g B (Airy-form branch at the links q -+ h/2)."""
    zq = T.zeta(q); v = {}
    for sg in (1, -1):
        qq = q + sg*h/2; zz = T.zeta(qq); gg = T.g(qq, zz)
        C, S, Cp, Sp_ = airy_cs((zz - zq)/h, zq, h)
        Q = h*gg*OB.B(qq)
        v[sg] = (gg*C + Q*Cp, gg*S + Q*Sp_)
    return T.a(q, h*h)*(v[-1][0]*v[1][1] - v[-1][1]*v[1][0])


def flux_F2(T, OB, q, hs=HS):
    """[h^2] F(q;h) by Richardson (F = 1 + h^2 F_2 + O(h^4))."""
    v = [(flux_disc(T, OB, q, h) - 1)/h**2 for h in hs]
    r = [(4*v[i + 1] - v[i])/3 for i in range(2)]
    return r[1]


def env(t):                                        # Olver's M/E for Ai
    return mp.sqrt(2)*mp.airyai(t) if t >= mp.mpf('-0.36605') else mp.sqrt(mp.airyai(t)**2 + mp.airybi(t)**2)


def exact_6j(m_, n_, abg, N, qhi, dps=90):
    """Natural-gauge exact 6j chain Y_k = sqrt(2k+1) Psi_k (seed from sympy at k_min, exact recurrence)."""
    with mp.workdps(dps):
        Ar, Br, Cr = (R(v)*N for v in abg)
        a6, b6, c6 = Ar - R(n_ + 1, 2), Br - R(m_ + 1, 2), Cr - R(1, 2)
        Psi = lambda f: (-1)**int(a6 + b6 + c6 + f + m_)*2*sp.sqrt(Ar*Br)*wigner_6j(a6, b6, c6, b6 + m_, a6 + n_, f)
        kmin = max(abs(R(m_)), abs(R(n_)))
        Am, Bm = mp.mpf(sp.N(Ar, dps)), mp.mpf(sp.N(Br, dps)); X = (Am**2 + Bm**2 - mp.mpf(sp.N(Cr, dps))**2)/(2*Am*Bm)
        mf, nf, k0 = (mp.mpf(sp.N(v, dps)) for v in (R(m_), R(n_), kmin))
        eta = lambda k: mp.sqrt((1 - k*k/(4*Am*Am))*(1 - k*k/(4*Bm*Bm)))
        A = lambda k: mp.sqrt((k*k - mf*mf)*(k*k - nf*nf))/(k*mp.sqrt(4*k*k - 1))*eta(k)
        B = lambda k: (2*k*(k + 1) + 1 - mf*mf - nf*nf)/(8*Am*Bm) + (mf*nf/(k*(k + 1)) if k > 0 else 0)
        Y = [mp.sqrt(2*k0 + 1)*mp.mpf(sp.N(Psi(kmin), dps))]
        Y.append(mp.sqrt(3)*mp.mpf(sp.N(Psi(1), dps)) if kmin == 0 else (X - B(k0))*Y[0]/A(k0 + 1))
        for j in range(1, int(qhi*N - k0) + 1):
            k = k0 + j; Y.append(((X - B(k))*Y[j] - A(k)*Y[j - 1])/A(k + 1))
        jt = len(Y) - 3
        dev = abs(Y[jt]/mp.sqrt(2*(k0 + jt) + 1) - mp.mpf(sp.N(Psi(kmin + jt), dps)))
    return Y, kmin, (Ar, Br, Cr), dev


def exact_3j(M_, S_, xr, N, qhi, dps=100):
    """Natural-gauge Hahn chain Y_L = sqrt(2L+1) Phi_L at the node 2 mu/N nearest xr (eq:PhiDef)."""
    M_, S_ = R(M_), R(S_)
    with mp.workdps(dps):
        jm, jp = (N - 1 - S_)/2, (N - 1 + S_)/2
        off = (jm + M_/2) % 1
        mu = off + int(round(float(xr*N/2 - off)))
        X = mp.mpf(sp.N(2*mu/N, dps)); h = mp.mpf(sp.N(1/R(N), dps))
        km = max(abs(M_), abs(S_))
        Y3 = lambda L: mp.sqrt(2*mp.mpf(sp.N(L, dps)) + 1)*mp.mpf(sp.N(
            (-1)**int(2*L + jp - mu - M_/2)*sp.sqrt(N)*wigner_3j(L, jm, jp, M_, mu - M_/2, -mu - M_/2), dps))
        Mf, Sf, k0 = (mp.mpf(sp.N(v, dps)) for v in (M_, S_, km))
        A = lambda k: mp.sqrt((k*k - Mf*Mf)*(k*k - Sf*Sf))/(k*mp.sqrt(4*k*k - 1))*mp.sqrt(1 - (k*h)**2)
        B = lambda k: Mf*Sf/(k*(k + 1)) if k > 0 else mp.mpf(0)
        Y = [Y3(km), Y3(km + 1)]
        for j in range(1, int(qhi*N - k0) + 1):
            k = k0 + j; Y.append(((X - B(k))*Y[j] - A(k)*Y[j - 1])/A(k + 1))
        dev = abs(Y[len(Y) - 3] - Y3(km + len(Y) - 3))
    return X, Y, km, dev


# ---------------- exact chain coefficients (formerly p3_chain_coefficients.py) ----------------


class Chain:
    """x A_{k+1} y_{k+1} ... natural gauge: A_k, B_k exact; sigma_k = k^2 h^2; hard edge e(sigma) = 0."""
    def __init__(self, kind, m_, n_, N, ab=None):
        self.kind, self.N = kind, mp.mpf(N); self.h = 1/self.N
        self.m, self.n = mp.mpf(m_), mp.mpf(n_); self.ab = ab
        self.kmin = max(abs(self.m), abs(self.n))

    def e(self, sg):
        if self.kind == 'hahn':
            return mp.sqrt(max(1 - sg, 0))
        if self.kind == 'bad':
            return max(1 - sg, 0)**2
        a, b = self.ab
        return mp.sqrt(max((1 - sg/(4*a*a))*(1 - sg/(4*b*b)), 0))

    def vq(self, q2):                       # v(q^2, h^2) of the class (b_0 = v(q^2, 0))
        if self.kind == 'hahn':
            return mp.mpf(0)
        if self.kind == 'bad':
            return q2
        a, b = self.ab
        return (2*q2 + self.h**2*(mp.mpf(1)/2 - self.m**2 - self.n**2))/(8*a*b)

    def A(self, k):
        k = mp.mpf(k)
        if k <= self.kmin:
            return mp.mpf(0)
        w = mp.sqrt(max((k*k - self.m**2)*(k*k - self.n**2), 0))/(k*mp.sqrt(4*k*k - 1))
        return w*self.e((k*self.h)**2)

    def B(self, k):
        k = mp.mpf(k)
        w = self.m*self.n/(k*(k + 1)) if k > 0 else mp.mpf(0)
        return w + self.vq(((k + mp.mpf(1)/2)*self.h)**2)

    def Psi(self, sg, tau):
        a_ = self.e(sg); b0 = self.vq(sg)
        return b0 + tau*a_

    def edge_sigma(self):
        if self.kind == 'hahn' or self.kind == 'bad':
            return mp.mpf(1)
        a, b = self.ab
        return 4*min(a, b)**2

    def turn(self, x):
        """smallest q>0 with D(q,x)=e^2-(x-b0)^2 = 0 (principal data), type tau; None if absent."""
        f = lambda q: self.e(q*q)**2 - (x - self.vq0(q*q))**2
        qmax = mp.sqrt(self.edge_sigma())
        qs = [qmax*j/4000 for j in range(1, 4000)]
        prev = f(qs[0])
        if prev <= 0:
            return None
        for q0, q1 in zip(qs, qs[1:]):
            if f(q1) <= 0:
                qt = mp.findroot(f, (q0, q1), solver='anderson', tol=1e-25)
                tau = 1 if x - self.vq0(qt*qt) > 0 else -1
                return qt, tau
        return None

    def vq0(self, q2):                      # b_0(q) = v(q^2, 0)
        if self.kind == 'hahn':
            return mp.mpf(0)
        if self.kind == 'bad':
            return q2
        a, b = self.ab
        return q2/(4*a*b)


def setup(kind, m_, n_, N, ab=None):
    """Chain with N shifted so that the hard edge k_max + 1 = N sqrt(edge sigma) lies on the rank lattice."""
    ch0 = Chain(kind, m_, n_, N, ab); em = mp.sqrt(ch0.edge_sigma())
    return Chain(kind, m_, n_, N + ((ch0.kmin - em*N) % 1)/em, ab)
