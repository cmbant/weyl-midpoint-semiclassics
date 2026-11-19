"""Helper (not a verifier): principal and second-order data of an even centered
Wigner--Jacobi chain from its deformation functions (e, v), in mpmath.

Implements eq:class-principal-data, eq:generic-second-weyl-density with
eq:class-a2b2, and eq:class-second-data of the article:
    p, S, Theta_q, Sigma_2, rho_2, Gamma_2, and the Darboux phase c_ab.
Used by verify_bulk_zero_rates.py and verify_rank_airy_second_order.py.
"""
import mpmath as mp
import sympy as sp

q, x, s, t = sp.symbols('q x s t', real=True)


def cab(a, b, T):
    """eq:wigner-darboux-phase-coeff."""
    a, b = mp.mpf(a), mp.mpf(b)
    return ((a*a + b*b - mp.mpf(1)/4)*mp.cos(T) - 2*a*b)/(2*mp.sin(T))


def dcab(a, b, T):
    a, b = mp.mpf(a), mp.mpf(b)
    return (-(a*a + b*b - mp.mpf(1)/4) + 2*a*b*mp.cos(T))/(2*mp.sin(T)**2)


class Chain:
    def __init__(self, e_expr, v_expr, a, b):
        self.a, self.b = a, b
        e = e_expr.subs(s, q**2)
        b0 = v_expr.subs({s: q**2, t: 0})
        dtv = sp.diff(v_expr, t).subs({s: q**2, t: 0})
        a0 = e/2
        cp = (x - b0)/e
        P = sp.acos(cp)
        sP = sp.sqrt(1 - cp**2)
        g = -sp.log(a0*sP)/2
        pq, pqq = sp.diff(P, q), sp.diff(P, q, 2)
        gq, gqq = sp.diff(g, q), sp.diff(g, q, 2)
        a0q, a0qq = sp.diff(a0, q), sp.diff(a0, q, 2)
        A, B = sp.nsimplify(a), sp.nsimplify(b)
        a2 = a0/q**2*(sp.Rational(1, 8) - (A**2 + B**2)/2)
        b2 = A*B/q**2 + dtv
        Sig = (a0*(gq**2 + gqq)*cp - a0*gq*pq*sP - a0/4*pq**2*cp - a0/3*pqq*sP
               + 2*a2*cp + a0q*gq*cp - a0q/2*pq*sP + a0qq/4*cp + b2)/(2*a0*sP)
        self.Sig = sp.lambdify((q, x), Sig, 'mpmath')
        self.p = sp.lambdify((q, x), P, 'mpmath')
        self.D = sp.lambdify((q, x), e**2 - (x - b0)**2, 'mpmath')
        self.e = sp.lambdify(q, e, 'mpmath')
        self.b0 = sp.lambdify(q, b0, 'mpmath')

    def S(self, Q, X):
        return mp.quad(lambda u: self.p(u, X), [0, Q])

    def Th(self, Q, X):
        return self.S(Q, X)/Q

    def rho2(self, Q, X):
        T = self.Th(Q, X)
        Tq = (self.p(Q, X) - T)/Q
        return self.Sig(Q, X) - (dcab(self.a, self.b, T)*Tq/Q - cab(self.a, self.b, T)/Q**2)

    def Gam2(self, Q, X, q0=mp.mpf('2e-3')):
        """Gamma_2 = int_0^Q rho_2; rho_2 is analytic in q^2, so [0,q0] is
        integrated from an even quadratic fit on q0, 2q0, 3q0."""
        r = [self.rho2(q0*k, X) for k in (1, 2, 3)]
        M = mp.matrix([[1, (q0*k)**2, (q0*k)**4] for k in (1, 2, 3)])
        c = mp.lu_solve(M, mp.matrix(r))
        I0 = c[0]*q0 + c[1]*q0**3/3 + c[2]*q0**5/5
        return I0 + mp.quad(lambda u: self.rho2(u, X), [q0, Q/2, Q])

    def rho2_at0(self, X, q0=mp.mpf('2e-3')):
        r = [self.rho2(q0*k, X) for k in (1, 2, 3)]
        M = mp.matrix([[1, (q0*k)**2, (q0*k)**4] for k in (1, 2, 3)])
        return mp.lu_solve(M, mp.matrix(r))[0]


def hahn(M, sig):
    return Chain(sp.sqrt(1 - s), sp.Integer(0), M, sig)


def racah(m, n, al, be):
    al, be = sp.nsimplify(al), sp.nsimplify(be)
    e = sp.sqrt(1 - s/(4*al**2))*sp.sqrt(1 - s/(4*be**2))
    v = (2*s + t*(sp.Rational(1, 2) - sp.nsimplify(m)**2 - sp.nsimplify(n)**2))/(8*al*be)
    return Chain(e, v, m, n)
