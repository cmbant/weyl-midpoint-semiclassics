#!/usr/bin/env python3
"""
Symbolic identities and numerical sympy-wigner_6j verification for Joint Rank-Position Uniformity (Centered Racah / 6j):
  Part A [symbolic identity checks]:
    1. Exact symbolic derivative d/drho S_6(rho, x) - omega_6(rho, x) == 0.
    2. Exact polynomial proof D_6^{5/2}*(d/drho tilde_Gamma^{(6)} - Sigma_2^{(6)}) == 0.
    3. Hahn degeneration limits tilde_Gamma^{(6)} -> tilde_Gamma^{(3)} and Lam^{-2} C_0^{(6)} -> C_0^{(3)}.
    4. Theorem 6.1(c) isosceles (A = B) pole-cancellation identities for A_2, A_3, and F_1(tv, 0) == F_2(tv, 0).
    5. All-chamber (4 chambers of (m, n)) fixed-rank O(S^{-2}) match vs Paper II cor:uniformNLO.
  Part B [numerical cross-checks]:
    Finite-precision check of square-root-free recurrence Phi_f(A, B, C) vs sympy wigner_6j,
    plus exact tetrahedral corner swap {a, b, c; b+m, a+n, f} == {a, a+n, f; b+m, b, c}.
"""

import math
import sympy as sp
from sympy.physics.wigner import wigner_6j


def epsilon_phase(a, b):
    return (-1.0 if (int(round(a - b)) % 2 == 1) else 1.0) if b <= a else 1.0


def edge_factor_R(p, X):
    p_int = int(round(p))
    return 1.0 if p_int <= 1 else math.sqrt(math.prod(1.0 + (-0.5 * (p_int - 1) + k) / X for k in range(p_int)))


def edge_factor_6j(A, B, C, m, n):
    p_min, p_max = int(round(abs(m - n))), int(round(abs(m + n)))
    return edge_factor_R(p_min, A - B + C) * edge_factor_R(p_min, -A + B + C) * edge_factor_R(p_max, A + B - C) * edge_factor_R(p_max, A + B + C)


def wigner_d_min_seed(a, b, theta):
    k0, p_p, p_m = max(abs(a), abs(b)), int(round(abs(a - b))), int(round(abs(a + b)))
    if k0 == 0:
        return 1.0
    pref = epsilon_phase(a, b) * math.sqrt(math.factorial(int(round(2 * k0))) / (math.factorial(p_p) * math.factorial(p_m)))
    return pref * (math.sin(0.5 * theta) ** p_p) * (math.cos(0.5 * theta) ** p_m)


def exact_phi_6j_chain(A, B, C, f_max, m, n):
    """Exact Phi_f(A, B, C) via Paper II eq:6jrec and lem:sqrtfreecarrier."""
    f0, x, out = max(abs(m), abs(n)), (A * A + B * B - C * C) / (2.0 * A * B), {}
    if f0 == 0:
        out[0] = 1.0
        if f_max >= 1:
            out[1] = (x - 1.0 / (8.0 * A * B)) / math.sqrt((1.0 - 1.0 / (4.0 * A * A)) * (1.0 - 1.0 / (4.0 * B * B)))
        f = 1.0
    else:
        d0 = wigner_d_min_seed(n, m, math.acos(max(-1.0, min(1.0, x))))
        pi_v, q = 1.0, f0
        while q > 0.25:
            pi_v, q = pi_v * (1.0 - (q * q) / (4.0 * A * A)) * (1.0 - (q * q) / (4.0 * B * B)), q - 1.0
        out[f0], f = edge_factor_6j(A, B, C, m, n) * d0 / math.sqrt(pi_v), float(f0)
    while f < f_max - 1e-9:
        qf, fp1 = (2.0 * f * (f + 1.0) + 1.0 - m * m - n * n) / (8.0 * A * B), f + 1.0
        lhs = (2.0 * f + 1.0) * (f * (f + 1.0) * (x - qf) - m * n) * out[f]
        if f > f0 + 1e-9:
            lhs -= (f + 1.0) * math.sqrt((f * f - m * m) * (f * f - n * n) * (1.0 - (f * f) / (4.0 * A * A)) * (1.0 - (f * f) / (4.0 * B * B))) * out[f - 1.0]
        denom = f * math.sqrt((fp1 * fp1 - m * m) * (fp1 * fp1 - n * n) * (1.0 - (fp1 * fp1) / (4.0 * A * A)) * (1.0 - (fp1 * fp1) / (4.0 * B * B)))
        out[fp1], f = lhs / denom, fp1
    return out


def run_part_a_symbolic():
    print("=== Part A: Exact Symbolic Identities (Centered Racah / 6j) [symbolic checks] ===")
    a, b, x, rho, m, n, f = sp.symbols("a b x rho m n f", real=True)
    a2, b2, omx2, s, Sig, tau = a**2, b**2, 1 - x**2, rho**2, m**2 + n**2, m * n
    gam2 = a2 + b2 - 2 * a * b * x
    D6, D6_p = omx2 - gam2 * s / (4 * a2 * b2), -gam2 * rho / (2 * a2 * b2)
    A1 = ((4 * Sig - 1) * x - 8 * tau) / (8 * omx2)
    num_A2 = (
        16 * a2 * b2 * (a2 + b2) * x * (x**2 - 7) + 16 * (a * b) ** 3 * (7 * x**2 + 5)
        + s * (24 * (a2**2 + b2**2) * x - 16 * a * b * (a2 + b2) * (2 * x**2 + 1) - 8 * a2 * b2 * x * (x**2 - 7))
        + s**2 * (-6 * (a2 + b2) * x + 3 * a * b * (3 * x**2 + 1))
    )
    A2 = num_A2 / (-96 * a2 * b2 * (4 * a2 - s) * (4 * b2 - s) * omx2)
    A3 = -5 * (a - b * x) * (b - a * x) * gam2 / (384 * a2**2 * b2**2 * omx2)
    lhs_rat = (
        A1 * (-D6**3 / s + sp.Rational(1, 2) * D6_p * D6**2 / rho) + sp.diff(A2 * rho, rho) * D6**2
        - sp.Rational(1, 2) * A2 * rho * D6_p * D6 + 3 * A3 * s * D6 - sp.Rational(3, 2) * A3 * rho**3 * D6_p
    )
    E_s, U_s = (4 * a2 - s) * (4 * b2 - s) / (16 * a2 * b2), x - s / (4 * a * b)
    dphi_part = 2 * gam2 * E_s - 2 * a * (a - b * x) * (1 - s / (4 * b2)) - 2 * b * (b - a * x) * (1 - s / (4 * a2))
    dom_part = s * (sp.diff(U_s, rho) * E_s - sp.Rational(1, 2) * U_s * sp.diff(E_s, rho)) / rho
    assert sp.expand(dphi_part - 2 * a * b * dom_part) == 0
    print("  [PASS] Exact derivative d/drho S_6(rho, x) - omega_6(rho) == 0 verified symbolically")

    E_r, E_rr = sp.diff(E_s, rho), sp.diff(E_s, rho, 2)
    P_num, g_r = -(sp.diff(U_s, rho) * E_s - sp.Rational(1, 2) * U_s * E_r) / E_s, -D6_p / (4 * D6)
    rhs_rat = (
        (U_s / 2) * (g_r**2 + sp.diff(g_r, rho)) * D6**2 - sp.Rational(1, 2) * g_r * P_num * D6**2 - (U_s / 8) * P_num**2 * D6
        - sp.Rational(1, 6) * (D6 * sp.diff(P_num, rho) - sp.Rational(1, 2) * D6_p * P_num) * D6
        + (sp.Rational(1, 8) - Sig / 2) / s * U_s * D6**2 + (E_r * U_s / (4 * E_s)) * g_r * D6**2 - (E_r / (8 * E_s)) * P_num * D6**2
        + (2 * E_rr * E_s - E_r**2) * U_s / (32 * E_s**2) * D6**2 + (tau / s + (sp.Rational(1, 2) - Sig) / (8 * a * b)) * D6**2
    )
    assert sp.expand(sp.numer(sp.together(lhs_rat - rhs_rat))) == 0
    print("  [PASS] Polynomial check D_6^{5/2}*(d/drho tilde_Gamma^{(6)} - Sigma_2^{(6)}) == 0")

    Lam, q, tv = sp.symbols("Lam q tv", positive=True)
    sub_h = {a: 1 / (2 * Lam), b: 1 - 1 / (2 * Lam), rho: q / Lam}
    assert sp.simplify(sp.limit(A2.subs(sub_h) / (Lam**2), Lam, sp.oo) - x * (7 - x**2 - 6 * q**2) / (24 * omx2 * (1 - q**2))) == 0
    assert sp.simplify(sp.limit(A3.subs(sub_h) / (Lam**4), Lam, sp.oo) - 5 * x / (24 * omx2)) == 0
    print("  [PASS] Hahn degeneration limit tilde_Gamma^{(6)} -> tilde_Gamma^{(3)} verified")
    # Hahn limit of the contact multiplier (eq:C0-6 -> eq:C0-3): Lam^{-2} C_0^{(6)} -> C_0^{(3)} with p_min -> p_+, p_max -> p_-,
    # delta = alpha - beta = 1/Lam - 1; the scaling (2 Lam)^{-2} would give C_0^{(3)}/4 (negative control).
    pmn, pmx, f0 = sp.symbols("pmn pmx f0", positive=True)
    dl = 1 / Lam - 1
    C06 = (2 * f0 + 1) / (24 * (1 - dl**2)**2) * (-(1 + dl**2) + 2 * dl**2 * (pmn - 1) * (2 * f0 + 1) / (1 - x)
                                                  + 2 * (pmx - 1) * (2 * f0 + 1) / (1 + x))
    C03 = (2 * f0 + 1) / 48 * (-1 + (pmn - 1) * (2 * f0 + 1) / (1 - x) + (pmx - 1) * (2 * f0 + 1) / (1 + x))
    lim = sp.limit(C06 / Lam**2, Lam, sp.oo)
    assert sp.simplify(lim - C03) == 0
    assert sp.simplify(sp.limit(C06 / (2 * Lam)**2, Lam, sp.oo) - C03 / 4) == 0     # the old scaling is off by exactly 4
    print("  [PASS] Hahn limit Lam^-2 C_0^(6) -> C_0^(3) verified ((2 Lam)^-2 scaling rejected)")

    A2_iso = sp.factor(A2.subs({a: sp.Rational(1, 2), b: sp.Rational(1, 2), x: 1 - 2 * tv}))
    A3_iso = sp.simplify(A3.subs({a: sp.Rational(1, 2), b: sp.Rational(1, 2), x: 1 - 2 * tv}))
    A2_closed = ((3 - 13 * tv + 4 * tv**2) - (6 - 22 * tv + 4 * tv**2) * s + 3 * (1 - 3 * tv) * s**2) / (24 * (1 - tv) * (1 - s) ** 2)
    assert sp.simplify(A2_iso - A2_closed) == 0 and sp.simplify(A3_iso + 5 * tv**2 / (6 * (1 - tv))) == 0
    F1_s0 = sp.simplify(((4 * Sig - 1) * (1 - 2 * tv) - 8 * tau) * sp.sqrt(1 - tv) / (16 * (1 - tv)) * (2 * sp.sqrt(1 - tv)))
    F2_s0 = sp.simplify(((Sig - sp.Rational(1, 4)) * (1 - 2 * tv) - 2 * tau) / 2)
    assert sp.simplify(F1_s0 - F2_s0) == 0
    print("  [PASS] Thm 6.1(c) isosceles A_2, A_3, and F_1(tv,0)==F_2(tv,0) pole cancellations verified symbolically")

    delta = sp.Symbol("delta", real=True)
    al, be, sin_th, lam, k = (1 + delta) / 2, (1 - delta) / 2, sp.sqrt(omx2), f * (f + 1), f + sp.Rational(1, 2)
    rho_AB = (1 / al**2 + 1 / be**2) / 8
    c0, c1 = (1 - Sig) / (8 * al * be) + rho_AB * tau, 1 / (4 * al * be) - rho_AB * x
    P_f = -c0 + c1 * (2 - 2 * Sig + 4 * tau * x - lam * omx2) / (3 * omx2)
    Q_f = rho_AB * lam / 2 + c1 * (tau - lam * x) / (3 * omx2)
    pair_min = (1 / al**2 + 1 / be**2 - (1 + x) / (al * be)) / (1 - x) ** 2
    pair_max = (1 / al**2 + 1 / be**2 + (1 - x) / (al * be)) / (1 + x) ** 2
    th2 = -(-1 / (4 * al * be) + rho_AB * x) / (3 * sin_th)
    W6_2 = ((2 * x * sin_th * th2) / omx2 + (al**2 + be**2 - 2 * al * be * x) / (4 * al**2 * be**2 * omx2)) / 4
    rho20 = (
        rho_AB * (9 * x - x**3 - 8 * Sig * x + 4 * tau * x**2 + 12 * tau)
        + (8 * Sig - 16 * tau * x - 2 * x**2 - 6) / (4 * al * be) + 12 * (sp.Rational(1, 2) - Sig) * omx2 / (8 * al * be)
    ) / (12 * sin_th**3)
    for p_mn, p_mx, f0 in [(m - n, m + n, m), (n - m, m + n, n), (n - m, -m - n, -m), (m - n, -m - n, -n)]:
        e2_6 = -p_mn * (p_mn**2 - 1) * pair_min / 48 - p_mx * (p_mx**2 - 1) * pair_max / 48
        om_s, c2f0, omd2 = (p_mn * (1 + x) - p_mx * (1 - x)) / 2, 2 * f0 + 1, 1 - delta**2
        C0_6 = (c2f0 / (24 * omd2**2)) * (-(1 + delta**2) + 2 * delta**2 * (p_mn - 1) * c2f0 / (1 - x) + 2 * (p_mx - 1) * c2f0 / (1 + x))
        assert sp.simplify((th2 * k**2 + rho20) - (-P_f / sin_th)) == 0
        assert sp.simplify((W6_2 * k**2 + C0_6 + th2 * k**2 * om_s / sin_th) - (Q_f - e2_6 - P_f * om_s / omx2)) == 0
    print("  [PASS] Fixed-rank O(S^{-2}) expansion matches Paper II cor:uniformNLO in all 4 chambers")


def run_part_b_exact_crosscheck():
    print("\n=== Part B: Numerical Recurrence Check & Exact Tetrahedral Duality [mixed checks] ===")
    cases = [
        (12, 9, 10, 0, 0, 4), (14, 10, 11, 2, -1, 6), (11, 13, 12, -1, 2, 6),
        (sp.Rational(23, 2), sp.Rational(17, 2), 9, sp.Rational(3, 2), sp.Rational(-1, 2), sp.Rational(11, 2)),
        (sp.Rational(21, 2), sp.Rational(21, 2), 3, sp.Rational(-3, 2), sp.Rational(-1, 2), sp.Rational(9, 2)),
    ]
    max_err = 0.0
    for a_s, b_s, c_s, m_s, n_s, f_s in cases:
        A, B, C = float(a_s + sp.Rational(n_s + 1, 2)), float(b_s + sp.Rational(m_s + 1, 2)), float(c_s + sp.Rational(1, 2))
        phis = exact_phi_6j_chain(A, B, C, float(f_s), float(m_s), float(n_s))
        f_cur = max(abs(m_s), abs(n_s))
        while f_cur <= f_s:
            phi_sp = ((-1) ** int(a_s + b_s + c_s + f_cur + n_s)) * 2.0 * math.sqrt(A * B) * float(wigner_6j(a_s, b_s, c_s, b_s + m_s, a_s + n_s, f_cur))
            max_err = max(max_err, abs(phis[float(f_cur)] - phi_sp))
            f_cur += 1
    assert max_err < 1e-13
    dual_left = wigner_6j(sp.Rational(41, 2), sp.Rational(41, 2), 2, 22, 20, sp.Rational(29, 2))
    dual_right = wigner_6j(sp.Rational(41, 2), 20, sp.Rational(29, 2), 22, sp.Rational(41, 2), 2)
    assert sp.simplify(dual_left - dual_right) == 0
    print(f"  [PASS] Max |Phi_f - sympy.wigner_6j| = {max_err:.3e}; tetrahedral corner swap exact (diff=0)")


if __name__ == "__main__":
    run_part_a_symbolic()
    run_part_b_exact_crosscheck()
    print("\nALL 6J SYMBOLIC, NUMERICAL & DUALITY CHECKS PASSED.")
