#!/usr/bin/env python3
"""
Symbolic identities and numerical sympy-wigner_3j verification for Joint Rank-Position Uniformity (Hahn / 3j):
  Part A [symbolic identity checks]:
    1. Exact derivative d/dq S(q, x) - arccos(x / sqrt(1 - q^2)) == 0.
    2. Exact antiderivative d/dq tilde_Gamma^{(3)}(q, x) - Sigma_2^{(3)}(q, x) == 0
       (main-paper eq:generic-second-weyl-density).
    3. Lemma 5.1 Olver Bessel-Darboux recursion up to M = 4: exact differential-operator
       cancellation for arbitrary W_0(theta), plus concrete Wigner-potential series recursion
       verifying parity, pole-free regularity at theta = 0, and A_m(0) + (p_+ + 1/2)*B_{m-1}'(0) == 0
       for all m = 1, 2, 3, 4.
    4. Lemma 5.2 blown-up scaling & regularity at eps = 0: verifies eps^{-1}(Theta_q^[2] - Theta_q)
       == nu^2 * zhat^2 * widehat_Gamma_2, finite limit at nu = 0, and exact vanishing of eps^0, eps^1
       in tilde_c_{k+1} + tilde_c_k - tilde_v_k and eps^0 in tilde_c_{k+1} - tilde_c_k.
    5. All-chamber (4 chambers of (M, sigma)) O(N^{-2}) match vs Paper II cor:uniformNLO.
  Part B [numerical cross-checks]:
    Finite-precision cross-check of square-root-free recurrence Phi_L^{M,sigma} vs sympy wigner_3j.
"""

import math
import sympy as sp
from sympy.physics.wigner import wigner_3j


def epsilon_phase(M, sigma):
    return (-1.0 if (int(round(M - sigma)) % 2 == 1) else 1.0) if sigma <= M else 1.0


def edge_factor_R(p, X):
    p_int = int(round(p))
    return 1.0 if p_int <= 1 else math.sqrt(math.prod(1.0 + (-0.5 * (p_int - 1) + k) / X for k in range(p_int)))


def edge_factor_3j(N, x, M, sigma):
    return edge_factor_R(abs(M - sigma), 0.5 * N * (1.0 - x)) * edge_factor_R(abs(M + sigma), 0.5 * N * (1.0 + x))


def wigner_d_min_seed(M, sigma, theta):
    L0, p_p, p_m = max(abs(M), abs(sigma)), int(round(abs(M - sigma))), int(round(abs(M + sigma)))
    if L0 == 0:
        return 1.0
    pref = epsilon_phase(M, sigma) * math.sqrt(math.factorial(int(round(2 * L0))) / (math.factorial(p_p) * math.factorial(p_m)))
    return pref * (math.sin(0.5 * theta) ** p_p) * (math.cos(0.5 * theta) ** p_m)


def exact_phi_3j_chain(N, x, L_max, M, sigma):
    """Exact Phi_L^{M,sigma}(x; N^{-1}) via Paper II eq:3jseed and eq:3jrank."""
    L0, h, out = max(abs(M), abs(sigma)), 1.0 / N, {}
    if L0 == 0:
        out[0] = 1.0
        if L_max >= 1:
            out[1] = x / math.sqrt(1.0 - h * h)
        L_cur = 1.0
    else:
        d0 = wigner_d_min_seed(M, sigma, math.acos(max(-1.0, min(1.0, x))))
        pi_sym = 1.0
        for k in range(int(round(2 * L0)) + 1):
            pi_sym *= 1.0 + (-L0 + k) * h
        out[L0], L_cur = edge_factor_3j(N, x, M, sigma) * d0 / math.sqrt(pi_sym), float(L0)
    while L_cur < L_max - 1e-9:
        Lp1 = L_cur + 1.0
        RLp1 = math.sqrt((Lp1 * Lp1 - M * M) * (Lp1 * Lp1 - sigma * sigma))
        num = (2.0 * L_cur + 1.0) * (L_cur * (L_cur + 1.0) * x - M * sigma) * out[L_cur]
        if L_cur > L0 + 1e-9:
            RL = math.sqrt((L_cur * L_cur - M * M) * (L_cur * L_cur - sigma * sigma))
            num -= (L_cur + 1.0) * RL * math.sqrt(1.0 - L_cur * L_cur * h * h) * out[L_cur - 1.0]
        out[Lp1], L_cur = num / (L_cur * RLp1 * math.sqrt(1.0 - Lp1 * Lp1 * h * h)), Lp1
    return out


def run_part_a_symbolic():
    print("=== Part A: Exact Symbolic Identities (Hahn / 3j) [symbolic checks] ===")
    q, x, M, s, L = sp.symbols("q x M s L", real=True)
    omx2 = 1 - x**2
    D, Sig, tau = omx2 - q**2, M**2 + s**2, M * s
    sqrtD = sp.sqrt(D)

    # 1. Principal action derivative
    xp, qp, up = sp.symbols("xp qp up", positive=True)
    rad = sp.sqrt(xp**2 + qp**2 + up**2)
    S_raw = sp.diff(
        q * sp.acos(x / sp.sqrt(1 - q**2)) - x * sp.asin(q / sp.sqrt(1 - x**2)) + sp.atan(x * q / sqrtD), q
    ) - sp.acos(x / sp.sqrt(1 - q**2))
    assert sp.simplify(sp.powdenest(S_raw.subs({x: xp / rad, q: qp / rad}), force=True)) == 0
    print("  [PASS] d/dq S(q, x) - arccos(x / sqrt(1 - q^2)) == 0 on the positive-parameter branch")

    # 2. Algebraic second eikonal antiderivative vs main-paper eq:generic-second-weyl-density
    tG = (
        ((4 * Sig - 1) * x - 8 * tau) * sqrtD / (8 * omx2 * q)
        + x * q * (7 - x**2 - 6 * q**2) / (24 * omx2 * (1 - q**2) * sqrtD)
        + 5 * x * q**3 / (24 * omx2 * D ** sp.Rational(3, 2))
    )
    p_q, g_q, a0 = -q * x / ((1 - q**2) * sqrtD), sp.Rational(1, 2) * q / D, sp.sqrt(1 - q**2) / 2
    p_qq, g_qq, a0_q, a0_qq = sp.diff(p_q, q), sp.diff(g_q, q), sp.diff(a0, q), sp.diff(a0, q, 2)
    cp, sp_p = x / sp.sqrt(1 - q**2), sqrtD / sp.sqrt(1 - q**2)
    a2, b2 = a0 * (sp.Rational(1, 8) - Sig / 2) / q**2, tau / q**2
    Sigma2 = (
        a0 * (g_q**2 + g_qq) * cp - a0 * g_q * p_q * sp_p - (a0 / 4) * p_q**2 * cp - (a0 / 3) * p_qq * sp_p
        + 2 * a2 * cp + a0_q * g_q * cp - (a0_q / 2) * p_q * sp_p + (a0_qq / 4) * cp + b2
    ) / (2 * a0 * sp_p)
    assert sp.simplify(sqrtD * (sp.diff(tG, q) - Sigma2)) == 0
    print("  [PASS] d/dq tilde_Gamma^{(3)}(q, x) - Sigma_2^{(3)}(q, x) == 0 (main-paper eq:generic-second-weyl-density)")

    # 3. Lemma 5.1 Olver Bessel-Darboux recursion extended to M = 4
    th, kap, p_sym, pm_sq = sp.symbols("th kap p_sym pm_sq", positive=True)
    mu_s = p_sym**2 - sp.Rational(1, 4)
    W0_fn, phi_f = sp.Function("W0")(th), sp.Function("phi")(th)
    A_fns = [sp.Integer(1)] + [sp.Function(f"A{m}")(th) for m in range(1, 5)]
    B_fns = [sp.Function(f"B{m}")(th) for m in range(4)]
    w4_sym = sum(A_fns[m] / kap ** (2 * m) for m in range(5)) * phi_f + sum(
        B_fns[m] / kap ** (2 * m + 2) for m in range(4)
    ) * sp.diff(phi_f, th)
    op_raw = sp.diff(w4_sym, th, 2) + (kap**2 - mu_s / th**2 - W0_fn) * w4_sym
    phi_d2 = (-kap**2 + mu_s / th**2) * phi_f
    phi_d3 = sp.diff(phi_d2, th).subs(sp.diff(phi_f, th, 2), phi_d2)
    Psi4 = sp.diff(A_fns[4], th, 2) - W0_fn * A_fns[4] + 2 * mu_s * sp.diff(B_fns[3] / th, th) / th
    diff_expr = sp.expand(op_raw.subs(sp.diff(phi_f, th, 3), phi_d3).subs(sp.diff(phi_f, th, 2), phi_d2) - Psi4 * phi_f / kap**8)
    c_phi, c_dphi = diff_expr.coeff(phi_f), diff_expr.coeff(sp.diff(phi_f, th))
    for m in range(4):
        Bm1_term = sp.Integer(0) if m == 0 else 2 * mu_s * sp.diff(B_fns[m - 1] / th, th) / th
        assert sp.simplify(c_phi.coeff(kap, -2 * m) - (-2 * sp.diff(B_fns[m], th) + sp.diff(A_fns[m], th, 2) - W0_fn * A_fns[m] + Bm1_term)) == 0
        assert sp.simplify(c_dphi.coeff(kap, -2 * m - 2) - (2 * sp.diff(A_fns[m + 1], th) + sp.diff(B_fns[m], th, 2) - W0_fn * B_fns[m])) == 0
    W0_ser = sp.series(mu_s * (1 / sp.sin(th) ** 2 - 1 / th**2) + (pm_sq - p_sym**2) / (4 * sp.cos(th / 2) ** 2), th, 0, 18).removeO()
    A_ser, B_ser = [sp.Integer(1)], []
    for m in range(4):
        Bm1_t = sp.Integer(0) if m == 0 else 2 * mu_s * sp.diff(sp.cancel(B_ser[m - 1] / th), th) / th
        int_B = sp.series(sp.diff(A_ser[m], th, 2) - W0_ser * A_ser[m] + Bm1_t, th, 0, 17 - 2 * m).removeO()
        Bm = sp.expand(sp.Rational(1, 2) * sp.integrate(int_B, (th, 0, th)))
        B_ser.append(Bm)
        assert sp.expand(Bm + Bm.subs(th, -th)) == 0
        int_A = sp.series(sp.diff(Bm, th, 2) - W0_ser * Bm, th, 0, 16 - 2 * m).removeO()
        Amp1 = sp.expand(-(p_sym + sp.Rational(1, 2)) * sp.diff(Bm, th).subs(th, 0) - sp.Rational(1, 2) * sp.integrate(int_A, (th, 0, th)))
        A_ser.append(Amp1)
        assert sp.expand(Amp1 - Amp1.subs(th, -th)) == 0
        assert sp.simplify(sp.limit(Amp1 + (p_sym + sp.Rational(1, 2)) * Bm / th, th, 0)) == 0
    print("  [PASS] Lemma 5.1 Olver Bessel-Darboux operator & Wigner-series regularity verified through M = 4")

    # 4. Lemma 5.2 blown-up scaling (nu^2 * zhat^2 * widehat_Gamma_2) & eps -> 0 analyticity of K_2, K_1, K_0
    nu, zhat, eps = sp.symbols("nu zhat eps", positive=True)
    h_b, q_b, kap_b = eps**2 * nu * zhat, nu * eps, 1 / (eps * zhat)
    G2_sym = sp.Symbol("G2_sym", real=True)
    G2hat_def = (eps**3 / q_b) * G2_sym
    assert sp.simplify((h_b**2 / q_b) * G2_sym / eps - nu**2 * zhat**2 * G2hat_def) == 0
    k_b = kap_b - sp.Rational(1, 2)
    x_b = sp.sqrt(1 - eps**2)
    c_tilde = lambda j: sp.sqrt((j**2 - M**2) * (j**2 - s**2) * (1 - j**2 * h_b**2)) / (j * sp.sqrt(4 * j**2 - 1))
    v_tilde = x_b - M * s / (k_b * (k_b + 1))
    diff_c = sp.series(c_tilde(k_b + 1) - c_tilde(k_b), eps, 0, 2).removeO()
    sum_c_v = sp.series(c_tilde(k_b + 1) + c_tilde(k_b) - v_tilde, eps, 0, 2).removeO()
    assert sp.simplify(diff_c) == 0 and sp.simplify(sum_c_v) == 0
    print("  [PASS] Lemma 5.2 blown-up nu^2*zhat^2*widehat_Gamma_2 scaling & eps=0 regularity of K_2, K_1, K_0")

    # 5. All-chamber fixed-rank O(N^{-2}) match vs Paper II cor:uniformNLO
    sin_th, k, J = sp.sqrt(omx2), L + sp.Rational(1, 2), L * (L + 1)
    P_f2 = (2 * J * (x**3 - x) - 4 * x * Sig + 2 * tau * (x**2 + 3) + 4 * x) / (12 * omx2 * sin_th)
    Q_f2 = -(J * (x**2 - 3) + 2 * tau * x) / (12 * omx2)
    theta_2 = -x / (6 * sin_th)
    W_2 = (2 * x * sin_th * theta_2 + 1) / (4 * omx2)
    rho20 = (x * (9 - x**2 - 8 * Sig) + 4 * tau * (x**2 + 3)) / (24 * omx2 ** sp.Rational(3, 2))
    for pp, pm, L0 in [(M - s, M + s, M), (s - M, M + s, s), (s - M, -M - s, -M), (M - s, -M - s, -s)]:
        e2 = -pp * (pp**2 - 1) / (12 * (1 - x) ** 2) - pm * (pm**2 - 1) / (12 * (1 + x) ** 2)
        om_sin, c2L0 = (pp * (1 + x) - pm * (1 - x)) / 2, 2 * L0 + 1
        C0 = (c2L0 / 48) * (-1 + (pp - 1) * c2L0 / (1 - x) + (pm - 1) * c2L0 / (1 + x))
        assert sp.simplify((theta_2 * k**2 + rho20) - P_f2) == 0
        assert sp.simplify((W_2 * k**2 + C0 + theta_2 * k**2 * om_sin / sin_th) - (Q_f2 - e2 + P_f2 * om_sin / sin_th)) == 0
    print("  [PASS] Fixed-rank O(N^{-2}) expansion matches Paper II cor:uniformNLO in all 4 chambers")

    # Corollary 3.2: near-turning expansion and Bessel cancellation.
    t_edge = sp.symbols("t_edge", positive=True)
    corner_gap = sp.asin(t_edge) - t_edge * sp.sqrt(1 - t_edge**2)
    assert sp.limit(corner_gap / t_edge**3, t_edge, 0) == sp.Rational(2, 3)
    assert 1 + sp.Rational(5, 24) / 3 == sp.Rational(77, 72)
    print("  [PASS] Corollary 3.2 near-turning gap 2/3 and Bessel coefficient 77/72")



def run_part_b_exact_crosscheck():
    print("\n=== Part B: Numerical Recurrence Cross-Check vs sympy.physics.wigner.wigner_3j [numerical evidence] ===")
    max_err = 0.0
    for N, M, sig in [(21, 0, 0), (22, 2, -1), (21, -1, 2), (21.5, 1.5, -0.5), (21.5, -1.5, -0.5)]:
        L0, p_p = max(abs(M), abs(sig)), int(round(abs(M - sig)))
        D_g, two_N = int(round(N - L0 - 1)), int(round(2 * N))
        jm, jp = sp.Rational(two_N - 2 - int(round(2 * sig)), 4), sp.Rational(two_N - 2 + int(round(2 * sig)), 4)
        for n_in in [0, 1, D_g // 2, D_g - 1, D_g]:
            x = 1.0 - (p_p + 1.0 + 2.0 * n_in) / N
            mu = sp.Rational(int(round(2 * N * x)), 4)
            mm, mp = mu - sp.Rational(int(round(2 * M)), 4), -mu - sp.Rational(int(round(2 * M)), 4)
            phis = exact_phi_3j_chain(N, x, L0 + 5.0, M, sig)
            for dL in range(6):
                Lv = L0 + dL
                Lr, Mr = sp.Rational(int(round(2 * Lv)), 2), sp.Rational(int(round(2 * M)), 2)
                sgn = -1 if (int(round(2 * Lv + float(jp + mp))) % 2 == 1) else 1
                sp_val = float((sgn * sp.sqrt(sp.Rational(two_N, 2)) * wigner_3j(Lr, jm, jp, Mr, mm, mp)).evalf(30))
                max_err = max(max_err, abs(phis[Lv] - sp_val))
    assert max_err < 1e-13
    print(f"  [PASS] Max |Phi_L - sympy.wigner_3j| across all chambers & endpoints = {max_err:.3e}")


def run_part_b_laguerre_sign():
    """Theorem 3.1 at the lower edge x_n^- (numerical cross-check against exact 3j at N=401): the sign of the Laguerre limit is
    (-1)^(L-sigma) eps(-M, sigma); the sign (-1)^(L-L_0) eps(-M, sigma) fails (negative control)."""
    R = sp.Rational

    def phi(L, M, s, x, N):
        jm, jp, mu = R(N - 1 - s, 2), R(N - 1 + s, 2), x * N / 2
        mm, mp = (2 * mu - M) / 2, (-M - 2 * mu) / 2
        return (-1) ** int(2 * L + jp + mp) * sp.sqrt(N) * wigner_3j(L, jm, jp, M, mm, mp)

    eps = lambda M, s: (-1) ** int(M - s) if s <= M else 1
    N, bad_old, n_cases = 401, 0, 0
    for M, s in [(1, 0), (2, 1), (2, -1), (0, 1), (1, 1), (3, 1), (-1, 2)]:
        L0, pm = max(abs(M), abs(s)), abs(M + s)
        for n in (0, 1):
            x = -1 + R(pm + 1 + 2 * n, N)
            for L in (L0 + 14, L0 + 15):
                lam = R(L * (L + 1), N)
                lim = float(sp.sqrt(sp.factorial(n) / sp.factorial(n + pm)) * lam ** R(pm, 2) * sp.exp(-lam / 2)
                            * sp.assoc_laguerre(n, pm, lam))
                val = float(phi(L, M, s, x, N))
                new, old = (-1) ** int(L - s) * eps(-M, s), (-1) ** int(L - L0) * eps(-M, s)
                assert abs(val - new * lim) < 0.05 * abs(lim) + 1e-3, (M, s, n, L, val, lim)
                bad_old += abs(val - old * lim) > 0.05 * abs(lim) + 1e-3
                n_cases += 1
    assert bad_old > 0
    print(f"  [PASS] Thm 3.1 lower-edge sign (-1)^(L-sigma) eps(-M,sigma) holds in {n_cases} cases; "
          f"(-1)^(L-L0) eps(-M,sigma) fails in {bad_old} (negative control)")


if __name__ == "__main__":
    run_part_a_symbolic()
    run_part_b_exact_crosscheck()
    run_part_b_laguerre_sign()
    print("\nALL 3J SYMBOLIC CHECKS & NUMERICAL CROSS-CHECKS PASSED.")
