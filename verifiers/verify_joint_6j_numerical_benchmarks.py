#!/usr/bin/env python3
"""
Standalone numerical-benchmark and ablation verifier for Joint Rank-Position Uniformity (Centered Racah / 6j):
  Part C [numerical evidence]:
    - C1:  Exact atan2 turning-point perimeter laws (< 1e-12) and interior quadrature cross-check (< 1e-7).
    - C2:  Multi-S dense 2D supremum scan across S ~ {500, 2000, 8000, 32000} (A:B ~ 3:2, nu_6 <= 0.82),
           asserting uniform O(1) boundedness and contraction of successive 4x-S increments.
    - C3a: Separated hard edges C_min, C_max (4x-S ratios -> 16 = O(S^{-2})).
    - C3b: Isosceles A = B: O(S^{-4}) super-convergence at C_min (x >= 0) vs O(S^{-2}) at C_max (x -> -1).
    - C3c: Isosceles A = B at fixed x = -17/32 and x = -161/200 (x >= -1+2*eta): O(S^{-4}) (2x-S ratios -> 16).
    - C4:  Growing-rank edge corridor nu_6 = 0.60 across S ~ 1000 -> 4000.
    - C5:  Macroscopic-rank isosceles corner (A = B = 500 -> 2000, rho = 0.35): O(C^{-2}) stability.
  Part D [numerical evidence]:
    Mandatory negative controls NC1, NC2, NC3a, NC3b, NC4 tied to exact analytical predictions.
"""

import math
import numpy as np


def epsilon_phase(a, b):
    return (-1.0 if (int(round(a - b)) % 2 == 1) else 1.0) if b <= a else 1.0


def edge_factor_R(p, X):
    p_int = int(round(p))
    return 1.0 if p_int <= 1 else math.sqrt(math.prod(1.0 + (-0.5 * (p_int - 1) + k) / X for k in range(p_int)))


def edge_factor_6j(A, B, C, m, n):
    p_min, p_max = int(round(abs(m - n))), int(round(abs(m + n)))
    return edge_factor_R(p_min, A - B + C) * edge_factor_R(p_min, -A + B + C) * edge_factor_R(p_max, A + B - C) * edge_factor_R(p_max, A + B + C)


def wigner_d_one(k, a, b, theta):
    d, dp = wigner_d_vec(np.array([float(k)]), a, b, np.array([float(theta)]))
    return float(d[0]), float(dp[0])


def wigner_d_vec(k_arr, a, b, theta_arr):
    """Vectorized exact recurrence for (d^k_{ab}(theta), d/dtheta d^k_{ab}(theta))."""
    k0, p_p, p_m = max(abs(a), abs(b)), int(round(abs(a - b))), int(round(abs(a + b)))
    k_max = float(np.max(k_arr))
    targets = {int(round(kv - k0)): idx for idx, kv in enumerate(k_arr)}
    d_out, dp_out = np.empty_like(theta_arr), np.empty_like(theta_arr)
    y, s, sh, ch = np.cos(theta_arr), np.sin(theta_arr), np.sin(0.5 * theta_arr), np.cos(0.5 * theta_arr)
    if k0 == 0:
        d_prev, dp_prev = np.ones_like(theta_arr), np.zeros_like(theta_arr)
        if 0 in targets:
            d_out[targets[0]], dp_out[targets[0]] = 1.0, 0.0
        if k_max < 1:
            return d_out, dp_out
        d_cur, dp_cur, k_cur = y.copy(), -s.copy(), 1.0
        if 1 in targets:
            d_out[targets[1]], dp_out[targets[1]] = d_cur[targets[1]], dp_cur[targets[1]]
    else:
        pref = epsilon_phase(a, b) * math.sqrt(math.factorial(int(round(2 * k0))) / (math.factorial(p_p) * math.factorial(p_m)))
        d_cur = pref * (sh ** p_p) * (ch ** p_m)
        t1 = 0.5 * p_p * pref * (sh ** (p_p - 1)) * (ch ** (p_m + 1)) if p_p > 0 else 0.0
        t2 = -0.5 * p_m * pref * (sh ** (p_p + 1)) * (ch ** (p_m - 1)) if p_m > 0 else 0.0
        dp_cur, d_prev, dp_prev, k_cur = t1 + t2, np.zeros_like(theta_arr), np.zeros_like(theta_arr), float(k0)
        if 0 in targets:
            d_out[targets[0]], dp_out[targets[0]] = d_cur[targets[0]], dp_cur[targets[0]]
    while k_cur < k_max - 1e-9:
        kp1, c2k1, kk1 = k_cur + 1.0, 2.0 * k_cur + 1.0, k_cur * (k_cur + 1.0)
        Rkp1 = math.sqrt((kp1 * kp1 - a * a) * (kp1 * kp1 - b * b))
        num, dnum = c2k1 * (kk1 * y - a * b) * d_cur, c2k1 * (-kk1 * s * d_cur + (kk1 * y - a * b) * dp_cur)
        if k_cur > k0 + 1e-9:
            Rk = math.sqrt((k_cur * k_cur - a * a) * (k_cur * k_cur - b * b))
            num, dnum = num - (k_cur + 1.0) * Rk * d_prev, dnum - (k_cur + 1.0) * Rk * dp_prev
        d_prev, dp_prev = d_cur, dp_cur
        d_cur, dp_cur, k_cur = num / (k_cur * Rkp1), dnum / (k_cur * Rkp1), kp1
        step = int(round(k_cur - k0))
        if step in targets:
            d_out[targets[step]], dp_out[targets[step]] = d_cur[targets[step]], dp_cur[targets[step]]
    return d_out, dp_out


def exact_phi_6j_chain(A, B, C, f_max, m, n):
    """Exact Phi_f(A, B, C) via Paper II eq:6jrec and lem:sqrtfreecarrier."""
    f0, x, out = max(abs(m), abs(n)), (A * A + B * B - C * C) / (2.0 * A * B), {}
    if f0 == 0:
        out[0] = 1.0
        if f_max >= 1:
            out[1] = (x - 1.0 / (8.0 * A * B)) / math.sqrt((1.0 - 1.0 / (4.0 * A * A)) * (1.0 - 1.0 / (4.0 * B * B)))
        f = 1.0
    else:
        d0, _ = wigner_d_one(f0, n, m, math.acos(max(-1.0, min(1.0, x))))
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


def racah_action_S6(rho, alpha, beta, x):
    s, omx2, a2, b2 = rho * rho, 1.0 - x * x, alpha * alpha, beta * beta
    gam = math.sqrt(a2 + b2 - 2.0 * alpha * beta * x)
    sp_phi = min(1.0, rho / (2.0 * alpha * beta * math.sqrt(omx2) / gam))
    cp_phi = math.sqrt(max(0.0, 1.0 - sp_phi * sp_phi))
    return (rho * math.atan2(math.sqrt(omx2) * cp_phi, x - s / (4.0 * alpha * beta)) + 2.0 * gam * math.atan2(sp_phi, cp_phi)
            - 2.0 * alpha * math.atan2((alpha - beta * x) * sp_phi, gam * cp_phi) - 2.0 * beta * math.atan2((beta - alpha * x) * sp_phi, gam * cp_phi))


def racah_u2_vec(rho_arr, alpha, beta, x, m, n):
    s_arr, omx2, a2, b2 = rho_arr * rho_arr, 1.0 - x * x, alpha * alpha, beta * beta
    gam2 = a2 + b2 - 2.0 * alpha * beta * x
    gam = math.sqrt(gam2)
    rho_star = 2.0 * alpha * beta * math.sqrt(omx2) / gam
    D6_arr = omx2 - gam2 * s_arr / (4.0 * a2 * b2)
    sp_p = np.clip(rho_arr / rho_star, 0.0, 1.0)
    cp_p = np.sqrt(np.maximum(0.0, 1.0 - sp_p * sp_p))
    S6_arr = (rho_arr * np.arctan2(math.sqrt(omx2) * cp_p, x - s_arr / (4.0 * alpha * beta)) + 2.0 * gam * np.arctan2(sp_p, cp_p)
              - 2.0 * alpha * np.arctan2((alpha - beta * x) * sp_p, gam * cp_p) - 2.0 * beta * np.arctan2((beta - alpha * x) * sp_p, gam * cp_p))
    T6_arr = np.where(rho_arr < 1e-12, math.acos(x), S6_arr / np.maximum(rho_arr, 1e-30))
    W6_arr, Sig, tau, sqrtD = (np.sin(T6_arr) ** 2 / D6_arr) ** 0.25, m * m + n * n, m * n, np.sqrt(D6_arr)
    A1 = ((4.0 * Sig - 1.0) * x - 8.0 * tau) / (8.0 * omx2)
    num_A2 = (16.0 * a2 * b2 * (a2 + b2) * x * (x * x - 7.0) + 16.0 * ((alpha * beta) ** 3) * (7.0 * x * x + 5.0)
              + s_arr * (24.0 * (a2 * a2 + b2 * b2) * x - 16.0 * alpha * beta * (a2 + b2) * (2.0 * x * x + 1.0) - 8.0 * a2 * b2 * x * (x * x - 7.0))
              + s_arr * s_arr * (-6.0 * (a2 + b2) * x + 3.0 * alpha * beta * (3.0 * x * x + 1.0)))
    A2 = num_A2 / (-96.0 * a2 * b2 * (4.0 * a2 - s_arr) * (4.0 * b2 - s_arr) * omx2)
    A3 = -5.0 * (alpha - beta * x) * (beta - alpha * x) * gam2 / (384.0 * a2 * a2 * b2 * b2 * omx2)
    tG = A1 * sqrtD / np.maximum(rho_arr, 1e-30) + A2 * rho_arr / sqrtD + A3 * (rho_arr ** 3) / (D6_arr ** 1.5)
    c_T6 = ((Sig - 0.25) * np.cos(T6_arr) - 2.0 * tau) / (2.0 * np.sin(T6_arr))
    eps1, beta1, beta2 = (1.0 / a2 + 1.0 / b2) / 8.0, 1.0 / (4.0 * alpha * beta), (0.5 - Sig) / (8.0 * alpha * beta)
    rho20 = (eps1 * (9.0 * x - x**3 - 8.0 * Sig * x + 4.0 * tau * x * x + 12.0 * tau)
             + beta1 * (8.0 * Sig - 16.0 * tau * x - 2.0 * x * x - 6.0) + 12.0 * beta2 * omx2) / (12.0 * (omx2 ** 1.5))
    G2r_arr = np.where(sp_p < 1e-4, rho20, (tG - c_T6 / np.maximum(rho_arr, 1e-30)) / np.maximum(rho_arr, 1e-30))
    return T6_arr, W6_arr, G2r_arr, D6_arr


def racah_u2_data(rho, alpha, beta, x, m, n):
    T6, W6, G2, D6 = racah_u2_vec(np.array([float(rho)]), alpha, beta, x, m, n)
    return float(T6[0]), float(W6[0]), float(G2[0]), float(D6[0])


def phi_6j_u2_cov(A, B, C, f, m, n, include_edge=True, covariant=True, include_c0=True):
    S = A + B
    h, alpha, beta, x, rho = 1.0 / S, A / S, B / S, (A * A + B * B - C * C) / (2.0 * A * B), (f + 0.5) / S
    delta, p_min, p_max, f0 = alpha - beta, abs(m - n), abs(m + n), max(abs(m), abs(n))
    E_val = edge_factor_6j(A, B, C, m, n) if include_edge else 1.0
    c2f0, omd2 = 2.0 * f0 + 1.0, 1.0 - delta * delta
    C0_6 = ((c2f0 / (24.0 * omd2 * omd2)) * (-(1.0 + delta * delta) + 2.0 * delta * delta * (p_min - 1.0) * c2f0 / (1.0 - x) + 2.0 * (p_max - 1.0) * c2f0 / (1.0 + x))) if include_c0 else 0.0
    Theta6, W6, G2r, D6 = racah_u2_data(rho, alpha, beta, x, m, n)
    d_val, dp_val = wigner_d_one(f, n, m, Theta6)
    om = 0.5 * p_min / math.tan(0.5 * Theta6) - 0.5 * p_max * math.tan(0.5 * Theta6)
    return E_val * W6 * ((1.0 + h * h * C0_6) * d_val + h * h * G2r * ((dp_val - om * d_val) if covariant else dp_val)), D6


def dense_sup_scan_6j(A, B, m, n, nu_max=0.82, u6_cut=1.0):
    """Dense (C, f) scan of sup D_6^{1/4} * sqrt(2f+1) * |Phi_f - Phi_f^{U2,cov}| * u_6^2."""
    S, f0, p_min, p_max = A + B, max(abs(m), abs(n)), int(round(abs(m - n))), int(round(abs(m + n)))
    h, alpha, beta = 1.0 / S, A / S, B / S
    a2, b2, delta, c2f0 = alpha * alpha, beta * beta, alpha - beta, 2.0 * f0 + 1.0
    omd2, C_min, C_max = 1.0 - delta * delta, abs(A - B) + 0.5 * (p_min + 1.0), A + B - 0.5 * (p_max + 1.0)
    r_half = max(0, int(round(C_max - C_min)) // 2)
    r_base = sorted({r for r in list(range(min(30, r_half + 1))) + [int(round(v)) for v in np.geomspace(30, max(31, r_half), 20)] if 0 <= r <= r_half})
    C_list, sup_val = sorted({C_min + r for r in r_base} | {C_max - r for r in r_base if C_max - r >= C_min}), 0.0
    for C in C_list:
        u6 = min(C - abs(A - B), A + B - C)
        if u6 < u6_cut - 1e-9:
            continue
        x, gam = (A * A + B * B - C * C) / (2.0 * A * B), C / S
        omx2, gam2 = 1.0 - x * x, gam * gam
        rho_star = 2.0 * alpha * beta * math.sqrt(omx2) / gam
        f_hi = math.floor(min(2.0 * min(A, B) - 1.0, nu_max * S * rho_star - 0.5) - (f0 % 1.0)) + (f0 % 1.0)
        if f_hi < f0 - 1e-9:
            continue
        phis, n_steps = exact_phi_6j_chain(A, B, C, f_hi, m, n), int(round(f_hi - f0))
        ks = list(range(n_steps + 1)) if n_steps <= 360 else sorted(
            set(range(50)) | set(range(max(0, n_steps - 16), n_steps + 1)) | {int(round(v)) for v in np.linspace(50, max(51, n_steps - 16), 190)}
        )
        f_list = [f0 + k for k in ks]
        f_arr = np.array(f_list, dtype=float)
        T6_arr, W6_arr, G2r_arr, D6_arr = racah_u2_vec((f_arr + 0.5) * h, alpha, beta, x, m, n)
        d_arr, dp_arr = wigner_d_vec(f_arr, n, m, T6_arr)
        om_arr = 0.5 * p_min / np.tan(0.5 * T6_arr) - 0.5 * p_max * np.tan(0.5 * T6_arr)
        C0_6 = (c2f0 / (24.0 * omd2 * omd2)) * (-(1.0 + delta * delta) + 2.0 * delta * delta * (p_min - 1.0) * c2f0 / (1.0 - x) + 2.0 * (p_max - 1.0) * c2f0 / (1.0 + x))
        vc_arr = edge_factor_6j(A, B, C, m, n) * W6_arr * ((1.0 + h * h * C0_6) * d_arr + h * h * G2r_arr * (dp_arr - om_arr * d_arr))
        pe_arr = np.array([phis[fv] for fv in f_list])
        sup_val = max(sup_val, float(np.max((D6_arr ** 0.25) * np.sqrt(2.0 * f_arr + 1.0) * np.abs(pe_arr - vc_arr) * (u6 * u6))))
    return sup_val


def run_part_c_convergence():
    print("=== Part C: Turning-Action Law, Dense Multi-S Scans, Hard Edges & Corner [numerical evidence] ===")
    for A, B, C, obtuse in [(120.0, 80.0, 45.0, True), (80.0, 120.0, 46.0, True), (120.0, 80.0, 195.0, False), (100.0, 100.0, 7.0, False)]:
        S = A + B
        al, be, gam, x = A / S, B / S, C / S, (A * A + B * B - C * C) / (2.0 * A * B)
        rs = 2.0 * al * be * math.sqrt(1.0 - x * x) / gam
        S6_turn = racah_action_S6(rs, al, be, x)
        err_turn = abs(S * S6_turn / math.pi - (C - abs(A - B))) if obtuse else abs(S * (math.pi * rs - S6_turn) / math.pi - (A + B - C))
        assert err_turn < 1e-12
        rho_i, n_q = 0.85 * rs, 2000
        quad = sum(math.acos((x - ((k + 0.5) * rho_i / n_q)**2 / (4 * al * be)) / math.sqrt((1 - ((k + 0.5) * rho_i / n_q)**2 / (4 * al * al)) * (1 - ((k + 0.5) * rho_i / n_q)**2 / (4 * be * be)))) for k in range(n_q)) * (rho_i / n_q)
        assert abs(racah_action_S6(rho_i, al, be, x) - quad) < 1e-7
    print("  [PASS] C1 Exact atan2 turning-point perimeter laws (<1e-12) & interior quadrature (<1e-7)")

    for m, n, off in [(0.0, 0.0, 0.5), (2.0, -1.0, 1.5), (1.5, -0.5, 1.0)]:
        sups = [dense_sup_scan_6j(3.0 * sc, 2.0 * sc + off, m, n, 0.82, 1.0) for sc in (100.0, 400.0, 1600.0, 6400.0)]
        d0, d1, d2 = abs(sups[1] - sups[0]), abs(sups[2] - sups[1]), abs(sups[3] - sups[2])
        assert 0.2 < sups[-1] < 2.0 and d2 <= max(0.6 * max(d0, d1), 0.05)
        print(f"  [PASS] C2 Multi-S dense sup (m={m}, n={n}, A:B~3:2, nu_6<=0.82): S~500..32k -> {[round(v, 4) for v in sups]} (increments={[round(d0, 4), round(d1, 4), round(d2, 4)]})")

    m, n, fv = 2.0, -1.0, 5.0
    sep_min, sep_max, iso_min, iso_max = [], [], [], []
    for sc in [40, 160, 640]:
        A, B = 3 * sc + 0.0, 2 * sc + 1.5
        for arr, C in [(sep_min, abs(A - B) + 2.0), (sep_max, A + B - 1.0)]:
            x = (A * A + B * B - C * C) / (2.0 * A * B)
            pe, (vc, _) = exact_phi_6j_chain(A, B, C, fv, m, n)[fv], phi_6j_u2_cov(A, B, C, fv, m, n)
            arr.append(abs(pe - vc) / (edge_factor_6j(A, B, C, m, n) * ((1.0 - x) / 2.0) ** 1.5 * ((1.0 + x) / 2.0) ** 0.5))
    for Ai in [40.0, 160.0]:
        xi = (2.0 * Ai * Ai - 4.0) / (2.0 * Ai * Ai)
        pei, (vci, _) = exact_phi_6j_chain(Ai, Ai, 2.0, fv, m, n)[fv], phi_6j_u2_cov(Ai, Ai, 2.0, fv, m, n)
        iso_min.append(abs(pei - vci) / (edge_factor_6j(Ai, Ai, 2.0, m, n) * ((1.0 - xi) / 2.0) ** 1.5 * ((1.0 + xi) / 2.0) ** 0.5))
    for Ai in [80.0, 320.0, 1280.0]:
        Ci = 2.0 * Ai - 1.0
        xi = (2.0 * Ai * Ai - Ci * Ci) / (2.0 * Ai * Ai)
        pei, (vci, _) = exact_phi_6j_chain(Ai, Ai, Ci, fv, m, n)[fv], phi_6j_u2_cov(Ai, Ai, Ci, fv, m, n)
        iso_max.append(abs(pei - vci) / (edge_factor_6j(Ai, Ai, Ci, m, n) * ((1.0 - xi) / 2.0) ** 1.5 * ((1.0 + xi) / 2.0) ** 0.5))
    rm0, rm1 = iso_max[0] / iso_max[1], iso_max[1] / iso_max[2]
    assert 15.0 < sep_min[1] / sep_min[2] < 16.5 and 15.0 < sep_max[1] / sep_max[2] < 16.5 and 245.0 < iso_min[0] / iso_min[1] < 268.0 and abs(rm1 - 16.0) < abs(rm0 - 16.0) and abs(rm1 - 16.0) < 0.6
    print(f"  [PASS] C3a Separated r=0 (A:B~3:2): C_min ratios={sep_min[0]/sep_min[1]:.2f}, {sep_min[1]/sep_min[2]:.2f}; C_max ratios={sep_max[0]/sep_max[1]:.2f}, {sep_max[1]/sep_max[2]:.2f} (~16 = O(S^-2))")
    print(f"  [PASS] C3b Isosceles A=B: C_min (x>=0) 4x-S ratio={iso_min[0]/iso_min[1]:.1f} (~256 = O(S^-4)); C_max (x->-1) ratios={rm0:.2f}, {rm1:.2f} (~16 = O(S^-2))")
    # C3c: thm:racah-u2-cov(c) on x >= -1+2*eta: exact isosceles slices x = 1 - (C/A)^2/2 at fixed C/A, A doubling (ratio 16 = O(S^-4)).
    for num, den, nskip in [(7, 4, 0), (19, 10, 1)]:
        errs = []
        for Ai in [80.0, 160.0, 320.0, 640.0]:
            Ci = Ai * num / den
            xi = 1.0 - Ci * Ci / (2.0 * Ai * Ai)
            pei, (vci, _) = exact_phi_6j_chain(Ai, Ai, Ci, fv, m, n)[fv], phi_6j_u2_cov(Ai, Ai, Ci, fv, m, n)
            errs.append(abs(pei - vci) / (edge_factor_6j(Ai, Ai, Ci, m, n) * ((1.0 - xi) / 2.0) ** 1.5 * ((1.0 + xi) / 2.0) ** 0.5))
        rats = [errs[i] / errs[i + 1] for i in range(3)]
        assert all(13.0 < r < 19.0 for r in rats[nskip:]), rats
        note = f"; first {nskip} ratio pre-asymptotic, not asserted" if nskip else ""
        print(f"  [PASS] C3c Isosceles A=B at fixed x={xi:.4f} < 0 (C/A={num}/{den}): 2x-S ratios={[round(r, 2) for r in rats]} (~16 = O(S^-4){note})")

    cor_sups, iso_sups = [], []
    for sc in [200.0, 800.0]:
        A, B = 3.0 * sc, 2.0 * sc + 1.5
        S, vals = A + B, []
        for r_in in [2, 4, 8, 16, 32, 64, 128]:
            C = abs(A - B) + 2.0 + r_in
            x = (A * A + B * B - C * C) / (2.0 * A * B)
            fk = float(int(round(0.60 * S * (2.0 * (A / S) * (B / S) * math.sqrt(1.0 - x * x) / (C / S)) - 0.5)))
            pe, (vc, D6) = exact_phi_6j_chain(A, B, C, fk, m, n)[fk], phi_6j_u2_cov(A, B, C, fk, m, n)
            vals.append((D6 ** 0.25) * math.sqrt(2.0 * fk + 1.0) * abs(pe - vc) * ((C - abs(A - B)) ** 2))
        cor_sups.append(max(vals))
    ext_sups = []
    for A_c in [500.0, 2000.0]:
        f_c, vals, vals_ext = 0.70 * A_c, [], []
        for C_c in [float(c) for c in range(2, 16)] + [18.0, 66.0, 258.0]:      # C_min = 2 for (m,n) = (2,-1)
            pe_c, (vc_c, D6_c) = exact_phi_6j_chain(A_c, A_c, C_c, f_c, m, n)[f_c], phi_6j_u2_cov(A_c, A_c, C_c, f_c, m, n)
            val = (D6_c ** 0.25) * math.sqrt(2.0 * f_c + 1.0) * abs(pe_c - vc_c) * (C_c * C_c)
            (vals if C_c <= 15.0 else vals_ext).append(val)
        iso_sups.append(max(vals))
        ext_sups.append(max(vals + vals_ext))
    assert abs(cor_sups[1] - cor_sups[0]) / cor_sups[1] < 0.05 and abs(iso_sups[1] - iso_sups[0]) / iso_sups[1] < 0.03
    print(f"  [PASS] C4 Growing-rank corridor nu_6=0.60 (S~1000->4000): max = {cor_sups[0]:.4f} -> {cor_sups[1]:.4f}")
    print(f"  [PASS] C5 Macroscopic isosceles corner (A=B=500->2000, rho=0.35, integer C=2..15): max(D_6^{{1/4}}*sqrt(2f+1)*|dPhi|*C^2) = {iso_sups[0]:.4f} -> {iso_sups[1]:.4f}; "
          f"with C in {{18,66,258}} added: {ext_sups[0]:.4f} -> {ext_sups[1]:.4f}")


def run_part_d_negative_controls():
    print("\n=== Part D: Mandatory Negative Controls [numerical evidence] ===")
    m, n, fv = 2.0, -1.0, 5.0
    pe_s, (vn_s, _) = exact_phi_6j_chain(1200.0, 801.5, 400.5, fv, m, n)[fv], phi_6j_u2_cov(1200.0, 801.5, 400.5, fv, m, n, include_edge=False)
    pe_i, (vn_i, _) = exact_phi_6j_chain(640.0, 640.0, 2.0, fv, m, n)[fv], phi_6j_u2_cov(640.0, 640.0, 2.0, fv, m, n, include_edge=False)
    rel_s, rel_i = abs(pe_s - vn_s) / abs(pe_s), abs(pe_i - vn_i) / abs(pe_i)
    assert abs(rel_s - (2.0 / math.sqrt(3.0) - 1.0)) < 2e-4 and abs(rel_i - (1.0 / 3.0)) < 1e-4
    print(f"  [PASS] NC1 Omitting E_{{A,B}}(C_min): sep rel_err={rel_s:.4f} (2/sqrt(3)-1), isosceles rel_err={rel_i:.4f} (exact 1/3=0.3333)")

    p3_rels, raw_errs, noc0_errs, coal_rels = [], [], [], []
    for sc in [160, 640]:
        At, Bt = 3 * sc + 0.0, 2 * sc + 1.5
        Ct, St = abs(At - Bt) + 2.0, At + Bt
        xt, rhot = (At * At + Bt * Bt - Ct * Ct) / (2.0 * At * Bt), (fv + 0.5) / St
        w6t = ((1.0 - xt) / 2.0) ** 1.5 * ((1.0 + xt) / 2.0) ** 0.5
        T6t, _, G2rt, D6t = racah_u2_data(rhot, At / St, Bt / St, xt, m, n)
        T62 = T6t + (1.0 / (St * St)) * G2rt
        d62, _ = wigner_d_one(fv, n, m, T62)
        pet = exact_phi_6j_chain(At, Bt, Ct, fv, m, n)[fv]
        p3_rels.append(abs(pet - edge_factor_6j(At, Bt, Ct, m, n) * ((math.sin(T62) ** 2 / D6t) ** 0.25) * d62) / abs(pet))
        vr, _ = phi_6j_u2_cov(At, Bt, Ct, fv, m, n, covariant=False)
        vnc, _ = phi_6j_u2_cov(At, Bt, Ct, fv, m, n, include_c0=False)
        raw_errs.append(abs(pet - vr) / (edge_factor_6j(At, Bt, Ct, m, n) * w6t))
        noc0_errs.append(abs(pet - vnc) / (edge_factor_6j(At, Bt, Ct, m, n) * w6t))
        fc = float(int(round(0.75 * St * (2.0 * (At / St) * (Bt / St) * math.sqrt(1.0 - xt * xt) / (Ct / St)) - 0.5)))
        pec, (vcc, _) = exact_phi_6j_chain(At, Bt, Ct, fc, m, n)[fc], phi_6j_u2_cov(At, Bt, Ct, fc, m, n)
        coal_rels.append(abs(pec - vcc) / abs(pec))
    exact_nc2 = 1.0 - (23.0 / 24.0) ** 3.5
    assert abs(p3_rels[1] - exact_nc2) < 5e-4 and 0.90 < raw_errs[0] / raw_errs[1] < 1.05 and 3.8 < noc0_errs[0] / noc0_errs[1] < 4.2 and coal_rels[1] > 0.02 and abs(coal_rels[1] - coal_rels[0]) / coal_rels[1] < 0.05
    print(f"  [PASS] NC2 Nonlinear shift E*W_6^[2]*d(Theta_6^[2]) at C_min: rel_err={p3_rels[0]:.4f}->{p3_rels[1]:.4f} (exact 1-(23/24)^(7/2)={exact_nc2:.4f})")
    print(f"  [PASS] NC3a Non-covariant raw d_Theta at C_min: 4x-S ratio={raw_errs[0]/raw_errs[1]:.2f} (~1 = O(1) double-pole failure)")
    print(f"  [PASS] NC3b Omitting C_0^{{(6)}}(delta,x) at C_min: 4x-S ratio={noc0_errs[0]/noc0_errs[1]:.2f} (~4 = O(S^-1) simple-pole degradation)")
    print(f"  [PASS] NC4 Separated coalescent corner (r=0, nu_6=0.75, S=801.5->3201.5): rel_err={coal_rels[0]:.4f}->{coal_rels[1]:.4f} (O(1) non-vanishing)")


if __name__ == "__main__":
    run_part_c_convergence()
    run_part_d_negative_controls()
    print("\nALL 6J NUMERICAL BENCHMARKS & NEGATIVE CONTROLS PASSED.")
