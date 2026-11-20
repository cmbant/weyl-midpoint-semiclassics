#!/usr/bin/env python3
"""Standalone numerical-benchmark & ablation verifier for Joint Rank-Position Uniformity (Hahn / 3j):
  Part C [numerical evidence]: C1a (multi-N sup N=500..32000), C1b (N=600), C2 (hard edge n=0),
    C3 (Hilb toe O(nu_L^4)), C4 (Zone IIb eps-uniform residual), C5 (interior q~0.35), C6a/b (Laguerre/Bessel).
  Part D [numerical evidence]: Negative controls NC1, NC2, NC3a, NC3b, NC4 tied to exact analytical predictions.
"""
import math
import numpy as np
from scipy.special import jv as bessel_j


def epsilon_phase(M, sigma):
    return (-1.0 if (int(round(M - sigma)) % 2 == 1) else 1.0) if sigma <= M else 1.0


def edge_factor_R(p, X):
    p_int = int(round(p))
    return 1.0 if p_int <= 1 else math.sqrt(math.prod(1.0 + (-0.5 * (p_int - 1) + k) / X for k in range(p_int)))


def edge_factor_3j(N, x, M, sigma):
    return edge_factor_R(abs(M - sigma), 0.5 * N * (1.0 - x)) * edge_factor_R(abs(M + sigma), 0.5 * N * (1.0 + x))


def wigner_d_one(L, M, sigma, theta):
    d, dp = wigner_d_vec(np.array([float(L)]), M, sigma, np.array([float(theta)]))
    return float(d[0]), float(dp[0])


def wigner_d_vec(L_arr, M, sigma, theta_arr):
    """Vectorized exact recurrence for (d^L_{M,sigma}(theta), d/dtheta d^L_{M,sigma}(theta))."""
    L0, p_p, p_m = max(abs(M), abs(sigma)), int(round(abs(M - sigma))), int(round(abs(M + sigma)))
    L_max = float(np.max(L_arr))
    targets = {int(round(Lv - L0)): idx for idx, Lv in enumerate(L_arr)}
    d_out, dp_out = np.empty_like(theta_arr), np.empty_like(theta_arr)
    y, s, sh, ch = np.cos(theta_arr), np.sin(theta_arr), np.sin(0.5 * theta_arr), np.cos(0.5 * theta_arr)
    if L0 == 0:
        d_prev, dp_prev = np.ones_like(theta_arr), np.zeros_like(theta_arr)
        if 0 in targets:
            d_out[targets[0]], dp_out[targets[0]] = 1.0, 0.0
        if L_max < 1:
            return d_out, dp_out
        d_cur, dp_cur, L_cur = y.copy(), -s.copy(), 1.0
        if 1 in targets:
            d_out[targets[1]], dp_out[targets[1]] = d_cur[targets[1]], dp_cur[targets[1]]
    else:
        pref = epsilon_phase(M, sigma) * math.sqrt(math.factorial(int(round(2 * L0))) / (math.factorial(p_p) * math.factorial(p_m)))
        d_cur = pref * (sh ** p_p) * (ch ** p_m)
        t1 = 0.5 * p_p * pref * (sh ** (p_p - 1)) * (ch ** (p_m + 1)) if p_p > 0 else 0.0
        t2 = -0.5 * p_m * pref * (sh ** (p_p + 1)) * (ch ** (p_m - 1)) if p_m > 0 else 0.0
        dp_cur, d_prev, dp_prev, L_cur = t1 + t2, np.zeros_like(theta_arr), np.zeros_like(theta_arr), float(L0)
        if 0 in targets:
            d_out[targets[0]], dp_out[targets[0]] = d_cur[targets[0]], dp_cur[targets[0]]
    while L_cur < L_max - 1e-9:
        Lp1, c2L1, LL1 = L_cur + 1.0, 2.0 * L_cur + 1.0, L_cur * (L_cur + 1.0)
        RLp1 = math.sqrt((Lp1 * Lp1 - M * M) * (Lp1 * Lp1 - sigma * sigma))
        num, dnum = c2L1 * (LL1 * y - M * sigma) * d_cur, c2L1 * (-LL1 * s * d_cur + (LL1 * y - M * sigma) * dp_cur)
        if L_cur > L0 + 1e-9:
            RL = math.sqrt((L_cur * L_cur - M * M) * (L_cur * L_cur - sigma * sigma))
            num, dnum = num - (L_cur + 1.0) * RL * d_prev, dnum - (L_cur + 1.0) * RL * dp_prev
        d_prev, dp_prev = d_cur, dp_cur
        d_cur, dp_cur, L_cur = num / (L_cur * RLp1), dnum / (L_cur * RLp1), Lp1
        step = int(round(L_cur - L0))
        if step in targets:
            d_out[targets[step]], dp_out[targets[step]] = d_cur[targets[step]], dp_cur[targets[step]]
    return d_out, dp_out


def exact_phi_3j_chain(N, x, L_max, M, sigma):
    """Exact Phi_L^{M,sigma}(x; N^{-1}) via Paper II eq:3jseed and eq:3jrank."""
    L0, h, out = max(abs(M), abs(sigma)), 1.0 / N, {}
    if L0 == 0:
        out[0] = 1.0
        if L_max >= 1:
            out[1] = x / math.sqrt(1.0 - h * h)
        L_cur = 1.0
    else:
        d0, _ = wigner_d_one(L0, M, sigma, math.acos(max(-1.0, min(1.0, x))))
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


def hahn_u2_vec(q_arr, x, M, sigma):
    omx2 = 1.0 - x * x
    D_arr, nu_arr = omx2 - q_arr * q_arr, q_arr / math.sqrt(omx2)
    sqrtD = np.sqrt(D_arr)
    Tq_arr = np.where(q_arr < 1e-12, math.acos(x), (q_arr * np.arccos(np.clip(x / np.sqrt(1.0 - q_arr * q_arr), -1.0, 1.0))
                      - x * np.arcsin(np.clip(nu_arr, -1.0, 1.0)) + np.arctan2(x * q_arr, sqrtD)) / np.maximum(q_arr, 1e-30))
    Wq_arr, Sig, tau = (np.sin(Tq_arr) ** 2 / D_arr) ** 0.25, M * M + sigma * sigma, M * sigma
    tG = (((4.0 * Sig - 1.0) * x - 8.0 * tau) * sqrtD / (8.0 * omx2 * np.maximum(q_arr, 1e-30))
          + x * q_arr * (7.0 - x * x - 6.0 * q_arr * q_arr) / (24.0 * omx2 * (1.0 - q_arr * q_arr) * sqrtD)
          + 5.0 * x * (q_arr ** 3) / (24.0 * omx2 * (D_arr ** 1.5)))
    c_Tq = ((Sig - 0.25) * np.cos(Tq_arr) - 2.0 * tau) / (2.0 * np.sin(Tq_arr))
    G2q_arr = np.where(nu_arr < 1e-5, (x * (9.0 - x * x - 8.0 * Sig) + 4.0 * tau * (x * x + 3.0)) / (24.0 * (omx2 ** 1.5)),
                       (tG - c_Tq / np.maximum(q_arr, 1e-30)) / np.maximum(q_arr, 1e-30))
    return Tq_arr, Wq_arr, G2q_arr, D_arr


def hahn_u2_data(q, x, M, sigma):
    Tq, Wq, G2, D = hahn_u2_vec(np.array([float(q)]), x, M, sigma)
    return float(Tq[0]), float(Wq[0]), float(G2[0]), float(D[0])


def phi_3j_u2_cov(N, x, L, M, sigma, include_edge=True, covariant=True, include_c0=True):
    h, q = 1.0 / N, (L + 0.5) / N
    p_p, p_m, L0 = abs(M - sigma), abs(M + sigma), max(abs(M), abs(sigma))
    c2L0 = 2.0 * L0 + 1.0
    E_val = edge_factor_3j(N, x, M, sigma) if include_edge else 1.0
    Theta_q, W_q, G2q, D = hahn_u2_data(q, x, M, sigma)
    d_val, dp_val = wigner_d_one(L, M, sigma, Theta_q)
    om = 0.5 * p_p / math.tan(0.5 * Theta_q) - 0.5 * p_m * math.tan(0.5 * Theta_q)
    C0 = ((c2L0 / 48.0) * (-1.0 + (p_p - 1.0) * c2L0 / (1.0 - x) + (p_m - 1.0) * c2L0 / (1.0 + x))) if include_c0 else 0.0
    return E_val * W_q * ((1.0 + h * h * C0) * d_val + h * h * G2q * ((dp_val - om * d_val) if covariant else dp_val)), D


def dense_sup_scan_3j(N, M, sigma, nu_max=0.82, u_min_cut=1.0):
    """Dense (x, L) scan of sup D^{1/4} * sqrt(2L+1) * |Phi_L - Phi_L^{U2,cov}| * u(x)^2."""
    L0, p_p, p_m = max(abs(M), abs(sigma)), int(round(abs(M - sigma))), int(round(abs(M + sigma)))
    c2L0, h, sup_val = 2.0 * L0 + 1.0, 1.0 / N, 0.0
    for sgn, p_e in [(1.0, p_p), (-1.0, p_m)]:
        n_eq = max(0, int(0.5 * N - 0.5 * (p_e + 1.0)))
        n_set = sorted({n for n in list(range(min(72, n_eq + 1))) + [int(round(v)) for v in np.geomspace(72, max(73, n_eq), 28)] if 0 <= n <= n_eq})
        for n_in in n_set:
            x = sgn * (1.0 - (p_e + 1.0 + 2.0 * n_in) / N)
            u_val = N * (1.0 - abs(x))
            if u_val < u_min_cut - 1e-9 or abs(x) >= 1.0:
                continue
            omx2 = 1.0 - x * x
            L_hi = math.floor(nu_max * N * math.sqrt(omx2) - 0.5 - (L0 % 1.0)) + (L0 % 1.0)
            if L_hi < L0 - 1e-9:
                continue
            phis = exact_phi_3j_chain(N, x, L_hi, M, sigma)
            n_steps = int(round(L_hi - L0))
            ks = list(range(n_steps + 1)) if n_steps <= 220 else sorted(
                set(range(45)) | set(range(max(0, n_steps - 16), n_steps + 1)) | {int(round(v)) for v in np.linspace(45, max(46, n_steps - 16), 165)}
            )
            L_list = [L0 + k for k in ks]
            L_arr = np.array(L_list, dtype=float)
            Tq_arr, Wq_arr, G2q_arr, D_arr = hahn_u2_vec((L_arr + 0.5) * h, x, M, sigma)
            d_arr, dp_arr = wigner_d_vec(L_arr, M, sigma, Tq_arr)
            om_arr = 0.5 * p_p / np.tan(0.5 * Tq_arr) - 0.5 * p_m * np.tan(0.5 * Tq_arr)
            C0 = (c2L0 / 48.0) * (-1.0 + (p_p - 1.0) * c2L0 / (1.0 - x) + (p_m - 1.0) * c2L0 / (1.0 + x))
            vc_arr = edge_factor_3j(N, x, M, sigma) * Wq_arr * ((1.0 + h * h * C0) * d_arr + h * h * G2q_arr * (dp_arr - om_arr * d_arr))
            pe_arr = np.array([phis[Lv] for Lv in L_list])
            sup_val = max(sup_val, float(np.max((D_arr ** 0.25) * np.sqrt(2.0 * L_arr + 1.0) * np.abs(pe_arr - vc_arr) * (u_val * u_val))))
    return sup_val


def run_part_c_convergence():
    print("=== Part C: Dense Multi-N 2D Scans & Asymptotic Regime Audits [numerical evidence] ===")
    max_sup_32k = 0.0
    for M, sig in [(0.0, 0.0), (2.0, -1.0), (1.5, -0.5)]:
        sups = [dense_sup_scan_3j(Nb + (0.5 if (sig % 1 != 0) else 0.0), M, sig, 0.82, 1.0) for Nb in (500.0, 2000.0, 8000.0, 32000.0)]
        max_sup_32k = max(max_sup_32k, sups[-1])
        d0, d1, d2 = abs(sups[1] - sups[0]), abs(sups[2] - sups[1]), abs(sups[3] - sups[2])
        assert 0.5 < sups[-1] < 3.5 and d1 <= d0 + 0.02 and d2 <= max(0.5 * d1, 0.025)
        print(f"  [PASS] C1a Multi-N dense sup (M={M}, sig={sig}, nu<=0.82): N=500..32k -> {[round(v, 4) for v in sups]} (increments={[round(d0, 4), round(d1, 4), round(d2, 4)]})")
    sup_600 = max(dense_sup_scan_3j(600.5 if (sig % 1 != 0) else 600.0, M, sig, 0.82, 9.0) for M, sig in [(0.0, 0.0), (2.0, -1.0), (1.5, -0.5)])
    assert 1.5 < sup_600 <= max_sup_32k + 0.05
    print(f"  [PASS] C1b Dense scan at N=600 (nu<=0.82, n_in>=4 i.e. u>=9): sup = {sup_600:.4f} (ceiling <= {max_sup_32k:.4f})")

    for M, sig, L in [(0.0, 0.0, 4.0), (2.0, -1.0, 5.0), (1.5, -0.5, 4.5)]:
        p_p, p_m, vals = int(round(abs(M - sig))), int(round(abs(M + sig))), []
        for base in [160, 640, 2560]:
            N = base + (0.5 if (sig % 1 != 0) else 0.0)
            x = 1.0 - (p_p + 1.0) / N
            w3j = ((1.0 - x) / 2.0) ** (0.5 * p_p) * ((1.0 + x) / 2.0) ** (0.5 * p_m)
            pe, (vc, _) = exact_phi_3j_chain(N, x, L, M, sig)[L], phi_3j_u2_cov(N, x, L, M, sig)
            vals.append(abs(pe - vc) / (edge_factor_3j(N, x, M, sig) * w3j))
        r0, r1 = vals[0] / vals[1], vals[1] / vals[2]
        assert abs(r1 - 16.0) < abs(r0 - 16.0) and abs(r1 - 16.0) < 0.5
        print(f"  [PASS] C2 Hard edge n=0 (M={M}, sig={sig}, L={L}): 4x-N ratios = {r0:.2f}, {r1:.2f} (~16 = O(N^-2)), terminal={vals[2]:.3e}")

    M, sig, hilb_errs = 2.0, -1.0, []
    for N in [80.0, 640.0, 5120.0]:
        L = float(int(round(2.0 * (N ** (1.0 / 3.0)))))
        x = 1.0 - (4.0 + 2.0 * max(1, int(round(0.5 * (4.5 * N / ((L + 0.5) ** 2) - 4.0))))) / N
        pe, (vc, D) = exact_phi_3j_chain(N, x, L, M, sig)[L], phi_3j_u2_cov(N, x, L, M, sig)
        hilb_errs.append((D ** 0.25) * math.sqrt(2.0 * L + 1.0) * abs(pe - vc))
    rh0, rh1 = hilb_errs[0] / hilb_errs[1], hilb_errs[1] / hilb_errs[2]
    assert abs(rh1 - 16.0) < abs(rh0 - 16.0) and abs(rh1 - 16.0) < 1.5
    print(f"  [PASS] C3 Hilb toe O(nu_L^4) (L~2*N^{{1/3}}, 8x-N): errs = {hilb_errs[0]:.2e}->{hilb_errs[1]:.2e}->{hilb_errs[2]:.2e} (ratios={rh0:.2f}, {rh1:.2f} ~ 8^{{4/3}}=16)")

    for nu_t in [0.30, 0.50, 0.80]:
        res_scaled = []
        for N_r in [500.0, 2000.0, 8000.0, 32000.0]:
            x_r = 1.0 - 24.0 / N_r
            eps_r, h_r = math.sqrt(1.0 - x_r * x_r), 1.0 / N_r
            heff_r, k_r = 1.0 / (N_r * eps_r * eps_r), float(int(round(nu_t * N_r * eps_r - 0.5)))
            U_vals = {}
            for kj in [k_r - 1.0, k_r, k_r + 1.0]:
                Tq, _, G2q, Dj = hahn_u2_data((kj + 0.5) * h_r, x_r, M, sig)
                Tq2 = Tq + h_r * h_r * G2q
                dj, _ = wigner_d_one(kj, M, sig, Tq2)
                U_vals[kj] = math.sqrt(2.0 * kj + 1.0) * ((math.sin(Tq2) ** 2 / Dj) ** 0.25) * dj
            Ak = lambda j: math.sqrt((j * j - M * M) * (j * j - sig * sig) * (1.0 - j * j * h_r * h_r)) / (j * math.sqrt(4.0 * j * j - 1.0))
            Rk = abs((x_r - M * sig / (k_r * (k_r + 1.0))) * U_vals[k_r] - Ak(k_r + 1.0) * U_vals[k_r + 1.0] - Ak(k_r) * U_vals[k_r - 1.0])
            res_scaled.append(Rk / ((eps_r ** 1.5) * (((k_r + 0.5) * h_r / eps_r) * (heff_r ** 3) + ((k_r + 0.5) * eps_r) ** (-8.0))))
        assert abs(res_scaled[-1] - res_scaled[-2]) <= max(0.5 * abs(res_scaled[-2] - res_scaled[-3]), 0.05)
        print(f"  [PASS] C4 Zone IIb eps-uniform residual (nu={nu_t:.2f}, eps=0.31->0.039): R_scaled={[round(v, 3) for v in res_scaled]}")

    int_errs = []
    for N_i in [200.0, 800.0, 3200.0]:
        L_i, n0, env = float(int(round(0.35 * N_i - 0.5))), int(round(0.20 * N_i - 2.0)), 0.0
        for dn in range(-4, 5):
            x_i = 1.0 - (4.0 + 2.0 * (n0 + dn)) / N_i
            pe_i, (vc_i, D_i) = exact_phi_3j_chain(N_i, x_i, L_i, M, sig)[L_i], phi_3j_u2_cov(N_i, x_i, L_i, M, sig)
            env = max(env, (D_i ** 0.25) * math.sqrt(2.0 * L_i + 1.0) * abs(pe_i - vc_i))
        int_errs.append(env)
    assert 14.0 < int_errs[0] / int_errs[1] < 18.5 and 15.0 < int_errs[1] / int_errs[2] < 17.0
    print(f"  [PASS] C5 Fixed interior ratio (q~0.35): env_err = {int_errs[0]:.2e}->{int_errs[1]:.2e}->{int_errs[2]:.2e} (ratios={int_errs[0]/int_errs[1]:.2f}, {int_errs[1]/int_errs[2]:.2f} ~ 16)")

    lag_errs, cor_bessel_errs, nu_c, u_c = [], [], 0.75, 4.0
    z_star_lim = u_c * (nu_c * math.sqrt(1.0 - nu_c * nu_c) + math.asin(nu_c))
    for N_l in [800.0, 3200.0, 12800.0]:
        L_l = float(int(round(math.sqrt(1.2 * N_l))))
        lam = L_l * (L_l + 1.0) / N_l
        pe_l = exact_phi_3j_chain(N_l, 1.0 - 6.0 / N_l, L_l, M, sig)[L_l]
        lag_errs.append(abs(pe_l - (-math.sqrt(1.0 / 24.0) * (lam ** 1.5) * math.exp(-0.5 * lam) * (4.0 - lam))))
        xc = 1.0 - u_c / N_l
        Lc = math.floor(nu_c * N_l * math.sqrt(1.0 - xc * xc) - 0.5)
        nv = (Lc + 0.5) / (N_l * math.sqrt(1.0 - xc * xc))
        s_nv = math.sqrt(1.0 - nv * nv)
        zs = u_c * (nv * s_nv + math.asin(nv))
        H0 = (35.0 * s_nv / (8.0 * nv) + nv / (4.0 * s_nv) + 5.0 * (nv ** 3) / (24.0 * (s_nv ** 3)) - 8.75 * u_c / zs) / nv
        lim_v = epsilon_phase(M, sig) * edge_factor_R(3, 2.0) * math.sqrt(zs / (2.0 * u_c * nv * s_nv)) * (float(bessel_j(3, zs)) - (nv * H0 / (2.0 * u_c)) * float(bessel_j(4, zs)))
        vc_c, _ = phi_3j_u2_cov(N_l, xc, float(Lc), M, sig)
        cor_bessel_errs.append(abs(vc_c - lim_v))
    assert 3.8 < lag_errs[0] / lag_errs[1] < 4.2 and 3.9 < cor_bessel_errs[1] / cor_bessel_errs[2] < 4.1
    print(f"  [PASS] C6a Laguerre corner (n=1, p_+=3, lam~1.2): |Phi-Lag| = {lag_errs[0]:.2e}->{lag_errs[1]:.2e}->{lag_errs[2]:.2e} (4x-N ratios={lag_errs[0]/lag_errs[1]:.2f}, {lag_errs[1]/lag_errs[2]:.2f} ~ 4)")
    print(f"  [PASS] C6b Cor 3.2 Bessel limit (z_*={z_star_lim:.6f}): |Phi_U2cov-lim(nu_L)| = {cor_bessel_errs[0]:.2e}->{cor_bessel_errs[1]:.2e}->{cor_bessel_errs[2]:.2e} (4x-N ratios={cor_bessel_errs[0]/cor_bessel_errs[1]:.2f}, {cor_bessel_errs[1]/cor_bessel_errs[2]:.2f} ~ 4)")


def run_part_d_negative_controls():
    print("\n=== Part D: Mandatory Negative Controls [numerical evidence] ===")
    N, M, sig, L = 1280.0, 2.0, -1.0, 5.0
    x0 = 1.0 - 4.0 / N
    pe = exact_phi_3j_chain(N, x0, L, M, sig)[L]
    v_no_E, _ = phi_3j_u2_cov(N, x0, L, M, sig, include_edge=False)
    rel_no_E = abs(pe - v_no_E) / abs(pe)
    assert abs(rel_no_E - (2.0 / math.sqrt(3.0) - 1.0)) < 1e-4
    print(f"  [PASS] NC1 Omitting E_N(x) at n=0 (p_+=3, N=1280): rel_err = {rel_no_E:.4f} (exact 2/sqrt(3)-1 = {2.0/math.sqrt(3.0)-1.0:.4f})")

    p3_rels, raw_errs, noc0_errs = [], [], []
    for N_t in [320.0, 1280.0]:
        xt, qt = 1.0 - 4.0 / N_t, (L + 0.5) / N_t
        w3j = ((1.0 - xt) / 2.0) ** 1.5 * ((1.0 + xt) / 2.0) ** 0.5
        Tqt, _, G2qt, Dt = hahn_u2_data(qt, xt, M, sig)
        Tq2 = Tqt + (1.0 / (N_t * N_t)) * G2qt
        d_q2, _ = wigner_d_one(L, M, sig, Tq2)
        pet = exact_phi_3j_chain(N_t, xt, L, M, sig)[L]
        p3_rels.append(abs(pet - edge_factor_3j(N_t, xt, M, sig) * ((math.sin(Tq2) ** 2 / Dt) ** 0.25) * d_q2) / abs(pet))
        v_raw, _ = phi_3j_u2_cov(N_t, xt, L, M, sig, covariant=False)
        v_noc0, _ = phi_3j_u2_cov(N_t, xt, L, M, sig, include_c0=False)
        raw_errs.append(abs(pet - v_raw) / (edge_factor_3j(N_t, xt, M, sig) * w3j))
        noc0_errs.append(abs(pet - v_noc0) / (edge_factor_3j(N_t, xt, M, sig) * w3j))
    exact_nc2 = 1.0 - (23.0 / 24.0) ** 3.5
    assert abs(p3_rels[1] - exact_nc2) < 5e-4 and 0.90 < raw_errs[0] / raw_errs[1] < 1.05 and 3.8 < noc0_errs[0] / noc0_errs[1] < 4.2
    print(f"  [PASS] NC2 Nonlinear shift E_N*W_q^[2]*d(Theta_q^[2]) at n=0: rel_err = {p3_rels[0]:.4f}->{p3_rels[1]:.4f} (exact 1-(23/24)^(7/2) = {exact_nc2:.4f})")
    print(f"  [PASS] NC3a Non-covariant raw d_Theta at n=0: 4x-N ratio = {raw_errs[0]/raw_errs[1]:.2f} (~1 = O(1) double-pole failure)")
    print(f"  [PASS] NC3b Omitting C_0^{{(3)}}(x) contact at n=0: 4x-N ratio = {noc0_errs[0]/noc0_errs[1]:.2f} (~4 = O(N^-1) simple-pole degradation)")

    nu_c, u_c = 0.75, 4.0
    s_nu, lam_s = math.sqrt(1.0 - nu_c * nu_c), 2.0 * u_c * nu_c * nu_c
    z_star = u_c * (nu_c * s_nu + math.asin(nu_c))
    H_0 = (35.0 * s_nu / (8.0 * nu_c) + nu_c / (4.0 * s_nu) + 5.0 * (nu_c ** 3) / (24.0 * (s_nu ** 3)) - 8.75 * u_c / z_star) / nu_c
    lag_lim = -math.sqrt(1.0 / 6.0) * (lam_s ** 1.5) * math.exp(-0.5 * lam_s)
    bes_lim = -edge_factor_R(3, 2.0) * math.sqrt(z_star / (2.0 * u_c * nu_c * s_nu)) * (float(bessel_j(3, z_star)) - (nu_c * H_0 / (2.0 * u_c)) * float(bessel_j(4, z_star)))
    exact_nc4 = abs(lag_lim - bes_lim) / abs(lag_lim)
    c_errs = []
    for N_c in [800.0, 3200.0, 12800.0]:
        xc = 1.0 - u_c / N_c
        L_c = float(int(round(nu_c * N_c * math.sqrt(1.0 - xc * xc) - 0.5)))
        pe_c, (vc_c, _) = exact_phi_3j_chain(N_c, xc, L_c, M, sig)[L_c], phi_3j_u2_cov(N_c, xc, L_c, M, sig)
        c_errs.append(abs(pe_c - vc_c) / abs(pe_c))
    assert abs(c_errs[2] - exact_nc4) < 1e-3
    print(f"  [PASS] NC4 Coalescent Laguerre corner (n=0, nu=0.75, N=800->12800): rel_err = {c_errs[0]:.4f}->{c_errs[1]:.4f}->{c_errs[2]:.4f} (exact = {exact_nc4:.4f})")


if __name__ == "__main__":
    run_part_c_convergence()
    run_part_d_negative_controls()
    print("\nALL 3J NUMERICAL BENCHMARKS & NEGATIVE CONTROLS PASSED.")
