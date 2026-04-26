from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, minimize_scalar


@dataclass
class SpringSolution:
    thickness: float
    arclength: float
    preload_torque: float
    stiffness: float
    stress_max: float
    stress_utilization: float
    radius_E: float
    radius_pre: float
    radius_R: float
    pitch_R: float
    n_revolutions: float
    active_constraints: list
    headroom: dict
    M_pre_max: float
    objective: str


# ---------------------------------------------------------------------------
# Geometric helpers — match SpiralTorsionSpring.calculate_* methods exactly
# ---------------------------------------------------------------------------

def _radius_E(t, radius_center, pitch_0):
    return radius_center + t / 2 + pitch_0


def _theta_IMD(t, radius_center, pitch_0):
    RE = _radius_E(t, radius_center, pitch_0)
    return 2 * np.pi * RE / (t + pitch_0)


def _arc_IMD(t, radius_center, pitch_0):
    RE = _radius_E(t, radius_center, pitch_0)
    theta_IMD = _theta_IMD(t, radius_center, pitch_0)
    return RE * theta_IMD / 2


def _theta_EMD(t, L, radius_center, pitch_0):
    arc_IMD = _arc_IMD(t, radius_center, pitch_0)
    theta_IMD = _theta_IMD(t, radius_center, pitch_0)
    arc_MD = L + arc_IMD
    theta_MD = np.sqrt(4 * np.pi * arc_MD / (t + pitch_0))
    return theta_MD - theta_IMD


def _theta_pre(t, L, radius_center, pitch_0, deltatheta_opt):
    return _theta_EMD(t, L, radius_center, pitch_0) - deltatheta_opt


def _radius_pre(t, L, radius_center, pitch_0, deltatheta_opt):
    RE = _radius_E(t, radius_center, pitch_0)
    tp = _theta_pre(t, L, radius_center, pitch_0, deltatheta_opt)
    if tp <= 0:
        return np.inf
    return 2 * L / tp - RE + t / 2


# ---------------------------------------------------------------------------
# Constraint surface evaluators
# ---------------------------------------------------------------------------

def _L_stress(t, M_pre, E, h, sigma_allow, deltatheta_opt):
    """Arc-length at which stress = sigma_allow, given t and M_pre."""
    denom = h * sigma_allow * t**2 - 6 * M_pre
    if denom <= 0:
        return np.inf
    return E * h * deltatheta_opt * t**3 / (2 * denom)


def _L_annulus(t, R_max, R_center, pitch_0):
    return np.pi * (R_max**2 - R_center**2) / (t + pitch_0)


def _M_pre_on_stress(t, L, E, h, sigma_allow, deltatheta_opt):
    """Preload torque on the stress-active surface."""
    return h * sigma_allow * t**2 / 6 - E * h * deltatheta_opt * t**3 / (12 * L)


def _L_Rmax(t, R_max, R_center, pitch_0, deltatheta_opt):
    """
    Return the largest L s.t. R_pre(t, L) <= R_max.

    Returns None when R_pre(L_annulus) <= R_max (radius never binds in range).
    Returns None when min R_pre > R_max (this t is entirely infeasible).

    Note: R_pre is NOT monotone in L. It falls from +inf near the geometric
    feasibility floor (theta_pre → 0+) to a minimum, then rises. We locate
    the minimum via minimize_scalar, then bracket the right crossing with brentq.
    """
    L_a = _L_annulus(t, R_max, R_center, pitch_0)
    theta_IMD = _theta_IMD(t, R_center, pitch_0)
    L_min_geo = max(
        1e-6,
        ((theta_IMD + deltatheta_opt)**2 * (t + pitch_0) / (4 * np.pi)
         - _arc_IMD(t, R_center, pitch_0)) * 1.001,
    )
    if L_min_geo >= L_a:
        return None

    rp_at_La = _radius_pre(t, L_a, R_center, pitch_0, deltatheta_opt)
    if rp_at_La != np.inf and rp_at_La <= R_max:
        return None  # radius never limits in the feasible arc-length range

    # R_pre has a minimum over L. Locate it to determine feasibility and bracket.
    def rp_safe(L):
        rp = _radius_pre(t, L, R_center, pitch_0, deltatheta_opt)
        return 1e20 if (rp == np.inf or np.isnan(rp)) else rp

    min_res = minimize_scalar(rp_safe, bounds=(L_min_geo, L_a), method='bounded',
                              options={'xatol': 1e-6})
    L_at_min = min_res.x
    rp_min = rp_safe(L_at_min)

    if rp_min > R_max + 1e-6:
        return None  # R_pre is always above R_max; this t is infeasible

    # rp_min <= R_max and rp_at_La > R_max → root lies in (L_at_min, L_a)
    def f(L):
        rp = _radius_pre(t, L, R_center, pitch_0, deltatheta_opt)
        return 1e20 if rp == np.inf else rp - R_max

    if abs(f(L_at_min)) < 1e-6:
        return L_at_min  # degenerate: touching point, rp_min ≈ R_max
    try:
        return brentq(f, L_at_min, L_a, xtol=1e-10)
    except ValueError:
        return L_at_min  # fallback for near-degenerate case


# ---------------------------------------------------------------------------
# Feasibility check (all five constraints)
# ---------------------------------------------------------------------------

def _is_feasible(t, L, M_pre, E, h, sigma_allow, deltatheta_opt,
                 R_max, R_center, pitch_0, t_min, t_hub, eps=1e-6):
    if t < t_min - eps or t > t_hub - eps:
        return False
    if L <= 0:
        return False
    RE = _radius_E(t, R_center, pitch_0)
    rp = _radius_pre(t, L, R_center, pitch_0, deltatheta_opt)
    if rp == np.inf or rp < RE - eps or rp > R_max + eps:
        return False
    if L > _L_annulus(t, R_max, R_center, pitch_0) + eps:
        return False
    deltatheta_R = 12 * L * M_pre / (E * h * t**3) + deltatheta_opt
    stress = E * t * deltatheta_R / (2 * L)
    if stress > sigma_allow + eps:
        return False
    return True


# ---------------------------------------------------------------------------
# Full solution builder
# ---------------------------------------------------------------------------

def _build_full_solution(t, L, M_pre, inputs, objective, active_constraints):
    E = inputs['elasticity']
    h = inputs['height']
    SF = inputs['safety_factor']
    sigma_Y = inputs['stress_yield']
    R_center = inputs['radius_center']
    pitch_0 = inputs['pitch_0']
    deltatheta_opt = inputs['deltatheta_opt']
    R_max = inputs['max_radius_pre']
    sigma_allow = SF * sigma_Y

    RE = _radius_E(t, R_center, pitch_0)
    theta_EMD = _theta_EMD(t, L, R_center, pitch_0)
    theta_pre_val = theta_EMD - deltatheta_opt
    rp = 2 * L / theta_pre_val - RE + t / 2

    deltatheta_R = 12 * L * M_pre / (E * h * t**3) + deltatheta_opt
    theta_E = theta_EMD - deltatheta_R
    rR = 2 * L / theta_E - RE + t / 2
    pitch_R = 2 * np.pi * (rR - RE) / theta_E
    n_rev = theta_E / (2 * np.pi)

    K = E * h * t**3 / (12 * L)
    stress = E * t * deltatheta_R / (2 * L)
    M_pre_max = (h * t**2 * 2 * L * sigma_allow - E * h * t**3 * deltatheta_opt) / (12 * L)

    headroom = {
        'stress_MPa': sigma_allow - stress,
        'radius_pre_mm': R_max - rp,
        'arclength_annulus_mm': _L_annulus(t, R_max, R_center, pitch_0) - L,
    }
    return SpringSolution(
        thickness=t,
        arclength=L,
        preload_torque=M_pre,
        stiffness=K,
        stress_max=stress,
        stress_utilization=stress / sigma_allow,
        radius_E=RE,
        radius_pre=rp,
        radius_R=rR,
        pitch_R=pitch_R,
        n_revolutions=n_rev,
        active_constraints=active_constraints,
        headroom=headroom,
        M_pre_max=M_pre_max,
        objective=objective,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def maximize_stiffness(inputs: dict) -> SpringSolution:
    """
    Maximize torsional stiffness K = E·h·t³/(12L) at fixed preload torque.

    At the optimum, the stress constraint is active plus one geometric constraint.
    Solves by scanning for roots of F(t) = R_pre(t, L_stress(t)) - R_max,
    then verifying all candidates against every constraint.
    """
    E = inputs['elasticity']
    h = inputs['height']
    SF = inputs['safety_factor']
    sigma_Y = inputs['stress_yield']
    R_max = inputs['max_radius_pre']
    R_center = inputs['radius_center']
    pitch_0 = inputs['pitch_0']
    deltatheta_opt = inputs['deltatheta_opt']
    M_pre = inputs['torque_pre']
    nozzle = inputs['nozzle_diameter']

    sigma_allow = SF * sigma_Y
    t_min = 2 * nozzle
    t_hub = 2 * R_center
    eps = 1e-9

    t_star = np.sqrt(6 * M_pre / (h * sigma_allow)) if M_pre > 0 else 0.0
    t_lo = max(t_min, t_star * 1.001)
    t_hi = t_hub - eps
    if t_lo >= t_hi:
        raise ValueError(f"No feasible thickness range: t_lo={t_lo:.4f} >= t_hi={t_hi:.4f}")

    def F(t):
        L = _L_stress(t, M_pre, E, h, sigma_allow, deltatheta_opt)
        if np.isinf(L) or L <= 0:
            return 1e20
        rp = _radius_pre(t, L, R_center, pitch_0, deltatheta_opt)
        if rp == np.inf:
            return 1e20
        return rp - R_max

    t_grid = np.linspace(t_lo, t_hi, 300)
    F_grid = np.array([F(t) for t in t_grid])
    valid = np.isfinite(F_grid) & (np.abs(F_grid) < 1e15)
    t_valid = t_grid[valid]
    F_valid = F_grid[valid]

    candidates = []

    # Primary: stress + R_pre active
    for i in np.where(np.diff(np.sign(F_valid)))[0]:
        try:
            t_c = brentq(F, t_valid[i], t_valid[i + 1], xtol=1e-10)
            L_c = _L_stress(t_c, M_pre, E, h, sigma_allow, deltatheta_opt)
            candidates.append(('stress+radius_pre', t_c, L_c))
        except ValueError:
            pass

    # Fallback: stress + annulus active
    def G(t):
        L_s = _L_stress(t, M_pre, E, h, sigma_allow, deltatheta_opt)
        if np.isinf(L_s):
            return np.inf
        return L_s - _L_annulus(t, R_max, R_center, pitch_0)

    G_grid = np.array([G(t) for t in t_grid])
    G_mask = np.isfinite(G_grid)
    t_Gv = t_grid[G_mask]
    G_Gv = G_grid[G_mask]
    for i in np.where(np.diff(np.sign(G_Gv)))[0]:
        try:
            t_c = brentq(G, t_Gv[i], t_Gv[i + 1], xtol=1e-10)
            L_c = _L_stress(t_c, M_pre, E, h, sigma_allow, deltatheta_opt)
            candidates.append(('stress+annulus', t_c, L_c))
        except ValueError:
            pass

    # Fallback: stress + hub boundary
    t_c = t_hub - eps
    L_c = _L_stress(t_c, M_pre, E, h, sigma_allow, deltatheta_opt)
    if not np.isinf(L_c) and L_c > 0:
        candidates.append(('stress+hub', t_c, L_c))

    # Filter feasible candidates; pick highest K (largest t on stress curve)
    best_K, best = -np.inf, None
    for label, t_c, L_c in candidates:
        if not _is_feasible(t_c, L_c, M_pre, E, h, sigma_allow, deltatheta_opt,
                            R_max, R_center, pitch_0, t_min, t_hub):
            continue
        K = E * h * t_c**3 / (12 * L_c)
        if K > best_K:
            best_K, best = K, (label, t_c, L_c)

    if best is None:
        raise ValueError("No feasible solution found — check inputs")

    label, t_opt, L_opt = best
    return _build_full_solution(t_opt, L_opt, M_pre, inputs, 'max_stiffness', [label])


def maximize_torque(inputs: dict) -> SpringSolution:
    """
    Maximize preload torque M_pre subject to stress and geometric constraints.

    On the stress-active surface, M_pre is maximised by maximising L at each t.
    Uses minimize_scalar over t with L = min(L_annulus, L_Rmax) as the upper bound.
    """
    E = inputs['elasticity']
    h = inputs['height']
    SF = inputs['safety_factor']
    sigma_Y = inputs['stress_yield']
    R_max = inputs['max_radius_pre']
    R_center = inputs['radius_center']
    pitch_0 = inputs['pitch_0']
    deltatheta_opt = inputs['deltatheta_opt']
    nozzle = inputs['nozzle_diameter']

    sigma_allow = SF * sigma_Y
    t_min = 2 * nozzle
    t_hub = 2 * R_center
    eps = 1e-9

    def L_upper(t):
        L_a = _L_annulus(t, R_max, R_center, pitch_0)
        L_R = _L_Rmax(t, R_max, R_center, pitch_0, deltatheta_opt)
        return L_a if L_R is None else min(L_a, L_R)

    def neg_M(t):
        L = L_upper(t)
        if L <= 0:
            return 0.0
        RE = _radius_E(t, R_center, pitch_0)
        rp = _radius_pre(t, L, R_center, pitch_0, deltatheta_opt)
        # Explicit R_max guard in case _L_Rmax missed a non-monotone region
        if rp == np.inf or rp < RE or rp > R_max + 1e-4:
            return 0.0
        M = _M_pre_on_stress(t, L, E, h, sigma_allow, deltatheta_opt)
        if M <= 0:
            return 0.0
        return -M

    # Coarse grid scan: minimize_scalar with bounded Brent's method requires the
    # objective to be unimodal. neg_M is flat at 0 for infeasible t (where M < 0
    # or rp > R_max), so the initial golden-section evaluations may all land in the
    # flat region and miss the feasible well. Pre-scan first to narrow the bracket.
    t_grid = np.linspace(t_min, t_hub - eps, 60)
    nm_grid = np.array([neg_M(t) for t in t_grid])

    feasible_mask = nm_grid < 0
    if not np.any(feasible_mask):
        raise ValueError("No feasible positive preload torque found — check inputs")

    best_idx = int(np.argmin(nm_grid))
    # Find the contiguous feasible region that contains best_idx
    lo_idx = best_idx
    while lo_idx > 0 and nm_grid[lo_idx - 1] < 0:
        lo_idx -= 1
    hi_idx = best_idx
    while hi_idx < len(nm_grid) - 1 and nm_grid[hi_idx + 1] < 0:
        hi_idx += 1

    t_lo_refine = t_grid[lo_idx]
    t_hi_refine = t_grid[hi_idx]

    if t_lo_refine < t_hi_refine:
        res = minimize_scalar(neg_M, bounds=(t_lo_refine, t_hi_refine), method='bounded',
                              options={'xatol': 1e-8})
        t_opt = res.x
    else:
        t_opt = t_grid[best_idx]

    L_opt = L_upper(t_opt)
    M_opt = _M_pre_on_stress(t_opt, L_opt, E, h, sigma_allow, deltatheta_opt)

    if M_opt <= 0:
        raise ValueError("No feasible positive preload torque found — check inputs")

    L_a = _L_annulus(t_opt, R_max, R_center, pitch_0)
    active = ['stress', 'annulus' if abs(L_opt - L_a) < 1e-4 else 'radius_pre']

    return _build_full_solution(t_opt, L_opt, M_opt, inputs, 'max_torque', active)
