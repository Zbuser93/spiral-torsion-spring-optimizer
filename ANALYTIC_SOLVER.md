# Spiral Torsion Spring — Analytic Optimizer Specification

A specification for a closed-form replacement of the SHGO-based optimizer in
`spiral_torsion_spring_optimizer.py`. Two objectives are supported:
maximizing stiffness `K` and maximizing preload torque `M_pre`. Both are
solved by a 1D root-find on thickness `t`, with arc-length `L` determined
on the active constraint curve.

All units: mm, N, MPa, radians.

---

## 1. Background and assumptions

This specification stays in the **straight-beam** regime — the same
constitutive law used in the existing optimizer:

- Moment-curvature: `M = E·I·κ` with `I = h·t³/12`
- Stress: `σ = E·t·Δθ_R / (2L)` (linear in distance from centroid)

A Winkler-Bach extension is sketched at the end and remains analytic, but is
not part of this initial implementation. Eventually the constitutive law will
be replaced by `curvebeam.solve(SpiralSegment, ...)` calls, at which point the
problem becomes numerical (still a small 2×2 system, but with each evaluation
costing ~1ms instead of being free).

The geometric chain (theta_EMD, theta_pre, radius_pre, etc.) is identical to
the existing optimizer — see `calculate_theta_EMD`, `calculate_radius_pre`
and friends in `spiral_torsion_spring_optimizer.py`. We replicate those
formulas inline below for self-containment.

---

## 2. Inputs

The user supplies:

| Symbol | Meaning | Units |
|---|---|---|
| `E` | Young's modulus | MPa |
| `σ_Y` | Yield stress | MPa |
| `SF` | Safety factor (multiplier on σ_Y) | — |
| `h` | Out-of-plane height | mm |
| `R_max` | Maximum permitted outer radius at preload | mm |
| `R_center` | Hub radius (inner clamp) | mm |
| `pitch_0` | Minimum coil-to-coil gap | mm |
| `Δθ_opt` | Required range of motion | rad |
| `M_pre_min` | Required minimum preload torque | N·mm |
| `nozzle` | Printer nozzle diameter | mm |

Decision variables: `t` (thickness) and `L` (effective arc-length). The
preload torque `M_pre` is either fixed (max-stiffness objective) or a
decision variable (max-torque objective); see below.

Derived constants:
```
σ_allow = SF · σ_Y
t_min   = 2 · nozzle                 (manufacturability floor)
t_hub   = 2 · R_center               (hub diameter ceiling)
```

---

## 3. The geometric chain (unchanged from existing optimizer)

These formulas express the spiral packing geometry. They depend on `(t, L)`
only and are independent of the constitutive law.

```
R_E       = R_center + t/2 + pitch_0          # outer radius of inner coil
                                                # (where the spring "begins")
θ_IMD     = 2π · R_E / (t + pitch_0)          # angle from origin to start of spring
arc_IMD   = R_E · θ_IMD / 2                    # arc-length from origin to start
arc_MD    = L + arc_IMD                        # total arc-length from origin
θ_MD      = sqrt(4π · arc_MD / (t + pitch_0)) # total winding angle from origin
θ_EMD     = θ_MD − θ_IMD                       # winding angle of effective spring
                                                # at fully-packed state
θ_pre     = θ_EMD − Δθ_opt                     # winding angle at preload state
R_pre     = 2L / θ_pre − R_E + t/2             # outer radius at preload state
```

**Key property:** `R_pre(t, L)` depends on `(t, L)` only — not on `M_pre`,
not on the constitutive law. The preload state is defined geometrically as
"wound enough that the next Δθ_opt of winding fully packs the spring at
pitch_0." The constitutive law determines what *force* is needed to hold
the spring in that state, but the state itself is geometric.

For numerical robustness, guard `θ_pre > 0`. If `θ_pre ≤ 0`, the spring
fully packs before reaching the preload state — that `(t, L)` pair is
infeasible.

---

## 4. The constraint surfaces

All five constraints in the original `cons_ms`, expressed as functions of
`(t, L)` for the straight-beam constitutive law:

### 4.1 Stress (depends on M_pre)

The bending stress at full deflection:
```
Δθ_pre  = 12 · M_pre · L / (E · h · t³)
Δθ_R    = Δθ_pre + Δθ_opt
σ_max   = E · t · Δθ_R / (2L)
```

Constraint: `σ_max ≤ σ_allow`. At equality, this defines a curve relating
`(t, L, M_pre)`. We use two specializations:

**Stress curve at fixed M_pre (used in max-stiffness objective):**

Solving `σ_max = σ_allow` for `L`:
```
L_stress(t; M_pre) = E · h · Δθ_opt · t³ / (2 · (h · σ_allow · t² − 6 · M_pre))
```
Has a vertical asymptote at `t* = sqrt(6 · M_pre / (h · σ_allow))`. Below
`t*`, preload alone exceeds yield — infeasible at any L.

**M_pre on the stress curve (used in max-torque objective):**

Solving `σ_max = σ_allow` for `M_pre`:
```
M_pre_on_stress(t, L) = h · σ_allow · t² / 6 − E · h · Δθ_opt · t³ / (12L)
```
Closed form, linear in `1/L`.

### 4.2 Max preload radius

`R_pre(t, L) ≤ R_max`. Geometric, depends on `(t, L)` only.

`R_pre` is monotone increasing in `L` at fixed `t`. The L-bound from this
constraint:
```
L_Rmax(t) = solve { R_pre(t, L) = R_max for L }
```
via `brentq` on a bracket that spans the feasible L range.

`R_pre` is *non-monotonic* in `t` at fixed `L`. This matters for the
max-stiffness solve: walking up the stress curve, `R_pre` first decreases,
then increases, crossing `R_max` twice. The optimum is at the second
(larger-t) crossing. See §5.1.

### 4.3 Positive radius

`R_pre(t, L) ≥ R_E`. Equivalently, the spring's outer end must be outside
its inner clamp. In practice this is satisfied with large margin whenever
the other constraints are satisfied; treat as a sanity check, not an
active constraint to solve against.

### 4.4 Annulus packing

Spring must fit in the annulus at minimum pitch:
```
L ≤ L_annulus(t) = π · (R_max² − R_center²) / (t + pitch_0)
```

### 4.5 Manufacturability and hub geometry

```
2 · nozzle ≤ t < 2 · R_center
```

---

## 5. Max-stiffness objective

**Inputs:** `M_pre` is fixed at `M_pre_min` (the user's required preload).

**Objective:** maximize `K = E · h · t³ / (12L)`.

### 5.1 Reduction to a 1D root-find

**Claim:** at the optimum, the stress and `R_pre = R_max` constraints are
simultaneously active.

**Proof sketch:** along the stress equality curve `L = L_stress(t)`,
substitute into the stiffness expression:
```
K_on_stress(t) = E · h · t³ / (12 · L_stress(t))
              = E · h · t³ · (h · σ_allow · t² − 6 · M_pre) / (6 · E · h · t³ · Δθ_opt)
              = (h · σ_allow · t² − 6 · M_pre) / (6 · Δθ_opt)
```

This is a clean quadratic in `t`, monotone increasing for `t > 0`. So along
the stress curve, larger `t` always means stiffer spring. The optimum is
therefore at the largest feasible `t`, which is bounded by either:

- `R_pre = R_max` (the typical case for well-conditioned designs), or
- `L_stress(t) = L_annulus(t)` (when the spring is allowed to be much
  larger than annulus packing permits — rare, treat as a fallback).

For the typical case, the optimum is the **larger of the two roots** of:
```
F(t) = R_pre(t, L_stress(t; M_pre)) − R_max = 0
```

`F(t)` has two roots because `R_pre` along the stress curve is non-monotonic
in `t` (drops, then climbs). The two roots bracket the t-window where
`R_pre ≤ R_max`. K is monotone increasing across this window, so the larger
root is the optimum.

### 5.2 Algorithm

```python
def maximize_stiffness(inputs):
    M_pre = inputs['M_pre_min']

    # 1. Compute stress curve asymptote and feasibility bracket
    t_star = sqrt(6 * M_pre / (h * sigma_allow))
    t_lo_search = max(t_min, t_star * 1.001)
    t_hi_search = t_hub - eps

    # 2. Find both roots of F(t) = R_pre(t, L_stress(t)) - R_max
    t_grid = linspace(t_lo_search, t_hi_search, 200)
    F_grid = [F(t) for t in t_grid]
    sign_changes = where(diff(sign(F_grid)) != 0)

    if len(sign_changes) < 2:
        # Could be: only one root (annulus binds before second crossing)
        # or: zero roots (R_pre never reaches R_max — spring is too small
        #                   for the envelope, no constraint from radius)
        # Handle these as fallbacks; see §5.3
        ...

    # 3. The optimum is the larger root
    i = sign_changes[-1]
    t_opt = brentq(F, t_grid[i], t_grid[i+1], xtol=1e-10)
    L_opt = L_stress(t_opt, M_pre)

    # 4. Verify all other constraints are satisfied (annulus, positive radius)
    assert L_opt <= L_annulus(t_opt) + eps
    assert R_pre(t_opt, L_opt) >= R_E(t_opt) + eps

    return t_opt, L_opt
```

Solve time: ~150 µs.

### 5.3 Fallback cases

**No roots of F(t):** `R_pre < R_max` everywhere on the stress curve. The
radius constraint is non-binding. Active constraints become stress + something
else. Walk through:
- Try stress + annulus: solve `L_stress(t) = L_annulus(t)`, take larger root.
- Try stress + hub: optimum is at `t = t_hub − eps`, `L = L_stress(t_hub)`.

**One root of F(t):** One side of the radius window is hidden by another
constraint. Re-bracket using the active constraint envelope.

For robustness, after the analytic root-find, evaluate the objective at a
small set of candidate points (intersection of every constraint pair) and
return the best feasible one. There are at most ~6 candidate intersections,
each costs a single 2×2 solve.

### 5.4 Recovering the unutilized elasticity

After solving, the user's prescribed `M_pre_min` may leave stress headroom
(if it isn't the binding constraint that determined `t_opt`). Use the
existing `calculate_torque_pre_max` formula to find the largest preload
torque that this spring geometry could deliver without exceeding stress:

```
M_pre_max = (h · t² · 2L · σ_allow − E · h · t³ · Δθ_opt) / (12L)
```

Report both `M_pre_min` (input) and `M_pre_max` (capacity) so the user can
choose to increase preload if they want.

---

## 6. Max-torque objective

**Inputs:** `M_pre_min` is ignored (the goal is to maximize `M_pre`).

**Decision variables:** `t`, `L`, and `M_pre`.

**Objective:** maximize `M_pre`.

### 6.1 Reduction to a 1D root-find

**Claim:** at the optimum, the stress constraint is active, and either
`R_pre = R_max` or `L = L_annulus(t)` is also active.

**Proof sketch:** if stress weren't active, you could increase `M_pre`
without violating any other constraint (all other constraints depend on
`(t, L)` only). So stress must bind.

On the stress curve, `M_pre = M_pre_on_stress(t, L) = h·σ_allow·t²/6 − E·h·Δθ_opt·t³/(12L)`.

For fixed `t`: `dM_pre/dL = E·h·Δθ_opt·t³/(12L²) > 0`. So you want `L` as
large as possible at every `t`. The active upper bound on `L` is
`min(L_annulus(t), L_Rmax(t))`.

For fixed `L`, taking `dM_pre/dt = 0`:
```
t_interior(L) = 4 · L · σ_allow / (3 · E · Δθ_opt)
```
`M_pre` has an interior peak in `t` at this value. But this peak is in the
*unconstrained* problem; the constrained optimum is generally at a boundary
imposed by `R_pre ≤ R_max` (which excludes thicker-and-larger springs).

### 6.2 Algorithm

```python
def maximize_torque(inputs):
    # 1. For each t, find the active upper bound on L
    def L_upper(t):
        L_a = L_annulus(t)
        L_R = L_Rmax(t)             # may be None if R_pre < R_max even at L_a
        return L_a if L_R is None else min(L_a, L_R)

    # 2. Find feasible t range (where M_pre_on_stress > 0 at L_upper(t))
    def feasible(t):
        L = L_upper(t)
        if L <= 0: return False
        M = M_pre_on_stress(t, L)
        if M <= 0: return False
        # Also check radius_pre >= R_E + eps and t_min <= t < t_hub
        if R_pre(t, L) < R_E(t): return False
        return True

    t_lo, t_hi = scan_for_feasible_range(...)

    # 3. Maximize M_pre_on_stress(t, L_upper(t)) over t
    res = minimize_scalar(
        lambda t: -M_pre_on_stress(t, L_upper(t)),
        bounds=(t_lo, t_hi),
        method='bounded',
        options={'xatol': 1e-8},
    )
    t_opt = res.x
    L_opt = L_upper(t_opt)
    M_opt = M_pre_on_stress(t_opt, L_opt)

    # 4. Identify which upper bound is active
    active = 'annulus' if L_opt == L_annulus(t_opt) else 'radius_pre'

    return t_opt, L_opt, M_opt, active
```

Solve time: ~few hundred µs (slower than max-stiffness because the inner
1D structure isn't as clean — `minimize_scalar` rather than `brentq`).

If you want to push it to a pure root-find: at the optimum, you have stress
+ one other constraint binding. Enumerate the two cases (stress + R_pre,
stress + annulus), solve each as a 2×2 algebraic system, evaluate `M_pre`
at each, return the best. Same structure as max-stiffness but with the
objective evaluated at the candidate solutions.

### 6.3 The two objectives can give very different springs

Worked example (inputs from `spiral_torsion_spring_optimizer.py`):
- `E=3100, σ_Y=85, h=12, R_max=70, R_center=15, pitch_0=0.5, Δθ_opt=3.14, SF=0.8`
- `M_pre_min = 2800` for max-stiffness; ignored for max-torque

| Objective | t [mm] | L [mm] | M_pre [N·mm] | K [N·mm/rad] |
|---|---|---|---|---|
| Max stiffness | 8.76 | 856.8 | 2800 (input) | 2431 |
| Max torque | 7.28 | 1199.9 | 4075 | 995 |

Max-torque uses a thinner, longer spring with much higher preload but
~2.4× less stiffness. Max-stiffness packs the energy into a thicker, shorter
spring at the radius limit.

---

## 7. Constraint pairs and 2×2 systems

For any well-conditioned design, the optimum is at the intersection of two
active constraints. Enumerating the relevant pairs:

| Pair | Equations | Notes |
|---|---|---|
| stress + R_pre | `σ_max = σ_allow`, `R_pre = R_max` | Most common |
| stress + annulus | `σ_max = σ_allow`, `L = L_annulus(t)` | When R_max is loose |
| stress + hub | `σ_max = σ_allow`, `t = t_hub − eps` | When both above are loose |
| stress + manuf. | `σ_max = σ_allow`, `t = t_min` | Degenerate (very thin) |
| R_pre + annulus | `R_pre = R_max`, `L = L_annulus(t)` | Stress not binding (rare) |

For a robust solver: solve each 2×2 system, filter for feasibility against
*all* constraints, evaluate objective at each candidate, return the best.
This handles fallback cases automatically without bespoke logic.

The `R_pre + annulus` case is interesting because it doesn't involve stress.
This happens when stress is non-binding because the prescribed preload is
small relative to what the geometry can carry. In that regime, the spring
"wants to be smaller" than allowed and stress headroom remains.

---

## 8. Output and reporting

The solver returns:

```python
@dataclass
class SpringSolution:
    # Decision variables
    thickness: float            # mm
    arclength: float            # mm
    preload_torque: float       # N*mm

    # Performance
    stiffness: float            # N*mm/rad
    stress_max: float           # MPa
    stress_utilization: float   # stress_max / sigma_allow, in [0, 1]

    # Geometric properties
    radius_E: float             # mm, inner coil
    radius_pre: float           # mm, outer coil at preload
    radius_R: float             # mm, outer coil at rest
    pitch_R: float              # mm, as-printed pitch
    n_revolutions: float

    # Diagnostics
    active_constraints: list[str]   # e.g. ['stress', 'radius_pre']
    headroom: dict                  # margin on each non-active constraint
    M_pre_max: float                # capacity if max-stiffness objective
                                    #   left elasticity unused
    objective: str                  # 'max_stiffness' or 'max_torque'
```

The `headroom` dict surfaces "you have X MPa of unused stress" or "the spring
is Y mm under R_max" so the user can see at a glance whether they could
ask for more.

---

## 9. Winkler-Bach extension (sketch, not for initial implementation)

If we replace the straight-beam constitutive law with Winkler-Bach, the
formulas pick up `R₀` (radius of curvature) as a parameter but stay
algebraic:

```
r_i = R₀ − t/2
r_o = R₀ + t/2
r_n = (r_o − r_i) / ln(r_o / r_i)            # neutral radius (always closed-form)
e   = R₀ − r_n                                # eccentricity
A   = h · t

# Moment-curvature (curvature change vs intrinsic):
M = E · A · e · (κ − κ₀) / κ₀                # κ₀ = 1/R₀
=> κ = κ₀ · (1 + M / (E · A · e))            # closed-form inverse

# Stiffness, integrated over varying R₀(s) along the spiral:
K = M / Δθ_pre = M / ∫₀ᴸ (M · κ₀(s) / (E · A · e(s))) ds
  = (E · A) / ∫₀ᴸ κ₀(s) / e(s) ds

# For linear R₀(s) = R_E + (R_pre − R_E) · s / L, the integral is
# closed-form (involves ln(R_pre / R_E) terms). Tractable.

# Stress at inner fiber (where it peaks for tight curvature):
σ_inner = E · (κ_inner − κ₀_inner) / κ₀_inner · (r_n_inner − r_i_inner) / r_i_inner
        = (M / A) · (r_n_inner − r_i_inner) / (e_inner · r_i_inner)
```

The 2×2 system at the optimum has the same structure but with these
expressions in place of the straight-beam ones. `brentq` and the constraint
enumeration logic are unchanged. Solve time stays in the µs range.

For very tight inner coils (`t / R_E > 0.3`), this captures the inner-fiber
stress amplification that straight-beam under-predicts by 20–40%. For looser
designs (`t / R_E < 0.1`), the correction is negligible.

When eventually replaced by `curvebeam.solve(SpiralSegment, ...)` calls, the
problem becomes numerical (each constraint evaluation is an ODE integration),
but the 2×2 system structure is preserved — `scipy.optimize.fsolve` on the
active-constraint pair, with constraint values computed by curvebeam. Solve
time goes to ~100ms (a few solve calls), still much faster than SHGO's
seconds.

---

## 10. Implementation checklist

- [ ] Module `analytic_solver.py` with two entry points: `maximize_stiffness(inputs)` and `maximize_torque(inputs)`
- [ ] Geometric chain replicated as standalone functions (no `SpiralTorsionSpring` class dependency for the inner math)
- [ ] Constraint surface evaluators: `L_stress`, `L_annulus`, `L_Rmax`, `R_pre`, `M_pre_on_stress`
- [ ] 2×2 intersection solvers for each constraint pair in §7
- [ ] Candidate evaluation and feasibility filter
- [ ] `SpringSolution` dataclass output
- [ ] Sanity-check against the existing `cons_ms` to verify we agree on which points are feasible
- [ ] Performance target: < 1ms total solve time
- [ ] Unit tests on the original example inputs (`M_pre_min = 2800`, etc.) confirming both objectives produce expected results
- [ ] Verify against SHGO: SHGO with very high `n` and `iters` should converge to the same answer as the analytic solver (within numerical tolerance)

---

## 11. Notes on the existing code

The existing `SpiralTorsionSpring` class wraps the geometry chain, the
constraint evaluation, and the SHGO optimizer in one place. The analytic
solver should:

1. Keep `SpiralTorsionSpring` as the public interface (so `to_dict()`,
   `verbose()`, etc. still work).
2. Add `maximize_stiffness_analytic` and `maximize_torque_analytic` class
   methods alongside the existing `maximize_stiffness` (which uses SHGO).
3. The analytic methods should populate the same instance attributes
   (`thickness`, `arclength_E`, `stiffness`, etc.) so downstream code
   (CAD generation, plotting) doesn't change.
4. Keep SHGO around as a fallback for inputs the analytic solver flags as
   pathological (e.g. zero feasible 2×2 intersections).
