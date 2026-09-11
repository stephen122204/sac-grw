# General-flux signed-gradient update (derivation before implementation)

## The update

For `u_t + f(u)_x = nu u_xx`, differentiate in `x` and set `w = u_x`:

    w_t + ( f'(u) w )_x = nu w_xx.

This is a transport-diffusion conservation law for `w` with velocity field
`f'(u)`. The signed particle measure `w^N = sum_i m_i delta(x - X_i)` therefore
advances by

    X_i <- X_i + f'( u_i ) dt + sqrt(2 nu dt) Z_i,     Z_i ~ N(0,1) iid,

with `u_i` a reconstruction of `u` at particle `i`. The paper's *inclusive*
schedule takes the value immediately to the RIGHT of the particle,

    u_i = u_right,i = u_{-inf} + sum_{j <= i} m_j        (sorted order),

so the general-flux inclusive velocity is

    v_i = f'( u_right,i ).

Specialisations, on the SAME schedule:

| flux            | `f'(u)` | `v_i`             |
|-----------------|---------|-------------------|
| Burgers `u^2/2` | `u`     | `u_right,i`       |
| cubic `u^3/3`   | `u^2`   | `u_right,i^2`     |

For Burgers this is exactly `u_left + cumsum(m)`, i.e. the existing production
velocity, so no transport discretisation changes between the two fluxes. That
is the point of the choice: only `f` varies.

## Scope and consistency

At particle `i` the field jumps by `m_i`, from `u_left,i = u_right,i - m_i` to
`u_right,i`. The conservative (Rankine-Hugoniot) speed of that jump is the
secant

    s_i = [ f(u_right,i) - f(u_left,i) ] / m_i.

The inclusive one-sided choice differs from it by

    Burgers:  u_right - s_i = m_i / 2,
    cubic:    u_right^2 - s_i = u_right m_i - m_i^2 / 3.

Both are `O(m_i) = O(1/N)`, so the inclusive scheme is consistent to first order
in particle mass and the discrepancy vanishes in the particle limit; at finite
`N` it is not exactly jump-conservative. This is a property of the *existing*
Burgers solver, inherited unchanged by the cubic case, which is precisely why
the comparison is controlled. Using secant velocities instead would change the
transport discretisation and would require a Burgers-secant bridge run at the
same parameters to stay interpretable; that is not done here.

## What cubic flux does and does not add

`f'(u) = u^2 >= 0` for every `u`. It vanishes at `u = 0` but does NOT change
sign there, so on this profile every particle drifts right or stands still:
cubic flux does **not** supply left/right propagation merely because the
solution crosses zero. What it does supply is a non-quadratic flux with a
genuine inflection, `f''(u) = 2u`, changing sign at `u = 0`, so the
characteristic speed is no longer an affine function of the reconstructed state
and the velocity is no longer the cumulative sum itself.

Relaxation speed (not exercised by the conditional-mean transport used in the
coupling study, but recorded for scope): the subcharacteristic condition reads
`a > max |f'(u)| = max u^2`, so on this profile `max|u| = 0.8` gives
`max|f'| = 0.64` and the study value `a = 2` satisfies it with margin.
