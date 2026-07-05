"""
Spike Island — Wilson-Cowan Neural Oscillator
===============================================
Two-population Wilson-Cowan model of excitatory and inhibitory neural
dynamics.  Produces oscillations, bistability, and fixed-point activity
depending on coupling strengths.

The core equations (continuous-time mean-field model):

    dE/dt = (-E + S(w_EE * E - w_EI * I + I_E)) / tau_E
    dI/dt = (-I + S(w_IE * E - w_II * I + I_I)) / tau_I

where S is a sigmoidal activation function, E and I are the mean
firing rates of the excitatory and inhibitory populations, and
the w_* parameters are synaptic coupling weights.

Key phenomena
-------------
- **Oscillations**: Strong E→I and I→E coupling with moderate E→E
  feedback produces sustained limit-cycle oscillations (analogous to
  gamma-range rhythms in cortex).
- **Bistability**: Strong recurrent excitation (high w_EE) produces
  two stable fixed points — a resting state and a persistent-activity
  state — modeling working memory or up/down states.
- **Fixed-point**: Weak coupling yields a single stable equilibrium.

References
----------
- Wilson, H.R. & Cowan, J.D. (1972). "Excitatory and inhibitory
  interactions in localized populations of model neurons."
  Biophysical Journal 12(1): 1–55.
- Ermentrout, G.B. & Terman, D. (2010). "Mathematical Foundations
  of Neuroscience." Springer, §10.3.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Activation function
# ---------------------------------------------------------------------------


def sigmoid(x: np.ndarray, theta: float = 1.0, r: float = 10.0) -> np.ndarray:
    """Sigmoidal activation function S(x) = 1 / (1 + exp(-r * (x - theta))).

    Parameters
    ----------
    x : np.ndarray
        Input values (can be scalars broadcast to arrays).
    theta : float
        Half-activation threshold (default 1.0).  When x == theta,
        the output is 0.5.
    r : float
        Slope / gain parameter (default 10.0).  Larger r makes the
        sigmoid steeper, approaching a step function.

    Returns
    -------
    np.ndarray
        Activation values in [0, 1].

    Notes
    -----
    This is the standard logistic sigmoid used in Wilson-Cowan models.
    It approximates the population firing rate as a function of net
    input current.  The parameter ``r`` controls how sharply the
    population transitions from silent to active.
    """
    return 1.0 / (1.0 + np.exp(-r * (x - theta)))


def sigmoid_scalar(x: float, theta: float = 1.0, r: float = 10.0) -> float:
    """Scalar version of :func:`sigmoid` for use in ODE right-hand side."""
    return float(1.0 / (1.0 + np.exp(-r * (x - theta))))


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WCParams:
    """Wilson-Cowan model parameters.

    Attributes
    ----------
    tau_E : float
        Time constant of the excitatory population (ms).  Typical: 80 ms.
    tau_I : float
        Time constant of the inhibitory population (ms).  Typical: 20 ms.
        Inhibitory neurons are faster, enabling gamma oscillations.
    w_EE : float
        Excitatory → excitatory coupling strength.  High values (≥ 1.2)
        produce bistability.
    w_EI : float
        Excitatory → inhibitory coupling strength.  Drives I population.
    w_IE : float
        Inhibitory → excitatory coupling strength (negative effect).
        Together with w_EI, controls oscillation frequency.
    w_II : float
        Inhibitory → inhibitory self-coupling.  Stabilizes I population.
    I_E : float
        External drive to excitatory population.  Controls the operating
        point along the nullcline.
    I_I : float
        External drive to inhibitory population.
    theta : float
        Sigmoid half-activation threshold (default 1.0).
    r : float
        Sigmoid gain / slope (default 10.0).
    """

    tau_E: float = 80.0
    tau_I: float = 20.0
    w_EE: float = 1.0
    w_EI: float = 1.0
    w_IE: float = 0.8
    w_II: float = 1.0
    I_E: float = 0.3
    I_I: float = 0.1
    theta: float = 1.0
    r: float = 10.0


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------


def wc_rhs(
    state: np.ndarray,
    params: WCParams,
    sigma: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Right-hand side of the Wilson-Cowan equations.

    Parameters
    ----------
    state : np.ndarray
        Current state [E, I] (firing rates, each in [0, 1]).
    params : WCParams
        Model parameters.
    sigma : float or None
        Additive noise standard deviation.  If set, Gaussian noise
        with this std is added to each derivative evaluation.
    rng : np.random.Generator or None
        Random number generator for noise.  Ignored if sigma is None.

    Returns
    -------
    np.ndarray
        Derivatives [dE/dt, dI/dt] in units of rate/ms.
    """
    E, I = state

    # Net inputs
    h_E = params.w_EE * E - params.w_EI * I + params.I_E
    h_I = params.w_IE * E - params.w_II * I + params.I_I

    # Sigmoidal activation
    s_E = sigmoid_scalar(h_E, params.theta, params.r)
    s_I = sigmoid_scalar(h_I, params.theta, params.r)

    # Derivatives
    dE = (-E + s_E) / params.tau_E
    dI = (-I + s_I) / params.tau_I

    # Optional noise
    if sigma is not None and rng is not None:
        dE += sigma * rng.normal()
        dI += sigma * rng.normal()

    return np.array([dE, dI])


# ---------------------------------------------------------------------------
# Numerical integration
# ---------------------------------------------------------------------------


def simulate_wc(
    params: WCParams,
    t_max: float = 2000.0,
    dt: float = 0.5,
    initial: tuple[float, float] | None = None,
    sigma: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the Wilson-Cowan model using forward Euler integration.

    Parameters
    ----------
    params : WCParams
        Model parameters.
    t_max : float
        Total simulation time in milliseconds (default 2 s).
    dt : float
        Integration timestep in milliseconds (default 0.5 ms).
        Must be smaller than tau_I for stability.
    initial : tuple of float or None
        Initial state (E_0, I_0).  Defaults to (0.05, 0.05).
    sigma : float or None
        Additive noise standard deviation per step.  Set to None for
        deterministic simulation.
    seed : int or None
        Random seed for reproducible noise.  Ignored if sigma is None.

    Returns
    -------
    t : np.ndarray
        Time vector in milliseconds (length N).
    state : np.ndarray
        State trajectory, shape (N, 2).  Columns are E and I.

    Notes
    -----
    Forward Euler is sufficient for Wilson-Cowan because the right-hand
    side is smooth and the system is low-dimensional.  For stiff variants
    (very different tau_E / tau_I ratios), consider RK4 or an implicit
    method.  The default dt = 0.5 ms is well below tau_I = 20 ms.
    """
    rng = np.random.default_rng(seed) if sigma is not None else None

    if initial is None:
        state = np.array([0.05, 0.05])
    else:
        state = np.array(initial, dtype=np.float64)

    n_steps = int(np.ceil(t_max / dt))
    t = np.arange(n_steps + 1) * dt
    trajectory = np.empty((n_steps + 1, 2), dtype=np.float64)
    trajectory[0] = state

    for k in range(n_steps):
        state = state + dt * wc_rhs(state, params, sigma=sigma, rng=rng)
        # Clamp to [0, 1] to prevent numerical blow-up
        state = np.clip(state, 0.0, 1.0)
        trajectory[k + 1] = state

    return t, trajectory


# ---------------------------------------------------------------------------
# Fixed-point analysis
# ---------------------------------------------------------------------------


def wc_fixed_points(
    params: WCParams,
    grid_resolution: int = 200,
) -> list[np.ndarray]:
    """Find fixed points of the Wilson-Cowan system by nullcline intersection.

    Parameters
    ----------
    params : WCParams
        Model parameters.
    grid_resolution : int
        Number of points on each nullcline grid (default 200).

    Returns
    -------
    list[np.ndarray]
        List of fixed points [E*, I*].  May be empty (no fixed point),
        contain one point (monostable), or two points (bistable).

    Notes
    -----
    The E-nullcline is defined by E = S(w_EE * E - w_EI * I + I_E).
    The I-nullcline is defined by I = S(w_IE * E - w_II * I + I_I).
    We solve each for I as a function of E (and vice versa) on a grid,
    then find intersections.
    """
    e_grid = np.linspace(0.0, 1.0, grid_resolution)

    # E-nullcline: solve for I given E
    # E = S(w_EE * E - w_EI * I + I_E)
    # => I = (w_EE * E + I_E - S^{-1}(E)) / w_EI
    # S^{-1}(y) = theta - (1/r) * ln((1-y)/y)
    def inv_sigmoid(y: np.ndarray) -> np.ndarray:
        y = np.clip(y, 1e-10, 1 - 1e-10)
        return params.theta - (1.0 / params.r) * np.log((1.0 - y) / y)

    e_null_I = (params.w_EE * e_grid + params.I_E - inv_sigmoid(e_grid)) / params.w_EI

    # I-nullcline: solve for I given E
    # I = S(w_IE * E - w_II * I + I_I)
    # => I = (w_IE * E + I_I - S^{-1}(I)) / (1 + w_II/r) ... iterative
    # Instead, compute I_null as I(E) by solving I = S(w_IE * E - w_II * I + I_I)
    # For each E, find I such that I = S(...)
    def i_nullcline(e_val: float) -> float:
        # Fixed-point iteration: I_{n+1} = S(w_IE * E - w_II * I_n + I_I)
        i_val = 0.5
        for _ in range(100):
            i_new = sigmoid_scalar(params.w_IE * e_val - params.w_II * i_val + params.I_I)
            if abs(i_new - i_val) < 1e-12:
                break
            i_val = i_new
        return i_val

    i_null_I = np.array([i_nullcline(e) for e in e_grid])

    # Find intersections: where e_null_I and i_null_I cross
    diff = e_null_I - i_null_I
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    fixed_points: list[np.ndarray] = []
    for idx in sign_changes:
        # Linear interpolation to find exact crossing
        e1, e2 = e_grid[idx], e_grid[idx + 1]
        d1, d2 = diff[idx], diff[idx + 1]
        # Crossing at e = e1 - d1 * (e2 - e1) / (d2 - d1)
        e_cross = e1 - d1 * (e2 - e1) / (d2 - d1) if (d2 - d1) != 0 else (e1 + e2) / 2
        i_cross = i_nullcline(e_cross)
        e_cross = np.clip(e_cross, 0.0, 1.0)
        i_cross = np.clip(i_cross, 0.0, 1.0)
        fixed_points.append(np.array([e_cross, i_cross]))

    # Remove duplicates (within tolerance)
    unique: list[np.ndarray] = []
    for fp in fixed_points:
        if not any(np.linalg.norm(fp - u) < 0.05 for u in unique):
            unique.append(fp)

    return unique


def wc_jacobian(
    state: np.ndarray,
    params: WCParams,
) -> np.ndarray:
    """Compute the Jacobian matrix of the Wilson-Cowan system at a given state.

    Parameters
    ----------
    state : np.ndarray
        Current state [E, I].
    params : WCParams
        Model parameters.

    Returns
    -------
    np.ndarray
        2×2 Jacobian matrix d(dE/dt, dI/dt) / d(E, I).

    Notes
    -----
    The Jacobian determines local stability:
    - If both eigenvalues have negative real parts → stable fixed point.
    - If any eigenvalue has positive real part → unstable.
    - Complex eigenvalues with negative real parts → damped oscillations.
    - Complex eigenvalues with positive real parts → unstable spiral.
    """
    E, I = state

    h_E = params.w_EE * E - params.w_EI * I + params.I_E
    h_I = params.w_IE * E - params.w_II * I + params.I_I

    # Derivative of sigmoid: S'(x) = r * S(x) * (1 - S(x))
    s_E = sigmoid_scalar(h_E, params.theta, params.r)
    s_I = sigmoid_scalar(h_I, params.theta, params.r)
    ds_E = params.r * s_E * (1.0 - s_E)
    ds_I = params.r * s_I * (1.0 - s_I)

    # Jacobian entries
    j11 = (-1.0 + params.w_EE * ds_E) / params.tau_E
    j12 = (-params.w_EI * ds_E) / params.tau_E
    j21 = (params.w_IE * ds_I) / params.tau_E  # Note: dI/dt uses tau_I
    j21 = (params.w_IE * ds_I) / params.tau_I
    j22 = (-1.0 - params.w_II * ds_I) / params.tau_I

    return np.array([[j11, j12], [j21, j22]])


def wc_stability(
    fixed_point: np.ndarray,
    params: WCParams,
) -> dict:
    """Analyze the stability of a fixed point via eigenvalue analysis.

    Parameters
    ----------
    fixed_point : np.ndarray
        Fixed point [E*, I*].
    params : WCParams
        Model parameters.

    Returns
    -------
    dict
        Stability analysis:
        - ``eigenvalues``: complex eigenvalues of the Jacobian (array of 2)
        - ``stable``: True if all eigenvalues have negative real parts
        - ``type``: one of "stable_node", "stable_spiral", "unstable_node",
          "unstable_spiral", "saddle"
        - ``e_star``: E* value
        - ``i_star``: I* value
        - ``trace``: trace of Jacobian
        - ``determinant``: determinant of Jacobian
    """
    J = wc_jacobian(fixed_point, params)
    eigenvalues = np.linalg.eigvals(J)
    tr = np.trace(J).real
    det = np.linalg.det(J).real

    real_parts = eigenvalues.real
    is_stable = bool(np.all(real_parts < 0))

    # Classify
    disc = tr * tr - 4 * det  # discriminant
    if det < 0:
        fp_type = "saddle"
    elif disc >= 0:
        fp_type = "stable_node" if is_stable else "unstable_node"
    else:
        fp_type = "stable_spiral" if is_stable else "unstable_spiral"

    return {
        "eigenvalues": eigenvalues,
        "stable": is_stable,
        "type": fp_type,
        "e_star": float(fixed_point[0]),
        "i_star": float(fixed_point[1]),
        "trace": float(tr),
        "determinant": float(det),
    }


def wc_bifurcation_scan(
    params: WCParams,
    vary_param: str,
    param_values: np.ndarray,
    t_max: float = 2000.0,
    transient_ms: float = 1000.0,
    dt: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Scan a parameter for bifurcation behavior.

    For each parameter value, simulate the system and record the
    maximum and minimum of E after the transient period.  This reveals
    transitions between fixed-point and oscillatory regimes.

    Parameters
    ----------
    params : WCParams
        Base model parameters.  The varied parameter is overridden
        for each simulation.
    vary_param : str
        Name of the parameter to vary (e.g., "w_EE", "I_E").
    param_values : np.ndarray
        Array of parameter values to scan.
    t_max : float
        Total simulation time in ms.
    transient_ms : float
        Transient period to discard before measuring.
    dt : float
        Integration timestep in ms.
    seed : int
        Random seed (for reproducibility).

    Returns
    -------
    np.ndarray
        Shape (len(param_values), 4).  Columns are:
        [param_value, E_min, E_max, E_mean] after the transient.

    Notes
    -----
    In a fixed-point regime, E_min ≈ E_max (single value).
    In an oscillatory regime, E_min < E_max (periodic oscillation).
    The gap between E_min and E_max reveals the oscillation amplitude.
    """
    results = np.empty((len(param_values), 4), dtype=np.float64)

    for idx, pval in enumerate(param_values):
        scan_params = dataclasses.replace(params, **{vary_param: pval})
        t, traj = simulate_wc(scan_params, t_max=t_max, dt=dt, seed=seed)

        # Discard transient
        steady_mask = t >= transient_ms
        e_steady = traj[steady_mask, 0]

        results[idx] = [pval, e_steady.min(), e_steady.max(), e_steady.mean()]

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_wc_phase_portrait(
    t: np.ndarray,
    trajectory: np.ndarray,
    params: WCParams,
    save_path: str = "plots/wc_phase_portrait.png",
) -> None:
    """Plot a phase portrait of the Wilson-Cowan system.

    Shows the trajectory in E-I space, nullclines, and fixed points.

    Parameters
    ----------
    t : np.ndarray
        Time vector from :func:`simulate_wc`.
    trajectory : np.ndarray
        State trajectory from :func:`simulate_wc`, shape (N, 2).
    params : WCParams
        Model parameters (used for nullcline computation).
    save_path : str
        Output file path for the PNG.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    E = trajectory[:, 0]
    I = trajectory[:, 1]

    # Plot trajectory
    ax.plot(E, I, color="#1e3a5f", linewidth=0.8, alpha=0.7)

    # Plot nullclines
    e_grid = np.linspace(0.0, 1.0, 300)

    def inv_sigmoid(y: np.ndarray) -> np.ndarray:
        y = np.clip(y, 1e-10, 1 - 1e-10)
        return params.theta - (1.0 / params.r) * np.log((1.0 - y) / y)

    e_null_I = (params.w_EE * e_grid + params.I_E - inv_sigmoid(e_grid)) / params.w_EI

    def i_nullcline_scalar(e_val: float) -> float:
        i_val = 0.5
        for _ in range(100):
            i_new = sigmoid_scalar(
                params.w_IE * e_val - params.w_II * i_val + params.I_I
            )
            if abs(i_new - i_val) < 1e-12:
                break
            i_val = i_new
        return i_val

    i_null_I = np.array([i_nullcline_scalar(e) for e in e_grid])

    ax.plot(e_grid, e_null_I, "--", color="#c45533", linewidth=1.5, label="E-nullcline")
    ax.plot(e_grid, i_null_I, "--", color="#2d9e5f", linewidth=1.5, label="I-nullcline")

    # Plot fixed points
    fps = wc_fixed_points(params)
    for fp in fps:
        stab = wc_stability(fp, params)
        color = "#2d9e5f" if stab["stable"] else "#c45533"
        marker = "o" if "node" in stab["type"] else "^"
        ax.plot(fp[0], fp[1], marker, markersize=10, color=color,
                label=f"FP: {stab['type']}")

    # Mark start and end
    ax.plot(trajectory[0, 0], trajectory[0, 1], "g.", markersize=12, label="Start")
    ax.plot(E[-1], I[-1], "r.", markersize=12, label="End")

    ax.set_xlabel("E (excitatory rate)", fontsize=11)
    ax.set_ylabel("I (inhibitory rate)", fontsize=11)
    ax.set_title("Wilson-Cowan Phase Portrait", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Phase portrait saved to {save_path}")


def plot_wc_timeseries(
    t: np.ndarray,
    trajectory: np.ndarray,
    save_path: str = "plots/wc_timeseries.png",
) -> None:
    """Plot E and I time series from a Wilson-Cowan simulation.

    Parameters
    ----------
    t : np.ndarray
        Time vector in milliseconds.
    trajectory : np.ndarray
        State trajectory, shape (N, 2).  Columns are E and I.
    save_path : str
        Output file path for the PNG.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(t / 1000, trajectory[:, 0], color="#1e3a5f", linewidth=0.8)
    axes[0].set_ylabel("E (excitatory)", fontsize=10)
    axes[0].set_title("Wilson-Cowan Dynamics", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(t / 1000, trajectory[:, 1], color="#c45533", linewidth=0.8)
    axes[1].set_ylabel("I (inhibitory)", fontsize=10)
    axes[1].set_xlabel("Time (s)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Time series saved to {save_path}")


def plot_wc_bifurcation(
    results: np.ndarray,
    vary_param: str,
    save_path: str = "plots/wc_bifurcation.png",
) -> None:
    """Plot a bifurcation diagram from a parameter scan.

    Parameters
    ----------
    results : np.ndarray
        Output from :func:`wc_bifurcation_scan`, shape (N, 4).
    vary_param : str
        Name of the varied parameter (used for axis label).
    save_path : str
        Output file path for the PNG.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    p_vals = results[:, 0]
    e_min = results[:, 1]
    e_max = results[:, 2]
    e_mean = results[:, 3]

    ax.fill_between(p_vals, e_min, e_max, alpha=0.3, color="#1e3a5f", label="E range")
    ax.plot(p_vals, e_mean, color="#c45533", linewidth=1.5, label="E mean")

    ax.set_xlabel(vary_param, fontsize=11)
    ax.set_ylabel("E (excitatory rate)", fontsize=11)
    ax.set_title(f"Bifurcation Scan — varying {vary_param}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Bifurcation diagram saved to {save_path}")


# ---------------------------------------------------------------------------
# Oscillation metrics
# ---------------------------------------------------------------------------


def wc_oscillation_metrics(
    t: np.ndarray,
    trajectory: np.ndarray,
    transient_ms: float = 1000.0,
) -> dict:
    """Compute oscillation metrics from a Wilson-Cowan time series.

    Parameters
    ----------
    t : np.ndarray
        Time vector in milliseconds.
    trajectory : np.ndarray
        State trajectory, shape (N, 2).  Columns are E and I.
    transient_ms : float
        Transient period to discard before computing metrics.

    Returns
    -------
    dict
        Oscillation metrics:
        - ``oscillating``: True if E shows sustained oscillations
        - ``frequency_hz``: Dominant oscillation frequency in Hz
        - ``amplitude``: Peak-to-peak amplitude of E during steady state
        - ``e_mean``: Mean E rate during steady state
        - ``e_std``: Standard deviation of E during steady state
        - ``i_mean``: Mean I rate during steady state
        - ``phase_lag_ms``: Estimated phase lag of I relative to E

    Notes
    -----
    Frequency is estimated via zero-crossing detection on the
    detrended E signal.  Phase lag is estimated via cross-correlation
    peak between E and I.
    """
    steady_mask = t >= transient_ms
    t_steady = t[steady_mask]
    E_steady = trajectory[steady_mask, 0]
    I_steady = trajectory[steady_mask, 1]

    # Detrend
    E_detrended = E_steady - np.mean(E_steady)
    I_detrended = I_steady - np.mean(I_steady)

    # Amplitude
    amplitude = float(E_steady.max() - E_steady.min())

    # Oscillation detection: if std > 5% of mean, likely oscillating
    e_mean = float(np.mean(E_steady))
    e_std = float(np.std(E_steady))
    oscillating = e_std > 0.05 * max(e_mean, 0.01)

    # Frequency via zero-crossing
    if oscillating and len(E_detrended) > 10:
        # Count positive zero crossings
        sign_changes = np.where(np.diff(np.sign(E_detrended)))[0]
        if len(sign_changes) >= 2:
            # Period is average distance between consecutive crossings
            crossings_ms = t_steady[sign_changes]
            periods = np.diff(crossings_ms)
            # Zero-crossing period is half the full period
            avg_period_ms = 2.0 * np.median(periods)
            frequency_hz = 1000.0 / avg_period_ms if avg_period_ms > 0 else 0.0
        else:
            frequency_hz = 0.0
    else:
        frequency_hz = 0.0

    # Phase lag via cross-correlation
    if oscillating and len(E_detrended) > 10:
        cross_corr = np.correlate(E_detrended, I_detrended, mode="full")
        lag_idx = np.argmax(cross_corr) - (len(E_detrended) - 1)
        dt_sample = t_steady[1] - t_steady[0] if len(t_steady) > 1 else 1.0
        phase_lag_ms = float(lag_idx * dt_sample)
    else:
        phase_lag_ms = 0.0

    return {
        "oscillating": oscillating,
        "frequency_hz": float(frequency_hz),
        "amplitude": amplitude,
        "e_mean": e_mean,
        "e_std": e_std,
        "i_mean": float(np.mean(I_steady)),
        "phase_lag_ms": phase_lag_ms,
    }


# ---------------------------------------------------------------------------
# Preset parameter sets
# ---------------------------------------------------------------------------


def wc_oscillatory() -> WCParams:
    """Return parameters that produce sustained oscillations.

    Strong E→I and I→E coupling with moderate E→E self-excitation
    creates a limit cycle.  This is analogous to gamma-range
    oscillations observed in cortical circuits.
    """
    return WCParams(
        tau_E=80.0,
        tau_I=20.0,
        w_EE=1.0,
        w_EI=1.0,
        w_IE=0.8,
        w_II=1.0,
        I_E=0.4,
        I_I=0.1,
    )


def wc_bistable() -> WCParams:
    """Return parameters that produce bistability (two stable fixed points).

    Strong recurrent excitation (w_EE > 1.2) creates two attracting
    states: a resting state near zero and a persistent-activity state.
    This models working memory or cortical up/down states.
    """
    return WCParams(
        tau_E=80.0,
        tau_I=20.0,
        w_EE=1.4,
        w_EI=1.0,
        w_IE=0.8,
        w_II=1.0,
        I_E=0.3,
        I_I=0.1,
    )


def wc_fixed_point() -> WCParams:
    """Return parameters that produce a single stable fixed point.

    Weak coupling produces a unique, stable equilibrium with no
    oscillations or bistability.  This is the baseline regime.
    """
    return WCParams(
        tau_E=80.0,
        tau_I=20.0,
        w_EE=0.8,
        w_EI=0.5,
        w_IE=0.4,
        w_II=0.8,
        I_E=0.3,
        I_I=0.1,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo() -> None:
    """Run a quick demo of the Wilson-Cowan oscillator."""
    import matplotlib.pyplot as plt

    duration = 3000.0  # 3 seconds

    # Oscillatory regime
    params_osc = wc_oscillatory()
    t_osc, traj_osc = simulate_wc(params_osc, t_max=duration, dt=0.5, seed=42)
    metrics = wc_oscillation_metrics(t_osc, traj_osc, transient_ms=1000.0)
    print(f"Oscillatory:  freq={metrics['frequency_hz']:.1f} Hz  "
          f"amp={metrics['amplitude']:.3f}  oscillating={metrics['oscillating']}")

    # Bistable regime — start from low initial condition
    params_bi = wc_bistable()
    t_bi, traj_bi = simulate_wc(params_bi, t_max=duration, dt=0.5,
                                 initial=(0.05, 0.05), seed=42)
    fps = wc_fixed_points(params_bi)
    print(f"Bistable:     {len(fps)} fixed point(s) found")
    for fp in fps:
        stab = wc_stability(fp, params_bi)
        print(f"  FP E={fp[0]:.3f} I={fp[1]:.3f}  type={stab['type']}")

    # Fixed-point regime
    params_fp = wc_fixed_point()
    t_fp, traj_fp = simulate_wc(params_fp, t_max=duration, dt=0.5, seed=42)
    metrics_fp = wc_oscillation_metrics(t_fp, traj_fp, transient_ms=1000.0)
    print(f"Fixed point:  oscillating={metrics_fp['oscillating']}  "
          f"E_mean={metrics_fp['e_mean']:.3f}")

    # Plot time series for oscillatory case
    plot_wc_timeseries(t_osc, traj_osc)

    # Plot phase portrait for oscillatory case
    plot_wc_phase_portrait(t_osc, traj_osc, params_osc)

    # Bifurcation scan
    w_ee_values = np.linspace(0.5, 1.5, 50)
    bif_results = wc_bifurcation_scan(params_osc, "w_EE", w_ee_values)
    plot_wc_bifurcation(bif_results, "w_EE")

    print("Wilson-Cowan demo complete.")


if __name__ == "__main__":
    demo()
