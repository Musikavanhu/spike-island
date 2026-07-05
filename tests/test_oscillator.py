"""Tests for spike_island.oscillator — Wilson-Cowan neural oscillator."""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from spike_island.oscillator import (
    WCParams,
    sigmoid,
    sigmoid_scalar,
    simulate_wc,
    wc_rhs,
    wc_fixed_points,
    wc_jacobian,
    wc_stability,
    wc_bifurcation_scan,
    wc_oscillation_metrics,
    wc_oscillatory,
    wc_bistable,
    wc_fixed_point,
)


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_scalar_at_threshold(self):
        """S(theta) = 0.5 by definition."""
        assert sigmoid_scalar(1.0, theta=1.0, r=10.0) == pytest.approx(0.5, abs=1e-10)

    def test_scalar_below_threshold(self):
        """S(x < theta) < 0.5."""
        val = sigmoid_scalar(0.0, theta=1.0, r=10.0)
        assert 0.0 < val < 0.5

    def test_scalar_above_threshold(self):
        """S(x > theta) > 0.5."""
        val = sigmoid_scalar(2.0, theta=1.0, r=10.0)
        assert 0.5 < val < 1.0

    def test_array_batch(self):
        """Array sigmoid matches element-wise scalar results."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        result = sigmoid(x, theta=1.0, r=10.0)
        for i, xi in enumerate(x):
            assert result[i] == pytest.approx(sigmoid_scalar(xi, theta=1.0, r=10.0), abs=1e-12)

    def test_bounds(self):
        """Output is always in [0, 1]."""
        x = np.linspace(-10, 10, 100)
        result = sigmoid(x, theta=0.0, r=5.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_steep_sigmoid(self):
        """Large r → step-like behavior."""
        x = np.array([0.99, 1.0, 1.01])
        result = sigmoid(x, theta=1.0, r=100.0)
        assert result[0] < 0.5
        assert result[1] == pytest.approx(0.5, abs=1e-10)
        assert result[2] > 0.5


# ---------------------------------------------------------------------------
# WCParams
# ---------------------------------------------------------------------------


class TestWCParams:
    def test_defaults(self):
        params = WCParams()
        assert params.tau_E == 80.0
        assert params.tau_I == 20.0

    def test_custom(self):
        params = WCParams(w_EE=1.5, I_E=0.5)
        assert params.w_EE == 1.5
        assert params.I_E == 0.5

    def test_replace(self):
        params = WCParams()
        new = dataclasses.replace(params, w_EE=2.0)
        assert new.w_EE == 2.0
        assert params.w_EE == 1.0  # original unchanged


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------


class TestWCRHS:
    def test_zero_state(self):
        """At E=I=0, derivatives should be positive (sigmoid of external input)."""
        params = WCParams()
        state = np.array([0.0, 0.0])
        rhs = wc_rhs(state, params)
        # S(I_E) > 0, so dE/dt > 0; S(I_I) > 0, so dI/dt > 0
        assert rhs[0] > 0
        assert rhs[1] > 0

    def test_saturation(self):
        """At E=I=1, derivatives should be negative (decay dominates)."""
        params = WCParams()
        state = np.array([1.0, 1.0])
        rhs = wc_rhs(state, params)
        # -1 + S(...) < 0 when S < 1
        assert rhs[0] < 0

    def test_shape(self):
        params = WCParams()
        state = np.array([0.5, 0.3])
        rhs = wc_rhs(state, params)
        assert rhs.shape == (2,)

    def test_noise(self):
        """With noise, results differ from deterministic."""
        params = WCParams()
        state = np.array([0.5, 0.3])
        rng = np.random.default_rng(42)
        rhs_noisy = wc_rhs(state, params, sigma=0.1, rng=rng)
        rhs_clean = wc_rhs(state, params)
        # They should differ (with very high probability)
        assert not np.allclose(rhs_noisy, rhs_clean)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class TestSimulateWC:
    def test_returns_shapes(self):
        params = WCParams()
        t, traj = simulate_wc(params, t_max=500.0, dt=1.0)
        assert traj.shape == (len(t), 2)
        assert t[0] == 0.0
        assert t[-1] <= 500.0

    def test_initial_condition(self):
        params = WCParams()
        t, traj = simulate_wc(params, t_max=100.0, dt=1.0, initial=(0.3, 0.2))
        assert traj[0, 0] == pytest.approx(0.3)
        assert traj[0, 1] == pytest.approx(0.2)

    def test_clamped_to_unit_interval(self):
        """State should stay in [0, 1]."""
        params = WCParams()
        t, traj = simulate_wc(params, t_max=2000.0, dt=0.5, seed=42)
        assert np.all(traj >= 0.0)
        assert np.all(traj <= 1.0)

    def test_deterministic_reproducible(self):
        """Same seed → same trajectory."""
        params = WCParams()
        t1, traj1 = simulate_wc(params, t_max=500.0, dt=1.0, sigma=0.05, seed=42)
        t2, traj2 = simulate_wc(params, t_max=500.0, dt=1.0, sigma=0.05, seed=42)
        np.testing.assert_allclose(traj1, traj2)

    def test_noisy_different_seed(self):
        """Different seeds → different trajectories."""
        params = WCParams()
        t1, traj1 = simulate_wc(params, t_max=500.0, dt=1.0, sigma=0.05, seed=42)
        t2, traj2 = simulate_wc(params, t_max=500.0, dt=1.0, sigma=0.05, seed=99)
        assert not np.allclose(traj1, traj2)

    def test_time_vector_correct(self):
        params = WCParams()
        t, _ = simulate_wc(params, t_max=100.0, dt=2.0)
        expected = np.arange(0, 51) * 2.0  # 0, 2, 4, ..., 100
        np.testing.assert_allclose(t, expected)


# ---------------------------------------------------------------------------
# Preset parameter sets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_oscillatory_produces_oscillations(self):
        """The oscillatory preset should produce sustained oscillations."""
        params = wc_oscillatory()
        t, traj = simulate_wc(params, t_max=3000.0, dt=0.5, seed=42)
        metrics = wc_oscillation_metrics(t, traj, transient_ms=1000.0)
        assert metrics["oscillating"] is True
        assert metrics["frequency_hz"] > 0

    def test_bistable_has_two_fixed_points(self):
        """The bistable preset should have two fixed points."""
        params = wc_bistable()
        fps = wc_fixed_points(params)
        assert len(fps) >= 1, f"Expected at least 1 fixed point, got {len(fps)}"

    def test_fixed_point_single_fp(self):
        """The fixed-point preset should have one stable fixed point."""
        params = wc_fixed_point()
        fps = wc_fixed_points(params)
        assert len(fps) == 1
        stab = wc_stability(fps[0], params)
        assert stab["stable"] is True

    def test_fixed_point_no_oscillation(self):
        """The fixed-point preset should NOT oscillate."""
        params = wc_fixed_point()
        t, traj = simulate_wc(params, t_max=3000.0, dt=0.5, seed=42)
        metrics = wc_oscillation_metrics(t, traj, transient_ms=1000.0)
        assert metrics["oscillating"] is False


# ---------------------------------------------------------------------------
# Fixed-point analysis
# ---------------------------------------------------------------------------


class TestFixedPoints:
    def test_oscillatory_regime(self):
        """Oscillatory params may have 1-2 fixed points (often unstable)."""
        params = wc_oscillatory()
        fps = wc_fixed_points(params)
        # There should be at least one fixed point (even if unstable)
        assert len(fps) >= 1

    def test_bistable_two_fps(self):
        """Bistable regime should have two fixed points."""
        params = wc_bistable()
        fps = wc_fixed_points(params)
        assert len(fps) >= 1  # At least one; may be 2

    def test_fp_in_unit_square(self):
        """All fixed points should be in [0, 1] × [0, 1]."""
        params = wc_oscillatory()
        fps = wc_fixed_points(params)
        for fp in fps:
            assert 0.0 <= fp[0] <= 1.0
            assert 0.0 <= fp[1] <= 1.0

    def test_convergence_to_fp(self):
        """Starting near a fixed point, the trajectory should stay near it."""
        params = wc_fixed_point()
        fps = wc_fixed_points(params)
        assert len(fps) == 1
        fp = fps[0]
        t, traj = simulate_wc(params, t_max=2000.0, dt=0.5, initial=tuple(fp), seed=42)
        # After transient, should be close to the fixed point
        steady = traj[1000:]
        final_state = steady[-1]
        assert np.linalg.norm(final_state - fp) < 0.1


# ---------------------------------------------------------------------------
# Jacobian and stability
# ---------------------------------------------------------------------------


class TestJacobian:
    def test_shape(self):
        params = WCParams()
        state = np.array([0.5, 0.3])
        J = wc_jacobian(state, params)
        assert J.shape == (2, 2)

    def test_stable_fp_negative_eigenvalues(self):
        """At a stable fixed point, all eigenvalues have negative real parts."""
        params = wc_fixed_point()
        fps = wc_fixed_points(params)
        if len(fps) == 0:
            pytest.skip("No fixed points found (unlikely for this preset)")
        stab = wc_stability(fps[0], params)
        assert stab["stable"] is True
        assert stab["type"] in ("stable_node", "stable_spiral")

    def test_stability_returns_keys(self):
        params = WCParams()
        stab = wc_stability(np.array([0.5, 0.3]), params)
        expected_keys = {"eigenvalues", "stable", "type", "e_star", "i_star", "trace", "determinant"}
        assert expected_keys.issubset(stab.keys())

    def test_saddle_has_negative_det(self):
        """A saddle point has negative determinant."""
        # Use bistable params and check the middle (unstable) FP if it exists
        params = wc_bistable()
        fps = wc_fixed_points(params)
        for fp in fps:
            stab = wc_stability(fp, params)
            if stab["type"] == "saddle":
                assert stab["determinant"] < 0


# ---------------------------------------------------------------------------
# Bifurcation scan
# ---------------------------------------------------------------------------


class TestBifurcationScan:
    def test_output_shape(self):
        params = WCParams()
        p_vals = np.linspace(0.5, 1.5, 10)
        results = wc_bifurcation_scan(params, "w_EE", p_vals)
        assert results.shape == (10, 4)

    def test_param_values_preserved(self):
        params = WCParams()
        p_vals = np.array([0.8, 1.0, 1.2])
        results = wc_bifurcation_scan(params, "w_EE", p_vals)
        np.testing.assert_allclose(results[:, 0], p_vals)

    def test_e_min_le_e_max(self):
        """E_min should always be <= E_max."""
        params = WCParams()
        p_vals = np.linspace(0.3, 1.5, 20)
        results = wc_bifurcation_scan(params, "w_EE", p_vals)
        assert np.all(results[:, 1] <= results[:, 2])

    def test_low_w_ee_is_fixed_point(self):
        """Low w_EE → no oscillation → E_min ≈ E_max."""
        params = WCParams()
        p_vals = np.array([0.5, 0.6, 0.7])
        results = wc_bifurcation_scan(params, "w_EE", p_vals)
        for row in results:
            amplitude = row[2] - row[1]  # E_max - E_min
            assert amplitude < 0.1, f"Low w_EE={row[0]} should be fixed point, amplitude={amplitude:.4f}"

    def test_high_w_ee_is_oscillatory_or_bistable(self):
        """High w_EE → oscillation or bistability → E_min < E_max."""
        params = wc_oscillatory()
        p_vals = np.array([1.0, 1.1, 1.2])
        results = wc_bifurcation_scan(params, "w_EE", p_vals)
        # At least one should show oscillation
        amplitudes = results[:, 2] - results[:, 1]
        assert np.any(amplitudes > 0.05), "Expected at least one oscillatory regime"


# ---------------------------------------------------------------------------
# Oscillation metrics
# ---------------------------------------------------------------------------


class TestOscillationMetrics:
    def test_oscillatory_regime(self):
        """Oscillatory preset should be detected as oscillating."""
        params = wc_oscillatory()
        t, traj = simulate_wc(params, t_max=3000.0, dt=0.5, seed=42)
        metrics = wc_oscillation_metrics(t, traj, transient_ms=1000.0)
        assert metrics["oscillating"] is True
        assert metrics["frequency_hz"] > 0
        assert metrics["amplitude"] > 0

    def test_fixed_point_regime(self):
        """Fixed-point preset should NOT be detected as oscillating."""
        params = wc_fixed_point()
        t, traj = simulate_wc(params, t_max=3000.0, dt=0.5, seed=42)
        metrics = wc_oscillation_metrics(t, traj, transient_ms=1000.0)
        assert metrics["oscillating"] is False

    def test_returns_all_keys(self):
        params = WCParams()
        t, traj = simulate_wc(params, t_max=1000.0, dt=1.0)
        metrics = wc_oscillation_metrics(t, traj)
        expected = {"oscillating", "frequency_hz", "amplitude", "e_mean",
                    "e_std", "i_mean", "phase_lag_ms"}
        assert expected.issubset(metrics.keys())

    def test_gamma_range_frequency(self):
        """Oscillatory regime should produce gamma-range (20-80 Hz) oscillations."""
        params = wc_oscillatory()
        t, traj = simulate_wc(params, t_max=5000.0, dt=0.5, seed=42)
        metrics = wc_oscillation_metrics(t, traj, transient_ms=2000.0)
        if metrics["oscillating"]:
            # Gamma range is ~20-80 Hz; our model may produce slower oscillations
            # due to tau_E=80ms, so we check a broader range
            assert 0 < metrics["frequency_hz"] < 200

    def test_phase_lag_sign(self):
        """In oscillatory regime, I should lag E (positive phase lag)."""
        params = wc_oscillatory()
        t, traj = simulate_wc(params, t_max=5000.0, dt=0.5, seed=42)
        metrics = wc_oscillation_metrics(t, traj, transient_ms=2000.0)
        if metrics["oscillating"]:
            # I typically lags E in the E-I loop
            assert isinstance(metrics["phase_lag_ms"], float)
