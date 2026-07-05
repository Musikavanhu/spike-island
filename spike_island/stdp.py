"""Spike-Timing Dependent Plasticity (STDP).

Implements the classic pair-based STDP learning rule where synaptic weights
change based on the relative timing of pre- and postsynaptic spikes.  Causal
correlations (pre before post) strengthen the synapse; anti-causal correlations
(post before pre) weaken it.

This module provides both a single-synapse model and a full network with
pairwise STDP learning across all connections.

References
----------
Bi & Poo (1998) "Synaptic modifications in cultured hippocampal neurons"
Gerstner & Kistler (2002) "Spiking Neuron Models"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class STDPParams:
    """Parameters controlling the STDP learning rule.

    Parameters
    ----------
    a_plus : float
        Amplitude of LTD (weight decrease for anti-causal pairing).
        Stored as a positive value; the actual change is ``-a_plus``.
    a_minus : float
        Amplitude of LTP (weight increase for causal pairing).
    tau_plus : float
        Time constant for the LTP window (ms).  Controls how far back
        a postsynaptic spike can "see" a presynaptic spike.
    tau_minus : float
        Time constant for the LTD window (ms).  Controls how far forward
        a presynaptic spike can "see" a postsynaptic spike.
    w_min : float
        Hard lower bound on synaptic weight.  Prevents weights from
        becoming negative.
    w_max : float
        Hard upper bound on synaptic weight.  Prevents unbounded growth.
    history_length : float
        Duration (ms) of the spike history buffer.  Spikes older than
        this are pruned to keep memory bounded.

    Typical values
    -------------
    a_plus=0.01, a_minus=0.012, tau_plus=20, tau_minus=20,
    w_min=0.0, w_max=1.0, history_length=100.
    """

    a_plus: float = 0.01
    a_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    w_min: float = 0.0
    w_max: float = 1.0
    history_length: float = 100.0


def stdp_window(dt: float, params: STDPParams) -> float:
    """Compute the weight change for a single pre/post spike pair.

    The STDP learning window is an asymmetric exponential function of
    the spike-time difference ``dt = t_post - t_pre``:

    - **dt > 0** (causal: pre fires before post):
      ``delta_w = +a_minus * exp(-dt / tau_plus)``  (LTP)

    - **dt < 0** (anti-causal: post fires before pre):
      ``delta_w = -a_plus * exp(+dt / tau_minus)``  (LTD)

    Parameters
    ----------
    dt : float
        Time difference ``t_post - t_pre`` in milliseconds.
        Positive means the presynaptic spike came first.
    params : STDPParams
        STDP hyperparameters.

    Returns
    -------
    float
        Weight change ``delta_w``.  Positive for LTP, negative for LTD.
    """
    if dt > 0:
        return params.a_minus * np.exp(-dt / params.tau_plus)
    else:
        return -params.a_plus * np.exp(dt / params.tau_minus)


@dataclass
class Synapse:
    """A single synapse with STDP learning.

    Maintains the current weight and a bounded history of pre/post
    spike timestamps.  When a spike is recorded, it triggers STDP
    updates against all recent spikes from the opposite side.

    Parameters
    ----------
    weight : float
        Initial synaptic weight.
    params : STDPParams
        STDP learning rule parameters.
    history_length : float
        Spike history retention window (ms).

    Attributes
    ----------
    weight : float
        Current synaptic weight (clipped to [w_min, w_max]).
    pre_spikes : list[float]
        Timestamps of recent presynaptic spikes (ms).
    post_spikes : list[float]
        Timestamps of recent postsynaptic spikes (ms).
    weight_history : list[tuple[float, float]]
        History of (timestamp, weight) after each spike event.
    """

    weight: float
    params: STDPParams
    history_length: float = 100.0
    pre_spikes: list[float] = field(default_factory=list)
    post_spikes: list[float] = field(default_factory=list)
    weight_history: list[tuple[float, float]] = field(default_factory=list)

    def _prune(self, current_time: float) -> None:
        """Remove spike timestamps older than ``history_length``."""
        cutoff = current_time - self.history_length
        self.pre_spikes = [t for t in self.pre_spikes if t > cutoff]
        self.post_spikes = [t for t in self.post_spikes if t > cutoff]

    def record_pre_spike(self, t: float) -> float:
        """Record a presynaptic spike and apply STDP against all recent post spikes.

        Parameters
        ----------
        t : float
            Spike timestamp in ms.

        Returns
        -------
        float
            Total weight change caused by this spike event.
        """
        self.pre_spikes.append(t)
        self._prune(t)

        total_delta = 0.0
        for t_post in self.post_spikes:
            dt = t_post - t  # post - pre
            delta_w = stdp_window(dt, self.params)
            total_delta += delta_w

        self.weight = float(np.clip(self.weight + total_delta, self.params.w_min, self.params.w_max))
        self.weight_history.append((t, self.weight))
        return total_delta

    def record_post_spike(self, t: float) -> float:
        """Record a postsynaptic spike and apply STDP against all recent pre spikes.

        Parameters
        ----------
        t : float
            Spike timestamp in ms.

        Returns
        -------
        float
            Total weight change caused by this spike event.
        """
        self.post_spikes.append(t)
        self._prune(t)

        total_delta = 0.0
        for t_pre in self.pre_spikes:
            dt = t - t_pre  # post - pre
            delta_w = stdp_window(dt, self.params)
            total_delta += delta_w

        self.weight = float(np.clip(self.weight + total_delta, self.params.w_min, self.params.w_max))
        self.weight_history.append((t, self.weight))
        return total_delta

    def record_spikes(
        self,
        pre_times: Sequence[float],
        post_times: Sequence[float],
    ) -> np.ndarray:
        """Process interleaved pre/post spike events in chronological order.

        This is the most realistic usage: spikes arrive from both sides
        and are processed in temporal order, each triggering STDP against
        the history of the other side.

        Parameters
        ----------
        pre_times : Sequence[float]
            Timestamps of presynaptic spikes (ms).
        post_times : Sequence[float]
            Timestamps of postsynaptic spikes (ms).

        Returns
        -------
        np.ndarray
            Weight trajectory after each spike event.  Length equals
            the total number of events (pre + post).
        """
        events: list[tuple[float, str]] = []
        for t in pre_times:
            events.append((float(t), "pre"))
        for t in post_times:
            events.append((float(t), "post"))

        events.sort(key=lambda x: (x[0], 0 if x[1] == "pre" else 1))

        weights = []
        for t, kind in events:
            if kind == "pre":
                self.record_pre_spike(t)
            else:
                self.record_post_spike(t)
            weights.append(self.weight)

        return np.array(weights, dtype=np.float64)


@dataclass
class STDPNetwork:
    """A network of neurons with pairwise STDP learning.

    Parameters
    ----------
    n_pre : int
        Number of presynaptic neurons.
    n_post : int
        Number of postsynaptic neurons.
    params : STDPParams
        STDP learning rule parameters (shared across all synapses).
    w_init : float
        Initial weight for all synapses.
    history_length : float
        Spike history retention window (ms).

    Attributes
    ----------
    weights : np.ndarray
        Synaptic weight matrix of shape (n_pre, n_post).
    weight_history : list[np.ndarray]
        Snapshots of the weight matrix at each recording interval.
    """

    n_pre: int
    n_post: int
    params: STDPParams = field(default_factory=STDPParams)
    w_init: float = 0.5
    history_length: float = 100.0

    weights: np.ndarray = field(init=False)
    weight_history: list[np.ndarray] = field(default_factory=list)
    _synapses: list[list[Synapse]] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize weight matrix and individual synapse objects."""
        self.weights = np.full((self.n_pre, self.n_post), self.w_init, dtype=np.float64)
        self._synapses = [
            [
                Synapse(
                    weight=self.w_init,
                    params=self.params,
                    history_length=self.history_length,
                )
                for _ in range(self.n_post)
            ]
            for _ in range(self.n_pre)
        ]

    def record_pre_spike(self, neuron: int, t: float) -> None:
        """Record a presynaptic spike from neuron ``neuron`` at time ``t``.

        Updates all synapses from this presynaptic neuron to all
        postsynaptic neurons using STDP.
        """
        for j in range(self.n_post):
            delta = self._synapses[neuron][j].record_pre_spike(t)
            self.weights[neuron, j] = self._synapses[neuron][j].weight

    def record_post_spike(self, neuron: int, t: float) -> None:
        """Record a postsynaptic spike from neuron ``neuron`` at time ``t``.

        Updates all synapses from all presynaptic neurons to this
        postsynaptic neuron using STDP.
        """
        for i in range(self.n_pre):
            delta = self._synapses[i][neuron].record_post_spike(t)
            self.weights[i, neuron] = self._synapses[i][neuron].weight

    def record_spikes(
        self,
        pre_times: list[tuple[int, float]],
        post_times: list[tuple[int, float]],
    ) -> None:
        """Process interleaved pre/post spike events across the network.

        Parameters
        ----------
        pre_times : list[tuple[int, float]]
            List of (neuron_index, timestamp_ms) for presynaptic spikes.
        post_times : list[tuple[int, float]]
            List of (neuron_index, timestamp_ms) for postsynaptic spikes.
        """
        events: list[tuple[float, str, int]] = []
        for neuron, t in pre_times:
            events.append((t, "pre", neuron))
        for neuron, t in post_times:
            events.append((t, "post", neuron))

        events.sort(key=lambda x: (x[0], 0 if x[1] == "pre" else 1))

        for t, kind, neuron in events:
            if kind == "pre":
                self.record_pre_spike(neuron, t)
            else:
                self.record_post_spike(neuron, t)

    def snapshot(self) -> None:
        """Save a snapshot of the current weight matrix."""
        self.weight_history.append(self.weights.copy())

    def get_stats(self) -> dict[str, float]:
        """Compute summary statistics of the current weight matrix.

        Returns
        -------
        dict
            Keys: 'mean', 'std', 'min', 'max', 'sparsity' (fraction at w_min).
        """
        sparsity = float(np.mean(self.weights == self.params.w_min))
        return {
            "mean": float(np.mean(self.weights)),
            "std": float(np.std(self.weights)),
            "min": float(np.min(self.weights)),
            "max": float(np.max(self.weights)),
            "sparsity": sparsity,
        }
