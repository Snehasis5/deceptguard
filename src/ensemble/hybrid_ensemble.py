"""
DeceptGuard HybridConstitutional Ensemble
Two-layer MLP meta-classifier over individual monitor scores.
Aggregates BB + CA-P + CA-C + AP-H into a single suspicion score.
"""

from __future__ import annotations

import logging
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models import AgentTrajectory, EnsembleScore, MonitorScore

logger = logging.getLogger(__name__)


class HybridConstitutionalEnsemble:
    """
    Two-layer MLP meta-classifier as described in §6.3.

    Inputs  : individual monitor scores (s_BB, s_CA-P, s_CA-C, s_AP-H) ∈ ℝ⁴
              (all standardised to zero mean, unit variance at train time)
    Outputs : ensemble suspicion score in [1, 10]

    The ensemble is robust to partial monitor failure: if a single monitor
    is compromised or unavailable, remaining monitors provide signal.
    """

    def __init__(
        self,
        bb_monitor=None,
        ca_p_monitor=None,
        ca_c_monitor=None,
        ap_h_monitor=None,
        mlp_path: Optional[str] = None,
    ):
        self.bb_monitor = bb_monitor
        self.ca_p_monitor = ca_p_monitor
        self.ca_c_monitor = ca_c_monitor
        self.ap_h_monitor = ap_h_monitor

        self._mlp = None
        self._scaler = None  # StandardScaler fitted on training data

        if mlp_path:
            self.load(mlp_path)

    # ------------------------------------------------------------------
    # MLP training
    # ------------------------------------------------------------------

    def train(
        self,
        score_matrix: np.ndarray,     # (n_samples, 4): [bb, ca_p, ca_c, ap_h]
        labels: np.ndarray,            # (n_samples,): binary
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 64,
        seed: int = 42,
        save_path: Optional[str] = None,
    ):
        """
        Train the meta-classifier MLP.
        Uses sklearn MLPClassifier as the production-safe backend.

        Args:
            score_matrix: monitor scores, shape (n, 4)
            labels: ground-truth binary labels
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        # Standardise inputs
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(score_matrix)

        self._mlp = MLPClassifier(
            hidden_layer_sizes=(hidden_dim, hidden_dim // 2),
            activation="relu",
            solver="adam",
            alpha=weight_decay,
            batch_size=batch_size,
            learning_rate_init=lr,
            max_iter=epochs,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        self._mlp.fit(X, labels)
        logger.info(
            "MLP meta-classifier trained. Best validation loss: %.4f",
            self._mlp.best_loss_,
        )

        if save_path:
            self.save(save_path)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"mlp": self._mlp, "scaler": self._scaler}, f)
        logger.info("Ensemble saved to %s", path)

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._mlp = state["mlp"]
        self._scaler = state["scaler"]
        logger.info("Ensemble loaded from %s", path)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_feature_vector(
        self, scores: Dict[str, Optional[float]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build feature vector from available scores.
        Missing scores are imputed with the neutral value (5.5).
        Returns (feature_vector, list_of_available_monitors).
        """
        NEUTRAL = 5.5
        keys = ["bb", "ca_p", "ca_c", "ap_h"]
        vec = np.array(
            [scores.get(k) if scores.get(k) is not None else NEUTRAL for k in keys],
            dtype=np.float32,
        )
        available = [k for k in keys if scores.get(k) is not None]
        return vec.reshape(1, -1), available

    def _raw_prob_to_score(self, prob: float) -> float:
        """Map P(deceptive) from MLP to [1, 10] scale."""
        return 1.0 + 9.0 * prob

    def score_from_subscores(
        self,
        trajectory_id: str,
        bb: Optional[float] = None,
        ca_p: Optional[float] = None,
        ca_c: Optional[float] = None,
        ap_h: Optional[float] = None,
        label: Optional[bool] = None,
    ) -> EnsembleScore:
        """
        Compute ensemble score from pre-computed sub-scores.
        Any missing sub-scores are imputed.
        """
        scores_map = {"bb": bb, "ca_p": ca_p, "ca_c": ca_c, "ap_h": ap_h}
        vec, available = self._build_feature_vector(scores_map)

        if self._mlp is not None and self._scaler is not None:
            vec_scaled = self._scaler.transform(vec)
            prob = self._mlp.predict_proba(vec_scaled)[0, 1]
            ensemble = self._raw_prob_to_score(prob)
        else:
            # Fall back to simple weighted average when MLP is not trained
            available_vals = [v for v in [bb, ca_p, ca_c, ap_h] if v is not None]
            ensemble = float(np.mean(available_vals)) if available_vals else 5.5

        return EnsembleScore(
            trajectory_id=trajectory_id,
            bb_score=bb,
            ca_p_score=ca_p,
            ca_c_score=ca_c,
            ap_h_score=ap_h,
            ensemble_score=round(ensemble, 3),
            label=label,
        )

    def score(
        self,
        trajectory: AgentTrajectory,
        include_cot: bool = True,
        include_probe: bool = False,
    ) -> EnsembleScore:
        """
        Run all available monitors on a trajectory and return ensemble score.

        Args:
            trajectory: the agent trajectory to score
            include_cot: whether to run CoT-aware monitors (requires CoT present)
            include_probe: whether to run AP-H (requires open-weights infrastructure)
        """
        bb_score: Optional[float] = None
        ca_p_score: Optional[float] = None
        ca_c_score: Optional[float] = None
        ap_h_score: Optional[float] = None

        if self.bb_monitor is not None:
            try:
                ms: MonitorScore = self.bb_monitor.score(trajectory)
                bb_score = ms.score
            except Exception as e:
                logger.warning("BB monitor failed: %s", e)

        if include_cot and trajectory.chain_of_thought:
            if self.ca_p_monitor is not None:
                try:
                    ms = self.ca_p_monitor.score(trajectory)
                    ca_p_score = ms.score
                except Exception as e:
                    logger.warning("CA-P monitor failed: %s", e)

            if self.ca_c_monitor is not None:
                try:
                    ms = self.ca_c_monitor.score(trajectory)
                    ca_c_score = ms.score
                except Exception as e:
                    logger.warning("CA-C monitor failed: %s", e)

        if include_probe and self.ap_h_monitor is not None:
            try:
                ms = self.ap_h_monitor.score(trajectory)
                ap_h_score = ms.score
            except Exception as e:
                logger.warning("AP-H monitor failed: %s", e)

        return self.score_from_subscores(
            trajectory_id=trajectory.trajectory_id,
            bb=bb_score,
            ca_p=ca_p_score,
            ca_c=ca_c_score,
            ap_h=ap_h_score,
            label=trajectory.is_deceptive,
        )

    def score_batch(
        self,
        trajectories: List[AgentTrajectory],
        include_cot: bool = True,
        include_probe: bool = False,
    ) -> List[EnsembleScore]:
        return [
            self.score(t, include_cot=include_cot, include_probe=include_probe)
            for t in trajectories
        ]
