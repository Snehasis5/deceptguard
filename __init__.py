"""
DeceptGuard: Constitutional Oversight Framework for Detecting Deception in LLM Agents
======================================================================================

Primary API entry point. Provides a unified interface for:
  - Scoring individual trajectories
  - Running the full benchmark evaluation
  - Training new probes and ensemble classifiers
  - Generating synthetic datasets

Example usage:

    import anthropic
    from deceptguard import DeceptGuard

    client = anthropic.Anthropic()
    dg = DeceptGuard(client)

    score = dg.score(trajectory)
    print(f"Suspicion score: {score.ensemble_score:.1f}/10")

    if score.ensemble_score >= 9.0:
        print("ALERT: Immediate intervention recommended")
    elif score.ensemble_score >= 7.0:
        print("FLAG: Queued for human review")
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .configs.config import DEFAULT_CONFIG, DeceptGuardConfig
from .src.ensemble.hybrid_ensemble import HybridConstitutionalEnsemble
from .src.models import AgentTrajectory, EnsembleScore, MonitorRegime
from .src.monitors.monitors import (
    BlackBoxMonitor,
    CoTAwarePromptedMonitor,
    ConsistencyProbingMonitor,
)
from .src.probes.activation_probe import ActivationProbeMonitor
from .src.taxonomy.definitions import TAXONOMY, get_category, get_categories_by_macro

logger = logging.getLogger(__name__)


class DeceptGuard:
    """
    Unified DeceptGuard interface.

    Automatically configures available monitors based on:
      - Whether the Anthropic API client is available (BB, CA-P, CA-C)
      - Whether open-weights infrastructure is available (AP-H)
    """

    def __init__(
        self,
        api_client=None,
        config: Optional[DeceptGuardConfig] = None,
        enable_cot: bool = True,
        enable_probe: bool = False,
        probe_path: Optional[str] = None,
        ensemble_path: Optional[str] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.client = api_client
        self.enable_cot = enable_cot
        self.enable_probe = enable_probe

        # Build monitors
        bb = None
        ca_p = None
        ca_c = None
        ap_h = None

        if api_client is not None:
            model = self.config.models.monitor_backbone
            bb = BlackBoxMonitor(api_client, model=model)
            if enable_cot:
                ca_p = CoTAwarePromptedMonitor(api_client, model=model)
                ca_c = ConsistencyProbingMonitor(api_client, model=model)

        if enable_probe:
            ap_h = ActivationProbeMonitor(
                probe_path=probe_path or self.config.probe_path,
                layer=self.config.probe.layer,
                device=self.config.probe.device,
            )

        self.ensemble = HybridConstitutionalEnsemble(
            bb_monitor=bb,
            ca_p_monitor=ca_p,
            ca_c_monitor=ca_c,
            ap_h_monitor=ap_h,
            mlp_path=ensemble_path or self.config.ensemble_path,
        )

        self.bb_monitor = bb
        self.ca_p_monitor = ca_p
        self.ca_c_monitor = ca_c
        self.ap_h_monitor = ap_h

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def score(self, trajectory: AgentTrajectory) -> EnsembleScore:
        """
        Score a single trajectory.

        Returns an EnsembleScore with:
          - Individual monitor scores (bb, ca_p, ca_c, ap_h)
          - Ensemble score in [1, 10]
          - Operational recommendation based on configured thresholds
        """
        return self.ensemble.score(
            trajectory,
            include_cot=self.enable_cot,
            include_probe=self.enable_probe,
        )

    def score_batch(
        self, trajectories: List[AgentTrajectory]
    ) -> List[EnsembleScore]:
        """Score a list of trajectories."""
        return self.ensemble.score_batch(
            trajectories,
            include_cot=self.enable_cot,
            include_probe=self.enable_probe,
        )

    def triage(self, score: EnsembleScore) -> str:
        """
        Map an ensemble score to an operational recommendation.
        Based on practitioner deployment guide (Appendix H).
        """
        if score.ensemble_score is None:
            return "unknown"
        if score.ensemble_score >= self.config.thresholds.immediate_intervention:
            return "pause_and_escalate"
        elif score.ensemble_score >= self.config.thresholds.flag_for_review:
            return "flag_for_review"
        else:
            return "pass"

    # ------------------------------------------------------------------
    # Taxonomy access
    # ------------------------------------------------------------------

    @staticmethod
    def get_taxonomy():
        return TAXONOMY

    @staticmethod
    def get_category(code: str):
        return get_category(code)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def train_ensemble(
        self,
        trajectories: List[AgentTrajectory],
        scores_bb: List[float],
        scores_ca_p: List[float],
        scores_ca_c: List[float],
        scores_ap_h: Optional[List[float]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Train the ensemble meta-classifier on pre-computed monitor scores.

        Args:
            trajectories: labeled training trajectories
            scores_*: parallel lists of monitor scores
        """
        import numpy as np

        labels = np.array([int(t.is_deceptive) for t in trajectories])
        n = len(labels)

        ap_h_arr = (
            np.array(scores_ap_h)
            if scores_ap_h else np.full(n, 5.5)
        )
        score_matrix = np.column_stack([
            np.array(scores_bb),
            np.array(scores_ca_p),
            np.array(scores_ca_c),
            ap_h_arr,
        ])

        self.ensemble.train(
            score_matrix,
            labels,
            hidden_dim=self.config.ensemble.hidden_dim,
            dropout=self.config.ensemble.dropout,
            lr=self.config.ensemble.lr,
            weight_decay=self.config.ensemble.weight_decay,
            epochs=self.config.ensemble.epochs,
            batch_size=self.config.ensemble.batch_size,
            save_path=save_path,
        )
        logger.info("Ensemble training complete.")

    def evaluate(
        self,
        trajectories: List[AgentTrajectory],
        scores: List[float],
        monitor_name: str,
        regime: MonitorRegime = MonitorRegime.BLACK_BOX,
    ):
        """Convenience wrapper for the evaluation harness."""
        from .src.benchmark.evaluator import DeceptArenaEvaluator
        evaluator = DeceptArenaEvaluator()
        return evaluator.evaluate(trajectories, scores, monitor_name, regime)
