"""
DeceptGuard Data Models
Trajectory, monitor score, and dataset structures.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CoTCondition(str, Enum):
    """
    Three CoT deception patterns used in DECEPTSYNTH.
    """
    HONEST = "honest"           # CoT faithfully reflects deceptive intent
    SANITIZED = "sanitized"     # CoT scrubbed of deceptive intent; subtle inconsistencies remain
    DECEPTIVE = "deceptive"     # CoT constructs a misleading narrative


class DeceptionIntensity(str, Enum):
    OVERT = "overt"             # easily detectable (~25% of positives)
    SEMI_COVERT = "semi_covert" # primary evaluation condition (~50%)
    COVERT = "covert"           # most sophisticated concealment (~25%)


class MonitorRegime(str, Enum):
    BLACK_BOX = "black_box"           # actions + outputs only
    COT_AWARE = "cot_aware"           # + chain-of-thought text
    ACTIVATION_PROBE = "activation_probe"  # + hidden-state vectors


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None


@dataclass
class AgentTrajectory:
    """
    A single agent trajectory, potentially containing internal CoT,
    tool calls, and user-facing output.
    """
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    environment: str = ""
    stated_task: str = ""
    chain_of_thought: Optional[str] = None   # may be None (BB) or present (CoT-aware)
    tool_calls: List[ToolCall] = field(default_factory=list)
    output: str = ""

    # Ground-truth annotation (present only in labeled data)
    is_deceptive: Optional[bool] = None
    deception_category: Optional[str] = None   # taxonomy code, e.g. "B3"
    cot_condition: Optional[CoTCondition] = None
    intensity: Optional[DeceptionIntensity] = None
    hidden_objective: Optional[str] = None     # kept private; not shown to monitors

    # Activation vector (extracted offline; populated by ActivationExtractor)
    activation_vector: Optional[List[float]] = None

    def to_black_box_view(self) -> "AgentTrajectory":
        """Return a copy with CoT and hidden fields stripped (BB monitor input)."""
        import copy
        view = copy.deepcopy(self)
        view.chain_of_thought = None
        view.hidden_objective = None
        view.activation_vector = None
        return view

    def to_cot_view(self) -> "AgentTrajectory":
        """Return a copy with hidden_objective stripped (CoT monitor input)."""
        import copy
        view = copy.deepcopy(self)
        view.hidden_objective = None
        view.activation_vector = None
        return view


@dataclass
class MonitorScore:
    """Suspicion score in [1, 10] with supporting rationale."""
    trajectory_id: str
    monitor_name: str
    regime: MonitorRegime
    score: float            # [1.0, 10.0]
    rationale: str = ""
    axis_scores: Dict[str, float] = field(default_factory=dict)   # per-axis breakdown
    raw_response: str = ""


@dataclass
class EnsembleScore:
    """Aggregated score from the HybridConstitutional ensemble."""
    trajectory_id: str
    bb_score: Optional[float] = None
    ca_p_score: Optional[float] = None
    ca_c_score: Optional[float] = None
    ap_h_score: Optional[float] = None
    ensemble_score: Optional[float] = None
    label: Optional[bool] = None  # ground truth (for evaluation)

    def available_scores(self) -> Dict[str, float]:
        result = {}
        if self.bb_score is not None:
            result["bb"] = self.bb_score
        if self.ca_p_score is not None:
            result["ca_p"] = self.ca_p_score
        if self.ca_c_score is not None:
            result["ca_c"] = self.ca_c_score
        if self.ap_h_score is not None:
            result["ap_h"] = self.ap_h_score
        return result


@dataclass
class EvaluationResult:
    """Full evaluation result for a benchmark run."""
    monitor_name: str
    regime: MonitorRegime
    pauroc: float           # primary metric, FPR ≤ 0.20
    full_auroc: float
    per_category_pauroc: Dict[str, float] = field(default_factory=dict)
    per_environment_pauroc: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    fpr_at_tpr_80: Optional[float] = None
    fpr_at_tpr_90: Optional[float] = None
    fpr_at_tpr_95: Optional[float] = None
