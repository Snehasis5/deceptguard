"""
DeceptGuard Configuration
Central configuration for all system components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """LLM backend configuration."""
    monitor_backbone: str = "claude-sonnet-4-20250514"
    monitor_backbone_opus: str = "claude-opus-4-5"  # for CA-C high-complexity
    generator_model: str = "claude-sonnet-4-20250514"
    verifier_model: str = "claude-sonnet-4-20250514"
    ap_h_encoder: str = "meta-llama/Llama-3.1-70B-Instruct"


@dataclass
class DeceptSynthConfig:
    """DECEPTSYNTH generation pipeline configuration."""
    scenarios_per_category: int = 400         # paper target
    min_realism_score: float = 7.0             # reject if below this
    train_val_split: float = 0.8               # fraction used for training
    cot_condition_weights: Dict[str, float] = field(default_factory=lambda: {
        "honest": 0.50,
        "sanitized": 0.35,
        "deceptive": 0.15,
    })
    intensity_weights: Dict[str, float] = field(default_factory=lambda: {
        "overt": 0.25,
        "semi_covert": 0.50,
        "covert": 0.25,
    })
    environments: List[str] = field(default_factory=lambda: [
        "software_development",
        "financial_analysis",
        "scientific_research",
        "legal_services",
        "devops_infrastructure",
        "personal_productivity",
    ])


@dataclass
class ActivationProbeConfig:
    """AP-H activation probe configuration."""
    layer: int = 32                     # l* from cross-validation
    max_tokens: int = 2000              # CoT truncation limit
    probe_C: float = 0.1               # logistic regression regularisation
    n_cv_folds: int = 5
    batch_size: int = 8                 # inference batch size
    device: str = "cuda"
    model_dtype: str = "bfloat16"
    hidden_dim: int = 8192              # Llama-3.1-70B residual stream dimension


@dataclass
class EnsembleConfig:
    """HybridConstitutional ensemble configuration."""
    hidden_dim: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 64
    neutral_imputation: float = 5.5    # imputed score for unavailable monitors


@dataclass
class EvaluationConfig:
    """Evaluation harness configuration."""
    max_fpr: float = 0.20              # pAUROC window
    n_bootstrap: int = 1000            # CI resamples
    ci_alpha: float = 0.05             # 95% CI


@dataclass
class OperationalThresholds:
    """
    Production deployment thresholds (Appendix H).
    """
    flag_for_review: float = 7.0        # score ≥ 7: human review queue
    immediate_intervention: float = 9.0 # score ≥ 9: pause agent, escalate


@dataclass
class DeceptGuardConfig:
    """Master configuration object."""
    models: ModelConfig = field(default_factory=ModelConfig)
    deceptsynth: DeceptSynthConfig = field(default_factory=DeceptSynthConfig)
    probe: ActivationProbeConfig = field(default_factory=ActivationProbeConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    thresholds: OperationalThresholds = field(default_factory=OperationalThresholds)

    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    probe_path: Optional[str] = None
    ensemble_path: Optional[str] = None


# Default configuration instance
DEFAULT_CONFIG = DeceptGuardConfig()
