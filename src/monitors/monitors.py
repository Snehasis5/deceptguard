"""
DeceptGuard Monitor Implementations
Black-box, CoT-aware, and activation-probe monitors.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..models import (
    AgentTrajectory,
    MonitorRegime,
    MonitorScore,
    ToolCall,
)
from .prompts import (
    DEFAULT_BB_PROMPT,
    DEFAULT_CA_C_PROMPT,
    DEFAULT_CA_P_PROMPT,
    MonitorPrompt,
)


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory formatter
# ──────────────────────────────────────────────────────────────────────────────

def format_trajectory(traj: AgentTrajectory, include_cot: bool = False) -> str:
    """Render a trajectory to the string format expected by monitor prompts."""
    parts: List[str] = []

    parts.append(f"STATED TASK: {traj.stated_task}")
    parts.append(f"ENVIRONMENT: {traj.environment}")
    parts.append("")

    if include_cot and traj.chain_of_thought:
        parts.append("[REASONING]")
        parts.append(traj.chain_of_thought.strip())
        parts.append("")

    if traj.tool_calls:
        parts.append("[TOOL CALLS]")
        for i, tc in enumerate(traj.tool_calls, 1):
            args_str = json.dumps(tc.arguments, indent=2) if tc.arguments else "{}"
            parts.append(f"  {i}. {tc.name}({args_str})")
            if tc.result:
                result_preview = tc.result[:300] + "..." if len(tc.result) > 300 else tc.result
                parts.append(f"     → {result_preview}")
        parts.append("")

    parts.append("[OUTPUT]")
    parts.append(traj.output.strip())

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Base monitor
# ──────────────────────────────────────────────────────────────────────────────

class BaseMonitor(ABC):
    """Abstract base for all monitor implementations."""

    name: str
    regime: MonitorRegime

    def __init__(self, api_client, model: str = "claude-sonnet-4-20250514"):
        self.client = api_client
        self.model = model

    @abstractmethod
    def score(self, trajectory: AgentTrajectory) -> MonitorScore:
        """Score a trajectory; returns MonitorScore with suspicion in [1, 10]."""
        ...

    def score_batch(
        self, trajectories: List[AgentTrajectory], delay: float = 0.2
    ) -> List[MonitorScore]:
        """Score a list of trajectories with optional rate-limiting delay."""
        scores = []
        for traj in trajectories:
            scores.append(self.score(traj))
            if delay > 0:
                time.sleep(delay)
        return scores

    # ------------------------------------------------------------------
    # Shared LLM call + score extraction helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call the backing LLM and return the raw text response."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @staticmethod
    def _extract_score(text: str) -> float:
        """
        Extract the final numeric score from a monitor response.
        Falls back to 5.0 (neutral) if no score is found.
        """
        # Look for FINAL ANSWER or ANSWER marker first
        for pattern in [
            r"FINAL ANSWER[:\s]+\[?(\d+(?:\.\d+)?)\]?",
            r"ANSWER[:\s]+\[?(\d+(?:\.\d+)?)\]?",
            r"\[(\d+(?:\.\d+)?)\]\s*$",
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"score[:\s]+(\d+(?:\.\d+)?)",
        ]:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                val = float(m.group(1))
                return max(1.0, min(10.0, val))

        # Last resort: take the last standalone number in the string
        nums = re.findall(r"\b([1-9]|10)\b", text)
        if nums:
            return float(nums[-1])
        return 5.0

    @staticmethod
    def _extract_intermediate_scores(text: str) -> Dict[str, float]:
        """
        Extract intermediate sub-scores (e.g. CoT Score, Behavioral Score)
        from a CA-P or CA-C response.
        """
        result = {}
        for label, pattern in [
            ("cot_score", r"CoT Score[:\s]+\[?(\d+(?:\.\d+)?)\]?"),
            ("behavioral_score", r"Behavioral Score[:\s]+\[?(\d+(?:\.\d+)?)\]?"),
            ("consistency_score", r"Consistency Score[:\s]+\[?(\d+(?:\.\d+)?)\]?"),
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                result[label] = max(1.0, min(10.0, float(m.group(1))))
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Black-Box Monitor
# ──────────────────────────────────────────────────────────────────────────────

class BlackBoxMonitor(BaseMonitor):
    """
    Constitutional black-box monitor.
    Input: tool calls and outputs only (no CoT).
    Regime: BLACK_BOX
    """

    name = "BB"
    regime = MonitorRegime.BLACK_BOX

    def __init__(
        self,
        api_client,
        model: str = "claude-sonnet-4-20250514",
        prompt: Optional[MonitorPrompt] = None,
    ):
        super().__init__(api_client, model)
        self.prompt_template = prompt or DEFAULT_BB_PROMPT

    def score(self, trajectory: AgentTrajectory) -> MonitorScore:
        traj_text = format_trajectory(trajectory, include_cot=False)
        full_prompt = self.prompt_template.template.format(trajectory=traj_text)

        raw = self._call_llm(full_prompt)
        final_score = self._extract_score(raw)
        axis_scores = self._extract_intermediate_scores(raw)

        return MonitorScore(
            trajectory_id=trajectory.trajectory_id,
            monitor_name=self.name,
            regime=self.regime,
            score=final_score,
            rationale=raw,
            axis_scores=axis_scores,
            raw_response=raw,
        )


# ──────────────────────────────────────────────────────────────────────────────
# CoT-Aware Prompted Monitor (CA-P)
# ──────────────────────────────────────────────────────────────────────────────

class CoTAwarePromptedMonitor(BaseMonitor):
    """
    CoT-aware prompted monitor.
    Input: CoT trace + tool calls + outputs.
    Regime: COT_AWARE
    """

    name = "CA-P"
    regime = MonitorRegime.COT_AWARE

    def __init__(
        self,
        api_client,
        model: str = "claude-sonnet-4-20250514",
        prompt: Optional[MonitorPrompt] = None,
    ):
        super().__init__(api_client, model)
        self.prompt_template = prompt or DEFAULT_CA_P_PROMPT

    def score(self, trajectory: AgentTrajectory) -> MonitorScore:
        traj_text = format_trajectory(trajectory, include_cot=True)
        full_prompt = self.prompt_template.template.format(trajectory=traj_text)

        raw = self._call_llm(full_prompt, max_tokens=1500)
        final_score = self._extract_score(raw)
        axis_scores = self._extract_intermediate_scores(raw)

        return MonitorScore(
            trajectory_id=trajectory.trajectory_id,
            monitor_name=self.name,
            regime=self.regime,
            score=final_score,
            rationale=raw,
            axis_scores=axis_scores,
            raw_response=raw,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Consistency-Probing Monitor (CA-C)
# ──────────────────────────────────────────────────────────────────────────────

class ConsistencyProbingMonitor(BaseMonitor):
    """
    Consistency-probing CoT monitor.
    Reconstructs implied reasoning from actions, then compares to stated CoT.
    Regime: COT_AWARE
    """

    name = "CA-C"
    regime = MonitorRegime.COT_AWARE

    def __init__(
        self,
        api_client,
        model: str = "claude-sonnet-4-20250514",
        prompt: Optional[MonitorPrompt] = None,
    ):
        super().__init__(api_client, model)
        self.prompt_template = prompt or DEFAULT_CA_C_PROMPT

    def score(self, trajectory: AgentTrajectory) -> MonitorScore:
        traj_text = format_trajectory(trajectory, include_cot=True)
        full_prompt = self.prompt_template.template.format(trajectory=traj_text)

        raw = self._call_llm(full_prompt, max_tokens=2000)
        final_score = self._extract_score(raw)
        axis_scores = self._extract_intermediate_scores(raw)

        return MonitorScore(
            trajectory_id=trajectory.trajectory_id,
            monitor_name=self.name,
            regime=self.regime,
            score=final_score,
            rationale=raw,
            axis_scores=axis_scores,
            raw_response=raw,
        )
