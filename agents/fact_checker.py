"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
Triggers HITL interrupt when confidence is below threshold.
"""

from pydantic import BaseModel
from agents.state import ResearchState


class ClaimVerdict(BaseModel):
    """Verification result for a single claim."""
    claim: str
    verdict: str  # "Supported" | "Unsupported" | "Inconclusive"
    evidence: str | None = None


class FactCheckReport(BaseModel):
    """Full verification report across all claims."""
    verdicts: list[ClaimVerdict]
    overall_confidence: float


def fact_checker_node(state: ResearchState) -> dict:
    """
    Verify the Analyst's response against trusted reference sources.

    TODO:
    - Extract claims from state["analysis"].
    - Query the 'fact-check-sources' Pinecone namespace for each claim.
    - Produce per-claim verdicts.
    - If confidence < threshold, trigger HITL interrupt.
    - Support Time Travel via state checkpointing.
    """
    # this is for testing purposes. Comment out for actual implementation:
    print("fact_checker called")
    return {'plan_step': state['plan_step'] + 1}
    # raise NotImplementedError
