"""
ResearchFlow — Graph State Definition

Defines the TypedDict that flows through the Supervisor StateGraph.
All nodes read from and write to this shared state.
"""

from typing import TypedDict, Literal
from utilities.plan_options import PlanStep


class ResearchState(TypedDict):
    """
    Shared state for the Supervisor graph.

    TODO: Expand these fields as your design evolves.

    Attributes:
        question: The original user research question.
        plan: Decomposed sub-tasks from the Planner node.
        plan_step: The current step in the plan to be executed.
        retrieved_chunks: Chunks returned by the Retriever agent.
        analysis: Synthesized response from the Analyst agent.
        fact_check_report: Verification report from the Fact-Checker agent.
        confidence_score: Overall confidence in the final answer (0.0–1.0).
        iteration_count: Number of self-refinement loops executed so far.
        HITL_threshold: Configurable confidence threshold where human interaction is required.
        max_refinement: Configurable number of times the agent is allowed to r
        scratchpad: Step-wise log of intermediate outputs for observability.
        user_id: Identifier for cross-thread memory via the Store interface.
    """
    question: str
    plan: list[PlanStep]
    plan_step: int
    retrieved_chunks: list[dict]
    analysis: dict
    fact_check_report: dict
    confidence_score: float
    iteration_count: int
    HITL_threshold: float
    max_refinement: int
    scratchpad: list[str]
    user_id: str
