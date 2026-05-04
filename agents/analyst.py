"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# from agents.state import ResearchState
from agents.state import ResearchState
from utilities.plan_options import PlanStep

# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single supporting citation."""

    source: str
    page_number: int | None = None
    excerpt: str


class AnalysisResult(BaseModel):
    """Pydantic model enforcing structured analyst output."""

    answer: str
    citations: list[Citation]
    confidence: float  # 0.0 – 1.0


def _format_chunks(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered, citeable block."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        page = f", p.{c['page_number']}" if c.get("page_number") else ""
        lines.append(f"[{i}] (source: {c['source']}{page})\n{c['chunk_text']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------------


def analyst_node(state: ResearchState) -> dict:
    """
    Synthesize retrieved chunks into a structured research response.

    TODO:
    - Build a prompt from the question, sub-task, and retrieved_chunks.
    - Invoke AWS Bedrock (e.g., Claude) with structured output enforcement.
    - Parse the response into an AnalysisResult.
    - Support streaming for real-time feedback.
    - Log actions to the scratchpad.

    Returns:
        Dict with "analysis" key containing the AnalysisResult as a dict,
        and "confidence_score" updated from the model's self-assessment.
    """
    print("analyst called")

    chunks = state.get("retrieved_chunks", [])
    log = [f"[analyst] synthesizing from {len(chunks)} chunks"]

    plan = state.get("plan", [])
    idx = state.get("current_subtask_index", 0)
    sub_task = plan[idx] if plan else state["question"]

    PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research analyst. Synthesize a precise answer to the user's "
                "question using ONLY the numbered context chunks below. Every factual "
                "claim must cite at least one chunk by its source filename and page. "
                "If the context does not support an answer, say so explicitly and set "
                "confidence below 0.4.\n\n"
                "Self-assess your confidence on a 0.0–1.0 scale where:\n"
                "  • 0.9+ = direct quote answers the question\n"
                "  • 0.6–0.9 = answer is supported by the context but requires inference\n"
                "  • <0.6 = context is partial, conflicting, or off-topic\n\n"
                "Output schema: return JSON with 'answer' (string), 'citations' "
                "(a JSON array of objects, each with 'source' and 'page_number'), "
                "and 'confidence' (number 0.0–1.0). Never return citations as a single string.",
            ),
            (
                "human",
                "Question: {question}\n\n"
                "Sub-task: {sub_task}\n\n"
                "Context chunks:\n{context_block}",
            ),
        ]
    )

    model_id = "amazon.nova-pro-v1:0"
    llm = ChatBedrock(
        model_id=model_id,
        region_name="us-east-1",
        model_kwargs={"max_tokens": 1024, "temperature": 0.2},
    )

    chain = PROMPT | llm.with_structured_output(AnalysisResult)

    response = chain.invoke(
        {
            "question": state["question"],
            "sub_task": sub_task,
            "context_block": _format_chunks(chunks),
        }
    )

    log.append(
        f"[analyst] confidence={response.confidence:.2f}, "
        f"citations={len(response.citations)}"
    )

    return {
        "analysis": response,
        "confidence_score": response.confidence,
        "scratchpad": log,
    }


if __name__ == "__main__":
    initial_state: ResearchState = {
        "question": "What are the effects of climate change on coral reefs?",
        "plan": [
            PlanStep(step="retriever_node"),
            PlanStep(step="analyst_node"),
            PlanStep(step="critique_node"),
        ],
        "plan_step": 0,
        "retrieved_chunks": [],
        "analysis": {},
        "fact_check_report": {},
        "confidence_score": 0.3,  # below threshold → retry
        "HITL_threshold": 0.7,
        "iteration_count": 0,
        "max_refinement": 1,  # only 1 retry allowed → will hit NotImplementedError
        "scratchpad": [],
        "user_id": "test_user",
    }

    analyst_node(initial_state)
