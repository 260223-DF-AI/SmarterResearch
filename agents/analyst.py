"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""

from langchain_aws import ChatBedrock
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

    prompt: str = f"""User question: {state["question"]}

Planner sub-tasks: {"\n".join(state["plan"])}

Retrieved chunks: {state["retrieved_chunks"]}
"""

    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    llm = ChatBedrock(
        model_id=model_id,
        region_name="us-east-1",
        streaming=True,
    )

    llm = llm.with_structured_output(AnalysisResult)

    response: str = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk

    print(f"{response=}, {type(response)=}")

    # return {"analysis": response, "confidence_score": "NYI"}
    return {"plan_step": state["plan_step"] + 1}


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
