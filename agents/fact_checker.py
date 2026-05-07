"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
Triggers HITL interrupt when confidence is below threshold.
"""

from pydantic import BaseModel, Field
import re
import os
from dotenv import load_dotenv
from pinecone import Pinecone, SearchQuery
from agents.analyst import AnalysisResult
from agents.state import ResearchState

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index_for_model(
        name=PINECONE_INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
    )
index = pc.Index(name=PINECONE_INDEX_NAME)

class ClaimVerdict(BaseModel):
    """Verification result for a single claim."""
    claim: str
    verdict: str = Field(pattern=r"^(Supported|Unsupported|Inconclusive)$")
    evidence: str | None = None


class FactCheckReport(BaseModel):
    """Full verification report across all claims."""
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)

class _SingleVerdict(BaseModel):
    """Schema the verdict-LLM is forced into."""
    verdict: str = Field(pattern=r"^(Supported|Unsupported|Inconclusive)$",
        description="Exactly one of: Supported, Unsupported, Inconclusive"
    )
    evidence: str = Field(description="A short quoted snippet from the evidence justifying the verdict")

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict fact-checker. Given a claim and supporting evidence, "
     "decide one of: Supported, Unsupported, Inconclusive.\n"
     "  • Supported = the evidence directly states or strongly implies the claim.\n"
     "  • Unsupported = the evidence contradicts the claim.\n"
     "  • Inconclusive = the evidence is silent on the claim.\n"
     "Quote a short snippet from the evidence as your justification.\n\n"
     "Output schema: return JSON with 'verdict' (one of the three labels above, "
     "exactly as spelled) and 'evidence' (a short string snippet from the input)."),
    ("human",
     "Claim: {claim}\n\nEvidence:\n{evidence}")
])

fact_check_llm = ChatOllama(model='llama3.2', temperature=0.05, format=_SingleVerdict.model_json_schema())

def _split_into_claims(answer: str) -> list[str]:
    """Heuristic claim extraction — split on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s for s in sentences if len(s) > 20]

def _verify_claim(claim: str) -> ClaimVerdict:
    results = index.search(namespace="review",query=SearchQuery(inputs={'text': claim}, top_k=3))

    if not results['result']['hits']:
        return ClaimVerdict(claim=claim, verdict="Inconclusive",
                            evidence="No supporting documents found.")
    
    evidence = "\n\n---\n\n".join(hit['fields']['chunk_text'] for hit in results['result']['hits'])
    
    verdict_chain = PROMPT | fact_check_llm
    result = _SingleVerdict.model_validate_json(verdict_chain.invoke({"claim": claim, "evidence": evidence}).content)
    
    return ClaimVerdict(claim=claim, verdict=result.verdict, evidence=result.evidence)


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
    log = ["[fact_checker] starting verification"]

    analysis: AnalysisResult = state.get("analysis") or {}
    answer = analysis.answer if analysis.answer else ""
    claims = _split_into_claims(answer)
    log.append(f"[fact_checker] extracted {len(claims)} claims")

    if not claims:
        report = FactCheckReport(verdicts=[], overall_confidence=0.0)
        return {
            "fact_check_report": report.model_dump(),
            "confidence_score": 0.0,
            "trigger_HITL": True,
            "scratchpad": log + ["[fact_checker] no claims, escalating to HITL"],
        }
    
    verdicts = [_verify_claim(c) for c in claims]
    counts = {"Supported": 0, "Unsupported": 0, "Inconclusive": 0}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1

    # Confidence = (supported - unsupported) / total, clamped to [0, 1].
    total = max(len(verdicts), 1)
    raw = (counts["Supported"] - counts["Unsupported"]) / total
    overall = max(0.0, min(1.0, raw))

    threshold = state.get("HITL_threshold", 0.6)
    trigger_HITL = overall < threshold or counts["Unsupported"] > 0

    report = FactCheckReport(verdicts=verdicts, overall_confidence=overall)
    log.append(
        f"[fact_checker] supported={counts['Supported']}, "
        f"unsupported={counts['Unsupported']}, inconclusive={counts['Inconclusive']}, "
        f"overall={overall:.2f}, hitl={trigger_HITL}"
    )

    return {
        "fact_check_report": report.model_dump(),
        "confidence_score": overall,
        "trigger_HITL": trigger_HITL,
        "scratchpad": log,
    }
