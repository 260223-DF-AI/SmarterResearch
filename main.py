"""
ResearchFlow — Main Entry Point

Parses CLI arguments and invokes the Supervisor graph to answer
a research question against the ingested document corpus.
"""

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.errors import GraphInterrupt

from agents.analyst import AnalysisResult
from agents.supervisor import build_supervisor_graph
from middleware.guardrails import detect_injection, sanitize_input
from middleware.pii_masking import mask_pii


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ResearchFlow: Adaptive Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="The research question to answer.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="default",
        help="User ID for cross-thread memory (Store interface).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable step-wise scratchpad logging.",
    )
    return parser.parse_args()

def main() -> None:
    """
    High-level flow:
    1. Load environment variables.
    2. Initialize the Supervisor graph (see agents/supervisor.py).
    3. Invoke the graph with the user's question.
    4. Print the structured research report.
    """
    load_dotenv()
    args = parse_args()

    # stop prompt injection & mask pii

    if detect_injection(args.question):
        print(
            "PROMPT INJECTION DETECTED. THIS IS EXPRESSLY FORBIDDEN UNDER SMARTERRESEARCH PROTOCOLS AND IS NOT RECEIVED KINDLY. PROGRAM EXITING POSTHASTE."
        )
        return
    question = mask_pii(sanitize_input(args.question))

    # build the initial graph state from args
    graph = build_supervisor_graph()

    # unique threads mean state is not collected across all invocations
    thread_id = f"cli-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": question,
        "user_id": args.user_id,
    }

    # invoke the graph and collect the final state
    
    try:
        result = graph.invoke(initial_state, config=config)
    except GraphInterrupt as interrupt:
        print("\n=== HUMAN-IN-THE-LOOP REVIEW REQUIRED ===")
        print(f"Reason: {interrupt}")

        graph_state = graph.get_state(config)
        current_analysis = graph_state.values.get("analysis", {})
        print("\nDraft answer:\n", current_analysis.get("answer", "<empty>"))
        decision = input("\nApprove answer as-is? [y/n]: ").strip().lower()
        if decision not in ["y", "yes", "yup", "uhuh", "mhm", "ya", "yuh"]:
            print("Rejected by reviewer. Aborting.")
            return

        # mark hitl approved in the state
        graph.update_state(config, {"needs_hitl": False, "confidence_score": 1.0})
        result = graph.invoke(None, config=config)

    analysis: AnalysisResult = result.get("analysis", {})
    safe_answer = mask_pii(analysis.answer if analysis.answer else "")

    # pretty-print the structured research report [thank you rich]
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(safe_answer)
    print("\nCITATIONS")
    for c in analysis.citations if analysis.citations else []:
        page = f", p.{c.page_number}" if c.page_number else ""
        print(f"  • {c.source}{page}: {c.excerpt[:120] if c.excerpt else ""}")
    print(f"\nCONFIDENCE: {result.get('confidence_score', 0.0):.2f}")
    print(f"ITERATIONS: {result.get('iteration_count', 0)}")

    if args.verbose:
        print("\nSCRATCHPAD")
        for line in result.get("scratchpad", []):
            print(" ", line)

    if result.get("fact_check_report"):
        print("\nFACT-CHECK REPORT")
        for v in result["fact_check_report"]["verdicts"]:
            print(f"  [{v['verdict']}] {v['claim'][:80]}")


if __name__ == "__main__":
    main()
