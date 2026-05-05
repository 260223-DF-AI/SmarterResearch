"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""
from pydantic import BaseModel, Field

from agents.state import ResearchState
from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node

from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeInterrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama

from memory.store import get_user_preferences, get_query_history, append_query

class Plan(BaseModel):
    """Steps and tasks from a broken down user query to answer the question."""
    steps: list[str] = Field(description=(
        "An ordered JSON array of sub-task strings (1–4 entries). "
        "Each element MUST be a string. Do NOT return a single concatenated string."
    ))

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You decompose research questions into 1–4 ordered, independently-"
     "answerable sub-tasks/steps. Prefer fewer, larger sub-tasks over many tiny "
     "ones. Each sub-task should be answerable from a single retrieval.\n\n"
     "Output schema: return JSON with a single key 'steps' whose value is "
     "a JSON array of strings. Never return a single concatenated string."),
    ("human",
     "User preferences: {preferences}\n"
     "Recent past questions from this user: {history}\n\n"
     "New question: {question}\n\n"
     "Return the steps as a JSON list of strings."),
])
HISTORY_LIM = 5

def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """
    print(state)
    user = state.get("user_id", "default")
    user_preferences = get_user_preferences(user)
    query_history = get_query_history(user, limit=HISTORY_LIM)
    append_query(user, state["question"])

    # llm initialization and setup
    # planning_llm = ChatBedrock(model="amazon.nova-pro-v1:0",
    #     region = "us-east-1", 
    #     model_kwargs={
    #         "max_tokens": 512,
    #         "temperature": 0.0
    # }) 
    # planning_chain = PROMPT | planning_llm.with_structured_output(Plan)
    # result: Plan = planning_chain.invoke({
    #     "question": state["question"],
    #     "preferences": user_preferences,
    #     "history": query_history or ["<none>"],
    # }) # type: ignore
    
    ollama_planning_llm = ChatOllama(model='llama3.2', temperature=0.05, format=Plan.model_json_schema())
    planning_chain = PROMPT | ollama_planning_llm
    result = Plan.model_validate_json(planning_chain.invoke({
        "question": state["question"],
        "preferences": user_preferences,
        "history": query_history or ["<none>"],
    }).content) # type: ignore

    print(result.steps)
    return {
        'plan': result.steps,
        'plan_step': 0,
        "iteration_count": 0,
        "needs_hitl": False,
        'scratchpad': [f"[planner] decomposed into {len(result.steps)} sub-tasks"]
    }


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.
    """
    # if(state['plan_step'] >= len(state['plan'])): # check END condition and route to END if satisfied
    #     return END
    # return state['plan'][state['plan_step']].step # return the next step in the plan
    if not state.get("retrieved_chunks"):
        return "retriever_node"
    if not state.get("analysis"):
        return "analyst_node"
    if not state.get("fact_check_report"):
        return "fact_checker_node"
    return "critique_node"

def critique_router(state: ResearchState) -> str:
    """Edge after critique_node — END if accepted, else loop."""
    confidence = state.get("confidence_score", 0.0)
    # threshold = float(os.environ.get("HITL_CONFIDENCE_THRESHOLD", 0.6))
    threshold = float(state.get("HITL_threshold", 0.6))
    if confidence >= threshold and not state.get("trigger_HITL"):
        return END
    return "retriever_node"


def critique_node(state: ResearchState) -> dict:
    """
    Evaluate the aggregated response and decide: accept, retry, or escalate.

    TODO:
    - Check confidence_score against the HITL threshold.
    - If below threshold and iterations < max, loop back for refinement.
    - If below threshold and iterations >= max, trigger HITL interrupt.
    - If above threshold, accept and route to END.
    - Increment iteration_count.
    """
    # assign variables
    confidence = state.get('confidence_score', 0.0)
    iteration = state.get("iteration_count", 0) + 1
    threshold = state.get('HITL_threshold', 0.6)
    max_iterations = state.get("max_refinement", 3)

    log = [f"[critique] iter={iteration}, conf={confidence:.2f}, "
           f"threshold={threshold}, max_iter={max_iterations}"]

    print('Critique node called') # for testing and debugging purposes only
    # Accept condition and action
    if(confidence >= threshold and state.get("trigger_HITL") == False):
        log.append("[critique] accepted")
        return {"iteration_count": iteration, "scratchpad": log}
    
    # Retry condition and action
    if(iteration < max_iterations):
        log.append("[critique] retrying — clearing analysis & fact_check")
        # increment the value of the plan step and loop back for refinement
        return {
            "iteration_count": iteration,
            "retrieved_chunks": [],          # forces retriever to re-fetch
            "analysis": {},                  # forces analyst to re-synthesize
            "fact_check_report": {},
            "scratchpad": log,
        }
        
    # Escelate condition and action
    # Trigger HITL interruption. Write middleware later
    log.append("[critique] max iterations reached — escalating to HITL")
    # NodeInterrupt pauses the graph; resume by graph.update_state(...).
    raise NodeInterrupt(
        f"Confidence {confidence:.2f} below threshold {threshold} "
        f"after {iteration} iterations. Human review required."
    )


def build_supervisor_graph():
    """
    Construct and compile the Supervisor StateGraph.

    TODO:
    - Instantiate StateGraph with ResearchState.
    - Add nodes: planner, retriever, analyst, fact_checker, critique.
    - Add edges and conditional edges (router).
    - Set entry point to planner.
    - Compile and return the graph.

    Returns:
        A compiled LangGraph that can be invoked with an initial state.
    """
    # Initialize a StateGraph on ResearchState
    supervisor = StateGraph(ResearchState)

    # Adding nodes in the order they would probably get called
    supervisor.add_node(planner_node)
    supervisor.add_node(retriever_node)
    supervisor.add_node(analyst_node)
    supervisor.add_node(fact_checker_node)
    supervisor.add_node(critique_node)
    
    # Initial edge
    supervisor.add_edge(START, 'planner_node')

    # Conditional edges routing from planner and critique nodes
    supervisor.add_conditional_edges('planner_node', router, 
        ['retriever_node', 'analyst_node', 'fact_checker_node', 'critique_node'])
    supervisor.add_conditional_edges('critique_node', critique_router, 
        ['retriever_node', END])
    
    # Linear edges routing through the graph for each subtask
    supervisor.add_edge("retriever_node", "analyst_node")
    supervisor.add_edge("analyst_node", "fact_checker_node")
    supervisor.add_edge("fact_checker_node", "critique_node")

    # Compile and return supervisor/orchestraion layer graph
    return supervisor.compile(checkpointer=MemorySaver())