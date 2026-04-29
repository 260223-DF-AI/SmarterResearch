"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""

from agents.state import ResearchState
from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node
from langgraph.graph import StateGraph, START, END

from langchain_aws import ChatBedrock

from utilities.plan_options import Plan

def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """
    # llm initialization and setup
    planning_llm = ChatBedrock(model="mistral.mistral-7b-instruct-v0:2",
        region = "us-east-1", 
        model_kwargs={
            "temperature": 0.1
    }) # type: ignore[call-arg]
    planning_llm = planning_llm.with_structured_output(Plan)

    query = f"""
You are a planner, use the plan-and-execute pattern to create a list of subtasks to achieve the user query.

record the steps in the plan in 'steps' 
record the reasoning for taking those steps in 'reasoning'

these are the actions that can be taken in the plan:
- 'retriever_node': Retrieve relevant document chunks for the current sub-task, 
- 'analyst_node': Synthesize retrieved chunks into a structured research response,
- 'fact_checker_node': Verify the Analyst's response against trusted reference sources,
- 'critique_node': Evaluate the aggregated response and decide: accept, retry, or escalate

this is the user query you are trying to achieve:
{state['question']}
"""

    result = planning_llm.invoke(query)

    print(result.steps) # type: ignore
    print(result.reasoning) # type: ignore
    
    state['plan'] = result.steps # type: ignore
    for i in range(len(result.reasoning)): # type: ignore
        state['scratchpad'].append(result.reasoning[i]) # type: ignore

    return dict(state)


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    step = state['plan'][state['plan_step']]
    raise NotImplementedError


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
    if(state['confidence_score'] < state['HITL_threshold']):
        if(state['iteration_count'] < state['max_refinement']):
            # TODO: loop back for refinement
            pass
        else:
            # TODO: trigger HITL interruption
            pass 
    else:
        # TODO: route to END
        pass

    # increment the value of the plan step
    state['iteration_count'] += 1
    raise NotImplementedError


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
    supervisor = StateGraph(ResearchState)

    # Adding nodes in the order they would probably get called
    supervisor.add_node(planner_node)
    supervisor.add_node(retriever_node)
    supervisor.add_node(analyst_node)
    supervisor.add_node(fact_checker_node)
    supervisor.add_node(critique_node)
    
    supervisor.add_edge(START, 'planner_node')
    supervisor.add_conditional_edges('planner_node', router, 
        ['retriever_node', 'analyst_node', 'fact_checker_node', 'critique_node'])
    supervisor.add_edge('critique_node', 'critique_node')
    supervisor.add_edge('critique_node', END)
    

    return supervisor.compile()
