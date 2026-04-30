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
from langchain_ollama import ChatOllama

from plan_options import Plan, PlanStep

def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """
    query = f"""
You are a planner, use the plan-and-execute pattern to create a list of subtasks to achieve the user query.

these are the actions that can be taken in the plan:
- 'retriever_node': Retrieve relevant document chunks for the current sub-task, 
- 'analyst_node': Synthesize retrieved chunks into a structured research response,
- 'fact_checker_node': Verify the Analyst's response against trusted reference sources,
- 'critique_node': Evaluate the aggregated response and decide: accept, retry, or escalate

this is the user query you are trying to achieve:
{state['question']}
"""
    # llm initialization and setup
    # planning_llm = ChatBedrock(model="amazon.nova-pro-v1:0",
    #     region = "us-east-1", 
    #     model_kwargs={
    #         "temperature": 0.05
    # }) 
    # planning_llm = planning_llm.with_structured_output(Plan)
    # result: Plan = planning_llm.invoke(query) # type: ignore
    
    ollama_planning_llm = ChatOllama(model='llama3.2', temperature=0.05, format=Plan.model_json_schema())
    result = Plan.model_validate_json(ollama_planning_llm.invoke(query).content) # type: ignore
    print(result)

    print(result.steps)
    print(result.reasoning)

    return {
        'plan': result.steps,
        'scratchpad': state['scratchpad'] + result.reasoning
    }


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    if(state['plan_step'] >= len(state['plan'])): # check END condition and route to END if satisfied
        return END
    return state['plan'][state['plan_step']].step # return the next step in the plan


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
    print('Critique node called') # for testing and debugging purposes only
    if(state['confidence_score'] < state['HITL_threshold']):
        # Retry condition and action
        if(state['iteration_count'] < state['max_refinement']):
            # TODO: loop back for refinement
            # increment the value of the plan step
            return {
                'plan_step': 0,
                'iteration_count': state['iteration_count'] + 1
            }
        
        # Escelate condition and action
        else:
            # TODO: trigger HITL interruption. Write middleware later
            pass 

    # Accept condition and action
    else:
        # Increment step count to trigger route to END
        # Need to route to end?
        return {'plan_step': state['plan_step'] + 1}

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

    # Conditional edges mapping all nodes to all non-planner nodes
    supervisor.add_conditional_edges('planner_node', router, 
        ['retriever_node', 'analyst_node', 'fact_checker_node', 'critique_node', END])
    supervisor.add_conditional_edges('retriever_node', router, 
        ['retriever_node', 'analyst_node', 'fact_checker_node', 'critique_node', END])
    supervisor.add_conditional_edges('analyst_node', router, 
        ['retriever_node', 'analyst_node', 'fact_checker_node', 'critique_node', END])
    supervisor.add_conditional_edges('fact_checker_node', router, 
        ['retriever_node', 'analyst_node', 'fact_checker_node', 'critique_node', END])
    supervisor.add_conditional_edges('critique_node', router, 
        ['retriever_node', 'analyst_node', 'fact_checker_node', 'critique_node', END])
    
    # Compile and return supervisor/orchestraion layer graph
    return supervisor.compile()

if __name__ == '__main__':
    graph = build_supervisor_graph()

    initial_state: ResearchState = {
        'question': 'What are the effects of climate change on coral reefs?',
        'plan': [PlanStep(step='retriever_node'), PlanStep(step='analyst_node'), PlanStep(step='fact_checker_node'), PlanStep(step='critique_node')],
        'plan_step': 0,
        'retrieved_chunks': [],
        'analysis': {},
        'fact_check_report': {},
        'confidence_score': 0.9,
        'iteration_count': 0,
        'HITL_threshold': 0.7,
        'max_refinement': 3,
        'scratchpad': [],
        'user_id': 'test_user'
    }
    result = graph.invoke(initial_state)
    print(result)
    
    initial_state: ResearchState = {
        'question': 'What are the discoveries on evolutionary algorithms in 2026 and how do they work?',
        'plan': [],
        'plan_step': 0,
        'retrieved_chunks': [],
        'analysis': {},
        'fact_check_report': {},
        'confidence_score': 0.9,
        'iteration_count': 0,
        'HITL_threshold': 0.7,
        'max_refinement': 3,
        'scratchpad': [],
        'user_id': 'test_user'
    }
    result = graph.invoke(initial_state)
    print(result)

    initial_state: ResearchState = {
        'question': 'What are the effects of climate change on coral reefs?',
        'plan': [PlanStep(step='retriever_node'), PlanStep(step='analyst_node'), PlanStep(step='critique_node')],
        'plan_step': 0,
        'retrieved_chunks': [],
        'analysis': {},
        'fact_check_report': {},
        'confidence_score': 0.3,   # below threshold → retry
        'HITL_threshold': 0.7,
        'iteration_count': 0,
        'max_refinement': 1,       # only 1 retry allowed → will hit NotImplementedError
        'scratchpad': [],
        'user_id': 'test_user'
    }

    result = graph.invoke(initial_state)
