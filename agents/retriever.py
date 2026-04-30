"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

from state import ResearchState
from plan_options import PlanStep
from pinecone import Pinecone, SearchQuery, SearchRerank
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index_for_model(
        name=PINECONE_INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
    )
index = pc.Index(name=PINECONE_INDEX_NAME)


def retriever_node(state: ResearchState) -> dict:
    """
    Retrieve relevant document chunks for the current sub-task.

    TODO:
    - Extract the current sub-task from state["plan"].
    - Query the Pinecone index with semantic search and metadata filters.
    - Apply context compression to reduce token noise.
    - Apply re-ranking to prioritize the most relevant results.
    - Return updated state with retrieved_chunks populated.
    - Log actions to the scratchpad.

    Returns:
        Dict with "retrieved_chunks" key containing a list of dicts,
        each with: content, relevance_score, source, page_number.
    """
    query = state["question"]

    results = index.search(namespace="research",
        query=SearchQuery(top_k=10, inputs={'text': query}),
        rerank={
            "model": "pinecone-rerank-v0",
            "top_n": 10,
            "rank_fields": ["chunk_text"]
        }
    )
    for hit in results['result']['hits']:
        # print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")
        print(f"{hit}")
    # this is for testing purposes. Comment out for actual implementation:
    # print("retriever called")
    return {'plan_step': state['plan_step'] + 1}
    # raise NotImplementedError

if __name__ == '__main__':
    initial_state: ResearchState = {
        'question': "What are the most recent developments in computer science?",
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
    retriever_node(initial_state)