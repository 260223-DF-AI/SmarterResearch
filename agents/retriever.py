"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

from agents.state import ResearchState
from pinecone import Pinecone, SearchQuery
from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrock
import cohere
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
co = cohere.Client(api_key=COHERE_API_KEY)

# llm initialization and setup
ollama_llm = ChatBedrock(model="amazon.nova-pro-v1:0",
    region = "us-east-1", 
    model_kwargs={
        "temperature": 0.05
}) 
# ollama_llm = ChatOllama(model='llama3.2', temperature=0.05)

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
    plan = state.get("plan", [])
    idx = state.get("plan_step", 0)
    query = plan[idx] if plan else state["question"]

    log = [f"[retriever] sub-task: {query!r}"]

    context = []
    docs = []

    # get semantic search results from pinecone
    results = index.search(namespace="research",
        query=SearchQuery(top_k=10, inputs={'text': query}, ),
        # rerank={
        #     # "model": "pinecone-rerank-v0",
        #     "model": "bge-reranker-v2-m3",
        #     "top_n": 10,
        #     "rank_fields": ["chunk_text"]
        # }
    )

    # formats each result as a dictionary and adds it to the list of context
    for hit in results['result']['hits']:
        print(hit)
        # compression query
        compression_query = f"""Clean the following string up to make it understandable.
        Extract and keep only the pieces relevant to the user query.
        Have each sentence on its own line.
        ONLY include the cleaned string and make your response as short as possible, no intro or outro.
        String to be cleaned: {hit['fields']['chunk_text']}
        User query: {query}"""
        # context compression: only extract information relevant to the user query
        # compressed = bedrock_llm.invoke(compression_query).content
        compressed = ollama_llm.invoke(compression_query).content
        print(compressed)
        docs.append(compressed)

        context.append({
            "content": compressed,
            "chunk_text": hit['fields']['chunk_text'],
            "relevance_score": round(hit['_score'], 3),
            "source": hit['fields']['source'],
            "page_number": hit['fields']['page']
        })
    log.append(f"retriever - Pinecone returned {len(context)} results")

    # rerank results with cohere to the user query on the compressed docs
    reranked_docs = co.rerank(
        query=query,
        documents=docs,
        top_n=5,  # controls how many reranked docs to return
        model="rerank-english-v3.0"
    )
    docs.clear()
    print(reranked_docs)

    # update semantic search results to reflect cohere rerank results
    for result in reranked_docs.results:
        context[result.index]['relevance_score'] = round(result.relevance_score, 3)
        docs.append(context[result.index])

    # update state
    return {
        # "plan_step": state["plan_step"] + 1,
        "retrieved_chunks": docs,
        "scratchpad": state['scratchpad'] + log
    }

    # this is for testing purposes. Comment out for actual implementation:
    # print("retriever called")
    # return {'plan_step': state['plan_step'] + 1}