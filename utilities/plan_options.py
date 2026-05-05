from typing import Literal, Optional
from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    """The step as an action to be performed (node to be executed) in a plan. Plans can use retriever node multiple times"""
    step: Literal['retriever_node', 'analyst_node', 'fact_checker_node', 'critique_node']

class Plan(BaseModel):
    """Tasks neccessary for a multiagentic AI system to perform to achieve a user query.
    Can use each node more than once if necessary."""
    steps: list[PlanStep]
    reasoning: list[str] = Field(description="Step by step reasoning for why this plan was chosen")