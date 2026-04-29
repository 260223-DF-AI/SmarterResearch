from typing import Literal
from pydantic import BaseModel

class PlanStep(BaseModel):
    step: Literal['retriever_node', 'analyst_node', 'fact_checker_node', 'critique_node']

class Plan(BaseModel):
    """Tasks neccessary for a multiagentic AI system to perform to achieve a user query"""
    steps: list[PlanStep]
    reasoning: list[str]