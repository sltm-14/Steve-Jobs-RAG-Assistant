"""
Pydantic Models: requests/responses
This module defines request schemas using pydantic for a text ingestion and question workflow
"""

from pydantic import BaseModel
from typing import Literal


# Request model for querying previous ingested data
class AskRequest(BaseModel): 
    """
    Attributes:
    -question (str): The user's question.
    -top_k (int): Number of top matching results to retrieve.( Defaults to 3.)
    -mode (Literal["evidence", "persona"]):
        Response strategy:
        - "evidence": answer based on retrieved evidence.
        - "persona": answer using a persona-style response.
        Defaults to "evidence".
    """
    question: str
    top_k: int = 3
    mode: Literal["evidence","persona"] = "persona"
