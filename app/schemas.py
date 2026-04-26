"""
Pydantic Models: requests/responses

This module defines request schemas using pydantic for a text ingestion and question workflow
"""

from pydantic import BaseModel
from typing import Literal

# Request model for ingesting content
class IngestTextRequest(BaseModel):
    """
    Attributes:
    -chunks (list[str]): List of text fragments or segments to be processed and stored.
    -source (str): Identifier or name of the origin of the text data.
    """
    chunks: list[str]
    source: str

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
    mode: Literal["evidence","persona"] = "evidence"
