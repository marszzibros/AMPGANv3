from anyio import Path
from typing import Annotated, TypedDict, Dict, List

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ExecutorState(TypedDict):
    agent: str
    execution_plan: str
    report: str
    step_reports: str
