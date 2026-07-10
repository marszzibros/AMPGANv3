from anyio import Path
from typing import Annotated, TypedDict, Dict, Literal, Optional, List

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from .ExecutorState import ExecutorState



class PlanState(TypedDict):
    # init value for stage is "Planning"
    stage: Literal["Executing", "Planning", "Generating", "Filtering",  "Verifying", "END"]

    executor: Optional[List[ExecutorState]]

    
    messages: Optional[List[str]]

    from_exec: bool


