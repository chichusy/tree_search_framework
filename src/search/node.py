# src/workload/node.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Node:
    node_id: int
    parent_id: Optional[int]
    depth: int

    prefix_text: str
    prefix_tokens_len: int

    children: List[int] = field(default_factory=list)
