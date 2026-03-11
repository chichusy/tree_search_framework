# src/core/trace/writer.py
from __future__ import annotations

import json
import os
import hashlib
from typing import Optional, List, Dict

from src.core.trace.schema import CallRecord


class TraceLogger:
    """
    负责把 CallRecord 写成 jsonl，并提供 token-level reuse 统计：
      - vs last call: 时间局部性
      - vs parent node: 树结构局部性
    """

    def __init__(self, out_path: str, mode: str = "w"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.out_path = out_path
        self._f = open(out_path, mode, encoding="utf-8")
        self._call_id = 0

        # last call input ids (time locality)
        self._last_input_ids: Optional[List[int]] = None

        # store prompt input ids for each node when that node is used as prompt
        # (tree locality: compare current prompt with parent node's prompt)
        self._node_input_ids: Dict[int, List[int]] = {}

    @property
    def next_call_id(self) -> int:
        self._call_id += 1
        return self._call_id

    def log_call(self, rec: CallRecord) -> None:
        self._f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None

    @staticmethod
    def _digest_input_ids(input_ids: List[int]) -> str:
        # faster+smaller than joining huge strings
        b = ",".join(map(str, input_ids)).encode("utf-8")
        return hashlib.sha1(b).hexdigest()

    @staticmethod
    def _lcp_len(a: Optional[List[int]], b: Optional[List[int]]) -> int:
        if not a or not b:
            return 0
        m = min(len(a), len(b))
        i = 0
        while i < m and a[i] == b[i]:
            i += 1
        return i

    def compute_locality(
        self,
        *,
        node_id: int,
        parent_node_id: Optional[int],
        input_ids: Optional[List[int]],
        remember_for_node: bool = True,
    ) -> dict:
        """
        返回一组标准字段（建议直接写到 CallRecord 顶层字段）：

        - input_len, input_ids_digest
        - lcp_last/reuse_last/miss_last
        - lcp_parent/reuse_parent/miss_parent

        remember_for_node:
          - True: 记录 node_id 对应的 input_ids（用于后续孩子节点计算 lcp_parent）
        """
        if not input_ids:
            # still update last + node mapping to keep behavior consistent
            self._last_input_ids = input_ids
            if remember_for_node:
                self._node_input_ids[node_id] = input_ids or []
            return {
                "input_len": 0,
                "input_ids_digest": "",
                "lcp_last": 0,
                "reuse_last": 0,
                "miss_last": 0,
                "lcp_parent": 0,
                "reuse_parent": 0,
                "miss_parent": 0,
            }

        digest = self._digest_input_ids(input_ids)

        # vs last call
        lcp_last = self._lcp_len(self._last_input_ids, input_ids)
        reuse_last = lcp_last
        miss_last = len(input_ids) - reuse_last

        # vs parent node (tree locality)
        parent_ids = self._node_input_ids.get(parent_node_id) if parent_node_id is not None else None
        lcp_parent = self._lcp_len(parent_ids, input_ids)
        reuse_parent = lcp_parent
        miss_parent = len(input_ids) - reuse_parent

        # update state
        self._last_input_ids = input_ids
        if remember_for_node:
            self._node_input_ids[node_id] = input_ids

        return {
            "input_len": len(input_ids),
            "input_ids_digest": digest,
            "lcp_last": int(lcp_last),
            "reuse_last": int(reuse_last),
            "miss_last": int(miss_last),
            "lcp_parent": int(lcp_parent),
            "reuse_parent": int(reuse_parent),
            "miss_parent": int(miss_parent),
        }