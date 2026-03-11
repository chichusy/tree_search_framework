from __future__ import annotations

import math
import random
import time
from typing import Dict, Optional, List, Tuple

from src.search.node import Node
from src.core.trace.writer import TraceLogger
from src.core.trace.schema import CallRecord

from src.core.reward.base import RewardScorer, RewardResult
from src.core.reward.none import NoneScorer

from src.core.terminal.base import TerminalChecker, TerminalDecision
from src.core.terminal.depth import MaxDepthTerminal


class MCTSWorkload:
    """
    rStar 风格的 MCTS workload，用于研究 LLM inference / KVCache 行为。
    """

    def __init__(
        self,
        lm,
        trace_logger: TraceLogger,
        tokenizer=None,
        max_depth: int = 6,
        branch_factor: int = 2,
        # exploration
        exploration_weight: float = 1.4,
        weight_scheduler: str = "const",  # "exp" | "lin" | "const"
        num_rollouts: int = 100,
        discount: float = 1.0,
        # generation
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_sequences: Optional[List[str]] = None,
        # reward
        reward_scorer: RewardScorer | None = None,
        terminal_checker: TerminalChecker | None = None,
        prm_agg: str = "mean",
        # prompt shaping
        fixed_prompt_tokens_per_step: int = 32,
        cache_key_mode: str = "node",  # currently only recorded in trace meta
        # trace text dumping
        save_prompt_text: bool = False,
        save_output_text: bool = False,
        save_prompt_max_chars: int = 0,   # 0 means no truncation; keep tail
        save_output_max_chars: int = 0,   # 0 means no truncation; keep head
    ):
        self.lm = lm
        self.logger = trace_logger
        self.tokenizer = tokenizer

        self.max_depth = max_depth
        self.branch_factor = branch_factor

        self.exploration_weight = exploration_weight
        self.weight_scheduler = weight_scheduler
        self.num_rollouts = num_rollouts
        self.discount = discount

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop_sequences = stop_sequences or []
        self.fixed_prompt_tokens_per_step = fixed_prompt_tokens_per_step

        # tree storage
        self.nodes: Dict[int, Node] = {}
        self.next_node_id = 0
        self.parent2children: Dict[int, List[int]] = {}

        # MCTS stats
        self.Q: Dict[int, float] = {}
        self.N: Dict[int, int] = {}

        # explored set
        self.explored_nodes: set[int] = set()

        # Reward scorer
        self.reward_scorer: RewardScorer = reward_scorer or NoneScorer()

        # Terminal checker (pluggable)
        self.terminal_checker: TerminalChecker = terminal_checker or MaxDepthTerminal(self.max_depth)

        # PRM aggregation (only used when reward_scorer.mode == "prm")
        self.prm_agg = (prm_agg or "mean").lower()

        self.root_text: Optional[str] = None
        self.cache_key_mode = cache_key_mode

        # trace text dumping
        self.save_prompt_text = save_prompt_text
        self.save_output_text = save_output_text
        self.save_prompt_max_chars = save_prompt_max_chars
        self.save_output_max_chars = save_output_max_chars

    def _next_call_id(self) -> int:
        return self.logger.next_call_id

    # --------------------------
    # Node utils
    # --------------------------
    def _new_node(self, parent: Optional[int], depth: int, prefix_text: str, prefix_tokens_len: int) -> int:
        self.next_node_id += 1
        nid = self.next_node_id
        self.nodes[nid] = Node(
            node_id=nid,
            parent_id=parent,
            depth=depth,
            prefix_text=prefix_text,
            prefix_tokens_len=prefix_tokens_len,
        )
        return nid

    def init_root(self, root_text: str, root_tokens_len: int) -> int:
        return self._new_node(None, 0, root_text, root_tokens_len)

    def _terminal_decision(self, nid: int) -> TerminalDecision:
        node = self.nodes[nid]
        return self.terminal_checker.decide(
            text=node.prefix_text,
            depth=node.depth,
            tokens_len=node.prefix_tokens_len,
            root_text=self.root_text or "",
        )

    def _is_terminal(self, nid: int) -> bool:
        return self._terminal_decision(nid).is_terminal

    # --------------------------
    # UCT helpers
    # --------------------------
    def _get_weight(self, rollout_id: int) -> float:
        if self.weight_scheduler == "exp":
            return self.exploration_weight * (0.1 ** (rollout_id / max(1, self.num_rollouts)))
        elif self.weight_scheduler == "lin":
            return self.exploration_weight * (1 - 0.9 * (rollout_id / max(1, self.num_rollouts)))
        else:
            return self.exploration_weight

    def _compute_uct(self, parent_id: int, child_id: int, rollout_id: int) -> float:
        n_child = self.N.get(child_id, 0)
        if n_child == 0:
            return float("inf")
        n_parent = max(1, self.N.get(parent_id, 1))
        q_child = self.Q.get(child_id, 0.0)
        w = self._get_weight(rollout_id)
        return (q_child / n_child) + w * math.sqrt(math.log(n_parent) / n_child)

    def _uct_select(self, parent_id: int, rollout_id: int) -> int:
        children = self.parent2children[parent_id]
        assert all(c in self.explored_nodes for c in children)
        return max(children, key=lambda c: self._compute_uct(parent_id, c, rollout_id))

    # --------------------------
    # LLM call + trace
    # --------------------------
    def _llm_call_create_child(self, rollout_id: int, parent_id: int, phase: str) -> int:
        parent = self.nodes[parent_id]
        prompt_text = parent.prefix_text

        # Tokenize prompt
        input_ids = None
        if self.tokenizer is not None:
            try:
                input_ids = self.tokenizer.encode(prompt_text)
                prompt_tokens_len = len(input_ids)
            except Exception:
                input_ids = None
                prompt_tokens_len = 0
        else:
            # fallback only for dummy / no-tokenizer mode
            prompt_tokens_len = parent.prefix_tokens_len + self.fixed_prompt_tokens_per_step

        # locality stats
        loc = self.logger.compute_locality(
            node_id=parent_id,
            parent_node_id=parent.parent_id,
            input_ids=input_ids,
            remember_for_node=True,
        )

        # LLM generate
        t0 = int(time.time() * 1000)

        try:
            res = self.lm.generate(
                prompt_text,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=self.stop_sequences if self.stop_sequences else None,
            )
        except TypeError:
            res = self.lm.generate(
                prompt_text,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        t1 = int(time.time() * 1000)

        out_text = getattr(res, "output_text", "") or ""
        out_tokens_len = int(getattr(res, "output_tokens_len", 0) or 0)

        # Optional dump prompt/output text
        prompt_text_saved = None
        output_text_saved = None

        if self.save_prompt_text:
            prompt_text_saved = prompt_text
            if (
                self.save_prompt_max_chars
                and self.save_prompt_max_chars > 0
                and len(prompt_text_saved) > self.save_prompt_max_chars
            ):
                prompt_text_saved = prompt_text_saved[-self.save_prompt_max_chars:]  # keep tail

        if self.save_output_text:
            output_text_saved = out_text
            if (
                output_text_saved
                and self.save_output_max_chars
                and self.save_output_max_chars > 0
                and len(output_text_saved) > self.save_output_max_chars
            ):
                output_text_saved = output_text_saved[:self.save_output_max_chars]  # keep head

        # Create child node
        child_prefix_text = prompt_text + "\n" + out_text

        # Create child node (token length)
        if self.tokenizer is not None:
            try:
                child_input_ids = self.tokenizer.encode(child_prefix_text)
                child_prefix_tokens_len = len(child_input_ids)
            except Exception:
                child_prefix_tokens_len = prompt_tokens_len + out_tokens_len
        else:
            child_prefix_tokens_len = prompt_tokens_len + out_tokens_len

        child_id = self._new_node(
            parent=parent_id,
            depth=parent.depth + 1,
            prefix_text=child_prefix_text,
            prefix_tokens_len=child_prefix_tokens_len,
        )

        self.parent2children.setdefault(parent_id, []).append(child_id)
        parent.children.append(child_id)

        # Trace record
        rec = CallRecord(
            call_id=self._next_call_id(),
            rollout_id=rollout_id,
            node_id=parent_id,
            parent_node_id=parent.parent_id,
            purpose=phase,
            input_len=loc["input_len"],
            output_len=out_tokens_len,
            input_ids_digest=loc["input_ids_digest"],
            lcp_last=loc["lcp_last"],
            reuse_last=loc["reuse_last"],
            miss_last=loc["miss_last"],
            lcp_parent=loc["lcp_parent"],
            reuse_parent=loc["reuse_parent"],
            miss_parent=loc["miss_parent"],
            t_start_ms=t0,
            t_end_ms=t1,
            meta={
                "created_child_id": child_id,
                "depth": parent.depth,
                "phase": phase,
                "cache_key_mode": self.cache_key_mode,
                "vllm_num_cached_tokens": int(getattr(res, "num_cached_tokens", 0) or 0),
                "stop_sequences": self.stop_sequences if self.stop_sequences else None,
            },
            prompt_text=prompt_text_saved,
            output_text=output_text_saved,
        )

        self.logger.log_call(rec)
        return child_id

    def _log_event(self, rollout_id: int, node_id: int, purpose: str, meta: dict):
        n = self.nodes[node_id]
        rec = CallRecord(
            call_id=self._next_call_id(),
            rollout_id=rollout_id,
            node_id=node_id,
            parent_node_id=n.parent_id,
            purpose=purpose,
            meta=meta,
        )
        self.logger.log_call(rec)

    # --------------------------
    # Reward
    # --------------------------
    def _aggregate_prm_scores(self, scores: List[float]) -> float:
        """
        Aggregate step-wise PRM scores in [0,1] into a scalar rollout reward.

        Supported:
          - mean: average
          - min:  minimum
          - prod: product
          - last: last score
          - terminal: only terminal score (same as last if you append terminal at end)
        """
        if not scores:
            return 0.0

        agg = (self.prm_agg or "mean").lower()
        if agg == "mean":
            return float(sum(scores) / max(1, len(scores)))
        if agg == "min":
            return float(min(scores))
        if agg == "prod":
            p = 1.0
            for s in scores:
                p *= float(s)
            return float(p)
        if agg == "last":
            return float(scores[-1])
        if agg == "terminal":
            return float(scores[-1])

        raise ValueError(f"Unknown prm_agg: {self.prm_agg}")

    def _compute_reward(self, full_path: List[int], rollout_id: int) -> Tuple[float, RewardResult]:
        leaf_id = full_path[-1]
        leaf_text = self.nodes[leaf_id].prefix_text
        mode = getattr(self.reward_scorer, "mode", "none")

        td: TerminalDecision = self._terminal_decision(leaf_id)
        term_name = getattr(self.terminal_checker, "name", "terminal_checker")
        term_reason = td.reason
        term_info = td.info

        if mode == "prm":
            # True PRM path scoring: step-wise + terminal
            step_rewards: List[float] = []
            step_raw_scores: List[float] = []

            # steps: full_path[1:-1] (exclude root, exclude leaf)
            for step_idx, nid in enumerate(full_path[1:-1], start=1):
                partial_text = self.nodes[nid].prefix_text
                step_res = self.reward_scorer.score_step(self.root_text, partial_text, step_idx)
                step_rewards.append(float(step_res.reward))
                if step_res.raw_score is not None:
                    step_raw_scores.append(float(step_res.raw_score))

            terminal_res = self.reward_scorer.score_terminal(self.root_text, leaf_text)

            # append terminal score as last element
            all_scores = list(step_rewards) + [float(terminal_res.reward)]
            agg_reward = self._aggregate_prm_scores(all_scores)

            raw_parts = list(step_raw_scores)
            if terminal_res.raw_score is not None:
                raw_parts.append(float(terminal_res.raw_score))
            raw_score = (sum(raw_parts) / len(raw_parts)) if raw_parts else None

            result = RewardResult(
                reward=float(agg_reward),
                mode="prm",
                raw_score=raw_score,
                extra={
                    **(terminal_res.extra or {}),
                    "prm_step_count": len(step_rewards),
                    "prm_terminal_reward": float(terminal_res.reward),
                    "prm_step_rewards": step_rewards,
                    "prm_agg": self.prm_agg,
                },
            )
        else:
            # none / orm
            result = self.reward_scorer.score_terminal(self.root_text, leaf_text)

        self._log_event(
            rollout_id=rollout_id,
            node_id=leaf_id,
            purpose="reward_model",
            meta={
                "rm_mode": result.mode,
                "reward": float(result.reward),
                "raw_score": result.raw_score,
                "term_name": term_name,
                "term_reason": term_reason,
                "term_info": term_info,
                **(result.extra or {}),
            },
        )
        return float(result.reward), result

    # --------------------------
    # MCTS phases
    # --------------------------
    def do_rollout(self, root_id: int, rollout_id: int) -> int:
        path_1 = self._select(root_id, rollout_id)
        leaf = path_1[-1]

        self._log_event(
            rollout_id,
            leaf,
            "select",
            {
                "selected_leaf_id": leaf,
                "selected_path": path_1,
                "selected_depth": self.nodes[leaf].depth,
            },
        )

        self._expand(leaf, rollout_id)
        path_2 = self._simulate(leaf, rollout_id)

        full_path = path_1 + path_2
        terminal_leaf = full_path[-1]

        reward, reward_res = self._compute_reward(full_path, rollout_id)

        td = self._terminal_decision(terminal_leaf)
        self._log_event(
            rollout_id,
            terminal_leaf,
            "rollout_summary",
            {
                "reward": float(reward),
                "rm_mode": reward_res.mode,
                "rm_raw_score": reward_res.raw_score,
                "leaf_depth": self.nodes[terminal_leaf].depth,
                "path_len": len(full_path),
                "term_name": getattr(self.terminal_checker, "name", "terminal_checker"),
                "term_reason": td.reason,
                "term_info": td.info,
            },
        )

        self._backpropagate(full_path, reward)
        return terminal_leaf

    def _select(self, node_id: int, rollout_id: int) -> List[int]:
        path = []
        cur = node_id
        while True:
            path.append(cur)
            if cur not in self.parent2children:
                return path
            children = self.parent2children[cur]
            unexplored = [c for c in children if c not in self.explored_nodes]
            if unexplored:
                nxt = random.choice(unexplored)
                path.append(nxt)
                return path
            cur = self._uct_select(cur, rollout_id)

    def _expand(self, node_id: int, rollout_id: int):
        if node_id in self.explored_nodes:
            return
        if self._is_terminal(node_id):
            self.explored_nodes.add(node_id)
            return
        need = self.branch_factor - len(self.parent2children.get(node_id, []))
        for _ in range(max(0, need)):
            self._llm_call_create_child(rollout_id, node_id, "expand")

    def _simulate(self, node_id: int, rollout_id: int) -> List[int]:
        path = []
        cur = node_id
        while True:
            if self._is_terminal(cur):
                return path
            if cur not in self.parent2children or not self.parent2children[cur]:
                need = self.branch_factor - len(self.parent2children.get(cur, []))
                for _ in range(max(0, need)):
                    self._llm_call_create_child(rollout_id, cur, "simulate")
            nxt = random.choice(self.parent2children[cur])
            path.append(nxt)
            cur = nxt

    def _backpropagate(self, path: List[int], reward: float):
        cur_reward = reward
        for nid in reversed(path):
            self.Q[nid] = self.Q.get(nid, 0.0) + cur_reward
            self.N[nid] = self.N.get(nid, 0) + 1
            self.explored_nodes.add(nid)
            cur_reward *= self.discount

    # --------------------------
    # Public API
    # --------------------------
    def run(self, root_text: str, root_tokens_len: int, num_rollouts: int) -> int:
        self.root_text = root_text
        self.num_rollouts = num_rollouts
        root = self.init_root(root_text, root_tokens_len)

        for r in range(num_rollouts):
            self.do_rollout(root, r)

        terminals = [nid for nid in self.nodes if self._is_terminal(nid) and self.N.get(nid, 0) > 0]
        if not terminals:
            terminals = [root]

        def value(nid: int) -> float:
            return self.Q.get(nid, 0.0) / max(1, self.N.get(nid, 1))

        best_leaf = max(terminals, key=value)
        best_value = value(best_leaf)

        best_path = []
        cur = best_leaf
        while cur is not None:
            best_path.append(cur)
            cur = self.nodes[cur].parent_id
        best_path.reverse()

        td = self._terminal_decision(best_leaf)
        self.logger.log_call(
            CallRecord(
                call_id=self._next_call_id(),
                rollout_id=-1,
                node_id=best_leaf,
                parent_node_id=self.nodes[best_leaf].parent_id,
                purpose="final",
                meta={
                    "best_leaf_id": best_leaf,
                    "best_path": best_path,
                    "best_value": best_value,
                    "num_nodes": len(self.nodes),
                    "num_rollouts": num_rollouts,
                    "term_name": getattr(self.terminal_checker, "name", "terminal_checker"),
                    "term_reason": td.reason,
                    "term_info": td.info,
                },
            )
        )
        return best_leaf