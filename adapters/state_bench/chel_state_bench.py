"""Chel on the STATE-Bench Agent Learning Track.

Build (offline, train trajectories only):

    python -m adapters.state_bench.chel_state_bench build \
        --train-dir STATE-Bench/datasets/train_task_trajectories/travel \
        --workspace learnings/chel/travel --model gpt-6-luna

  1. Every tool call in every train trajectory becomes one H transition:
       obs      = facts the agent had already seen in that episode (flattened
                  scalar fields of earlier tool results, latest value wins) plus
                  this call's arguments as `arg_<name>`;
       action   = {"kind": <tool name>};
       next_obs = the flattened scalar fields of the tool result.
  2. Model-authored rule induction (chel/learn.py) proposes rules per tool; the
     ledger accepts only rules with zero counterexamples across all of H.

Retrieve (inference time, read-only): mix `ChelLearnings` into a STATE-Bench
`StateBenchAgent` subclass.  `retrieve_learnings(query, top_k)` returns the
best-matching verified rules with their support counts.

Imports nothing from STATE-Bench.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "chel"))

from ledger import FORBIDDEN_FRAGMENTS, MAX_FIELDS, MAX_STR, Ledger, LedgerError, is_num  # noqa: E402

CONTEXT_FIELDS = 200


def field_name(path):
    name = re.sub(r"[^a-z0-9_]+", "_", path.casefold()).strip("_")
    if not name or not name[0].isalpha():
        name = "f_" + name
    return name[:64].rstrip("_")


def flatten(value, prefix=""):
    """Flatten nested JSON to scalar fields; lists become `n_<name>` counts."""
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}_{key}" if prefix else str(key)))
    elif isinstance(value, list):
        if prefix:
            out[field_name("n_" + prefix)] = len(value)
    elif prefix:
        name = field_name(prefix)
        if any(fragment in name for fragment in FORBIDDEN_FRAGMENTS):
            return {}
        if value is None or type(value) is bool or (type(value) is int and is_num(value)):
            out[name] = value
        elif type(value) is float and is_num(value):
            out[name] = int(value) if value.is_integer() else value
        elif isinstance(value, str):
            out[name] = value[:MAX_STR]
    return out


def trajectory_transitions(conversation):
    context, out = {}, []
    for message in conversation:
        for call in (message.get("tool_calls") or []) if message.get("role") == "assistant" else []:
            args = {field_name("arg_" + k): v for k, v in flatten(call.get("arguments") or {}).items()}
            obs = dict(list(context.items())[-CONTEXT_FIELDS:])
            obs.update(args)
            if len(obs) > MAX_FIELDS:
                obs = dict(list(obs.items())[-MAX_FIELDS:])
            result = call.get("result")
            next_obs = flatten(result if isinstance(result, dict) else {"result": result})
            if len(next_obs) > MAX_FIELDS:
                next_obs = dict(list(next_obs.items())[:MAX_FIELDS])
            out.append({"obs": obs, "action": {"kind": field_name(str(call.get("name") or "unknown"))}, "next_obs": next_obs})
            for key, value in next_obs.items():
                context.pop(key, None)
                context[key] = value
    return out


def build(train_dir, workspace, complete, *, rounds=3, min_support=2, log=print, limit=None):
    from learn import induce

    train_dir, workspace = pathlib.Path(train_dir), pathlib.Path(workspace)
    files = sorted(train_dir.glob("*.json"))[: limit or None]
    workspace.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.create(workspace, min_support=min_support)
    transitions = []
    for path in files:
        transitions.extend(trajectory_transitions(json.loads(path.read_text()).get("conversation") or []))
    ledger.record(transitions)
    log(f"recorded {len(transitions)} transitions from {len(files)} trajectories")
    trace = induce(ledger, complete, rounds=rounds, log=log)
    summary = {"trajectories": len(files), "transitions": len(transitions), "status": ledger.load().status(), "induction": trace}
    (workspace / "build-trace.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def render(rule):
    return (f"{rule['text']} (Chel rule {rule['id']}: verified on {rule['support']} past tool results with no exceptions)")


class ChelLearnings:
    """Mixin for a STATE-Bench StateBenchAgent subclass (read-only retrieval).

    The ledger for a domain is ``$CHEL_STATE_BENCH_ROOT/<domain>`` unless the
    subclass sets ``chel_root``.
    """

    chel_root = None
    _cache = {}

    def _chel_ledger(self):
        domain = getattr(getattr(self, "runtime_context", None), "domain", None) or os.environ.get("STATE_BENCH_DOMAIN")
        workspace = pathlib.Path(self.chel_root or os.environ["CHEL_STATE_BENCH_ROOT"]) / domain
        key = str(workspace)
        if key not in ChelLearnings._cache:
            ChelLearnings._cache[key] = Ledger(workspace).load()
        return ChelLearnings._cache[key]

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        try:
            return [render(rule) for rule in self._chel_ledger().recall(query[:2000], top_k)]
        except LedgerError:
            return []


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build Chel learnings from STATE-Bench train trajectories")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--train-dir", required=True)
    b.add_argument("--workspace", required=True)
    b.add_argument("--model", default="gpt-6-luna")
    b.add_argument("--reasoning-effort", default="medium")
    b.add_argument("--rounds", type=int, default=3)
    b.add_argument("--limit", type=int, default=None)
    r = sub.add_parser("recall")
    r.add_argument("--workspace", required=True)
    r.add_argument("query")
    r.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args(argv)
    if args.command == "build":
        from llm import openai_completer
        complete = openai_completer(args.model, reasoning_effort=args.reasoning_effort)
        summary = build(args.train_dir, args.workspace, complete, rounds=args.rounds, limit=args.limit)
        summary["model"], summary["model_usage"] = args.model, complete.usage.as_dict()
        (pathlib.Path(args.workspace) / "build-trace.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "induction"}, indent=2))
    else:
        for rule in Ledger(args.workspace).load().recall(args.query, args.top_k):
            print(render(rule))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
