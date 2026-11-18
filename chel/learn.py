"""Model-authored rule induction over a Chel ledger (offline, model-agnostic).

For each action kind (e.g. a tool name) with enough history, a model is shown a
compact dossier -- which result fields exist, how they vary, and a sample of
transitions -- and proposes rules.  Each rule goes through the unchanged ledger
gate (zero counterexamples across ALL of H, minimum support).  Rejected rules
come back with the gate's reasons and counterexamples, and the model may repair
them for a bounded number of rounds.  Every proposal, accepted or not, is
logged in the ledger's revision chain: that chain is the learning trace.

`complete(system, prompt) -> str` is any callable returning a JSON object as
text; the ledger never calls a model itself.
"""
from __future__ import annotations

import json
from collections import Counter

from ledger import LedgerError

SYSTEM = """You are Chel's offline rule learner. You study recorded transitions of one action
(obs = what was known before, action = what was done, next_obs = what came back) and propose
RULES that predict next_obs fields. A rule is accepted only if it holds on EVERY matching
recorded transition (zero counterexamples) and matches at least {min_support} of them.

Return one JSON object: {"rules": [RULE, ...]} with at most {max_rules} rules.
RULE = {"id": "short-unique-id", "text": "one plain sentence a support agent can act on",
        "when": {...}, "then": {...}}
when: maps "obs.<field>" or "action.<field>" to a literal (equality), a list of literals
      (membership), or one comparator {"ne"|"lt"|"le"|"gt"|"ge": value}. All must hold.
      Always include "action.kind".
then: maps a next_obs field to {"set": literal} | {"copy": "obs.<field>"} |
      {"expr": E}. E = {"const": v} | {"ref": "obs.<field>"} |
      {"op": OP, "args": [E, ...]} with OP in add sub mul div (2 args), min max (2+),
      round (1-2: value, digits), abs (1), eq ne lt le gt ge (2), and or (1+), not (1),
      if (3: condition, then, else). Numbers match within 0.01.
What to learn, in priority order:
1. Policy outcomes: fees, refunds, compensation, prices, point costs, eligibility, and
   which inputs they depend on -- ideally as one formula (e.g. max(50, 0.15 * price)).
2. Behavioural effects that matter for acting: e.g. "with confirm=false the result is a
   preview and nothing is written", status transitions, errors and their conditions.
Do NOT propose: echoes (copying an argument into the same-named result field), fields that
are constant for every row and say nothing about policy, or rules that only restate long
policy text. The "text" must state exactly what the when/then clause checks, in words a
support agent can act on -- never add advice the clause does not verify.
Return only the JSON object."""


def _short(value, limit=60):
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def dossier(history, kind, sample=24, max_fields=40):
    rows = [t for t in history if t["action"].get("kind") == kind]
    result_fields = Counter(f for t in rows for f in t["next_obs"])
    obs_fields = Counter(f for t in rows for f in t["obs"])
    lines = [f"Action kind: {kind}. Recorded transitions: {len(rows)}.", "", "next_obs fields (present in N rows; distinct values):"]
    for field, count in result_fields.most_common(max_fields):
        values = Counter(_short(t["next_obs"][field], 30) for t in rows if field in t["next_obs"])
        lines.append(f"- {field}: {count}; {len(values)} distinct; e.g. {', '.join(v for v, _ in values.most_common(4))}")
    # Show input fields that vary across this action's rows and are usually present.
    varying = []
    for field, count in obs_fields.items():
        if count >= max(2, len(rows) // 3):
            distinct = len({_short(t["obs"].get(field)) for t in rows})
            if distinct > 1:
                varying.append((field.startswith("arg_"), count, field))
    varying.sort(key=lambda x: (not x[0], -x[1], x[2]))
    shown = [f for _, _, f in varying[:max_fields]]
    lines.append("")
    lines.append(f"Sample transitions (showing {len(shown)} varying obs fields and all scalar next_obs fields):")
    step = max(1, len(rows) // sample)
    for t in rows[::step][:sample]:
        o = {f: t["obs"][f] for f in shown if f in t["obs"]}
        n = {f: v for f, v in t["next_obs"].items() if f in dict(result_fields.most_common(max_fields))}
        lines.append(f"obs={json.dumps(o, ensure_ascii=False)} -> next_obs={json.dumps(n, ensure_ascii=False)}")
    return "\n".join(lines)


def induce(ledger, complete, *, kinds=None, rounds=3, max_rules=8, log=print):
    """Run bounded propose/repair rounds per action kind.  Returns a trace list."""
    ledger.load()
    counts = Counter(t["action"].get("kind") for t in ledger.history)
    kinds = kinds or [k for k, n in counts.most_common() if k is not None and n >= ledger.min_support]
    system = SYSTEM.replace("{min_support}", str(ledger.min_support)).replace("{max_rules}", str(max_rules))
    trace = []
    for kind in kinds:
        ledger.load()
        existing = [f"[{rid}] {e['rule']['text']}" for rid, e in ledger.active().items()
                    if e["rule"]["when"].get("action.kind") == kind]
        prompt = dossier(ledger.history, kind) + "\n\nAlready accepted rules for this action:\n" + ("\n".join(existing) or "(none)")
        item = {"kind": kind, "transitions": counts[kind], "rounds": []}
        for round_number in range(rounds):
            try:
                raw = json.loads(complete(system, prompt))
                rules = raw.get("rules") if isinstance(raw, dict) else None
                if not isinstance(rules, list):
                    raise ValueError("response has no rules list")
            except (ValueError, TypeError) as exc:
                item["rounds"].append({"error": str(exc)[:300]})
                prompt += f"\n\nYour last answer was not usable ({exc}). Return {{\"rules\": [...]}}."
                continue
            feedback, accepted = [], 0
            for rule in rules[:max_rules]:
                try:
                    result = ledger.propose({"op": "add", "rule": rule, "rationale": f"induce {kind} round {round_number + 1}"})
                except LedgerError as exc:
                    result = {"accepted": False, "reasons": [str(exc)], "counterexamples": []}
                rid = rule.get("id") if isinstance(rule, dict) else None
                if result["accepted"]:
                    accepted += 1
                    feedback.append(f"[{rid}] ACCEPTED (support {result['support']})")
                else:
                    ce = "; ".join(_short(c, 700) for c in result.get("counterexamples", [])[:2])
                    feedback.append(f"[{rid}] REJECTED: {'; '.join(result['reasons'])[:400]}"
                                    + (f" Counterexamples: {ce}" if ce else ""))
            item["rounds"].append({"proposed": len(rules[:max_rules]), "accepted": accepted})
            log(f"  {kind} round {round_number + 1}: {accepted}/{len(rules[:max_rules])} accepted")
            if all(line.endswith(")") and "ACCEPTED" in line for line in feedback):
                break
            prompt += ("\n\nGate results for your last proposal:\n" + "\n".join(feedback) +
                       "\nRepair rejected rules (add the missing condition, fix the formula, or narrow the claim) "
                       "using NEW ids, or propose different rules. Do not resubmit accepted rules.")
        trace.append(item)
    return trace
