#!/usr/bin/env python3
"""Memento3-style executable memory / world model for one agent attempt.

Design follows Memento 3 (Zhao et al., arXiv:2610.11794, https://arxiv.org/abs/2610.11794);
the deterministic faithfulness check, full-history replay and error routing are Chel's own.

Notation (both forms are used deliberately so text search works either way):

  H            episodic, append-only, subject-visible interaction history:
               (obs, action, next_obs) transitions the subject itself recorded.
  ψ (psi)      human-readable semantic rulebook: the subject's current beliefs,
               each rule carrying plain-language text plus a small declarative
               when/then clause that gives it reference semantics.
  θ (theta)    executable world model that mechanizes ψ (psi).  It is written in
               a bounded JSON expression language interpreted here, never as
               arbitrary generated Python, so it terminates, has no I/O, and is
               auditable.
  φ (phi)      frozen model parameters.  Nothing in this module trains weights.
  Φ (Phi)      a future population of competing (ψ, θ) hypotheses.  Not
               implemented; `Hypothesis` and `behavior_signature` are the
               single-model seams such a population would group/sample by.

Acceptance gate (identical for every candidate kind, including simplification):

  1. θ (theta) is semantically faithful to ψ (psi) according to a
     `FaithfulnessVerifier` (default: structural traceability + differential
     agreement on a bounded deterministic probe set derived only from H and ψ).
  2. Replay of the COMPLETE history H reproduces every recorded next
     observation exactly under both ψ (psi) and θ (theta).  A candidate that
     explains the newest transition by breaking an older one is rejected.

On a mismatch the route is explicit:
  BELIEF_ERROR          ψ (psi) predicts the wrong observation -> revise rulebook.
  IMPLEMENTATION_ERROR  ψ (psi) is right but θ (theta) diverges from it -> repair code.

Isolation: all state lives in `<attempt workspace>/memento/`.  The module reads
no evaluator data, authoritative world files, server internals, Docker state,
credentials, environment variables, network, or clock.  Field names that look
like privileged channels are rejected as a tripwire; the trusted adapter that
produces transitions must still supply only ordinary HUD/gameplay-visible data.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import sys


SCHEMA = 1
MEMENTO_DIR = "memento"
FIELD = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
RULE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# Tripwire only: privileged-looking channel names never enter H, ψ (psi), or θ (theta).
FORBIDDEN_FRAGMENTS = (
    "evaluator", "scorer", "score", "authoritative", "server", "docker", "credential",
    "secret", "token", "password", "hidden", "solution", "world_file", "chunk", "boundary",
)
INT_LIMIT = 2**31 - 1
MAX_STR = 256
MAX_FIELDS = 64
MAX_HISTORY = 100_000
MAX_RULES = 256
MAX_LIST = 64
MAX_TEXT = 2000
MAX_NODES = 4096
MAX_DEPTH = 32
DEFAULT_PROBE_LIMIT = 2048
MAX_REPORTED = 20
GENESIS = hashlib.sha256(b"fishbowl-memento-H-v1").hexdigest()

BELIEF_ERROR = "BELIEF_ERROR"
IMPLEMENTATION_ERROR = "IMPLEMENTATION_ERROR"
CONSISTENT = "CONSISTENT"
KINDS = ("theta_repair", "psi_revision", "simplification")
COMPARATORS = ("ne", "lt", "le", "gt", "ge")
OPS = {  # name -> exact arity, or negative minimum arity
    "eq": 2, "ne": 2, "lt": 2, "le": 2, "gt": 2, "ge": 2,
    "and": -1, "or": -1, "not": 1, "in": -2, "add": 2, "sub": 2, "if": 3,
}
UNDEFINED = {"__undefined__": True}  # cannot collide: FIELD forbids leading "_"
EMPTY_PSI = {"schema": SCHEMA, "rules": []}
EMPTY_THETA = {"schema": SCHEMA, "cases": []}


class MementoError(ValueError):
    pass


def fail(message):
    raise MementoError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def same(a, b):
    """Strict equality: True never equals 1, "1" never equals 1."""
    return type(a) is type(b) and a == b


def is_int(value):
    return type(value) is int


def scalar_ok(value):
    if value is None or type(value) is bool:
        return True
    if is_int(value):
        return abs(value) <= INT_LIMIT
    return type(value) is str and len(value) <= MAX_STR


def check_field(name, where):
    if not isinstance(name, str) or not FIELD.fullmatch(name):
        fail(f"{where}: invalid field name {name!r}")
    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment in name:
            fail(f"{where}: field {name!r} looks like a privileged channel ({fragment}); only subject-visible data is allowed")


def check_ref(ref, where):
    if not isinstance(ref, str) or "." not in ref:
        fail(f"{where}: reference must be obs.<field> or action.<field>, got {ref!r}")
    scope, field = ref.split(".", 1)
    if scope not in ("obs", "action"):
        fail(f"{where}: reference scope must be obs or action, got {ref!r}")
    check_field(field, where)


def exact_keys(value, keys, where):
    if not isinstance(value, dict) or set(value) != set(keys):
        fail(f"{where}: expected object with exactly keys {sorted(keys)}")


# --------------------------------------------------------------------------- H

def validate_state(state, where):
    if not isinstance(state, dict) or len(state) > MAX_FIELDS:
        fail(f"{where}: must be an object with at most {MAX_FIELDS} fields")
    for key, value in state.items():
        check_field(key, where)
        if not scalar_ok(value):
            fail(f"{where}.{key}: values must be null, bool, bounded int, or short string")
    return dict(state)


def validate_transition(transition):
    """One H entry: what the subject saw, what it did, and what it saw next."""
    exact_keys(transition, ("obs", "action", "next_obs"), "transition")
    return {key: validate_state(transition[key], f"transition.{key}") for key in ("obs", "action", "next_obs")}


def chain_digest(previous, transition):
    return hashlib.sha256((previous + canonical(transition)).encode()).hexdigest()


# --------------------------------------------------------------- ψ (psi) rulebook

def validate_psi(psi):
    exact_keys(psi, ("schema", "rules"), "ψ (psi)")
    if psi["schema"] != SCHEMA or not isinstance(psi["rules"], list) or len(psi["rules"]) > MAX_RULES:
        fail("ψ (psi): unsupported schema or too many rules")
    seen = set()
    for index, rule in enumerate(psi["rules"]):
        where = f"ψ (psi) rule {index}"
        exact_keys(rule, ("id", "text", "when", "then"), where)
        if not isinstance(rule["id"], str) or not RULE_ID.fullmatch(rule["id"]) or rule["id"] in seen:
            fail(f"{where}: invalid or duplicate id {rule['id']!r}")
        seen.add(rule["id"])
        if not isinstance(rule["text"], str) or not rule["text"].strip() or len(rule["text"]) > MAX_TEXT:
            fail(f"{where}: text must be a non-empty human-readable sentence")
        if not isinstance(rule["when"], dict) or not isinstance(rule["then"], dict):
            fail(f"{where}: when/then must be objects")
        for ref, cond in rule["when"].items():
            check_ref(ref, where)
            if isinstance(cond, list):
                if not cond or len(cond) > MAX_LIST or not all(scalar_ok(v) for v in cond):
                    fail(f"{where}: membership list must hold 1..{MAX_LIST} scalars")
            elif isinstance(cond, dict):
                if len(cond) != 1 or next(iter(cond)) not in COMPARATORS:
                    fail(f"{where}: comparator must be one of {COMPARATORS}")
                op, operand = next(iter(cond.items()))
                if not scalar_ok(operand) or (op != "ne" and not is_int(operand)):
                    fail(f"{where}: invalid comparator operand")
            elif not scalar_ok(cond):
                fail(f"{where}: invalid condition literal")
        for field, effect in rule["then"].items():
            check_field(field, where)
            if not isinstance(effect, dict) or len(effect) != 1:
                fail(f"{where}: effect must be one of set/delta/copy")
            kind, arg = next(iter(effect.items()))
            if kind == "set" and scalar_ok(arg):
                continue
            if kind == "delta" and is_int(arg) and abs(arg) <= INT_LIMIT:
                continue
            if kind == "copy":
                check_ref(arg, where)
                continue
            fail(f"{where}: invalid effect {effect!r}")
    return psi


MISSING = object()


def lookup(ref, obs, action):
    scope, field = ref.split(".", 1)
    return (obs if scope == "obs" else action).get(field, MISSING)


def compare(op, a, b):
    if a is MISSING or b is MISSING:
        return False
    if op == "eq":
        return same(a, b)
    if op == "ne":
        return not same(a, b)
    if not (is_int(a) and is_int(b)):
        return False
    return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[op]


def condition_holds(cond, value):
    if value is MISSING:
        return False
    if isinstance(cond, list):
        return any(same(value, option) for option in cond)
    if isinstance(cond, dict):
        op, operand = next(iter(cond.items()))
        return compare(op, value, operand)
    return same(value, cond)


def psi_predict(psi, obs, action):
    """Reference semantics of ψ (psi): first matching rule wins; unmentioned fields persist."""
    for rule in psi["rules"]:
        if all(condition_holds(c, lookup(r, obs, action)) for r, c in rule["when"].items()):
            nxt = dict(obs)
            for field, effect in sorted(rule["then"].items()):
                kind, arg = next(iter(effect.items()))
                if kind == "set":
                    nxt[field] = arg
                elif kind == "copy":
                    value = lookup(arg, obs, action)
                    if value is MISSING:
                        return UNDEFINED, rule["id"]
                    nxt[field] = value
                else:
                    current = obs.get(field, MISSING)
                    if not is_int(current) or abs(current + arg) > INT_LIMIT:
                        return UNDEFINED, rule["id"]
                    nxt[field] = current + arg
            return nxt, rule["id"]
    return dict(obs), None


# ------------------------------------------------------- θ (theta) executable model

def validate_expr(expr, depth, counter, where):
    counter[0] += 1
    if depth > MAX_DEPTH or counter[0] > MAX_NODES:
        fail(f"{where}: θ (theta) expression exceeds depth/size bound")
    if not isinstance(expr, dict):
        fail(f"{where}: expression must be an object")
    if set(expr) == {"const"}:
        if not scalar_ok(expr["const"]):
            fail(f"{where}: invalid constant")
    elif set(expr) in ({"obs"}, {"action"}):
        check_field(next(iter(expr.values())), where)
    elif set(expr) == {"op", "args"}:
        arity = OPS.get(expr["op"]) if isinstance(expr["op"], str) else None
        args = expr["args"]
        if arity is None or not isinstance(args, list):
            fail(f"{where}: unknown operator {expr.get('op')!r}")
        if (arity >= 0 and len(args) != arity) or (arity < 0 and len(args) < -arity):
            fail(f"{where}: wrong arity for {expr['op']}")
        for arg in args:
            validate_expr(arg, depth + 1, counter, where)
    else:
        fail(f"{where}: expression must be const, obs, action, or op/args")


def validate_theta(theta):
    exact_keys(theta, ("schema", "cases"), "θ (theta)")
    if theta["schema"] != SCHEMA or not isinstance(theta["cases"], list) or len(theta["cases"]) > MAX_RULES:
        fail("θ (theta): unsupported schema or too many cases")
    counter = [0]
    for index, case in enumerate(theta["cases"]):
        where = f"θ (theta) case {index}"
        exact_keys(case, ("implements", "guard", "set"), where)
        if not isinstance(case["implements"], str) or not RULE_ID.fullmatch(case["implements"]):
            fail(f"{where}: implements must name a ψ (psi) rule id")
        validate_expr(case["guard"], 0, counter, where)
        if not isinstance(case["set"], dict):
            fail(f"{where}: set must be an object")
        for field, expr in case["set"].items():
            check_field(field, where)
            validate_expr(expr, 0, counter, where)
    return theta


class _Undefined(Exception):
    pass


def evaluate(expr, obs, action):
    if "const" in expr:
        return expr["const"]
    if "obs" in expr:
        return obs.get(expr["obs"], MISSING)
    if "action" in expr:
        return action.get(expr["action"], MISSING)
    op, args = expr["op"], expr["args"]
    if op == "and":
        return all(evaluate(a, obs, action) is True for a in args)
    if op == "or":
        return any(evaluate(a, obs, action) is True for a in args)
    if op == "not":
        return evaluate(args[0], obs, action) is not True
    if op == "if":
        return evaluate(args[1] if evaluate(args[0], obs, action) is True else args[2], obs, action)
    values = [evaluate(a, obs, action) for a in args]
    if op == "in":
        return values[0] is not MISSING and any(same(values[0], v) for v in values[1:])
    if op in ("add", "sub"):
        a, b = values
        if not (is_int(a) and is_int(b)):
            raise _Undefined()
        result = a + b if op == "add" else a - b
        if abs(result) > INT_LIMIT:
            raise _Undefined()
        return result
    return compare(op, *values)


def theta_predict(theta, obs, action):
    """Execute θ (theta): first case whose guard is exactly True fires; other fields persist."""
    fired = None
    try:
        for case in theta["cases"]:
            if evaluate(case["guard"], obs, action) is True:
                fired = case["implements"]
                nxt = dict(obs)
                for field in sorted(case["set"]):
                    value = evaluate(case["set"][field], obs, action)
                    if value is MISSING or not scalar_ok(value):
                        raise _Undefined()
                    nxt[field] = value
                return nxt, fired
        return dict(obs), None
    except _Undefined:
        return UNDEFINED, fired


# ------------------------------------------------------------- verification gate

class Hypothesis:
    """One (ψ, θ) pair.  A future population Φ (Phi) would hold many of these."""

    __slots__ = ("psi", "theta")

    def __init__(self, psi, theta):
        self.psi = validate_psi(psi)
        self.theta = validate_theta(theta)

    def key(self):
        return digest([self.psi, self.theta])

    def predict(self, obs, action):
        return theta_predict(self.theta, obs, action)[0]


def probe_states(history, psi, limit=DEFAULT_PROBE_LIMIT):
    """Deterministic bounded probe inputs built only from H and ψ (psi) literals."""
    domains = {"obs": {}, "action": {}}

    def add(scope, field, value):
        bucket = domains[scope].setdefault(field, {})
        bucket[canonical(value)] = value
        if is_int(value):
            for neighbor in (value - 1, value + 1):
                if abs(neighbor) <= INT_LIMIT:
                    bucket[canonical(neighbor)] = neighbor

    for t in history:
        for scope, key in (("obs", "obs"), ("obs", "next_obs"), ("action", "action")):
            for field, value in t[key].items():
                add(scope, field, value)
    for rule in psi["rules"]:
        for ref, cond in rule["when"].items():
            scope, field = ref.split(".", 1)
            if isinstance(cond, list):
                values = cond
            elif isinstance(cond, dict):
                values = list(cond.values())
            else:
                values = [cond]
            for value in values:
                add(scope, field, value)
        for field, effect in rule["then"].items():
            if "set" in effect:
                add("obs", field, effect["set"])
    base = [(t["obs"], t["action"]) for t in history] or [({}, {})]
    probes, seen = [], set()

    def push(obs, action):
        key = canonical([obs, action])
        if key not in seen and len(probes) < limit:
            seen.add(key)
            probes.append((obs, action))

    for obs, action in base:
        push(obs, action)
    for obs, action in base:
        for scope in ("obs", "action"):
            state = obs if scope == "obs" else action
            for field in sorted(domains[scope]):
                variants = [dict(state, **{field: v}) for _, v in sorted(domains[scope][field].items())]
                if field in state:
                    variants.append({k: v for k, v in state.items() if k != field})
                for variant in variants:
                    push(variant, action) if scope == "obs" else push(obs, variant)
        if len(probes) >= limit:
            break
    return probes


class FaithfulnessVerifier:
    """Interface: decide whether θ (theta) faithfully implements ψ (psi).

    Replacements (e.g. a later model-judge) may add checks but must never relax
    the gate: `verify_candidate` always also performs full-history replay.
    """

    def check(self, psi, theta, history):  # pragma: no cover - interface
        raise NotImplementedError


class DifferentialFaithfulnessVerifier(FaithfulnessVerifier):
    """Structural traceability plus differential agreement on bounded probes."""

    def __init__(self, probe_limit=DEFAULT_PROBE_LIMIT):
        self.probe_limit = probe_limit

    def check(self, psi, theta, history):
        failures = []
        rule_ids = [r["id"] for r in psi["rules"]]
        implemented = {c["implements"] for c in theta["cases"]}
        for case_index, case in enumerate(theta["cases"]):
            if case["implements"] not in rule_ids:
                failures.append({"kind": "untraceable_case", "case": case_index, "implements": case["implements"]})
        for rule_id in rule_ids:
            if rule_id not in implemented:
                failures.append({"kind": "unimplemented_rule", "rule": rule_id})
        probes = probe_states(history, psi, self.probe_limit)
        for index, (obs, action) in enumerate(probes):
            psi_next, psi_rule = psi_predict(psi, obs, action)
            theta_next, theta_rule = theta_predict(theta, obs, action)
            if canonical(psi_next) != canonical(theta_next) or psi_rule != theta_rule:
                failures.append({
                    "kind": "probe_divergence", "probe": index, "obs": obs, "action": action,
                    "psi": psi_next, "psi_rule": psi_rule, "theta": theta_next, "theta_rule": theta_rule,
                })
        return {"faithful": not failures, "probes": len(probes), "failures": failures}


def diagnose(psi, theta, transition):
    """Route one transition: CONSISTENT, BELIEF_ERROR (ψ wrong), or IMPLEMENTATION_ERROR (θ ≠ ψ)."""
    expected = canonical(transition["next_obs"])
    psi_next, _ = psi_predict(psi, transition["obs"], transition["action"])
    theta_next, _ = theta_predict(theta, transition["obs"], transition["action"])
    if canonical(psi_next) != expected:
        route = BELIEF_ERROR
    elif canonical(theta_next) != expected:
        route = IMPLEMENTATION_ERROR
    else:
        route = CONSISTENT
    return {"route": route, "expected": transition["next_obs"], "psi_prediction": psi_next, "theta_prediction": theta_next}


def expr_nodes(expr):
    return 1 + sum(expr_nodes(a) for a in expr.get("args", []))


def description_length(psi, theta):
    """MDL-style structural size; human prose is excluded so wording is not 'simplification'."""
    size = sum(1 + len(r["when"]) + len(r["then"]) for r in psi["rules"])
    for case in theta["cases"]:
        size += 1 + expr_nodes(case["guard"]) + sum(1 + expr_nodes(e) for e in case["set"].values())
    return size


def behavior_signature(hypothesis, probes):
    """Hash of predictions on shared probes; a future Φ (Phi) would group by this, not by source."""
    return digest([hypothesis.predict(obs, action) for obs, action in probes])


def history_digest(history):
    value = GENESIS
    for t in history:
        value = chain_digest(value, t)
    return value


def verify_candidate(psi, theta, history, verifier=None):
    """The single acceptance gate: faithfulness AND exact full-history replay."""
    validate_psi(psi)
    validate_theta(theta)
    verifier = verifier or DifferentialFaithfulnessVerifier()
    faith = verifier.check(psi, theta, history)
    replay_failures = []
    for index, transition in enumerate(history):
        diagnosis = diagnose(psi, theta, transition)
        if diagnosis["route"] != CONSISTENT:
            replay_failures.append(dict(diagnosis, t=index))
    routes = {f["route"] for f in replay_failures}
    if not faith["faithful"]:
        routes.add(IMPLEMENTATION_ERROR)
    if BELIEF_ERROR in routes:
        route = "REVISE_PSI"
    elif IMPLEMENTATION_ERROR in routes:
        route = "REPAIR_THETA"
    else:
        route = "ACCEPT"
    reasons = [f"replay t={f['t']} {f['route']}" for f in replay_failures[:MAX_REPORTED]]
    reasons += [f"unfaithful {f['kind']}" + (f" probe={f['probe']}" if "probe" in f else "") for f in faith["failures"][:MAX_REPORTED]]
    return {
        "passed": route == "ACCEPT",
        "route": route,
        "faithful": faith["faithful"],
        "probes": faith["probes"],
        "faithfulness_failure_count": len(faith["failures"]),
        "faithfulness_failures": faith["failures"][:MAX_REPORTED],
        "replay_ok": not replay_failures,
        "replay_failure_count": len(replay_failures),
        "replay_failures": replay_failures[:MAX_REPORTED],
        "reasons": reasons,
        "history_len": len(history),
        "history_digest": history_digest(history),
        "psi_sha256": digest(psi),
        "theta_sha256": digest(theta),
        "description_length": description_length(psi, theta),
    }


def render_psi(psi):
    lines = [
        "# ψ (psi) rulebook",
        "",
        "Generated view of `psi.json`. Rules are checked in order; the first matching rule",
        "determines the next observation. Fields a rule does not mention stay unchanged.",
        "",
    ]
    if not psi["rules"]:
        lines.append("_No rules yet: everything is predicted to stay unchanged._")
    for rule in psi["rules"]:
        lines += [f"## {rule['id']}", "", rule["text"].strip(), ""]
        when = "; ".join(f"`{r}` {canonical(c)}" for r, c in sorted(rule["when"].items())) or "always"
        then = "; ".join(f"`{f}` {canonical(e)}" for f, e in sorted(rule["then"].items())) or "nothing changes"
        lines += [f"- when: {when}", f"- then: {then}", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def pretty(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# --------------------------------------------------------------- workspace store

class MementoMemory:
    """H, ψ (psi), θ (theta), verification, and provenance under `<workspace>/memento/`.

    Authority: `history.jsonl`, `revisions.jsonl`, and content-addressed
    `objects/`.  `psi.json`, `psi.md`, `theta.json`, and `verification.json` are
    deterministic views that `load` re-derives and checks.  No timestamps or host
    identifiers are stored, so identical operation sequences produce identical bytes.
    """

    def __init__(self, workspace, verifier=None):
        self.workspace = pathlib.Path(workspace).resolve(strict=True)
        self.root = self.workspace / MEMENTO_DIR
        self.verifier = verifier or DifferentialFaithfulnessVerifier()
        self.history = []
        self.digests = [GENESIS]
        self.revisions = []
        self.psi = EMPTY_PSI
        self.theta = EMPTY_THETA

    # -- confined I/O --------------------------------------------------------
    def _check_dirs(self):
        for directory in (self.root, self.root / "objects"):
            info = os.lstat(directory)
            if not stat.S_ISDIR(info.st_mode):
                fail(f"{directory} must be a real directory, not a symlink")

    def _path(self, name):
        self._check_dirs()
        path = self.root / name
        if path.parent not in (self.root, self.root / "objects") or ".." in pathlib.PurePath(name).parts:
            fail(f"path escapes memento store: {name}")
        return path

    def _read(self, name):
        fd = os.open(self._path(name), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, encoding="utf-8") as handle:
            return handle.read()

    def _write(self, name, text):
        path = self._path(name)
        temporary = path.with_name(f".{path.name}.tmp")
        if os.path.lexists(temporary):
            os.unlink(temporary)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)

    def _append(self, name, line):
        fd = os.open(self._path(name), os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _put_object(self, value):
        name = f"objects/{digest(value)}.json"
        text = canonical(value) + "\n"
        if os.path.lexists(self._path(name)):
            if self._read(name) != text:
                fail(f"content-addressed object mismatch: {name}")
            return
        self._write(name, text)

    def _get_object(self, sha):
        if not re.fullmatch(r"[0-9a-f]{64}", sha or ""):
            fail("invalid object hash")
        value = json.loads(self._read(f"objects/{sha}.json"))
        if digest(value) != sha:
            fail(f"object {sha} does not match its hash")
        return value

    # -- lifecycle -------------------------------------------------------------
    @classmethod
    def create(cls, workspace, verifier=None):
        memory = cls(workspace, verifier)
        if os.path.lexists(memory.root):
            fail(f"memento store already exists: {memory.root}")
        os.mkdir(memory.root, 0o755)
        os.mkdir(memory.root / "objects", 0o755)
        memory._write("history.jsonl", "")
        memory._write("revisions.jsonl", "")
        result = verify_candidate(EMPTY_PSI, EMPTY_THETA, [], memory.verifier)
        memory._log("init", EMPTY_PSI, EMPTY_THETA, result, [], "empty initial model", True)
        return memory

    @classmethod
    def load(cls, workspace, verifier=None):
        memory = cls(workspace, verifier)
        memory._load()
        return memory

    def _load(self):
        for number, line in enumerate(self._read("history.jsonl").splitlines()):
            transition = validate_transition(json.loads(line))
            if canonical(transition) != line:
                fail(f"history line {number} is not canonical")
            self.history.append(transition)
            self.digests.append(chain_digest(self.digests[-1], transition))
        accepted = None
        for number, line in enumerate(self._read("revisions.jsonl").splitlines()):
            entry = json.loads(line)
            if canonical(entry) != line or entry.get("rev") != number:
                fail(f"revision line {number} is not canonical/sequential")
            expected_prev = digest(self.revisions[-1]) if self.revisions else None
            if entry.get("prev_entry_sha256") != expected_prev:
                fail(f"revision {number} breaks the provenance hash chain")
            length = entry.get("history_len")
            if not is_int(length) or not 0 <= length <= len(self.history) or self.digests[length] != entry.get("history_digest"):
                fail(f"revision {number} history prefix no longer matches H (H must be append-only)")
            psi = validate_psi(self._get_object(entry["psi_sha256"]))
            theta = validate_theta(self._get_object(entry["theta_sha256"]))
            if entry.get("accepted"):
                accepted = (psi, theta)
            self.revisions.append(entry)
        if accepted is None:
            fail("no accepted revision")
        self.psi, self.theta = accepted
        for name, text in self._views().items():
            if self._read(name) != text:
                fail(f"{name} does not match the last accepted revision; propose changes instead of editing views")

    def _views(self):
        return {
            "psi.json": pretty(self.psi),
            "psi.md": render_psi(self.psi),
            "theta.json": pretty(self.theta),
            "verification.json": pretty(self.verify()),
        }

    def _log(self, kind, psi, theta, result, reasons, rationale, accepted):
        self._put_object(psi)
        self._put_object(theta)
        entry = {
            "rev": len(self.revisions),
            "kind": kind,
            "accepted": accepted,
            "base_rev": self.current_rev(),
            "psi_sha256": result["psi_sha256"],
            "theta_sha256": result["theta_sha256"],
            "history_len": result["history_len"],
            "history_digest": result["history_digest"],
            "description_length": result["description_length"],
            "route": result["route"],
            "reasons": reasons,
            "replay_failures": result["replay_failures"],
            "faithfulness_failures": result["faithfulness_failures"],
            "rationale": rationale,
            "prev_entry_sha256": digest(self.revisions[-1]) if self.revisions else None,
        }
        self._append("revisions.jsonl", canonical(entry))
        self.revisions.append(entry)
        if accepted:
            self.psi, self.theta = psi, theta
        for name, text in self._views().items():
            self._write(name, text)
        return entry

    # -- public API ------------------------------------------------------------
    def current_rev(self):
        for entry in reversed(self.revisions):
            if entry["accepted"]:
                return entry["rev"]
        return None

    def hypothesis(self):
        return Hypothesis(self.psi, self.theta)

    def verify(self):
        return verify_candidate(self.psi, self.theta, self.history, self.verifier)

    def record(self, transition):
        """Append one subject-visible transition to H and diagnose the current model on it."""
        transition = validate_transition(transition)
        if len(self.history) >= MAX_HISTORY:
            fail("history bound reached")
        self._append("history.jsonl", canonical(transition))
        self.history.append(transition)
        self.digests.append(chain_digest(self.digests[-1], transition))
        diagnosis = diagnose(self.psi, self.theta, transition)
        result = self.verify()
        self._write("verification.json", pretty(result))
        return dict(diagnosis, t=len(self.history) - 1, model_route=result["route"])

    def propose(self, kind, theta, psi=None, rationale=""):
        """Submit a candidate successor.  Every kind passes the identical gate."""
        if kind not in KINDS:
            fail(f"kind must be one of {KINDS}")
        if not isinstance(rationale, str) or len(rationale) > MAX_TEXT:
            fail("rationale must be a short string")
        reasons = []
        if kind == "theta_repair":
            if psi is not None and digest(psi) != digest(self.psi):
                reasons.append("theta_repair must keep ψ (psi) unchanged; use psi_revision for belief errors")
            psi = self.psi
        elif psi is None:
            fail(f"{kind} requires a ψ (psi) candidate")
        result = verify_candidate(psi, theta, self.history, self.verifier)
        reasons += result["reasons"]
        if kind == "theta_repair":
            if result["route"] == "REVISE_PSI":
                reasons.append("belief error: ψ (psi) mispredicts history, so no θ (theta) repair can be accepted")
            if digest(theta) == digest(self.theta):
                reasons.append("theta_repair must change θ (theta)")
        if kind == "psi_revision" and digest(psi) == digest(self.psi):
            reasons.append("psi_revision must change ψ (psi)")
        if kind == "simplification" and result["description_length"] >= description_length(self.psi, self.theta):
            reasons.append("simplification must strictly reduce description length")
        accepted = not reasons
        entry = self._log(kind, psi, theta, result, reasons, rationale, accepted)
        return {"accepted": accepted, "rev": entry["rev"], "route": result["route"], "reasons": reasons, "result": result}

    def simplification_candidates(self, max_candidates=32):
        """Bounded deterministic pruning: drop one rule with its cases, or one redundant case."""
        out = []
        rules, cases = self.psi["rules"], self.theta["cases"]
        for index, rule in enumerate(rules):
            psi = {"schema": SCHEMA, "rules": rules[:index] + rules[index + 1:]}
            theta = {"schema": SCHEMA, "cases": [c for c in cases if c["implements"] != rule["id"]]}
            out.append((f"drop rule {rule['id']}", psi, theta))
        for index in range(len(cases)):
            out.append((f"drop θ (theta) case {index}", self.psi, {"schema": SCHEMA, "cases": cases[:index] + cases[index + 1:]}))
        return out[:max_candidates]

    def auto_simplify(self, max_candidates=32):
        """MDL hook: pick the smallest candidate passing the gate, then submit it through `propose`."""
        current = description_length(self.psi, self.theta)
        best = None
        evaluated = []
        for label, psi, theta in self.simplification_candidates(max_candidates):
            result = verify_candidate(psi, theta, self.history, self.verifier)
            evaluated.append({"candidate": label, "passed": result["passed"], "description_length": result["description_length"]})
            if result["passed"] and result["description_length"] < current:
                if best is None or result["description_length"] < best[0]:
                    best = (result["description_length"], label, psi, theta)
        if best is None:
            return {"accepted": False, "evaluated": evaluated, "reasons": ["no simpler candidate passed the gate"]}
        outcome = self.propose("simplification", best[3], best[2], f"auto_simplify: {best[1]}")
        return dict(outcome, evaluated=evaluated)


# ----------------------------------------------------------------------- CLI

def read_workspace_json(workspace, name):
    """CLI inputs must already live inside the attempt workspace (no outside reads)."""
    path = (pathlib.Path(workspace).resolve(strict=True) / name).resolve(strict=True)
    if pathlib.Path(workspace).resolve(strict=True) not in path.parents:
        fail("input file must be inside the attempt workspace")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Memento3-style H / ψ (psi) / θ (theta) memory for one attempt workspace")
    ap.add_argument("--workspace", required=True)
    commands = ap.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("status")
    record = commands.add_parser("record")
    record.add_argument("transition", help="workspace-relative JSON file {obs, action, next_obs}")
    propose = commands.add_parser("propose")
    propose.add_argument("kind", choices=KINDS)
    propose.add_argument("candidate", help="workspace-relative JSON file {psi?, theta, rationale}")
    simplify = commands.add_parser("simplify")
    simplify.add_argument("--max-candidates", type=int, default=32)
    args = ap.parse_args(argv)
    try:
        if args.command == "init":
            memory = MementoMemory.create(args.workspace)
            output, ok = {"initialized": True, "rev": memory.current_rev()}, True
        else:
            memory = MementoMemory.load(args.workspace)
            if args.command == "status":
                output = memory.verify()
                ok = output["passed"]
            elif args.command == "record":
                output = memory.record(read_workspace_json(args.workspace, args.transition))
                ok = output["route"] == CONSISTENT
            elif args.command == "propose":
                candidate = read_workspace_json(args.workspace, args.candidate)
                if not isinstance(candidate, dict):
                    fail("candidate must be an object")
                output = memory.propose(args.kind, candidate.get("theta"), candidate.get("psi"), candidate.get("rationale", ""))
                ok = output["accepted"]
            else:
                output = memory.auto_simplify(max(0, min(args.max_candidates, 256)))
                ok = output["accepted"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"MEMENTO_ERROR {exc}", file=sys.stderr)
        return 2
    print(canonical(output))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
