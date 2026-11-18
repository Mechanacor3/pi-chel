#!/usr/bin/env python3
"""Agent-defined verbs: policy, sandbox, example gate, rule use, and safe redefinition."""
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "chel"))
import ledger as lg  # noqa: E402
from verbs import VerbError, VerbRuntime, check_source  # noqa: E402

DAYS = {"name": "days_between", "params": ["start", "end"], "doc": "Whole days from start to end (ISO dates or datetimes).",
        "source": "def days_between(start, end):\n    return (date.fromisoformat(end[:10]) - date.fromisoformat(start[:10])).days\n",
        "examples": [{"args": ["2026-06-01", "2026-06-08"], "result": 7}, {"args": ["2026-06-01T10:00:00", "2026-06-01T23:00:00"], "result": 0}]}


def change(booked, departs, fee):
    return {"obs": {"booked_at": booked, "departure_time": departs}, "action": {"kind": "change"}, "next_obs": {"change_fee": fee}}


H = [change("2026-06-01T09:00:00", "2026-06-20T10:00:00", 0), change("2026-06-01T09:00:00", "2026-06-05T10:00:00", 75),
     change("2026-05-01T09:00:00", "2026-06-30T10:00:00", 0), change("2026-06-10T09:00:00", "2026-06-12T10:00:00", 75)]
RULE = {"id": "advance-fee", "text": "Changes booked 7+ days before departure are free; otherwise $75.",
        "when": {"action.kind": "change"},
        "then": {"change_fee": {"expr": {"op": "if", "args": [
            {"op": "ge", "args": [{"op": "call", "verb": "days_between", "args": [{"ref": "obs.booked_at"}, {"ref": "obs.departure_time"}]}, {"const": 7}]},
            {"const": 0}, {"const": 75}]}}}}


class PolicyAndSandbox(unittest.TestCase):
    def test_escape_attempts_are_rejected_statically(self):
        for source in ("def f(x):\n    import os\n    return 1\n", "def f(x):\n    return x.__class__\n",
                       "def f(x):\n    return open('a')\n", "def f(x):\n    return (lambda: 1)()\n",
                       "def f(x):\n    while True:\n        pass\n", "def f(x):\n    return getattr(x, 'y')\n",
                       "def f(x):\n    return 2 ** 64\n", "def f(x):\n    global y\n    return 1\n", "def g(x):\n    return 1\n"):
            with self.assertRaises(VerbError, msg=source):
                check_source("f", ["x"], source)

    def test_runaway_and_bad_return_values_fail_safely(self):
        verbs = {"spin": {"params": ["x"], "source": "def spin(x):\n    n = 0\n    for i in range(1000000000):\n        n += i\n    return n\n"},
                 "listy": {"params": ["x"], "source": "def listy(x):\n    return [x]\n"}}
        with VerbRuntime(verbs) as rt:
            with self.assertRaisesRegex(VerbError, "CPU time"):
                rt.call("spin", [1])
            with self.assertRaisesRegex(VerbError, "must return"):
                rt.call("listy", [1])
            self.assertEqual(rt.calls, 2)


class LedgerVerbs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = lg.Ledger.create(pathlib.Path(self.tmp.name))
        self.ledger.record(H)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rule_needs_the_verb_then_passes_the_full_gate(self):
        self.assertIn("unknown verb", self.ledger.propose({"op": "add", "rule": RULE})["reasons"][0])
        self.assertTrue(self.ledger.propose({"op": "define_verb", "verb": DAYS})["accepted"])
        result = self.ledger.propose({"op": "add", "rule": RULE})
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["support"], 4)
        reloaded = lg.Ledger(pathlib.Path(self.tmp.name)).load()
        self.assertEqual(sorted(reloaded.verbs), ["days_between"])
        self.assertTrue((reloaded.dir / "verbs" / "days_between.py").exists())
        wrong = {**RULE, "id": "five-day", "then": {"change_fee": {"expr": {"op": "if", "args": [
            {"op": "ge", "args": [{"op": "call", "verb": "days_between", "args": [{"ref": "obs.booked_at"}, {"ref": "obs.departure_time"}]}, {"const": 3}]},
            {"const": 0}, {"const": 75}]}}}}
        self.assertEqual(self.ledger.propose({"op": "add", "rule": wrong})["counterexamples"][0]["index"], 1)
        self.assertEqual(reloaded.recall("advance days change fee", 1)[0]["id"], "advance-fee")

    def test_examples_gate_and_safe_redefinition(self):
        bad = {**DAYS, "examples": [{"args": ["2026-06-01", "2026-06-08"], "result": 8}, DAYS["examples"][1]]}
        out = self.ledger.propose({"op": "define_verb", "verb": bad})
        self.assertFalse(out["accepted"])
        self.assertIn("expected 8", out["reasons"][0])
        self.assertFalse(self.ledger.propose({"op": "define_verb", "verb": {**DAYS, "examples": DAYS["examples"][:1]}})["accepted"])
        self.ledger.propose({"op": "define_verb", "verb": DAYS})
        self.ledger.propose({"op": "add", "rule": RULE})
        always_zero = {**DAYS, "source": DAYS["source"].replace(".days\n", ".days * 0\n"),
                       "examples": [{"args": ["2026-06-01", "2026-06-08"], "result": 0}, {"args": ["2026-06-01", "2026-06-05"], "result": 0}]}
        out = self.ledger.propose({"op": "define_verb", "verb": always_zero})
        self.assertFalse(out["accepted"])
        self.assertIn("would break active rule advance-fee", out["reasons"][0])
        kinds = [(r["kind"], r["accepted"]) for r in lg.Ledger(pathlib.Path(self.tmp.name)).load().revisions if r["kind"] == "verb"]
        self.assertEqual(kinds, [("verb", False), ("verb", False), ("verb", True), ("verb", False)])

    def test_new_history_rechecks_verb_rules(self):
        self.ledger.propose({"op": "define_verb", "verb": DAYS})
        self.ledger.propose({"op": "add", "rule": RULE})
        out = self.ledger.record([change("2026-06-01T00:00:00", "2026-06-03T00:00:00", 100)])
        self.assertEqual(out["falsified"], ["advance-fee"])


if __name__ == "__main__":
    unittest.main()
