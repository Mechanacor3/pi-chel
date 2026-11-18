#!/usr/bin/env python3
"""Tests for the open-world Chel rule ledger."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "chel"))
import ledger as lg  # noqa: E402


def cancel(cabin, route, price, fee, status="confirmed"):
    return {"obs": {"cabin_class": cabin, "route_type": route, "price_paid": price, "status": status},
            "action": {"kind": "cancel_booking"},
            "next_obs": {"cancellation_fee": fee, "status": "cancelled"}}


H = [cancel("economy", "domestic", 300, 50), cancel("economy", "domestic", 600, 90.0),
     cancel("economy", "international", 500, 100), cancel("business", "international", 2000, 160)]

FEE_RULE = {"id": "R1", "text": "Domestic economy cancellation fee is the larger of $50 and 15% of the fare.",
            "when": {"action.kind": "cancel_booking", "obs.cabin_class": "economy", "obs.route_type": "domestic"},
            "then": {"cancellation_fee": {"expr": {"op": "max", "args": [
                {"const": 50}, {"op": "mul", "args": [{"ref": "obs.price_paid"}, {"const": 0.15}]}]}}}}


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = pathlib.Path(self.tmp.name)
        self.ledger = lg.Ledger.create(self.ws, min_support=2)
        self.ledger.record(H)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rule_with_formula_is_accepted_with_support(self):
        result = self.ledger.propose({"op": "add", "rule": FEE_RULE})
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["support"], 2)
        self.assertEqual(self.ledger.load().active()["R1"]["support"], 2)

    def test_counterexample_rejects_and_is_reported(self):
        rule = {**FEE_RULE, "id": "R2", "when": {"action.kind": "cancel_booking", "obs.cabin_class": "economy"}}
        result = self.ledger.propose({"op": "add", "rule": rule})
        self.assertFalse(result["accepted"])
        self.assertIn("contradicted", result["reasons"][0])
        ce = result["counterexamples"][0]
        self.assertEqual(ce["index"], 2)
        self.assertEqual(ce["mismatch"]["cancellation_fee"]["actual"], 100)
        self.assertEqual(ce["context"]["obs.route_type"], "international")

    def test_insufficient_support_and_vacuous_rules_are_rejected(self):
        rule = {"id": "R3", "text": "Business international fee is 8%.",
                "when": {"obs.cabin_class": "business"},
                "then": {"cancellation_fee": {"expr": {"op": "mul", "args": [{"ref": "obs.price_paid"}, {"const": 0.08}]}}}}
        self.assertIn("at least 2", self.ledger.propose({"op": "add", "rule": rule})["reasons"][0])
        never = {"id": "R4", "text": "Nothing.", "when": {"obs.cabin_class": "first"}, "then": {"status": {"set": "x"}}}
        self.assertFalse(self.ledger.propose({"op": "add", "rule": never})["accepted"])
        empty = {"id": "R5", "text": "No claim.", "when": {}, "then": {}}
        self.assertIn("then must claim", self.ledger.propose({"op": "add", "rule": empty})["reasons"][0])

    def test_duplicate_claims_and_reused_ids_are_rejected(self):
        self.assertTrue(self.ledger.propose({"op": "add", "rule": FEE_RULE})["accepted"])
        again = self.ledger.propose({"op": "add", "rule": {**FEE_RULE, "id": "R9", "text": "Same claim, new words."}})
        self.assertIn("already makes exactly this claim", again["reasons"][-1])
        reused = self.ledger.propose({"op": "add", "rule": {**FEE_RULE, "then": {"status": {"set": "cancelled"}}}})
        self.assertIn("already used", reused["reasons"][0])

    def test_new_history_falsifies_and_support_grows(self):
        self.ledger.propose({"op": "add", "rule": FEE_RULE})
        status_rule = {"id": "S1", "text": "Cancelling sets status to cancelled.", "when": {"action.kind": "cancel_booking"},
                       "then": {"status": {"set": "cancelled"}}}
        self.assertTrue(self.ledger.propose({"op": "add", "rule": status_rule})["accepted"])
        out = self.ledger.record([cancel("economy", "domestic", 1000, 150)])
        self.assertEqual(out["falsified"], [])
        self.assertEqual(self.ledger.load().active()["R1"]["support"], 3)
        out = self.ledger.record([cancel("economy", "domestic", 400, 75)])
        self.assertEqual(out["falsified"], ["R1"])
        reloaded = lg.Ledger(self.ws).load()
        self.assertEqual(reloaded.rules["R1"]["state"], "falsified")
        self.assertEqual(reloaded.rules["R1"]["counterexample"]["index"], 5)
        self.assertEqual(reloaded.rules["S1"]["support"], 6)

    def test_revise_and_retract_keep_provenance(self):
        self.ledger.propose({"op": "add", "rule": FEE_RULE})
        better = {**FEE_RULE, "id": "R1b", "text": "Domestic economy fee: max($50, 15% of fare), any status.",
                  "when": {**FEE_RULE["when"], "obs.status": "confirmed"}}
        self.assertTrue(self.ledger.propose({"op": "revise", "target": "R1", "rule": better})["accepted"])
        self.assertEqual(self.ledger.rules["R1"]["state"], "superseded")
        self.assertTrue(self.ledger.propose({"op": "retract", "target": "R1b", "reason": "too narrow"})["accepted"])
        self.assertFalse(self.ledger.propose({"op": "retract", "target": "R1b", "reason": "again"})["accepted"])
        kinds = [r["kind"] for r in lg.Ledger(self.ws).load().revisions]
        self.assertEqual(kinds, ["add", "revise", "retract", "retract"])

    def test_tampering_with_history_or_revisions_is_detected(self):
        self.ledger.propose({"op": "add", "rule": FEE_RULE})
        path = self.ws / lg.LEDGER_DIR / "history.jsonl"
        lines = path.read_text().splitlines()
        lines[0] = lines[0].replace('"price_paid":300', '"price_paid":301')
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaises(lg.LedgerError):
            lg.Ledger(self.ws).load()

    def test_invalid_expressions_and_privileged_fields_fail_closed(self):
        bad = [
            {**FEE_RULE, "id": "B1", "then": {"cancellation_fee": {"expr": {"op": "pow", "args": []}}}},
            {**FEE_RULE, "id": "B2", "then": {"cancellation_fee": {"expr": {"op": "add", "args": [{"const": 1}]}}}},
            {**FEE_RULE, "id": "B3", "when": {"obs.api_key": "x"}},
            {**FEE_RULE, "id": "B4", "when": {"evil.path": 1}},
        ]
        for rule in bad:
            self.assertFalse(self.ledger.propose({"op": "add", "rule": rule})["accepted"], rule["id"])
        with self.assertRaises(lg.LedgerError):
            self.ledger.record([{"obs": {"secret_value": 1}, "action": {"kind": "x"}, "next_obs": {}}])
        deep = {"const": 1}
        for _ in range(12):
            deep = {"op": "abs", "args": [deep]}
        self.assertFalse(self.ledger.propose({"op": "add", "rule": {**FEE_RULE, "id": "B5", "then": {"x": {"expr": deep}}}})["accepted"])

    def test_undefined_predictions_abstain(self):
        rule = {"id": "D1", "text": "Fee divided by zero is undefined.", "when": {"action.kind": "cancel_booking"},
                "then": {"cancellation_fee": {"expr": {"op": "div", "args": [{"ref": "obs.price_paid"}, {"const": 0}]}}}}
        result = self.ledger.propose({"op": "add", "rule": rule})
        self.assertFalse(result["accepted"])
        self.assertEqual(result["abstain"], 4)

    def test_recall_ranks_by_query_and_views_are_deterministic(self):
        self.ledger.propose({"op": "add", "rule": FEE_RULE})
        self.ledger.propose({"op": "add", "rule": {"id": "S1", "text": "Cancelling sets status to cancelled.",
                                                   "when": {"action.kind": "cancel_booking"}, "then": {"status": {"set": "cancelled"}}}})
        top = self.ledger.recall("what is the fee to cancel a domestic economy fare", 1)
        self.assertEqual(top[0]["id"], "R1")
        before = (self.ws / lg.LEDGER_DIR / "rules.json").read_bytes()
        lg.Ledger(self.ws).load()._views()
        self.assertEqual(before, (self.ws / lg.LEDGER_DIR / "rules.json").read_bytes())

    def test_cli_round_trip(self):
        cli = [sys.executable, "-I", str(ROOT / "chel" / "ledger.py"), "--workspace", str(self.ws)]
        out = subprocess.run([*cli, "propose"], input=json.dumps({"op": "add", "rule": FEE_RULE}),
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        status = json.loads(subprocess.run([*cli, "status"], capture_output=True, text=True).stdout)
        self.assertEqual(status["rules"], {"active": 1})


if __name__ == "__main__":
    unittest.main()
