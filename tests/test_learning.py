#!/usr/bin/env python3
"""Rule induction loop and the STATE-Bench adapter, with a fake model."""
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "chel"))
sys.path.insert(0, str(ROOT))
import learn  # noqa: E402
from adapters.state_bench import chel_state_bench as cb  # noqa: E402
from ledger import Ledger  # noqa: E402


def conversation(price, fee):
    return [{"role": "user", "content": "cancel please"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "get_booking", "arguments": {"booking_id": "BK-1"},
                 "result": {"booking_id": "BK-1", "cabin_class": "economy", "price_paid": price, "fare": {"base": 1.5}, "legs": [1, 2]}},
                {"name": "cancel_booking", "arguments": {"booking_id": "BK-1", "confirm": False},
                 "result": {"status": "preview", "cancellation_fee": fee, "api_key": "x"}}]}]


GOOD = {"id": "fee", "text": "Economy cancel fee is max(50, 15% of price paid).",
        "when": {"action.kind": "cancel_booking", "obs.cabin_class": "economy"},
        "then": {"cancellation_fee": {"expr": {"op": "max", "args": [{"const": 50}, {"op": "mul", "args": [{"ref": "obs.price_paid"}, {"const": 0.15}]}]}}}}
FLAT = {**GOOD, "id": "flat", "then": {"cancellation_fee": {"set": 50}}}


class AdapterTest(unittest.TestCase):
    def test_transitions_flatten_context_args_and_drop_privileged_fields(self):
        t = cb.trajectory_transitions(conversation(600, 90.0))
        self.assertEqual([x["action"]["kind"] for x in t], ["get_booking", "cancel_booking"])
        self.assertEqual(t[0]["next_obs"]["fare_base"], 1.5)
        self.assertEqual(t[0]["next_obs"]["n_legs"], 2)
        self.assertEqual(t[1]["obs"]["price_paid"], 600)
        self.assertIs(t[1]["obs"]["arg_confirm"], False)
        self.assertEqual(t[1]["next_obs"]["cancellation_fee"], 90)  # integral floats become ints
        self.assertNotIn("api_key", t[1]["next_obs"])

    def test_build_induces_repairs_and_retrieves(self):
        prompts = []

        def complete(system, prompt):
            prompts.append(prompt)
            if "Action kind: cancel_booking" not in prompt:
                return json.dumps({"rules": []})
            if "REJECTED" not in prompt:
                return json.dumps({"rules": [FLAT]})
            return json.dumps({"rules": [GOOD]})

        with tempfile.TemporaryDirectory() as tmp:
            train = pathlib.Path(tmp) / "travel"
            train.mkdir()
            for i, (price, fee) in enumerate(((300, 50), (600, 90), (1000, 150))):
                (train / f"{i}.json").write_text(json.dumps({"conversation": conversation(price, fee)}))
            summary = cb.build(train, pathlib.Path(tmp) / "out" / "travel", complete, rounds=3, log=lambda *_: None)
            self.assertEqual(summary["transitions"], 6)
            self.assertEqual(summary["status"]["rules"], {"active": 1})
            cancel = [i for i in summary["induction"] if i["kind"] == "cancel_booking"][0]
            self.assertEqual([r["accepted"] for r in cancel["rounds"]], [0, 1])
            self.assertTrue(any("Counterexamples" in p for p in prompts))
            revisions = Ledger(pathlib.Path(tmp) / "out" / "travel").load().revisions
            self.assertEqual([(r["rule"]["id"], r["accepted"]) for r in revisions if r["kind"] == "add"], [("flat", False), ("fee", True)])

            class Agent(cb.ChelLearnings):
                chel_root = str(pathlib.Path(tmp) / "out")

                class runtime_context:
                    domain = "travel"
            got = Agent().retrieve_learnings("economy cancellation fee", 3)
            self.assertEqual(len(got), 1)
            self.assertIn("verified on 3 past tool results", got[0])

    def test_dossier_shows_varying_inputs_and_results(self):
        history = []
        for price, fee in ((300, 50), (600, 90)):
            history.extend(cb.trajectory_transitions(conversation(price, fee)))
        text = learn.dossier(history, "cancel_booking")
        self.assertIn("Recorded transitions: 2", text)
        self.assertIn("cancellation_fee", text)
        self.assertIn("price_paid", text)


if __name__ == "__main__":
    unittest.main()
