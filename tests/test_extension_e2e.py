#!/usr/bin/env python3
"""End-to-end: real Pi 1.0.0 + extensions/chel.ts + scripted fake model (Docker; skips otherwise)."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "e2e"))
sys.path.insert(0, str(HERE.parent / "chel"))
from harness import FakeUpstream, PiSession, docker_ready, plan_script  # noqa: E402
from ledger import Ledger  # noqa: E402

FEE = {"id": "fee-economy", "text": "Economy cancellation fee is the larger of $50 and 15% of the fare.",
       "when": {"action.kind": "lookup_fee", "obs.arg_cabin": "economy"},
       "then": {"fee": {"expr": {"op": "max", "args": [{"const": 50}, {"op": "mul", "args": [{"ref": "obs.arg_price"}, {"const": 0.15}]}]}}}}
WRONG = {**FEE, "id": "fee-flat", "text": "Economy fee is always $50.", "then": {"fee": {"set": 50}}}

ASSISTANT = {
    "Check fees": [[("lookup_fee", {"cabin": "economy", "price": 300}), ("lookup_fee", {"cabin": "economy", "price": 600})]],
    "Learn": [[("ledger_propose", {"op": "add", "rule_json": json.dumps(WRONG)}),
               ("ledger_propose", {"op": "add", "rule_json": json.dumps(FEE)})],
              [("scratch_write", {"path": "notes.md", "content": "fees: max(50, 15%)\n"}),
               ("scratch_write", {"path": "chel-ledger/history.jsonl", "content": "{}\n", "mode": "append"}),
               ("scratch_commit", {"message": "learned economy fee rule"})]],
    "What is the economy cancellation fee": [],
}

EPISODE = {
    "Decision step 1": [[("lookup_fee", {"cabin": "economy", "price": 1})], [("submit_action", {"action": "fly"})],
                        [("submit_action", {"action": "harvest"})]],
}


@unittest.skipUnless(docker_ready(), "needs docker and the e2e image (see tests/e2e/Dockerfile)")
class ChelPiEndToEnd(unittest.TestCase):
    def start(self, plans, env):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = pathlib.Path(self.tmp.name)
        self.upstream = FakeUpstream(plan_script(plans))
        self.pi = PiSession("extensions/chel.ts", self.work, self.upstream.port, env)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.upstream.close)
        self.addCleanup(self.pi.close)

    def test_assistant_mode_records_verifies_and_injects(self):
        self.start(ASSISTANT, {"CHEL_RECORD": "1"})
        self.pi.prompt("Check fees for two economy fares.")
        ws = self.work / ".pi" / "chel"
        ledger = Ledger(ws).load()
        self.assertEqual(len(ledger.history), 2)
        self.assertEqual(ledger.history[0]["obs"], {"arg_cabin": "economy", "arg_price": 300})
        self.assertEqual(ledger.history[1]["next_obs"]["fee"], 90)

        self.pi.prompt("Learn from what you saw.")
        results = {}
        for end in self.pi.tool_ends():
            results.setdefault(end["toolName"], []).append(end)
        flat, formula = (json.loads(e["result"]["content"][0]["text"]) for e in results["ledger_propose"])
        self.assertFalse(flat["accepted"])
        self.assertEqual(flat["counterexamples"][0]["mismatch"]["fee"]["actual"], 90)
        self.assertTrue(formula["accepted"], formula)
        writes = results["scratch_write"]
        self.assertFalse(writes[0].get("isError"))
        self.assertTrue(writes[1].get("isError"))  # the ledger is read-only to the model
        log = subprocess.run(["git", "-C", str(ws), "log", "--format=%s"], capture_output=True, text=True).stdout
        self.assertIn("learned economy fee rule", log)

        self.pi.prompt("What is the economy cancellation fee for a $400 fare?")
        injected = json.dumps(self.upstream.requests[-1]["messages"][-1])
        self.assertIn("Chel verified rules", injected)
        self.assertIn("fee-economy", injected)
        self.assertNotIn("fee-flat", injected)  # the rejected rule was never accepted

    def test_episode_mode_is_driver_controlled(self):
        self.start(EPISODE, {"CHEL_MODE": "episode", "CHEL_WORKSPACE": "/work"})
        subprocess.run(["git", "init", "-q", str(self.work)], check=True)
        self.pi.prompt('/chel-decision {"step": 1, "actions": ["observe", "harvest"]}')
        self.pi.prompt("Decision step 1: choose.")
        ends = self.pi.tool_ends()
        self.assertTrue(ends[0]["isError"])  # non-Chel tools are blocked in episode mode
        self.assertIn("not a Chel tool", json.dumps(ends[0]["result"]))
        self.assertTrue(ends[1]["isError"])  # 'fly' is not an allowed action
        self.assertEqual(ends[2]["result"]["details"], {"step": 1, "action": "harvest"})
        tools = {t["function"]["name"] for t in self.upstream.requests[0]["tools"]}
        self.assertTrue({"memento_propose", "submit_action", "ledger_propose"} <= tools)


if __name__ == "__main__":
    unittest.main()
