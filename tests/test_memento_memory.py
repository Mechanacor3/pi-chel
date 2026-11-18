#!/usr/bin/env python3
"""Focused tests for the Memento3-style H / ψ (psi) / θ (theta) prototype."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memento_memory as mm  # noqa: E402


def obs(inventory, target, reach=True):
    return {"inventory_dirt": inventory, "target_block": target, "target_in_reach": reach}


def step(before, kind, after):
    return {"obs": before, "action": {"kind": kind}, "next_obs": after}


T0 = step(obs(0, "grass"), "harvest", obs(1, "air"))              # harvest grass -> +1 dirt
T1 = step(obs(1, "air"), "observe", obs(1, "air"))                # looking changes nothing
T2 = step(obs(1, "dirt", False), "harvest", obs(1, "dirt", False))  # out of reach: no effect
T3 = step(obs(1, "stone"), "harvest", obs(1, "stone"))            # stone is not harvestable


def E(op, *args):
    return {"op": op, "args": list(args)}


def C(value):
    return {"const": value}


OBS = lambda f: {"obs": f}  # noqa: E731
ACT = lambda f: {"action": f}  # noqa: E731
HARVEST_TEXT = "Harvesting grass or dirt within reach adds one dirt to my inventory and leaves air."


def psi_of(*rules):
    return {"schema": 1, "rules": list(rules)}


def theta_of(*cases):
    return {"schema": 1, "cases": list(cases)}


def harvest_rule(targets=("dirt", "grass"), rule_id="R1"):
    when = {"action.kind": "harvest", "obs.target_in_reach": True}
    if targets is not None:
        when["obs.target_block"] = list(targets)
    return {"id": rule_id, "text": HARVEST_TEXT, "when": when,
            "then": {"inventory_dirt": {"delta": 1}, "target_block": {"set": "air"}}}


def harvest_case(targets=("dirt", "grass"), rule_id="R1", inventory=None, extra_guard=None):
    guard = [E("eq", ACT("kind"), C("harvest")), E("eq", OBS("target_in_reach"), C(True))]
    if targets is not None:
        guard.append(E("in", OBS("target_block"), *[C(t) for t in targets]))
    if extra_guard is not None:
        guard.append(extra_guard)
    return {"implements": rule_id, "guard": E("and", *guard),
            "set": {"inventory_dirt": inventory or E("add", OBS("inventory_dirt"), C(1)),
                    "target_block": C("air")}}


GOOD_PSI = psi_of(harvest_rule())
GOOD_THETA = theta_of(harvest_case())


class MementoTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.temporary.name)
        self.workspace = self.base / "attempt" / "workspace"
        self.workspace.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def fresh(self, *transitions):
        memory = mm.MementoMemory.create(self.workspace)
        for transition in transitions:
            memory.record(transition)
        return memory


class ReplayTest(MementoTestCase):
    def test_consistent_model_replays_complete_history(self):
        memory = self.fresh()
        first = memory.record(T0)
        self.assertEqual(first["route"], mm.BELIEF_ERROR)  # empty ψ (psi) predicts no change
        outcome = memory.propose("psi_revision", GOOD_THETA, GOOD_PSI, "grass harvest gave dirt")
        self.assertTrue(outcome["accepted"], outcome["reasons"])
        for transition in (T1, T2, T3):
            self.assertEqual(memory.record(transition)["route"], mm.CONSISTENT)
        result = memory.verify()
        self.assertTrue(result["passed"])
        self.assertTrue(result["replay_ok"] and result["faithful"])
        self.assertEqual(result["history_len"], 4)

    def test_newer_rule_that_breaks_older_observation_is_rejected(self):
        memory = self.fresh(T0)
        naive = memory.propose("psi_revision", theta_of(harvest_case(None)), psi_of(harvest_rule(None)), "harvest gives dirt")
        self.assertTrue(naive["accepted"], naive["reasons"])
        memory.record(T1)
        memory.record(T2)
        latest = memory.record(T3)
        self.assertEqual(latest["route"], mm.BELIEF_ERROR)
        accepted_rev = memory.current_rev()

        # Explains stone (newest) but breaks the older grass harvest at t=0.
        dirt_only = memory.propose("psi_revision", theta_of(harvest_case(("dirt",))), psi_of(harvest_rule(("dirt",))), "only dirt?")
        self.assertFalse(dirt_only["accepted"])
        self.assertEqual(dirt_only["route"], "REVISE_PSI")
        self.assertEqual([f["t"] for f in dirt_only["result"]["replay_failures"]], [0])
        self.assertEqual(memory.current_rev(), accepted_rev)
        self.assertEqual(memory.psi, psi_of(harvest_rule(None)))

        fixed = memory.propose("psi_revision", GOOD_THETA, GOOD_PSI, "stone does not yield dirt")
        self.assertTrue(fixed["accepted"], fixed["reasons"])
        self.assertTrue(memory.verify()["replay_ok"])


class RoutingTest(MementoTestCase):
    def test_diagnose_routes_belief_and_implementation_errors_distinctly(self):
        wrong_belief = mm.diagnose(psi_of(harvest_rule(None)), theta_of(harvest_case(None)), T3)
        self.assertEqual(wrong_belief["route"], mm.BELIEF_ERROR)
        broken_code = theta_of(harvest_case(inventory=C(1)))  # ψ (psi) is right, θ (theta) hard-codes 1
        wrong_code = mm.diagnose(GOOD_PSI, broken_code, step(obs(4, "grass"), "harvest", obs(5, "air")))
        self.assertEqual(wrong_code["route"], mm.IMPLEMENTATION_ERROR)
        self.assertEqual(mm.diagnose(GOOD_PSI, GOOD_THETA, T0)["route"], mm.CONSISTENT)

    def test_implementation_error_requires_theta_repair_and_keeps_psi(self):
        memory = self.fresh(T0, T1)
        # θ (theta) silently refuses to harvest at >= 5 dirt; probes built from H never reach 5.
        # Phrased as not(ge) rather than lt so that on the "inventory_dirt absent" probe the guard
        # still fires and θ (theta) is UNDEFINED via R1 exactly like ψ (psi).  An `lt` cap would
        # instead fall through to "no rule" there, which the verifier correctly flags as unfaithful.
        capped = theta_of(harvest_case(extra_guard=E("not", E("ge", OBS("inventory_dirt"), C(5)))))
        self.assertTrue(memory.propose("psi_revision", capped, GOOD_PSI, "harvest rule")["accepted"])
        diagnosis = memory.record(step(obs(7, "grass"), "harvest", obs(8, "air")))
        self.assertEqual(diagnosis["route"], mm.IMPLEMENTATION_ERROR)
        self.assertEqual(diagnosis["model_route"], "REPAIR_THETA")
        psi_before = mm.digest(memory.psi)

        changed_psi = memory.propose("theta_repair", GOOD_THETA, psi_of(harvest_rule(("grass",))), "sneak a belief change")
        self.assertFalse(changed_psi["accepted"])
        self.assertIn("theta_repair must keep ψ (psi) unchanged", changed_psi["reasons"][0])

        repaired = memory.propose("theta_repair", GOOD_THETA, rationale="remove unfounded cap")
        self.assertTrue(repaired["accepted"], repaired["reasons"])
        self.assertEqual(mm.digest(memory.psi), psi_before)
        self.assertEqual(memory.revisions[-1]["kind"], "theta_repair")

    def test_belief_error_cannot_be_patched_by_theta_repair(self):
        memory = self.fresh(T0)
        memory.propose("psi_revision", theta_of(harvest_case(None)), psi_of(harvest_rule(None)), "naive")
        self.assertEqual(memory.record(T3)["route"], mm.BELIEF_ERROR)
        # θ (theta) that happens to match all history but no longer implements ψ (psi).
        patched = memory.propose("theta_repair", GOOD_THETA, rationale="patch code instead of belief")
        self.assertFalse(patched["accepted"])
        self.assertEqual(patched["route"], "REVISE_PSI")
        self.assertTrue(any("belief error" in reason for reason in patched["reasons"]))


class FaithfulnessTest(MementoTestCase):
    def test_unfaithful_theta_rejected_even_when_replay_matches(self):
        memory = self.fresh(T0)
        hard_coded = theta_of(harvest_case(inventory=C(1)))  # matches 0 -> 1 only
        outcome = memory.propose("psi_revision", hard_coded, GOOD_PSI, "looks right on history")
        self.assertFalse(outcome["accepted"])
        self.assertTrue(outcome["result"]["replay_ok"])
        self.assertFalse(outcome["result"]["faithful"])
        self.assertEqual(outcome["route"], "REPAIR_THETA")
        self.assertEqual(memory.psi, mm.EMPTY_PSI)

    def test_untraceable_or_missing_implementation_is_rejected(self):
        memory = self.fresh(T0)
        mislabeled = theta_of(harvest_case(rule_id="R9"))
        outcome = memory.propose("psi_revision", mislabeled, GOOD_PSI, "wrong label")
        self.assertFalse(outcome["accepted"])
        kinds = {f["kind"] for f in outcome["result"]["faithfulness_failures"]}
        self.assertIn("untraceable_case", kinds)
        self.assertIn("unimplemented_rule", kinds)

    def test_privileged_fields_never_enter_memory(self):
        memory = self.fresh()
        for bad in ({"evaluator_state": 1}, {"server_tick": 3}, {"hidden_x": 4}):
            with self.assertRaises(mm.MementoError):
                memory.record({"obs": bad, "action": {"kind": "observe"}, "next_obs": bad})
        with self.assertRaises(mm.MementoError):
            memory.propose("psi_revision", theta_of(), psi_of({"id": "R1", "text": "peek", "when": {"obs.score_hint": 1}, "then": {}}))
        self.assertEqual(memory.history, [])
        self.assertEqual(memory._read("history.jsonl"), "")


class PersistenceTest(MementoTestCase):
    def scenario(self, workspace):
        memory = mm.MementoMemory.create(workspace)
        memory.record(T0)
        memory.propose("psi_revision", theta_of(harvest_case(inventory=C(1))), GOOD_PSI, "rejected attempt")
        memory.propose("psi_revision", GOOD_THETA, GOOD_PSI, "accepted belief")
        memory.record(T3)
        return memory

    def snapshot(self, workspace):
        root = workspace / "memento"
        return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    def test_persistence_is_deterministic_and_reload_is_exact(self):
        other = self.base / "other" / "workspace"
        other.mkdir(parents=True)
        first = self.scenario(self.workspace)
        self.scenario(other)
        self.assertEqual(self.snapshot(self.workspace), self.snapshot(other))
        before = self.snapshot(self.workspace)
        loaded = mm.MementoMemory.load(self.workspace)
        self.assertEqual(loaded.history, first.history)
        self.assertEqual(loaded.psi, first.psi)
        self.assertEqual(loaded.theta, first.theta)
        self.assertEqual(loaded.revisions, first.revisions)
        self.assertEqual(loaded.verify(), first.verify())
        self.assertEqual(self.snapshot(self.workspace), before)  # load never writes
        self.assertEqual([r["accepted"] for r in loaded.revisions], [True, False, True])
        self.assertIn(HARVEST_TEXT, (self.workspace / "memento/psi.md").read_text())

    def test_all_file_access_stays_inside_attempt_workspace(self):
        opened = []
        active = [True]

        def hook(event, args):
            if active[0] and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
                opened.append(pathlib.Path(os.fsdecode(args[0])).resolve())

        sys.addaudithook(hook)
        try:
            self.scenario(self.workspace)
            loaded = mm.MementoMemory.load(self.workspace)
            loaded.auto_simplify()
        finally:
            active[0] = False
        self.assertTrue(opened)
        root = self.workspace.resolve() / "memento"
        outside = [p for p in opened if root not in p.parents]
        self.assertEqual(outside, [])
        self.assertEqual(sorted(p.name for p in self.workspace.iterdir()), ["memento"])

    def test_symlink_and_tampering_are_rejected_on_load(self):
        self.scenario(self.workspace)
        history = self.workspace / "memento/history.jsonl"
        original = history.read_text()

        outside = self.base / "outside.jsonl"
        outside.write_text(original)
        history.unlink()
        history.symlink_to(outside)
        with self.assertRaises(OSError):
            mm.MementoMemory.load(self.workspace)
        history.unlink()

        lines = original.splitlines()
        lines[0] = mm.canonical(step(obs(0, "grass"), "harvest", obs(2, "air")))  # rewrite the past
        history.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(mm.MementoError, "append-only"):
            mm.MementoMemory.load(self.workspace)

        history.write_text(original)
        (self.workspace / "memento/psi.json").write_text(mm.pretty(mm.EMPTY_PSI))
        with self.assertRaisesRegex(mm.MementoError, "propose changes"):
            mm.MementoMemory.load(self.workspace)


class SimplificationTest(MementoTestCase):
    OBSERVE_RULE = {"id": "R0", "text": "Observing changes nothing.", "when": {"action.kind": "observe"}, "then": {}}
    OBSERVE_CASE = {"implements": "R0", "guard": E("eq", ACT("kind"), C("observe")), "set": {}}

    def verbose_memory(self):
        memory = self.fresh(T0, T1, T2, T3)
        psi = psi_of(self.OBSERVE_RULE, harvest_rule())
        theta = theta_of(self.OBSERVE_CASE, harvest_case())
        self.assertTrue(memory.propose("psi_revision", theta, psi, "verbose but true")["accepted"])
        return memory

    def test_auto_simplify_prefers_smaller_verified_model(self):
        memory = self.verbose_memory()
        before = mm.description_length(memory.psi, memory.theta)
        probes = mm.probe_states(memory.history, memory.psi)
        behavior = mm.behavior_signature(memory.hypothesis(), probes)
        outcome = memory.auto_simplify()
        self.assertTrue(outcome["accepted"], outcome)
        self.assertEqual(memory.psi, GOOD_PSI)
        self.assertEqual(memory.theta, GOOD_THETA)
        self.assertLess(mm.description_length(memory.psi, memory.theta), before)
        self.assertEqual(mm.behavior_signature(memory.hypothesis(), probes), behavior)  # same behavior, shorter
        self.assertEqual(memory.revisions[-1]["kind"], "simplification")
        failed = [e["candidate"] for e in outcome["evaluated"] if not e["passed"]]
        self.assertIn("drop rule R1", failed)  # dropping R1 would break replay of t=0
        self.assertFalse(memory.auto_simplify()["accepted"])

    def test_simplification_cannot_bypass_replay(self):
        memory = self.verbose_memory()
        rev = memory.current_rev()
        smaller_but_wrong = memory.propose("simplification", theta_of(self.OBSERVE_CASE), psi_of(self.OBSERVE_RULE), "drop harvest")
        self.assertFalse(smaller_but_wrong["accepted"])
        self.assertFalse(smaller_but_wrong["result"]["replay_ok"])
        self.assertIn(0, [f["t"] for f in smaller_but_wrong["result"]["replay_failures"]])
        not_smaller = memory.propose("simplification", memory.theta, memory.psi, "same size")
        self.assertFalse(not_smaller["accepted"])
        self.assertIn("simplification must strictly reduce description length", not_smaller["reasons"])
        self.assertEqual(memory.current_rev(), rev)


class CliTest(MementoTestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, HERE / "memento_memory.py", "--workspace", self.workspace, *map(str, args)],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_cli_round_trip_and_workspace_confinement(self):
        self.assertEqual(self.cli("init").returncode, 0)
        (self.workspace / "t0.json").write_text(json.dumps(T0))
        (self.workspace / "cand.json").write_text(json.dumps({"psi": GOOD_PSI, "theta": GOOD_THETA, "rationale": "cli"}))
        self.assertEqual(self.cli("record", "t0.json").returncode, 1)  # BELIEF_ERROR
        self.assertEqual(self.cli("propose", "psi_revision", "cand.json").returncode, 0)
        status = self.cli("status")
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertTrue(json.loads(status.stdout)["passed"])
        outside = self.base / "outside.json"
        outside.write_text(json.dumps(T0))
        escaped = self.cli("record", outside)
        self.assertEqual(escaped.returncode, 2)
        self.assertIn("inside the attempt workspace", escaped.stderr)


if __name__ == "__main__":
    unittest.main()
