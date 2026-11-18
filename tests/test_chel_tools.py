#!/usr/bin/env python3
"""Offline tests for the Chel Pi-brain tool backend (`chel_tools.py`)."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chel_tools as ct  # noqa: E402
import memento_memory as mm  # noqa: E402
from test_memento_memory import GOOD_PSI, GOOD_THETA, T0, T1, harvest_case, theta_of  # noqa: E402


def snapshot(root):
    out = {}
    for path in sorted(pathlib.Path(root).rglob("*")):
        if ".git" in path.parts:
            continue
        out[str(path.relative_to(root))] = path.read_bytes() if path.is_file() and not path.is_symlink() else None
    return out


class ChelToolsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.temporary.name)
        self.ws = self.base / "workspace"
        self.ws.mkdir()
        mm.MementoMemory.create(self.ws)

    def tearDown(self):
        self.temporary.cleanup()

    def tool(self, name, **args):
        return ct.run_tool(str(self.ws), {"tool": name, "args": args})

    def ok(self, name, **args):
        out = self.tool(name, **args)
        self.assertTrue(out["ok"], out)
        return out["result"]

    def refused(self, name, fragment, **args):
        out = self.tool(name, **args)
        self.assertFalse(out["ok"], out)
        self.assertIn(fragment, out["error"])
        return out

    # -- scratch ---------------------------------------------------------------
    def test_write_read_append_list_delete_roundtrip(self):
        self.ok("scratch_write", path="notes/plan.md", content="dig grass\n")
        self.refused("scratch_write", "file exists", path="notes/plan.md", content="x")
        self.ok("scratch_write", path="notes/plan.md", content="then place\n", mode="append")
        self.assertEqual(self.ok("scratch_read", path="notes/plan.md")["content"], "dig grass\nthen place\n")
        self.assertEqual(self.ok("scratch_read", path="notes/plan.md", offset=4, limit=5)["content"], "grass")
        listing = self.ok("scratch_list")
        paths = {e["path"] for e in listing["entries"]}
        self.assertIn("notes/plan.md", paths)
        self.assertIn("memento/psi.md", paths)
        self.ok("scratch_write", path="notes/plan.md", content="v2", mode="overwrite")
        self.assertEqual((self.ws / "notes/plan.md").read_text(), "v2")
        self.ok("scratch_delete", path="notes/plan.md")
        self.ok("scratch_delete", path="notes")
        self.assertFalse((self.ws / "notes").exists())
        self.assertEqual([p.name for p in self.ws.iterdir() if p.name.startswith(".chel-tmp")], [])

    def test_rejects_escapes_hidden_names_and_memento_writes(self):
        before = snapshot(self.ws)
        for bad in ("../outside", "/etc/passwd", "a/../../x", ".git/config", ".pi/extensions/x.ts",
                    "notes/.gitattributes", "AGENTS.md/..", "a//b", "a b", "x" * 65, "/".join("a" * 9)):
            self.refused("scratch_write", "", path=bad, content="x")
            self.refused("scratch_read", "", path=bad)
        self.refused("scratch_write", "read-only", path="memento/psi.json", content="{}", mode="overwrite")
        self.refused("scratch_delete", "read-only", path="memento/history.jsonl")
        self.assertEqual(snapshot(self.ws), before)
        mm.MementoMemory.load(self.ws)  # store still intact

    def test_never_follows_symlinks(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("host secret")
        os.symlink(outside, self.ws / "link")
        os.symlink(outside / "secret.txt", self.ws / "file-link")
        self.refused("scratch_read", "cannot open", path="link/secret.txt")
        self.refused("scratch_read", "cannot open", path="file-link")
        self.refused("scratch_write", "cannot open", path="link/new.txt", content="x")
        self.refused("scratch_write", "not a regular file", path="file-link", content="x", mode="overwrite")
        self.assertEqual(sorted(p.name for p in outside.iterdir()), ["secret.txt"])
        self.assertEqual((outside / "secret.txt").read_text(), "host secret")

    def test_quotas(self):
        self.refused("scratch_write", "exceeds", path="big.txt", content="x" * (ct.MAX_FILE_BYTES + 1))
        self.ok("scratch_write", path="a.txt", content="x" * (ct.MAX_FILE_BYTES - 1))
        self.refused("scratch_write", "would exceed", path="a.txt", content="yy", mode="append")
        self.refused("scratch_write", "NUL", path="b.txt", content="a\x00b")
        old = ct.MAX_FILES
        try:
            files, _ = ct.Workspace(str(self.ws)).usage()
            ct.MAX_FILES = files
            self.refused("scratch_write", "quota", path="one-more.txt", content="x")
        finally:
            ct.MAX_FILES = old

    def test_unknown_tool_and_arguments(self):
        self.assertFalse(ct.run_tool(str(self.ws), {"tool": "bash", "args": {"command": "id"}})["ok"])
        self.refused("scratch_read", "unknown arguments", path="memento/psi.md", follow=True)
        self.refused("scratch_read", "missing", )

    def test_commit_disables_hooks_and_commits_scratch(self):
        git = lambda *a: subprocess.run(["git", "-C", str(self.ws), *a], check=True, capture_output=True, text=True)  # noqa: E731
        git("init", "-q")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "base")
        hook = self.ws / ".git/hooks/pre-commit"
        hook.write_text(f"#!/bin/sh\ntouch {self.base}/HOOK_RAN\nexit 1\n")
        hook.chmod(0o755)
        self.ok("scratch_write", path="notes/a.md", content="hello")
        self.refused("scratch_commit", "single line", message="a\nb")
        out = self.ok("scratch_commit", message="notes after step 1")
        self.assertTrue(out["committed"])
        self.assertFalse((self.base / "HOOK_RAN").exists())
        self.assertEqual(git("status", "--porcelain").stdout, "")
        self.assertIn("memento/history.jsonl", git("ls-files").stdout)
        self.assertEqual(self.ok("scratch_commit", message="again"), {"committed": False, "reason": "nothing to commit"})

    # -- memento ---------------------------------------------------------------
    def test_propose_goes_through_gate_and_records_rejections(self):
        memory = mm.MementoMemory.load(self.ws)
        memory.record(T0)
        memory.record(T1)
        bad = self.ok("memento_propose", kind="psi_revision", psi_json=json.dumps(GOOD_PSI),
                      theta_json=json.dumps(theta_of(harvest_case(inventory={"const": 1}))),
                      rationale="hard-coded inventory")
        self.assertFalse(bad["accepted"])
        self.assertEqual(bad["route"], "REPAIR_THETA")
        self.assertEqual(mm.MementoMemory.load(self.ws).current_rev(), 0)
        good = self.ok("memento_propose", kind="psi_revision", psi_json=json.dumps(GOOD_PSI),
                       theta_json=json.dumps(GOOD_THETA), rationale="harvest adds dirt")
        self.assertTrue(good["accepted"], good)
        status = self.ok("memento_status")
        self.assertEqual(status["psi"], GOOD_PSI)
        self.assertEqual(status["verification"]["route"], "ACCEPT")
        self.assertEqual(len(mm.MementoMemory.load(self.ws).revisions), 3)  # init + rejected + accepted

    def test_propose_argument_contract(self):
        self.refused("memento_propose", "omit psi_json", kind="theta_repair", psi_json="{}", theta_json="{}", rationale="r")
        self.refused("memento_propose", "requires psi_json", kind="psi_revision", theta_json="{}", rationale="r")
        self.refused("memento_propose", "not valid JSON", kind="theta_repair", theta_json="{nope", rationale="r")
        self.refused("memento_propose", "kind", kind="rewrite_history", theta_json="{}", rationale="r")
        self.refused("memento_propose", "before the gate", kind="theta_repair",
                     theta_json=json.dumps({"schema": 1, "cases": [{"implements": "R1", "guard": {"op": "exec"}}]}),
                     rationale="r")

    def test_history_paging_and_simplify(self):
        memory = mm.MementoMemory.load(self.ws)
        for t in (T0, T1, T1):
            memory.record(t)
        page = self.ok("memento_history", offset=1, limit=1)
        self.assertEqual(page["total"], 3)
        self.assertEqual(page["transitions"], [dict(t=1, **T1)])
        self.refused("memento_history", "limit", limit=1000)
        self.assertIn("accepted", self.ok("memento_simplify", max_candidates=4))

    def test_trusted_cli_record_view_and_tool(self):
        script = str(HERE / "chel_tools.py")
        run = lambda cmd, payload=None: subprocess.run(  # noqa: E731
            [sys.executable, "-I", script, "--workspace", str(self.ws), cmd],
            input="" if payload is None else json.dumps(payload), capture_output=True, text=True)
        rec = run("record", T0)
        self.assertEqual(rec.returncode, 0, rec.stderr)
        self.assertEqual(json.loads(rec.stdout)["history_len"], 1)
        self.assertEqual(json.loads(run("view").stdout)["history_tail"], [T0])
        tool = run("tool", {"tool": "memento_history", "args": {}})
        self.assertEqual(json.loads(tool.stdout)["result"]["total"], 1)
        self.assertEqual(run("record", {"obs": {"score": 1}, "action": {}, "next_obs": {}}).returncode, 2)
        garbage = run("tool")
        self.assertFalse(json.loads(garbage.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
