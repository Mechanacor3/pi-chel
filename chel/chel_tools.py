#!/usr/bin/env python3
"""Bounded tool backend for the Chel Pi brain (runs inside the brain container).

The Pi extension `pi-ext/fishbowl-chel.ts` is a thin adapter: every model tool
call becomes one invocation of

    python3 -I /opt/fishbowl/chel_tools.py --workspace /workspace tool  < {"tool": NAME, "args": {...}}

and the JSON printed on stdout is the tool result.  Two further subcommands are
for the trusted host driver only (via `docker exec`, never exposed as tools):

    ... record  < {"obs": ..., "action": ..., "next_obs": ...}   append one H transition
    ... view                                                     bounded prompt view

Policy (all enforced here, so it is testable without Node):

* Memento ψ/θ changes go only through `MementoMemory.propose` / `auto_simplify`,
  i.e. the unchanged acceptance gate.  The subject cannot append to H.
* Scratch files: UTF-8 text inside the workspace only; every path component must
  match SAFE_COMPONENT (so no hidden names: `.git`, `.pi`, `.gitattributes`, ...);
  `memento/` is read-only; every open walks directory fds with O_NOFOLLOW; per-file,
  file-count, and total-byte quotas.
* `scratch_commit` runs git with hooks, fsmonitor, signing, and global/system
  config disabled.

Nothing here reads the environment, network, clock, evaluator, or world state.
"""

import argparse
import json
import os
import re
import secrets
import stat
import subprocess
import sys

# Run with `python3 -I`: import only from this file's own (root-owned, read-only)
# directory, never from the working directory or the subject workspace.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from memento_memory import KINDS, MAX_TEXT, MEMENTO_DIR, MementoMemory, canonical, description_length, digest  # noqa: E402

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_DEPTH = 8
MAX_PATH = 512
MAX_FILE_BYTES = 65_536
MAX_FILES = 256
MAX_TOTAL_BYTES = 1_048_576
MAX_READ = 65_536
DEFAULT_READ = 16_384
MAX_LIST = 256
MAX_HISTORY_PAGE = 64
MAX_SIMPLIFY = 32
MAX_JSON_ARG = 262_144
MAX_COMMIT_MESSAGE = 200
HISTORY_TAIL = 8
MAX_REPORTED = 10
GIT_CONFIG = (
    "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c", "commit.gpgsign=false",
    "-c", "user.name=chel", "-c", "user.email=chel@fishbowl.invalid", "-c", "core.autocrlf=false",
)
GIT_ENV = {"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "HOME": "/nonexistent", "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0"}
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class ToolError(ValueError):
    """A refusal reported to the model as an ordinary failed tool result."""


# ------------------------------------------------------------------ arguments

def _args(args, allowed, required=()):
    if not isinstance(args, dict):
        raise ToolError("arguments must be an object")
    extra = set(args) - set(allowed)
    if extra:
        raise ToolError(f"unknown arguments: {sorted(extra)}")
    missing = [k for k in required if k not in args]
    if missing:
        raise ToolError(f"missing arguments: {missing}")
    return args


def _int(value, name, low, high, default):
    if value is None:
        return default
    if type(value) is not int or not low <= value <= high:
        raise ToolError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _str(value, name, limit, allow_empty=False):
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ToolError(f"{name} must be a{'' if allow_empty else ' non-empty'} string")
    if len(value.encode("utf-8")) > limit:
        raise ToolError(f"{name} exceeds {limit} bytes")
    return value


def _json_arg(value, name):
    text = _str(value, name, MAX_JSON_ARG)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolError(f"{name} is not valid JSON: {exc.msg} at {exc.pos}") from None


def split_path(path, allow_root=False):
    if path in ("", ".") and allow_root:
        return []
    _str(path, "path", MAX_PATH)
    parts = path.split("/")
    if len(parts) > MAX_DEPTH:
        raise ToolError(f"path deeper than {MAX_DEPTH} components")
    for part in parts:
        if not SAFE_COMPONENT.fullmatch(part):
            raise ToolError(f"path component {part!r} is not allowed (letters, digits, '_', '-', '.'; no leading dot)")
    return parts


def _writable(parts):
    if parts[0] == MEMENTO_DIR:
        raise ToolError("memento/ is read-only here; change ψ/θ with memento_propose")


# ----------------------------------------------------------- confined file I/O

class Workspace:
    def __init__(self, path):
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode):
            raise ToolError("workspace must be a real directory")
        self.path = path
        self.fd = os.open(path, DIR_FLAGS)

    def close(self):
        os.close(self.fd)

    def open_dir(self, parts, create=False):
        """Return an fd for the directory `parts`, never following a symlink."""
        fd = os.dup(self.fd)
        try:
            for part in parts:
                try:
                    nxt = os.open(part, DIR_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    if not create:
                        raise ToolError(f"no such directory: {'/'.join(parts)}") from None
                    os.mkdir(part, 0o755, dir_fd=fd)
                    nxt = os.open(part, DIR_FLAGS, dir_fd=fd)
                except (NotADirectoryError, OSError) as exc:
                    raise ToolError(f"cannot open directory component {part!r}: {exc.strerror}") from None
                os.close(fd)
                fd = nxt
            return fd
        except BaseException:
            os.close(fd)
            raise

    def lstat(self, dir_fd, name):
        try:
            return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def usage(self):
        """(files, bytes) over everything a scratch tool can create (excludes .git and hidden)."""
        files = total = 0
        for _root, dirs, names, rootfd in os.fwalk(".", dir_fd=self.fd, follow_symlinks=False):
            dirs[:] = [d for d in dirs if SAFE_COMPONENT.fullmatch(d)]
            for name in names:
                if not SAFE_COMPONENT.fullmatch(name):
                    continue
                info = os.stat(name, dir_fd=rootfd, follow_symlinks=False)
                files += 1
                total += info.st_size if stat.S_ISREG(info.st_mode) else 0
        return files, total


def _read_file(ws, parts, offset, limit):
    dfd = ws.open_dir(parts[:-1])
    try:
        try:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
        except FileNotFoundError:
            raise ToolError(f"no such file: {'/'.join(parts)}") from None
        except OSError as exc:
            raise ToolError(f"cannot open {'/'.join(parts)}: {exc.strerror}") from None
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ToolError("not a regular file")
            size = os.fstat(handle.fileno()).st_size
            handle.seek(offset)
            data = handle.read(limit)
    finally:
        os.close(dfd)
    return data, size


def scratch_list(ws, args):
    args = _args(args, ("path",))
    parts = split_path(args.get("path", ""), allow_root=True)
    dfd = ws.open_dir(parts)
    entries, truncated = [], False
    try:
        for root, dirs, names, rootfd in os.fwalk(".", dir_fd=dfd, follow_symlinks=False):
            dirs[:] = sorted(d for d in dirs if SAFE_COMPONENT.fullmatch(d))
            prefix = "/".join(parts + ([] if root == "." else root[2:].split("/")))
            for name in sorted(names) + [d + "/" for d in dirs]:
                bare = name.rstrip("/")
                if not SAFE_COMPONENT.fullmatch(bare):
                    continue
                if len(entries) >= MAX_LIST:
                    truncated = True
                    break
                info = os.stat(bare, dir_fd=rootfd, follow_symlinks=False)
                kind = "dir" if name.endswith("/") else ("file" if stat.S_ISREG(info.st_mode) else "other")
                entry = {"path": f"{prefix}/{bare}" if prefix else bare, "type": kind}
                if kind == "file":
                    entry["bytes"] = info.st_size
                entries.append(entry)
            if truncated:
                break
    finally:
        os.close(dfd)
    files, total = ws.usage()
    return {"entries": entries, "truncated": truncated,
            "quota": {"files": files, "max_files": MAX_FILES, "bytes": total, "max_bytes": MAX_TOTAL_BYTES}}


def scratch_read(ws, args):
    args = _args(args, ("path", "offset", "limit"), ("path",))
    parts = split_path(args["path"])
    offset = _int(args.get("offset"), "offset", 0, 2**31 - 1, 0)
    limit = _int(args.get("limit"), "limit", 1, MAX_READ, DEFAULT_READ)
    data, size = _read_file(ws, parts, offset, limit)
    return {"path": "/".join(parts), "offset": offset, "bytes": size,
            "content": data.decode("utf-8", errors="replace"), "eof": offset + len(data) >= size}


def scratch_write(ws, args):
    args = _args(args, ("path", "content", "mode"), ("path", "content"))
    parts = split_path(args["path"])
    _writable(parts)
    mode = args.get("mode", "create")
    if mode not in ("create", "overwrite", "append"):
        raise ToolError("mode must be create, overwrite, or append")
    content = _str(args["content"], "content", MAX_FILE_BYTES, allow_empty=True)
    if "\x00" in content:
        raise ToolError("content must be text (no NUL)")
    new = content.encode("utf-8")
    dfd = ws.open_dir(parts[:-1], create=True)
    try:
        info = ws.lstat(dfd, parts[-1])
        old = b""
        if info is not None:
            if not stat.S_ISREG(info.st_mode):
                raise ToolError("target exists and is not a regular file")
            if mode == "create":
                raise ToolError("file exists; use mode overwrite or append")
            if mode == "append":
                old, _ = _read_file(ws, parts, 0, MAX_FILE_BYTES + 1)
        data = old + new
        if len(data) > MAX_FILE_BYTES:
            raise ToolError(f"file would exceed {MAX_FILE_BYTES} bytes")
        files, total = ws.usage()
        files += 0 if info is not None else 1
        total += len(data) - (info.st_size if info is not None else 0)
        if files > MAX_FILES or total > MAX_TOTAL_BYTES:
            raise ToolError(f"workspace quota exceeded ({files}/{MAX_FILES} files, {total}/{MAX_TOTAL_BYTES} bytes)")
        temp = f".chel-tmp-{secrets.token_hex(8)}"
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(temp, parts[-1], src_dir_fd=dfd, dst_dir_fd=dfd)
        except BaseException:
            try:
                os.unlink(temp, dir_fd=dfd)
            except FileNotFoundError:
                pass
            raise
    finally:
        os.close(dfd)
    return {"path": "/".join(parts), "bytes": len(data), "mode": mode}


def scratch_delete(ws, args):
    args = _args(args, ("path",), ("path",))
    parts = split_path(args["path"])
    _writable(parts)
    dfd = ws.open_dir(parts[:-1])
    try:
        info = ws.lstat(dfd, parts[-1])
        if info is None:
            raise ToolError(f"no such file: {'/'.join(parts)}")
        if stat.S_ISDIR(info.st_mode):
            try:
                os.rmdir(parts[-1], dir_fd=dfd)
            except OSError:
                raise ToolError("only empty directories can be deleted") from None
        else:
            os.unlink(parts[-1], dir_fd=dfd)
    finally:
        os.close(dfd)
    return {"deleted": "/".join(parts)}


def _git(ws, *argv):
    # safe.directory names only this exact workspace (command-line scope), so the
    # attempt repo works even when the host provisioned it under another uid.
    return subprocess.run(["git", "-C", ws.path, *GIT_CONFIG, "-c", f"safe.directory={os.path.realpath(ws.path)}", *argv], env=GIT_ENV, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)


def scratch_commit(ws, args):
    args = _args(args, ("message",), ("message",))
    message = _str(args["message"], "message", MAX_COMMIT_MESSAGE)
    if any(ord(ch) < 32 for ch in message):
        raise ToolError("message must be a single line")
    info = ws.lstat(ws.fd, ".git")
    if info is None or not stat.S_ISDIR(info.st_mode):
        raise ToolError("workspace is not a git repository")
    added = _git(ws, "add", "-A", "--", ".")
    if added.returncode:
        raise ToolError(f"git add failed: {added.stderr.strip()[-500:]}")
    if _git(ws, "diff", "--cached", "--quiet").returncode == 0:
        return {"committed": False, "reason": "nothing to commit"}
    done = _git(ws, "commit", "--no-verify", "--quiet", "-m", message)
    if done.returncode:
        raise ToolError(f"git commit failed: {done.stderr.strip()[-500:]}")
    head = _git(ws, "rev-parse", "HEAD").stdout.strip()
    return {"committed": True, "head": head}


# --------------------------------------------------------------- memento tools

def _memory(ws):
    if ws.lstat(ws.fd, MEMENTO_DIR) is None:
        raise ToolError("memento store is not initialized for this attempt")
    return MementoMemory.load(ws.path)


def _verification_summary(result):
    return {
        "route": result["route"], "passed": result["passed"], "replay_ok": result["replay_ok"],
        "faithful": result["faithful"], "history_len": result["history_len"],
        "history_digest": result["history_digest"], "description_length": result["description_length"],
        "reasons": result["reasons"][:MAX_REPORTED],
        "replay_failures": result["replay_failures"][:MAX_REPORTED],
        "faithfulness_failures": result["faithfulness_failures"][:MAX_REPORTED],
    }


def memory_view(memory):
    """Deterministic, bounded view shared by `view` and `memento_status`."""
    return {
        "schema": 1,
        "rev": memory.current_rev(),
        "psi": memory.psi,
        "theta": memory.theta,
        "theta_sha256": digest(memory.theta),
        "description_length": description_length(memory.psi, memory.theta),
        "verification": _verification_summary(memory.verify()),
        "history_tail": memory.history[-HISTORY_TAIL:],
        "revisions": len(memory.revisions),
    }


def memento_status(ws, args):
    _args(args, ())
    return memory_view(_memory(ws))


def memento_history(ws, args):
    args = _args(args, ("offset", "limit"))
    memory = _memory(ws)
    offset = _int(args.get("offset"), "offset", 0, 2**31 - 1, 0)
    limit = _int(args.get("limit"), "limit", 1, MAX_HISTORY_PAGE, 32)
    return {"offset": offset, "total": len(memory.history),
            "transitions": [dict(t=offset + i, **tr) for i, tr in enumerate(memory.history[offset:offset + limit])]}


def memento_propose(ws, args):
    args = _args(args, ("kind", "psi_json", "theta_json", "rationale"), ("kind", "theta_json", "rationale"))
    kind = args["kind"]
    if kind not in KINDS:
        raise ToolError(f"kind must be one of {list(KINDS)}")
    rationale = _str(args["rationale"], "rationale", MAX_TEXT)
    theta = _json_arg(args["theta_json"], "theta_json")
    psi = None
    if "psi_json" in args:
        if kind == "theta_repair":
            raise ToolError("theta_repair keeps ψ fixed; omit psi_json")
        psi = _json_arg(args["psi_json"], "psi_json")
    elif kind == "psi_revision":
        raise ToolError("psi_revision requires psi_json")
    memory = _memory(ws)
    try:
        outcome = memory.propose(kind, theta, psi, rationale)
    except ValueError as exc:
        raise ToolError(f"candidate rejected before the gate: {exc}") from None
    return {"accepted": outcome["accepted"], "rev": outcome["rev"], "route": outcome["route"],
            "reasons": outcome["reasons"][:MAX_REPORTED], "gate": _verification_summary(outcome["result"])}


def memento_simplify(ws, args):
    args = _args(args, ("max_candidates",))
    limit = _int(args.get("max_candidates"), "max_candidates", 1, MAX_SIMPLIFY, MAX_SIMPLIFY)
    outcome = _memory(ws).auto_simplify(limit)
    out = {"accepted": outcome["accepted"], "reasons": outcome.get("reasons", [])[:MAX_REPORTED],
           "evaluated": outcome.get("evaluated", [])[:MAX_SIMPLIFY]}
    if "rev" in outcome:
        out["rev"] = outcome["rev"]
    return out


TOOLS = {
    "memento_status": memento_status,
    "memento_history": memento_history,
    "memento_propose": memento_propose,
    "memento_simplify": memento_simplify,
    "scratch_list": scratch_list,
    "scratch_read": scratch_read,
    "scratch_write": scratch_write,
    "scratch_delete": scratch_delete,
    "scratch_commit": scratch_commit,
}


def run_tool(workspace, request):
    """Execute one tool request; never raises for model-caused errors."""
    try:
        if not isinstance(request, dict) or set(request) - {"tool", "args"} or request.get("tool") not in TOOLS:
            raise ToolError(f"unknown tool; available: {sorted(TOOLS)}")
        ws = Workspace(workspace)
        try:
            return {"ok": True, "result": TOOLS[request["tool"]](ws, request.get("args") or {})}
        finally:
            ws.close()
    except (ToolError, ValueError, OSError) as exc:
        return {"ok": False, "error": str(exc)[:2000]}


# ------------------------------------------------------------------------ CLI

def _stdin_json(limit=MAX_JSON_ARG * 3):
    data = sys.stdin.buffer.read(limit + 1)
    if len(data) > limit:
        raise ToolError("request too large")
    return json.loads(data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("command", choices=("tool", "record", "view", "init"))
    args = ap.parse_args(argv)
    if args.command == "tool":
        try:
            request = _stdin_json()
        except (ValueError, ToolError) as exc:
            print(canonical({"ok": False, "error": f"invalid request: {exc}"}))
            return 0
        print(canonical(run_tool(args.workspace, request)))
        return 0
    # Trusted-driver subcommands: errors are fatal for the run (exit 2).
    try:
        if args.command == "init":
            if os.path.lexists(os.path.join(args.workspace, MEMENTO_DIR)):
                memory = MementoMemory.load(args.workspace)
            else:
                memory = MementoMemory.create(args.workspace)
            out = {"rev": memory.current_rev(), "history_len": len(memory.history)}
        elif args.command == "record":
            memory = MementoMemory.load(args.workspace)
            out = memory.record(_stdin_json())
            out["history_len"] = len(memory.history)
            out["history_digest"] = memory.digests[-1]
        else:
            out = memory_view(MementoMemory.load(args.workspace))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"CHEL_TOOLS_ERROR {exc}", file=sys.stderr)
        return 2
    print(canonical(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
