# pi-chel

**Chel** is a memory plugin for the [Pi coding agent](https://pi.dev) built on one rule:
*a belief is kept only if it holds on everything the agent has recorded.*

The agent writes beliefs as small, executable rules. Before a rule is accepted, Chel replays
the complete recorded history against it. A single counterexample rejects it, and the agent
gets that counterexample back so it can repair the rule. New history re-checks every
accepted rule, and a rule that stops holding is marked *falsified*, never silently deleted.

- **Rule ledger** (open world, `chel/ledger.py`). Each rule states a condition and what a tool
  or environment will return, using literals, copies, or a small arithmetic expression language.
  For example, *"economy cancellation fee is max($50, 15% of fare)"* becomes:
  ```json
  {"id": "fee-economy", "text": "Economy cancellation fee is the larger of $50 and 15% of the fare.",
   "when": {"action.kind": "lookup_fee", "obs.arg_cabin": "economy"},
   "then": {"fee": {"expr": {"op": "max", "args": [{"const": 50},
            {"op": "mul", "args": [{"ref": "obs.arg_price"}, {"const": 0.15}]}]}}}}
  ```
  A rule is accepted only with zero counterexamples across all history and enough supporting
  cases (default 2).
- **Memento** (closed world, `chel/memento_memory.py`). A ψ rulebook plus an executable θ
  model that must reproduce *every* field of *every* recorded transition. Mismatches are routed
  as `BELIEF_ERROR` (fix ψ) or `IMPLEMENTATION_ERROR` (fix θ), and simplifications must pass the
  same gate. This suits small, fully observed state machines such as games.
- **Scratch notes**: a bounded, Git-committed notes folder the agent owns.

Everything is plain JSON files with hash chains. Identical operations produce byte-identical
stores, and nothing calls the network.

## Install

Requires Pi 1.0+, `python3` (3.10 or newer, standard library only) and `git` on `PATH`.

```bash
pi install git:github.com/Mechanacor3/pi-chel
# or try it for one session:
pi -e git:github.com/Mechanacor3/pi-chel
```

## Use

**Assistant mode** (the default) adds these tools next to your normal ones: `ledger_status`,
`ledger_recall`, `ledger_propose` and `scratch_*`. Before each run, Chel injects the accepted
rules most relevant to your prompt as a short hidden message.

| Variable | Default | Meaning |
|---|---|---|
| `CHEL_WORKSPACE` | `<cwd>/.pi/chel` | Where the ledger, notes and Git history live |
| `CHEL_RECORD` | off | `1` records every non-Chel tool result that is a JSON object as history (`arg_*` inputs, result fields) |
| `CHEL_MODE` | `assistant` | `episode` for the testbed protocol below |
| `CHEL_PYTHON` | `python3` | Interpreter for the backend |

With `CHEL_RECORD=1`, ask the agent to "learn rules from what you have seen". It proposes rules
with `ledger_propose`, receives counterexamples for the wrong ones, and repairs them.

**Episode mode** (`CHEL_MODE=episode`) is a typed-action protocol for testbeds: one action per
decision, driven by a trusted host over Pi RPC.

```
/chel-decision {"step": 1, "actions": ["observe", "harvest", "finish"]}   # opens a decision (no model call)
<prompt with the observation>                                              # the agent thinks, uses memory tools,
                                                                           # and ends with submit_action
/chel-wrapup                                                               # end of attempt: learn, commit, no action
```

Only Chel tools may run in episode mode. Tool calls and proposals are budgeted per decision,
and the driver re-validates the submitted action from the tool result's `details.action`.
Memento's history H is appended only by the driver (`chel/chel_tools.py record`), never by the
model. `prompts/chel-system.md` is a matching system prompt.

## Offline learning from trajectories

`chel/learn.py` runs model-authored rule induction over a ledger. It takes any
`complete(system, prompt) -> json_text` function; `chel/llm.py` supplies one for the OpenAI
Responses API using only the standard library. Every proposal, accepted or rejected, is logged
in `chel-ledger/revisions.jsonl`, and that log is the learning trace.

### STATE-Bench Agent Learning Track

`adapters/state_bench/chel_state_bench.py` turns every tool call in the
[STATE-Bench](https://github.com/microsoft/STATE-Bench) train trajectories into one transition:
facts seen earlier in the episode plus the call's arguments, the tool name, and the result
fields. It then runs induction and exposes `ChelLearnings.retrieve_learnings` for a
`StateBenchAgent` subclass.

```bash
OPENAI_API_KEY=... python3 -m adapters.state_bench.chel_state_bench build \
  --train-dir STATE-Bench/datasets/train_task_trajectories/travel --workspace learnings/chel/travel
```

```python
from state_bench.agents.state_bench import StateBenchAgent
from adapters.state_bench.chel_state_bench import ChelLearnings

class ChelAgent(ChelLearnings, StateBenchAgent):
    chel_root = "learnings/chel"          # <root>/<domain>/chel-ledger
```

Results are in [Results](#results).

## Safety notes

- The extension is a policy layer, not a sandbox. Run untrusted agents in a container whose only
  writable mount is the Chel workspace.
- The model can never write `memento/` or `chel-ledger/` directly: scratch tools refuse those
  paths, and history is appended only by trusted code.
- Scratch paths cannot be hidden names, symlinks or escapes. Git runs with hooks, fsmonitor,
  signing, and global/system configuration disabled.
- Field names that look like privileged channels (`secret`, `credential`, `api_key`, …) are
  refused as a tripwire. Do not record data the agent should not see.

## Tests

```bash
python3 -m unittest discover -s tests -t .
# Pi end-to-end tests (real Pi 1.0.0 in Docker against a scripted fake model):
docker build -t pi-plugin-e2e:pi-1.0.0 tests/e2e && python3 -m unittest tests.test_extension_e2e
```

## Background and credits

Chel's Memento store follows the design of **Memento 3**
([arXiv:2610.11794](https://arxiv.org/abs/2610.11794)). A frozen-model agent keeps a Markdown
rulebook ψ of hypotheses about its environment and compiles it into an executable Python model θ
in a Git-versioned workspace. On a prediction mismatch it chooses to keep the model, revise the
rulebook, or repair only the code. A candidate is accepted only if the model judges θ faithful to
ψ *and* exact replay reproduces the entire recorded history. An MDL bias prefers shorter
descriptions. Chel keeps that loop, including the belief-versus-implementation routing and the
simplification preference.

What Chel changes or adds:
- **θ is not free Python.** It is a small, terminating JSON expression language, and its
  faithfulness to ψ is checked deterministically (structural traceability plus differential
  agreement on a bounded probe set) instead of by a model judgment. That trades expressiveness
  for safety and auditability inside isolated agents.
- **Open-world rule ledger** (`chel/ledger.py`). Independent, partial rules that must hold on all
  history with no counterexample. This is our extension for tool-using agents whose
  observations are too wide to predict in full.
- Nothing here is the paper authors' code. Any errors are ours.

If you use Chel, please cite the paper:

```bibtex
@misc{zhao2026memento3,
  title         = {Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks},
  author        = {Zhao, Haoyu and Yu, Zhengxu and He, Zhiyuan and Fang, Meng and Tutunov, Rasul and
                   Bou-Ammar, Haitham and Luo, Weilin and Wang, Jun},
  year          = {2026},
  eprint        = {2610.11794},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2610.11794}
}
```

Evaluation uses [STATE-Bench](https://github.com/microsoft/STATE-Bench) (MIT) and runs on
[Pi](https://pi.dev).

## Results

STATE-Bench pilot results will be added here.

## License

MIT
