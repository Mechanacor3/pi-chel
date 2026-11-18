/**
 * Chel — verified-belief memory for Pi.
 *
 * All memory policy lives in the Python backend (chel/chel_tools.py, chel/ledger.py,
 * chel/memento_memory.py), which is stdlib-only and tested offline.  This extension
 * is a thin adapter plus a small per-run budget.  It is a policy/UX layer, not a
 * security boundary: run untrusted agents inside a container whose only writable
 * mount is the Chel workspace.
 *
 * Modes (CHEL_MODE):
 *
 *   assistant (default)  For ordinary Pi sessions.  Adds the rule-ledger and scratch
 *                        tools next to your normal tools.  Before each run, the most
 *                        relevant verified rules are injected as a short message.  With
 *                        CHEL_RECORD=1, every non-Chel tool result that is a JSON object
 *                        is recorded as ledger history {args -> result}, and active rules
 *                        are re-checked against it.
 *
 *   episode              The typed-action testbed protocol (one action per decision).
 *                        A trusted driver sends extension commands, which Pi handles
 *                        without a model call:
 *                          /chel-decision {"step": N, "actions": ["a", "b", ...]}
 *                          /chel-wrapup
 *                        Only Chel tools may run, budgets reset per decision, and the
 *                        model ends each decision with `submit_action`; the driver
 *                        re-validates `details.action` from the tool result.
 *
 * Workspace: CHEL_WORKSPACE, default `<cwd>/.pi/chel` (created on first use).
 */

import { StringEnum, Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const TOOLS_PY = process.env.CHEL_TOOLS ?? process.env.FISHBOWL_CHEL_TOOLS ?? resolve(HERE, "../chel/chel_tools.py");
const PYTHON = process.env.CHEL_PYTHON ?? "python3";
const MODE = (process.env.CHEL_MODE ?? "assistant") === "episode" ? "episode" : "assistant";
const RECORD = process.env.CHEL_RECORD === "1";
const MAX_CALLS_PER_DECISION = 16;
const MAX_PROPOSALS_PER_DECISION = 4;
const MAX_CALLS_WRAPUP = 24;
const MAX_CALLS_ASSISTANT_RUN = 48;
const MAX_RESULT_CHARS = 16_000;
const INJECT_RULES = 5;

const MEMENTO_TOOLS = ["memento_status", "memento_history", "memento_propose", "memento_simplify"] as const;
const SCRATCH_TOOLS = ["scratch_list", "scratch_read", "scratch_write", "scratch_delete", "scratch_commit"] as const;
const LEDGER_TOOLS = ["ledger_status", "ledger_recall", "ledger_propose"] as const;
type BackendTool = (typeof MEMENTO_TOOLS)[number] | (typeof SCRATCH_TOOLS)[number] | (typeof LEDGER_TOOLS)[number];
const CHEL_TOOLS = new Set<string>([
	...LEDGER_TOOLS,
	...SCRATCH_TOOLS,
	...(MODE === "episode" ? [...MEMENTO_TOOLS, "submit_action"] : []),
]);
const PROPOSAL_TOOLS = new Set<string>(["memento_propose", "memento_simplify", "ledger_propose"]);

type Mode = "idle" | "decision" | "wrapup" | "assistant";
const state = {
	mode: (MODE === "assistant" ? "assistant" : "idle") as Mode,
	step: 0,
	actions: new Set<string>(),
	calls: 0,
	proposals: 0,
	submitted: null as string | null,
	workspace: process.env.CHEL_WORKSPACE ?? process.env.FISHBOWL_CHEL_WORKSPACE ?? "",
};

function text(value: unknown) {
	let body = JSON.stringify(value, null, 1);
	if (body.length > MAX_RESULT_CHARS) {
		body = `${body.slice(0, MAX_RESULT_CHARS)}\n…[truncated; page with offset/limit]`;
	}
	return [{ type: "text" as const, text: body }];
}

function python(args: string[], input: string) {
	const run = spawnSync(PYTHON, ["-I", TOOLS_PY, "--workspace", state.workspace, ...args], {
		input,
		encoding: "utf8",
		timeout: 120_000,
		maxBuffer: 4 * 1024 * 1024,
		env: { PATH: process.env.PATH ?? "/usr/bin:/bin", LC_ALL: "C.UTF-8" },
	});
	if (run.error || run.status !== 0) {
		throw new Error(`chel backend failed: ${run.error?.message ?? run.stderr?.slice(-500) ?? run.status}`);
	}
	return run.stdout;
}

function backend(tool: string, args: Record<string, unknown>) {
	const out = JSON.parse(python(["tool"], JSON.stringify({ tool, args }))) as { ok: boolean; result?: unknown; error?: string };
	if (!out.ok) throw new Error(out.error ?? "refused");
	return { content: text(out.result), details: out.result };
}

function backendTool(name: BackendTool, label: string, description: string, parameters: any) {
	return {
		name,
		label,
		description,
		parameters,
		async execute(_id: string, params: Record<string, unknown>) {
			// Drop undefined optionals so the backend's strict argument check sees only real fields.
			const args = Object.fromEntries(Object.entries(params ?? {}).filter(([, v]) => v !== undefined));
			return backend(name, args);
		},
	};
}

// ---- assistant-mode recording: JSON tool results become ledger history ------------

function fieldName(path: string) {
	let name = path.toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
	if (!/^[a-z]/.test(name)) name = `f_${name}`;
	return name.slice(0, 64).replace(/_+$/, "");
}

function flatten(value: unknown, prefix = "", out: Record<string, unknown> = {}) {
	if (value && typeof value === "object" && !Array.isArray(value)) {
		for (const [k, v] of Object.entries(value)) flatten(v, prefix ? `${prefix}_${k}` : k, out);
	} else if (Array.isArray(value)) {
		if (prefix) out[fieldName(`n_${prefix}`)] = value.length;
	} else if (prefix) {
		if (value === null || typeof value === "boolean") out[fieldName(prefix)] = value;
		else if (typeof value === "number" && Number.isFinite(value) && Math.abs(value) <= 1e12) out[fieldName(prefix)] = value;
		else if (typeof value === "string") out[fieldName(prefix)] = value.slice(0, 256);
	}
	return out;
}

function capFields(obj: Record<string, unknown>) {
	return Object.fromEntries(Object.entries(obj).slice(0, 256));
}

function ensureWorkspace(cwd: string) {
	if (!state.workspace) state.workspace = join(cwd, ".pi", "chel");
	if (!existsSync(state.workspace)) mkdirSync(state.workspace, { recursive: true });
	if (!existsSync(join(state.workspace, ".git"))) {
		spawnSync("git", ["init", "-q", state.workspace], { encoding: "utf8" });
	}
	python(["ledger-init"], "");
	if (MODE === "episode" && !existsSync(join(state.workspace, "memento"))) python(["init"], "");
}

const optInt = (description: string) => Type.Optional(Type.Integer({ minimum: 0, description }));

export default function (pi: ExtensionAPI) {
	// ---------------------------------------------------------------- ledger tools
	pi.registerTool(
		backendTool(
			"ledger_status",
			"Chel ledger status",
			"Show your verified rule ledger: how much history it holds and which rules are currently accepted.",
			Type.Object({}),
		),
	);
	pi.registerTool(
		backendTool(
			"ledger_recall",
			"Chel ledger recall",
			"Find accepted rules relevant to a query. Every returned rule held on all recorded history with no exceptions.",
			Type.Object({ query: Type.String(), top_k: optInt("1-20, default 5") }),
		),
	);
	pi.registerTool(
		backendTool(
			"ledger_propose",
			"Chel ledger propose",
			"Propose a rule (op=add), replace one (op=revise, target) or retract one (op=retract, target, reason). " +
				"A rule is {id, text, when, then}: `when` maps obs.<field>/action.<field> to a literal, a list, or " +
				"{ne|lt|le|gt|ge: v}; `then` maps a result field to {set: v}, {copy: 'obs.f'} or {expr: E} " +
				"(E: const, ref, add sub mul div min max round abs, eq ne lt le gt ge, and or not, if). " +
				"It is accepted only with zero counterexamples across ALL recorded history and enough support; " +
				"rejections return the counterexamples.",
			Type.Object({
				op: StringEnum(["add", "revise", "retract"] as const),
				rule_json: Type.Optional(Type.String({ description: "the rule as JSON text (add/revise)" })),
				target: Type.Optional(Type.String({ description: "rule id (revise/retract)" })),
				reason: Type.Optional(Type.String({ description: "why (retract)" })),
				rationale: Type.Optional(Type.String()),
			}),
		),
	);

	// ---------------------------------------------------------------- scratch tools
	pi.registerTool(
		backendTool(
			"scratch_list",
			"Scratch list",
			"List files in your persistent Chel workspace (memento/ and chel-ledger/ are read-only). Shows quota usage.",
			Type.Object({ path: Type.Optional(Type.String({ description: "directory, default workspace root" })) }),
		),
	);
	pi.registerTool(
		backendTool(
			"scratch_read",
			"Scratch read",
			"Read a text file from your Chel workspace.",
			Type.Object({ path: Type.String(), offset: optInt("byte offset"), limit: optInt("bytes, max 65536") }),
		),
	);
	pi.registerTool(
		backendTool(
			"scratch_write",
			"Scratch write",
			"Write a UTF-8 text note into your persistent Chel workspace (max 64 KiB/file, 256 files, 1 MiB total). " +
				"Path components: letters, digits, '_', '-', '.', no leading dot.",
			Type.Object({
				path: Type.String(),
				content: Type.String(),
				mode: Type.Optional(StringEnum(["create", "overwrite", "append"] as const)),
			}),
		),
	);
	pi.registerTool(
		backendTool("scratch_delete", "Scratch delete", "Delete one file or empty directory from your Chel workspace.", Type.Object({ path: Type.String() })),
	);
	pi.registerTool(
		backendTool(
			"scratch_commit",
			"Scratch commit",
			"Commit every Chel workspace change (notes, ledger, memento) to its Git repository.",
			Type.Object({ message: Type.String({ description: "one line, max 200 bytes" }) }),
		),
	);

	// ---------------------------------------------------------------- episode protocol
	if (MODE === "episode") {
		pi.registerTool(
			backendTool(
				"memento_status",
				"Memento status",
				"Show your current accepted rulebook ψ (psi), executable model θ (theta), full-history verification, and the latest transitions of H.",
				Type.Object({}),
			),
		);
		pi.registerTool(
			backendTool(
				"memento_history",
				"Memento history",
				"Page through H, your append-only history of {obs, action, next_obs} transitions (max 64 per call).",
				Type.Object({ offset: optInt("first transition index"), limit: optInt("number of transitions, 1-64") }),
			),
		);
		pi.registerTool(
			backendTool(
				"memento_propose",
				"Memento propose",
				"Propose a successor model. It is accepted only if θ faithfully implements ψ AND both replay ALL of H exactly. " +
					"kind=psi_revision: new beliefs (psi_json + theta_json) for a BELIEF_ERROR. " +
					"kind=theta_repair: fix code only (theta_json; ψ unchanged) for an IMPLEMENTATION_ERROR. " +
					"kind=simplification: a strictly smaller equivalent pair. Rejections are recorded and return reasons.",
				Type.Object({
					kind: StringEnum(["psi_revision", "theta_repair", "simplification"] as const),
					psi_json: Type.Optional(Type.String({ description: "ψ document as JSON text: {schema:1, rules:[...]}" })),
					theta_json: Type.String({ description: "θ document as JSON text: {schema:1, cases:[...]}" }),
					rationale: Type.String({ description: "why this candidate is better" }),
				}),
			),
		);
		pi.registerTool(
			backendTool(
				"memento_simplify",
				"Memento simplify",
				"Try bounded automatic prunings of the current model; accepts the smallest one that still passes the full gate.",
				Type.Object({ max_candidates: optInt("1-32") }),
			),
		);
		pi.registerTool({
			name: "submit_action",
			label: "Submit action",
			description: "Choose exactly one of the allowed actions for the current decision. This ends your turn.",
			parameters: Type.Object({ action: Type.String({ description: "one of the allowed action names" }) }),
			async execute(_id: string, params: { action: string }) {
				if (state.mode !== "decision") throw new Error("no decision is open");
				if (state.submitted !== null) throw new Error("an action was already submitted for this decision");
				if (!state.actions.has(params.action)) {
					throw new Error(`action must be one of: ${[...state.actions].join(", ")}`);
				}
				state.submitted = params.action;
				return {
					content: [{ type: "text" as const, text: `submitted ${params.action}` }],
					details: { step: state.step, action: params.action },
					terminate: true,
				};
			},
		});

		pi.registerCommand("chel-decision", {
			description: "Trusted driver: open one decision",
			handler: async (args: string) => {
				const spec = JSON.parse(args) as { step: number; actions: string[] };
				if (!Number.isInteger(spec.step) || spec.step < 1 || !Array.isArray(spec.actions) || spec.actions.length === 0) {
					throw new Error("invalid decision spec");
				}
				state.mode = "decision";
				state.step = spec.step;
				state.actions = new Set(spec.actions.map(String));
				state.calls = 0;
				state.proposals = 0;
				state.submitted = null;
			},
		});
		pi.registerCommand("chel-wrapup", {
			description: "Trusted driver: end of attempt",
			handler: async () => {
				state.mode = "wrapup";
				state.actions = new Set();
				state.calls = 0;
				state.proposals = 0; // may still learn from the final observation
				state.submitted = null;
			},
		});
	}

	// ---------------------------------------------------------------- lifecycle + budgets
	pi.on("session_start", async (_event, ctx) => {
		if (MODE === "assistant") ensureWorkspace(ctx.cwd);
		else if (!state.workspace) state.workspace = "/workspace";
	});

	pi.on("before_agent_start", async (event) => {
		if (MODE !== "assistant") return undefined;
		state.calls = 0;
		state.proposals = 0;
		try {
			const out = JSON.parse(python(["tool"], JSON.stringify({ tool: "ledger_recall", args: { query: event.prompt.slice(0, 2000) || "rules", top_k: INJECT_RULES } })));
			const rules = (out.ok ? out.result.rules : []) as { id: string; text: string; support: number }[];
			if (rules.length === 0) return undefined;
			const lines = rules.map((r) => `- [${r.id}] ${r.text} (held on ${r.support} recorded results, no exceptions)`);
			return {
				message: {
					customType: "chel-rules",
					content: `Chel verified rules relevant to this request:\n${lines.join("\n")}`,
					display: false,
				},
			};
		} catch {
			return undefined;
		}
	});

	pi.on("tool_call", async (event) => {
		const name = event.toolName;
		if (state.mode === "assistant") {
			if (!CHEL_TOOLS.has(name)) return undefined; // the user's own tools are not Chel's business
			if (state.calls >= MAX_CALLS_ASSISTANT_RUN) return { block: true, reason: "Chel tool budget for this run is spent" };
			if (PROPOSAL_TOOLS.has(name) && state.proposals >= 3 * MAX_PROPOSALS_PER_DECISION) {
				return { block: true, reason: "Chel proposal budget for this run is spent" };
			}
			state.calls += 1;
			if (PROPOSAL_TOOLS.has(name)) state.proposals += 1;
			return undefined;
		}
		if (!CHEL_TOOLS.has(name)) return { block: true, reason: `${name} is not a Chel tool` };
		if (state.mode === "idle") return { block: true, reason: "no decision is open" };
		if (state.mode === "decision" && state.submitted !== null) {
			return { block: true, reason: "action already submitted; wait for the next decision" };
		}
		if (state.mode === "wrapup" && name === "submit_action") {
			return { block: true, reason: "the attempt is over; only commit your notes" };
		}
		const limit = state.mode === "wrapup" ? MAX_CALLS_WRAPUP : MAX_CALLS_PER_DECISION;
		if (state.calls >= limit && name !== "submit_action") {
			return { block: true, reason: `tool budget (${limit}) for this decision is spent; call submit_action now` };
		}
		if (PROPOSAL_TOOLS.has(name) && state.proposals >= MAX_PROPOSALS_PER_DECISION) {
			return { block: true, reason: `proposal budget (${MAX_PROPOSALS_PER_DECISION}) for this decision is spent` };
		}
		state.calls += 1;
		if (PROPOSAL_TOOLS.has(name)) state.proposals += 1;
		return undefined;
	});

	pi.on("tool_result", async (event) => {
		if (MODE !== "assistant" || !RECORD || event.isError || CHEL_TOOLS.has(event.toolName)) return undefined;
		const body = event.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		let result: unknown;
		try {
			result = JSON.parse(body);
		} catch {
			return undefined;
		}
		if (!result || typeof result !== "object" || Array.isArray(result)) return undefined;
		const args = Object.fromEntries(Object.entries(flatten(event.input ?? {})).map(([k, v]) => [fieldName(`arg_${k}`), v]));
		const transition = { obs: capFields(args), action: { kind: fieldName(event.toolName) }, next_obs: capFields(flatten(result)) };
		try {
			python(["ledger-record"], `${JSON.stringify(transition)}\n`);
		} catch {
			// Recording is best-effort; a refused transition (e.g. a privileged-looking field) is skipped.
		}
		return undefined;
	});
}
