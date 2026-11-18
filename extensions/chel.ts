/**
 * Fishbowl Chel extension for Pi 1.0.0 (baked read-only into the Chel brain image).
 *
 * This is a policy/UX layer, NOT the security boundary.  It runs inside the Pi
 * process with the subject's permissions; the container contract and the
 * host-side seal audit (audit-memento-attempt.py) are the real gates.  All file
 * and Memento policy lives in chel_tools.py, which is tested offline.
 *
 * Protocol with the trusted host driver (pi-chel-decision-driver.py):
 *   /chel-decision {"step": N, "actions": ["observe", "harvest", "finish"]}
 *       opens one decision: resets budgets and sets the allowed action set.
 *   /chel-wrapup
 *       end of attempt: memento/scratch tools stay available (fresh budget, so
 *       Chel can still learn from the final observation); submit_action is refused.
 * Extension commands are handled by Pi without a model call; the model cannot
 * invoke them.  The chosen action is reported through the `submit_action`
 * tool result (`details.action`), which the driver re-validates.
 */

import { StringEnum, Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawnSync } from "node:child_process";

const WORKSPACE = process.env.FISHBOWL_CHEL_WORKSPACE ?? "/workspace";
const TOOLS_PY = process.env.FISHBOWL_CHEL_TOOLS ?? "/opt/fishbowl/chel_tools.py";
const PYTHON = "/usr/bin/python3";
const MAX_CALLS_PER_DECISION = 16;
const MAX_PROPOSALS_PER_DECISION = 4;
const MAX_CALLS_WRAPUP = 24;
const MAX_RESULT_CHARS = 16_000;

const BACKEND_TOOLS = [
	"memento_status",
	"memento_history",
	"memento_propose",
	"memento_simplify",
	"scratch_list",
	"scratch_read",
	"scratch_write",
	"scratch_delete",
	"scratch_commit",
] as const;
const ALL_TOOLS = new Set<string>([...BACKEND_TOOLS, "submit_action"]);
const PROPOSAL_TOOLS = new Set<string>(["memento_propose", "memento_simplify"]);

type Mode = "idle" | "decision" | "wrapup";
const state = {
	mode: "idle" as Mode,
	step: 0,
	actions: new Set<string>(),
	calls: 0,
	proposals: 0,
	submitted: null as string | null,
};

function text(value: unknown) {
	let body = JSON.stringify(value, null, 1);
	if (body.length > MAX_RESULT_CHARS) {
		body = `${body.slice(0, MAX_RESULT_CHARS)}\n…[truncated; page with offset/limit]`;
	}
	return [{ type: "text" as const, text: body }];
}

function backend(tool: string, args: Record<string, unknown>) {
	const run = spawnSync(PYTHON, ["-I", TOOLS_PY, "--workspace", WORKSPACE, "tool"], {
		input: JSON.stringify({ tool, args }),
		encoding: "utf8",
		timeout: 120_000,
		maxBuffer: 4 * 1024 * 1024,
		env: { PATH: "/usr/bin:/bin", LC_ALL: "C.UTF-8" },
	});
	if (run.error || run.status !== 0) {
		throw new Error(`tool backend failed: ${run.error?.message ?? run.stderr?.slice(-500) ?? run.status}`);
	}
	const out = JSON.parse(run.stdout) as { ok: boolean; result?: unknown; error?: string };
	if (!out.ok) {
		throw new Error(out.error ?? "refused");
	}
	return { content: text(out.result), details: out.result };
}

function backendTool(name: (typeof BACKEND_TOOLS)[number], label: string, description: string, parameters: any) {
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

const optInt = (description: string) => Type.Optional(Type.Integer({ minimum: 0, description }));

export default function (pi: ExtensionAPI) {
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
	pi.registerTool(
		backendTool(
			"scratch_list",
			"Scratch list",
			"List files in your persistent workspace (memento/ is listed but read-only). Shows quota usage.",
			Type.Object({ path: Type.Optional(Type.String({ description: "directory, default workspace root" })) }),
		),
	);
	pi.registerTool(
		backendTool(
			"scratch_read",
			"Scratch read",
			"Read a text file from your workspace.",
			Type.Object({
				path: Type.String(),
				offset: optInt("byte offset"),
				limit: optInt("bytes, max 65536"),
			}),
		),
	);
	pi.registerTool(
		backendTool(
			"scratch_write",
			"Scratch write",
			"Write a UTF-8 text note into your persistent workspace (max 64 KiB/file, 256 files, 1 MiB total). " +
				"Path components: letters, digits, '_', '-', '.', no leading dot. memento/ is read-only.",
			Type.Object({
				path: Type.String(),
				content: Type.String(),
				mode: Type.Optional(StringEnum(["create", "overwrite", "append"] as const)),
			}),
		),
	);
	pi.registerTool(
		backendTool(
			"scratch_delete",
			"Scratch delete",
			"Delete one file or empty directory from your workspace (not memento/).",
			Type.Object({ path: Type.String() }),
		),
	);
	pi.registerTool(
		backendTool(
			"scratch_commit",
			"Scratch commit",
			"Commit every workspace change (notes and memento) to your memory repository. Uncommitted work is lost at the end of the attempt.",
			Type.Object({ message: Type.String({ description: "one line, max 200 bytes" }) }),
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

	pi.on("tool_call", async (event) => {
		const name = event.toolName;
		if (!ALL_TOOLS.has(name)) return { block: true, reason: `${name} is not a Fishbowl tool` };
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
