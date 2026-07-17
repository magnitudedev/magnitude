import { createHash, randomUUID } from "node:crypto"
import { Effect, Option, Schema } from "effect"
import type { LlamaApplyTemplateRequest, LlamaApplyTemplateResponse, LlamaServerError } from "./server"
import {
  type LlamaCppReasoningOption,
  type LlamaCppReasoningTemplateInspection,
  type LlamaCppTemplateReasoningControl,
} from "../model-properties"
export * from "../model-properties"

type LlamaCppProbeRequestShape = "without_tools" | "with_tools" | "post_tool_result"
type LlamaCppApplyTemplateOutcome =
  | { readonly _tag: "Rendered"; readonly promptHash: string }
  | { readonly _tag: "Rejected"; readonly status: Option.Option<number>; readonly reason: string }

export class LlamaCppReasoningDiscoveryError extends Schema.TaggedError<LlamaCppReasoningDiscoveryError>()(
  "LlamaCppReasoningDiscoveryError",
  { message: Schema.String },
) {}

export const LLAMA_REASONING_PROBE_PROTOCOL_VERSION = "llama-apply-template-v1"
const SYMBOLIC_CANDIDATES = ["none", "no_think", "minimal", "low", "medium", "high", "max"] as const
const SHAPES: readonly LlamaCppProbeRequestShape[] = ["without_tools", "with_tools", "post_tool_result"]

const hash = (value: string): string => createHash("sha256").update(value).digest("hex")
const key = (shape: LlamaCppProbeRequestShape, run: number, control: LlamaCppTemplateReasoningControl): string =>
  `${shape}:${run}:${JSON.stringify(control)}`
const kwargsFor = (control: LlamaCppTemplateReasoningControl): Readonly<Record<string, unknown>> | undefined => {
  switch (control._tag) {
    case "Omitted": return undefined
    case "EnableThinkingKwarg": return { enable_thinking: control.enabled }
    case "ReasoningEffortKwarg": return { reasoning_effort: control.value }
    case "EnableThinkingAndReasoningEffortKwarg": return { enable_thinking: control.enabled, reasoning_effort: control.value }
  }
}

const requestFor = (
  shape: LlamaCppProbeRequestShape,
  nonce: string,
  control: LlamaCppTemplateReasoningControl,
): LlamaApplyTemplateRequest => {
  const user = { role: "user", content: `reasoning-probe-${nonce}` }
  const tool = {
    type: "function",
    function: { name: "probe_tool", description: "Probe tool", parameters: { type: "object", properties: {} } },
  }
  const messages: readonly Record<string, unknown>[] = shape === "post_tool_result"
    ? [
        { role: "system", content: "You are Magnitude's coding agent." },
        user,
        { role: "assistant", content: "", tool_calls: [{ id: "probe_call", type: "function", function: { name: "probe_tool", arguments: "{}" } }] },
        { role: "tool", tool_call_id: "probe_call", content: "probe-result" },
      ]
    : [{ role: "system", content: "You are Magnitude's coding agent." }, user]
  return {
    messages,
    ...(shape === "without_tools" ? {} : { tools: [tool], toolChoice: "auto" }),
    ...(kwargsFor(control) === undefined ? {} : { chatTemplateKwargs: kwargsFor(control)! }),
  }
}

const semanticEffort = (value: string): string => {
  if (value === "none" || value === "no_think") return "None"
  return value.charAt(0).toUpperCase() + value.slice(1)
}

export interface LlamaReasoningTemplateRenderer {
  readonly render: (request: LlamaApplyTemplateRequest) => Effect.Effect<LlamaApplyTemplateResponse, LlamaServerError>
}

export const discoverLlamaCppReasoning = (
  renderer: LlamaReasoningTemplateRenderer,
): Effect.Effect<LlamaCppReasoningTemplateInspection, LlamaCppReasoningDiscoveryError> => Effect.gen(function* () {
  const outcomes = new Map<string, LlamaCppApplyTemplateOutcome>()
  const sentinelA = `__magnitude_invalid_a_${randomUUID()}__`
  const sentinelB = `__magnitude_invalid_b_${randomUUID()}__`
  const baseControls: LlamaCppTemplateReasoningControl[] = [
    { _tag: "EnableThinkingKwarg", enabled: false },
    { _tag: "EnableThinkingKwarg", enabled: true },
    ...SYMBOLIC_CANDIDATES.map((value) => ({ _tag: "ReasoningEffortKwarg" as const, value })),
    { _tag: "ReasoningEffortKwarg", value: sentinelA },
    { _tag: "ReasoningEffortKwarg", value: sentinelB },
  ]

  const observe = (shape: LlamaCppProbeRequestShape, run: number, nonce: string, control: LlamaCppTemplateReasoningControl) =>
    renderer.render(requestFor(shape, nonce, control)).pipe(
      Effect.match({
        onFailure: (error): LlamaCppApplyTemplateOutcome => ({
          _tag: "Rejected",
          status: error.status,
          reason: error.reason,
        }),
        onSuccess: ({ prompt }): LlamaCppApplyTemplateOutcome => prompt.includes(`reasoning-probe-${nonce}`)
          ? { _tag: "Rendered", promptHash: hash(prompt) }
          : { _tag: "Rejected", status: Option.none(), reason: "The template did not preserve the probe nonce" },
      }),
      Effect.tap((outcome) => Effect.sync(() => {
        outcomes.set(key(shape, run, control), outcome)
      })),
    )

  for (let run = 1; run <= 2; run++) {
    const nonce = randomUUID()
    for (const shape of SHAPES) {
      const before = yield* observe(shape, run, nonce, { _tag: "Omitted" })
      for (const control of baseControls) {
        yield* observe(shape, run, nonce, control)
      }
      const after = yield* observe(shape, run, nonce, { _tag: "Omitted" })
      if (before._tag !== "Rendered" || after._tag !== "Rendered" || before.promptHash !== after.promptHash) {
        return yield* new LlamaCppReasoningDiscoveryError({ message: "The /apply-template baseline was unstable" })
      }
    }
  }

  const cells = SHAPES.flatMap((shape) => [1, 2].map((run) => ({ shape, run })))
  const outcome = (cell: { shape: LlamaCppProbeRequestShape; run: number }, control: LlamaCppTemplateReasoningControl) =>
    outcomes.get(key(cell.shape, cell.run, control))
  const allRendered = (control: LlamaCppTemplateReasoningControl) => cells.every((cell) => outcome(cell, control)?._tag === "Rendered")
  const equal = (left: LlamaCppTemplateReasoningControl, right: LlamaCppTemplateReasoningControl) => cells.every((cell) => {
    const a = outcome(cell, left)
    const b = outcome(cell, right)
    return a?._tag === "Rendered" && b?._tag === "Rendered" && a.promptHash === b.promptHash
  })
  const allRejected = (control: LlamaCppTemplateReasoningControl) => cells.every((cell) => outcome(cell, control)?._tag === "Rejected")

  const omitted: LlamaCppTemplateReasoningControl = { _tag: "Omitted" }
  const off: LlamaCppTemplateReasoningControl = { _tag: "EnableThinkingKwarg", enabled: false }
  const on: LlamaCppTemplateReasoningControl = { _tag: "EnableThinkingKwarg", enabled: true }
  const efforts: LlamaCppReasoningOption[] = [{ reasoningEffort: "Default", control: omitted }]
  const add = (reasoningEffort: string, control: LlamaCppTemplateReasoningControl) => {
    const existing = efforts.find((entry) => entry.reasoningEffort === reasoningEffort)
    if (!existing) {
      efforts.push({ reasoningEffort, control })
      return true
    }
    if (!equal(existing.control, control)) return false
    return true
  }

  const toggleEffective = allRendered(off) && allRendered(on) && !equal(off, on)
  if (toggleEffective) {
    if (!equal(off, omitted) && !add("None", off)) {
      return yield* new LlamaCppReasoningDiscoveryError({ message: "Distinct toggle branches mapped to the same semantic effort" })
    }
    if (!equal(on, omitted) && !add("Enabled", on)) {
      return yield* new LlamaCppReasoningDiscoveryError({ message: "Distinct toggle branches mapped to the same semantic effort" })
    }
  }

  const sentinelControlA: LlamaCppTemplateReasoningControl = { _tag: "ReasoningEffortKwarg", value: sentinelA }
  const sentinelControlB: LlamaCppTemplateReasoningControl = { _tag: "ReasoningEffortKwarg", value: sentinelB }
  const sentinelsReject = allRejected(sentinelControlA) && allRejected(sentinelControlB)
  const sentinelsShareFallback = allRendered(sentinelControlA) && allRendered(sentinelControlB) && equal(sentinelControlA, sentinelControlB)
  const passThrough = allRendered(sentinelControlA) && allRendered(sentinelControlB) && !equal(sentinelControlA, sentinelControlB)

  if (!passThrough && (sentinelsReject || sentinelsShareFallback)) {
    for (const value of SYMBOLIC_CANDIDATES) {
      const control: LlamaCppTemplateReasoningControl = { _tag: "ReasoningEffortKwarg", value }
      if (!allRendered(control) || equal(control, omitted)) continue
      if (sentinelsShareFallback && equal(control, sentinelControlA)) continue
      if (!add(semanticEffort(value), control)) {
        return yield* new LlamaCppReasoningDiscoveryError({ message: `Distinct controls mapped to effort ${semanticEffort(value)}` })
      }
    }
  }

  // Probe symbolic controls under each effective toggle branch. Only add a
  // combined control when the standalone symbolic value was not already valid.
  if (toggleEffective) {
    for (const enabled of [false, true]) {
      const combinedSentinelA: LlamaCppTemplateReasoningControl = { _tag: "EnableThinkingAndReasoningEffortKwarg", enabled, value: sentinelA }
      const combinedSentinelB: LlamaCppTemplateReasoningControl = { _tag: "EnableThinkingAndReasoningEffortKwarg", enabled, value: sentinelB }
      for (let run = 1; run <= 2; run++) {
        const nonce = randomUUID()
        for (const shape of SHAPES) {
          const branch: LlamaCppTemplateReasoningControl = { _tag: "EnableThinkingKwarg", enabled }
          const before = yield* observe(shape, run, nonce, branch)
          yield* observe(shape, run, nonce, combinedSentinelA)
          const middle = yield* observe(shape, run, nonce, branch)
          yield* observe(shape, run, nonce, combinedSentinelB)
          const after = yield* observe(shape, run, nonce, branch)
          if (before._tag !== "Rendered" || middle._tag !== "Rendered" || after._tag !== "Rendered"
            || before.promptHash !== middle.promptHash || middle.promptHash !== after.promptHash) {
            return yield* new LlamaCppReasoningDiscoveryError({ message: "The interaction-probe baseline was unstable" })
          }
        }
      }
      const combinedPassThrough = allRendered(combinedSentinelA)
        && allRendered(combinedSentinelB)
        && !equal(combinedSentinelA, combinedSentinelB)
      const combinedClosedDomain = (allRejected(combinedSentinelA) && allRejected(combinedSentinelB))
        || (allRendered(combinedSentinelA) && allRendered(combinedSentinelB) && equal(combinedSentinelA, combinedSentinelB))
      if (combinedPassThrough || !combinedClosedDomain) continue
      for (const value of SYMBOLIC_CANDIDATES) {
        const combined: LlamaCppTemplateReasoningControl = { _tag: "EnableThinkingAndReasoningEffortKwarg", enabled, value }
        let distinctAcrossCorpus = true
        for (let run = 1; run <= 2; run++) {
          const nonce = randomUUID()
          for (const shape of SHAPES) {
            const before = yield* observe(shape, run, nonce, { _tag: "EnableThinkingKwarg", enabled })
            const candidate = yield* observe(shape, run, nonce, combined)
            const after = yield* observe(shape, run, nonce, { _tag: "EnableThinkingKwarg", enabled })
            if (before._tag !== "Rendered" || after._tag !== "Rendered" || before.promptHash !== after.promptHash) {
              return yield* new LlamaCppReasoningDiscoveryError({ message: "The interaction-probe baseline was unstable" })
            }
            if (candidate._tag !== "Rendered" || candidate.promptHash === before.promptHash) distinctAcrossCorpus = false
          }
        }
        if (distinctAcrossCorpus) {
          const name = semanticEffort(value)
          if (!efforts.some((entry) => entry.reasoningEffort === name)) efforts.push({ reasoningEffort: name, control: combined })
        }
      }
    }
  }

  return {
    probeProtocolVersion: LLAMA_REASONING_PROBE_PROTOCOL_VERSION,
    options: efforts,
  }
})
