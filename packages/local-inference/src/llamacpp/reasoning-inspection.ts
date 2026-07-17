import { createHash, randomUUID } from "node:crypto"
import { Effect, Option, Schema } from "effect"
import { ReasoningEffortSchema } from "@magnitudedev/ai"
import type { LlamaApplyTemplateRequest, LlamaApplyTemplateResponse, LlamaServerError } from "./server"
import type { LlamaCppReasoningTemplateInspection } from "../model-properties"
import {
  LLAMA_CPP_NATIVE_REASONING_EFFORT_CANDIDATES,
  llamaCppReasoningDefinitionForNativeValue,
  type LlamaCppNativeReasoningEffort,
} from "./reasoning-policy"
import {
  buildLlamaCppReasoningProfile,
  LlamaCppReasoningTemplateOptionsSchema,
  type LlamaCppReasoningTemplateOptions,
} from "./reasoning-profile"

type LlamaCppInspectionRequestShape = "without_tools" | "with_tools" | "post_tool_result"
type LlamaCppApplyTemplateOutcome =
  | { readonly _tag: "Rendered"; readonly promptHash: string }
  | { readonly _tag: "Rejected"; readonly status: Option.Option<number>; readonly reason: string }

export const LlamaCppVerifiedSymbolicReasoningEffortSchema = Schema.Struct({
  reasoningEffort: ReasoningEffortSchema,
  templateOptions: LlamaCppReasoningTemplateOptionsSchema,
  baselineEquivalent: Schema.Boolean,
})
export type LlamaCppVerifiedSymbolicReasoningEffort =
  typeof LlamaCppVerifiedSymbolicReasoningEffortSchema.Type

/** Verified capabilities of one loaded chat template. This is transient inspection output. */
export const LlamaCppReasoningInspectionFactsSchema = Schema.Struct({
  enableThinkingToggle: Schema.Boolean,
  symbolicEfforts: Schema.Array(LlamaCppVerifiedSymbolicReasoningEffortSchema),
})
export type LlamaCppReasoningInspectionFacts = typeof LlamaCppReasoningInspectionFactsSchema.Type

export class LlamaCppReasoningInspectionError extends Schema.TaggedError<LlamaCppReasoningInspectionError>()(
  "LlamaCppReasoningInspectionError",
  { message: Schema.String },
) {}

export interface LlamaReasoningTemplateRenderer {
  readonly render: (request: LlamaApplyTemplateRequest) => Effect.Effect<LlamaApplyTemplateResponse, LlamaServerError>
}

const SHAPES: readonly LlamaCppInspectionRequestShape[] = ["without_tools", "with_tools", "post_tool_result"]
const emptyTemplateOptions = (): LlamaCppReasoningTemplateOptions => ({
  enableThinking: Option.none(),
  reasoningEffort: Option.none(),
})
const toggleOptions = (enabled: boolean): LlamaCppReasoningTemplateOptions => ({
  enableThinking: Option.some(enabled),
  reasoningEffort: Option.none(),
})
const symbolicOptions = (value: string, enabled: Option.Option<boolean> = Option.none()): LlamaCppReasoningTemplateOptions => ({
  enableThinking: enabled,
  reasoningEffort: Option.some(value),
})
const hash = (value: string): string => createHash("sha256").update(value).digest("hex")
const controlKey = (options: LlamaCppReasoningTemplateOptions): string => JSON.stringify({
  enableThinking: Option.getOrNull(options.enableThinking),
  reasoningEffort: Option.getOrNull(options.reasoningEffort),
})
const key = (shape: LlamaCppInspectionRequestShape, run: number, options: LlamaCppReasoningTemplateOptions): string =>
  `${shape}:${run}:${controlKey(options)}`

const templateKwargs = (options: LlamaCppReasoningTemplateOptions): Readonly<Record<string, unknown>> => ({
  ...Option.match(options.enableThinking, {
    onNone: () => ({}),
    onSome: (enableThinking) => ({ enable_thinking: enableThinking }),
  }),
  ...Option.match(options.reasoningEffort, {
    onNone: () => ({}),
    onSome: (reasoningEffort) => ({ reasoning_effort: reasoningEffort }),
  }),
})

const requestFor = (
  shape: LlamaCppInspectionRequestShape,
  nonce: string,
  options: LlamaCppReasoningTemplateOptions,
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
  const kwargs = templateKwargs(options)
  return {
    messages,
    ...(shape === "without_tools" ? {} : { tools: [tool], toolChoice: "auto" }),
    ...(Object.keys(kwargs).length === 0 ? {} : { chatTemplateKwargs: kwargs }),
  }
}

export const inspectLlamaCppReasoning = (
  renderer: LlamaReasoningTemplateRenderer,
): Effect.Effect<LlamaCppReasoningInspectionFacts, LlamaCppReasoningInspectionError> => Effect.gen(function* () {
  const outcomes = new Map<string, LlamaCppApplyTemplateOutcome>()
  const invalidValueA = `__magnitude_invalid_a_${randomUUID()}__`
  const invalidValueB = `__magnitude_invalid_b_${randomUUID()}__`
  const baseline = emptyTemplateOptions()
  const off = toggleOptions(false)
  const on = toggleOptions(true)
  const baseCases = [
    off,
    on,
    ...LLAMA_CPP_NATIVE_REASONING_EFFORT_CANDIDATES.map((value) => symbolicOptions(value)),
    symbolicOptions(invalidValueA),
    symbolicOptions(invalidValueB),
  ]

  const inspectCase = (
    shape: LlamaCppInspectionRequestShape,
    run: number,
    nonce: string,
    options: LlamaCppReasoningTemplateOptions,
  ) => renderer.render(requestFor(shape, nonce, options)).pipe(
    Effect.match({
      onFailure: (error): LlamaCppApplyTemplateOutcome => ({
        _tag: "Rejected",
        status: error.status,
        reason: error.reason,
      }),
      onSuccess: ({ prompt }): LlamaCppApplyTemplateOutcome => prompt.includes(`reasoning-probe-${nonce}`)
        ? { _tag: "Rendered", promptHash: hash(prompt) }
        : { _tag: "Rejected", status: Option.none(), reason: "The template did not preserve the inspection nonce" },
    }),
    Effect.tap((outcome) => Effect.sync(() => outcomes.set(key(shape, run, options), outcome))),
  )

  for (let run = 1; run <= 2; run++) {
    const nonce = randomUUID()
    for (const shape of SHAPES) {
      const before = yield* inspectCase(shape, run, nonce, baseline)
      for (const inspectionCase of baseCases) yield* inspectCase(shape, run, nonce, inspectionCase)
      const after = yield* inspectCase(shape, run, nonce, baseline)
      if (before._tag !== "Rendered" || after._tag !== "Rendered" || before.promptHash !== after.promptHash) {
        return yield* new LlamaCppReasoningInspectionError({ message: "The /apply-template baseline was unstable" })
      }
    }
  }

  const cells = SHAPES.flatMap((shape) => [1, 2].map((run) => ({ shape, run })))
  const outcome = (cell: { shape: LlamaCppInspectionRequestShape; run: number }, options: LlamaCppReasoningTemplateOptions) =>
    outcomes.get(key(cell.shape, cell.run, options))
  const allRendered = (options: LlamaCppReasoningTemplateOptions) =>
    cells.every((cell) => outcome(cell, options)?._tag === "Rendered")
  const allRejected = (options: LlamaCppReasoningTemplateOptions) =>
    cells.every((cell) => outcome(cell, options)?._tag === "Rejected")
  const equal = (left: LlamaCppReasoningTemplateOptions, right: LlamaCppReasoningTemplateOptions) => cells.every((cell) => {
    const a = outcome(cell, left)
    const b = outcome(cell, right)
    return a?._tag === "Rendered" && b?._tag === "Rendered" && a.promptHash === b.promptHash
  })

  const enableThinkingToggle = allRendered(off) && allRendered(on) && !equal(off, on)
  const invalidA = symbolicOptions(invalidValueA)
  const invalidB = symbolicOptions(invalidValueB)
  const invalidsReject = allRejected(invalidA) && allRejected(invalidB)
  const invalidsShareFallback = allRendered(invalidA) && allRendered(invalidB) && equal(invalidA, invalidB)
  const openPassThrough = allRendered(invalidA) && allRendered(invalidB) && !equal(invalidA, invalidB)
  const discovered = new Map<typeof ReasoningEffortSchema.Type, LlamaCppVerifiedSymbolicReasoningEffort>()

  const add = (nativeValue: LlamaCppNativeReasoningEffort, options: LlamaCppReasoningTemplateOptions) => {
    const definition = llamaCppReasoningDefinitionForNativeValue(nativeValue)
    const candidate = {
      reasoningEffort: definition.reasoningEffort,
      templateOptions: options,
      baselineEquivalent: equal(options, baseline),
    }
    const existing = discovered.get(definition.reasoningEffort)
    if (!existing) {
      discovered.set(definition.reasoningEffort, candidate)
      return true
    }
    return equal(existing.templateOptions, options)
  }

  if (!openPassThrough && (invalidsReject || invalidsShareFallback)) {
    for (const nativeValue of LLAMA_CPP_NATIVE_REASONING_EFFORT_CANDIDATES) {
      const options = symbolicOptions(nativeValue)
      if (!allRendered(options)) continue
      if (invalidsShareFallback && equal(options, invalidA)) continue
      if (!add(nativeValue, options)) {
        const effort = llamaCppReasoningDefinitionForNativeValue(nativeValue).reasoningEffort
        return yield* new LlamaCppReasoningInspectionError({ message: `Distinct controls mapped to effort ${effort}` })
      }
    }
  }

  if (enableThinkingToggle) {
    for (const enabled of [false, true]) {
      const branch = toggleOptions(enabled)
      const combinedInvalidA = symbolicOptions(invalidValueA, Option.some(enabled))
      const combinedInvalidB = symbolicOptions(invalidValueB, Option.some(enabled))
      for (let run = 1; run <= 2; run++) {
        const nonce = randomUUID()
        for (const shape of SHAPES) {
          const before = yield* inspectCase(shape, run, nonce, branch)
          yield* inspectCase(shape, run, nonce, combinedInvalidA)
          const middle = yield* inspectCase(shape, run, nonce, branch)
          yield* inspectCase(shape, run, nonce, combinedInvalidB)
          const after = yield* inspectCase(shape, run, nonce, branch)
          if (before._tag !== "Rendered" || middle._tag !== "Rendered" || after._tag !== "Rendered"
            || before.promptHash !== middle.promptHash || middle.promptHash !== after.promptHash) {
            return yield* new LlamaCppReasoningInspectionError({ message: "The interaction-inspection baseline was unstable" })
          }
        }
      }
      const combinedPassThrough = allRendered(combinedInvalidA)
        && allRendered(combinedInvalidB)
        && !equal(combinedInvalidA, combinedInvalidB)
      const combinedClosedDomain = (allRejected(combinedInvalidA) && allRejected(combinedInvalidB))
        || (allRendered(combinedInvalidA) && allRendered(combinedInvalidB) && equal(combinedInvalidA, combinedInvalidB))
      if (combinedPassThrough || !combinedClosedDomain) continue

      for (const nativeValue of LLAMA_CPP_NATIVE_REASONING_EFFORT_CANDIDATES) {
        const options = symbolicOptions(nativeValue, Option.some(enabled))
        let distinctAcrossCorpus = true
        for (let run = 1; run <= 2; run++) {
          const nonce = randomUUID()
          for (const shape of SHAPES) {
            const before = yield* inspectCase(shape, run, nonce, branch)
            const candidate = yield* inspectCase(shape, run, nonce, options)
            const after = yield* inspectCase(shape, run, nonce, branch)
            if (before._tag !== "Rendered" || after._tag !== "Rendered" || before.promptHash !== after.promptHash) {
              return yield* new LlamaCppReasoningInspectionError({ message: "The interaction-inspection baseline was unstable" })
            }
            if (candidate._tag !== "Rendered" || candidate.promptHash === before.promptHash) distinctAcrossCorpus = false
          }
        }
        if (!distinctAcrossCorpus) continue
        const definition = llamaCppReasoningDefinitionForNativeValue(nativeValue)
        const preferredBranch = definition.semantics._tag === "Disabled" ? !enabled : enabled
        if (!discovered.has(definition.reasoningEffort) || preferredBranch) {
          discovered.set(definition.reasoningEffort, {
            reasoningEffort: definition.reasoningEffort,
            templateOptions: options,
            baselineEquivalent: equal(options, baseline),
          })
        }
      }
    }
  }

  return {
    enableThinkingToggle,
    symbolicEfforts: [...discovered.values()],
  }
})

/** Inspect a loaded template and resolve its complete public reasoning profile. */
export const discoverLlamaCppReasoning = (
  renderer: LlamaReasoningTemplateRenderer,
): Effect.Effect<LlamaCppReasoningTemplateInspection, LlamaCppReasoningInspectionError> =>
  inspectLlamaCppReasoning(renderer).pipe(
    Effect.map((facts) => ({ profile: buildLlamaCppReasoningProfile(facts) })),
  )
