import { Schema } from "effect"
import { ReasoningEffortSchema, type ReasoningEffort } from "@magnitudedev/ai"

export const LlamaCppNativeReasoningEffortSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(256),
  Schema.brand("LlamaCppNativeReasoningEffort"),
)
export type LlamaCppNativeReasoningEffort = typeof LlamaCppNativeReasoningEffortSchema.Type

export const LlamaCppReasoningSemanticsSchema = Schema.Union(
  Schema.TaggedStruct("Disabled", {}),
  Schema.TaggedStruct("Budgeted", { tokens: Schema.NonNegativeInt }),
  Schema.TaggedStruct("Unbounded", {}),
)
export type LlamaCppReasoningSemantics = typeof LlamaCppReasoningSemanticsSchema.Type

export const LlamaCppReasoningEffortDefinitionSchema = Schema.Struct({
  reasoningEffort: ReasoningEffortSchema,
  nativeValues: Schema.Array(LlamaCppNativeReasoningEffortSchema).pipe(Schema.minItems(1)),
  semantics: LlamaCppReasoningSemanticsSchema,
  preferredDefault: Schema.Boolean,
})
export type LlamaCppReasoningEffortDefinition = typeof LlamaCppReasoningEffortDefinitionSchema.Type

export const LlamaCppReasoningPolicySchema = Schema.Array(LlamaCppReasoningEffortDefinitionSchema).pipe(
  Schema.minItems(1),
  Schema.filter((definitions) =>
    new Set(definitions.map(({ reasoningEffort }) => reasoningEffort)).size === definitions.length,
  { message: () => "llama.cpp reasoning policy must contain unique public efforts" }),
  Schema.filter((definitions) => {
    const nativeValues = definitions.flatMap((definition) => definition.nativeValues)
    return new Set(nativeValues).size === nativeValues.length
  }, { message: () => "llama.cpp reasoning policy must contain unique native values" }),
  Schema.filter((definitions) =>
    definitions.filter(({ preferredDefault }) => preferredDefault).length === 1,
  { message: () => "llama.cpp reasoning policy must contain one preferred default" }),
  Schema.filter((definitions) =>
    definitions.filter(({ semantics }) => semantics._tag === "Disabled").length === 1,
  { message: () => "llama.cpp reasoning policy must contain one disabled effort" }),
  Schema.filter((definitions) => definitions.every((definition) =>
    !definition.preferredDefault || definition.semantics._tag !== "Disabled"),
  { message: () => "the preferred default must enable reasoning" }),
)

const native = LlamaCppNativeReasoningEffortSchema.make
const effort = ReasoningEffortSchema.make

/** The only source of llama.cpp effort names, aliases, ordering, defaults, and budgets. */
export const LLAMA_CPP_REASONING_POLICY = Schema.decodeUnknownSync(LlamaCppReasoningPolicySchema)([
  {
    reasoningEffort: effort("none"),
    nativeValues: [native("none"), native("off"), native("no_think")],
    semantics: { _tag: "Disabled" },
    preferredDefault: false,
  },
  {
    reasoningEffort: effort("minimal"),
    nativeValues: [native("minimal")],
    semantics: { _tag: "Budgeted", tokens: 1_024 },
    preferredDefault: false,
  },
  {
    reasoningEffort: effort("low"),
    nativeValues: [native("low")],
    semantics: { _tag: "Budgeted", tokens: 1_024 },
    preferredDefault: false,
  },
  {
    reasoningEffort: effort("medium"),
    nativeValues: [native("medium")],
    semantics: { _tag: "Budgeted", tokens: 2_048 },
    preferredDefault: false,
  },
  {
    reasoningEffort: effort("high"),
    nativeValues: [native("high")],
    semantics: { _tag: "Budgeted", tokens: 4_096 },
    preferredDefault: true,
  },
  {
    reasoningEffort: effort("extra_high"),
    nativeValues: [native("extra_high"), native("extra-high"), native("xhigh"), native("very_high")],
    semantics: { _tag: "Budgeted", tokens: 8_192 },
    preferredDefault: false,
  },
  {
    reasoningEffort: effort("max"),
    nativeValues: [native("max")],
    semantics: { _tag: "Unbounded" },
    preferredDefault: false,
  },
])

const definitionsByNativeValue = new Map(
  LLAMA_CPP_REASONING_POLICY.flatMap((definition) =>
    definition.nativeValues.map((nativeValue) => [nativeValue, definition] as const)),
)
const definitionsByReasoningEffort = new Map(
  LLAMA_CPP_REASONING_POLICY.map((definition) => [definition.reasoningEffort, definition] as const),
)

export const LLAMA_CPP_NATIVE_REASONING_EFFORT_CANDIDATES =
  LLAMA_CPP_REASONING_POLICY.flatMap((definition) => definition.nativeValues)

export const llamaCppReasoningDefinitionForNativeValue = (
  nativeValue: LlamaCppNativeReasoningEffort,
): LlamaCppReasoningEffortDefinition => definitionsByNativeValue.get(nativeValue)!

export const llamaCppReasoningDefinitionForEffort = (
  reasoningEffort: ReasoningEffort,
): LlamaCppReasoningEffortDefinition => definitionsByReasoningEffort.get(reasoningEffort)!

export const llamaCppDisabledReasoningDefinition = (): LlamaCppReasoningEffortDefinition =>
  LLAMA_CPP_REASONING_POLICY.find(({ semantics }) => semantics._tag === "Disabled")!

export const llamaCppPreferredReasoningDefinition = (): LlamaCppReasoningEffortDefinition =>
  LLAMA_CPP_REASONING_POLICY.find(({ preferredDefault }) => preferredDefault)!
