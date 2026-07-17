import { Option, Schema } from "effect"
import { ReasoningEffortSchema, type ReasoningEffort } from "@magnitudedev/ai"
import type { LlamaCppReasoningInspectionFacts } from "./reasoning-inspection"
import {
  llamaCppReasoningDefinitionForEffort,
  llamaCppDisabledReasoningDefinition,
  llamaCppPreferredReasoningDefinition,
  type LlamaCppReasoningEffortDefinition,
} from "./reasoning-policy"

const LlamaCppReasoningTemplateOptionsEncodedSchema = Schema.Struct({
  enableThinking: Schema.optionalWith(Schema.NullOr(Schema.Boolean), { exact: true }),
  reasoningEffort: Schema.optionalWith(
    Schema.NullOr(Schema.String.pipe(Schema.minLength(1), Schema.maxLength(256))),
    { exact: true },
  ),
})

const LlamaCppReasoningTemplateOptionsTypeSchema = Schema.Struct({
  enableThinking: Schema.OptionFromSelf(Schema.Boolean),
  reasoningEffort: Schema.OptionFromSelf(Schema.String.pipe(Schema.minLength(1), Schema.maxLength(256))),
})

/**
 * Omits absent template controls in newly encoded JSON while accepting the
 * nullable representation written by development builds before this format
 * was corrected.
 */
export const LlamaCppReasoningTemplateOptionsSchema = Schema.transform(
  LlamaCppReasoningTemplateOptionsEncodedSchema,
  LlamaCppReasoningTemplateOptionsTypeSchema,
  {
    decode: (encoded) => ({
      enableThinking: Option.fromNullable(encoded.enableThinking),
      reasoningEffort: Option.fromNullable(encoded.reasoningEffort),
    }),
    encode: (options) => ({
      ...Option.match(options.enableThinking, {
        onNone: () => ({}),
        onSome: (enableThinking) => ({ enableThinking }),
      }),
      ...Option.match(options.reasoningEffort, {
        onNone: () => ({}),
        onSome: (reasoningEffort) => ({ reasoningEffort }),
      }),
    }),
  },
)
export type LlamaCppReasoningTemplateOptions = typeof LlamaCppReasoningTemplateOptionsSchema.Type

export const LlamaCppThinkingBudgetSchema = Schema.Union(
  Schema.TaggedStruct("Disabled", {}),
  Schema.TaggedStruct("Enabled", { tokens: Schema.NonNegativeInt }),
)
export type LlamaCppThinkingBudget = typeof LlamaCppThinkingBudgetSchema.Type

export const LlamaCppReasoningEffortMappingSchema = Schema.Struct({
  reasoningEffort: ReasoningEffortSchema,
  templateOptions: LlamaCppReasoningTemplateOptionsSchema,
  thinkingBudget: LlamaCppThinkingBudgetSchema,
})
export type LlamaCppReasoningEffortMapping = typeof LlamaCppReasoningEffortMappingSchema.Type

const LlamaCppReasoningEffortMappingsSchema = Schema.Array(LlamaCppReasoningEffortMappingSchema).pipe(
  Schema.minItems(1),
  Schema.filter(
    (mappings) => new Set(mappings.map((mapping) => mapping.reasoningEffort)).size === mappings.length,
    { message: () => "Reasoning profile efforts must be unique" },
  ),
)

export const LlamaCppReasoningProfileSchema = Schema.Struct({
  defaultReasoningEffort: ReasoningEffortSchema,
  effortMappings: LlamaCppReasoningEffortMappingsSchema,
}).pipe(Schema.filter(
  (profile) => profile.effortMappings.some(
    (mapping) => mapping.reasoningEffort === profile.defaultReasoningEffort,
  ),
  { message: () => "Reasoning profile must contain its default effort" },
))
export type LlamaCppReasoningProfile = typeof LlamaCppReasoningProfileSchema.Type

export const resolveLlamaCppReasoningEffort = (
  profile: LlamaCppReasoningProfile,
  reasoningEffort: ReasoningEffort,
): Option.Option<LlamaCppReasoningEffortMapping> => Option.fromNullable(
  profile.effortMappings.find((mapping) => mapping.reasoningEffort === reasoningEffort),
)

const thinkingBudget = (
  definition: LlamaCppReasoningEffortDefinition,
): LlamaCppThinkingBudget => definition.semantics._tag === "Budgeted"
  ? { _tag: "Enabled", tokens: definition.semantics.tokens }
  : { _tag: "Disabled" }

const mapping = (
  reasoningEffort: ReasoningEffort,
  templateOptions: LlamaCppReasoningTemplateOptions,
): LlamaCppReasoningEffortMapping => ({
  reasoningEffort,
  templateOptions,
  thinkingBudget: thinkingBudget(llamaCppReasoningDefinitionForEffort(reasoningEffort)),
})

export const buildLlamaCppReasoningProfile = (
  facts: LlamaCppReasoningInspectionFacts,
): LlamaCppReasoningProfile => {
  if (facts.symbolicEfforts.length === 0) {
    const disabled = llamaCppDisabledReasoningDefinition()
    if (!facts.enableThinkingToggle) {
      return {
        defaultReasoningEffort: disabled.reasoningEffort,
        effortMappings: [mapping(disabled.reasoningEffort, {
          enableThinking: Option.none(),
          reasoningEffort: Option.none(),
        })],
      }
    }
    const enabled = llamaCppPreferredReasoningDefinition()
    return {
      defaultReasoningEffort: enabled.reasoningEffort,
      effortMappings: [
        mapping(disabled.reasoningEffort, {
          enableThinking: Option.some(false),
          reasoningEffort: Option.none(),
        }),
        mapping(enabled.reasoningEffort, {
          enableThinking: Option.some(true),
          reasoningEffort: Option.none(),
        }),
      ],
    }
  }

  const hasDisabledEffort = facts.symbolicEfforts.some(({ reasoningEffort }) =>
    llamaCppReasoningDefinitionForEffort(reasoningEffort).semantics._tag === "Disabled")
  const symbolicMappings = facts.symbolicEfforts.map(({ reasoningEffort, templateOptions }) => {
    const definition = llamaCppReasoningDefinitionForEffort(reasoningEffort)
    return mapping(reasoningEffort, {
      enableThinking: facts.enableThinkingToggle
        ? Option.some(definition.semantics._tag !== "Disabled")
        : templateOptions.enableThinking,
      reasoningEffort: templateOptions.reasoningEffort,
    })
  })
  const disabled = llamaCppDisabledReasoningDefinition()
  const effortMappings = facts.enableThinkingToggle && !hasDisabledEffort
    ? [mapping(disabled.reasoningEffort, {
        enableThinking: Option.some(false),
        reasoningEffort: Option.none(),
      }), ...symbolicMappings]
    : symbolicMappings
  const preferred = effortMappings.find(({ reasoningEffort }) =>
    llamaCppReasoningDefinitionForEffort(reasoningEffort).preferredDefault)
  const baseline = facts.symbolicEfforts.find(({ baselineEquivalent }) => baselineEquivalent)

  return {
    defaultReasoningEffort: preferred?.reasoningEffort
      ?? baseline?.reasoningEffort
      ?? effortMappings[0]!.reasoningEffort,
    effortMappings,
  }
}
