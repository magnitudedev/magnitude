import { agentCore, contextSweep, defineExperiment, icn, type ModelArtifactDefinition } from "../src/experiment"

// Every request may generate up to this many tokens; the server context is
// sized as prompt budget + output budget so generation is never squeezed.
const OUTPUT_BUDGET = 32_768
const SMOKE_PROMPT_BUDGET = 8_192
const LONG_CONTEXT_REQUEST_TIMEOUT_MS = 3_600_000

const requestPolicy = (promptBudget: number) => ({
  contextTokensPerSequence: promptBudget + OUTPUT_BUDGET,
  parallelSequences: 1,
  maxOutputTokens: OUTPUT_BUDGET,
  temperature: 1,
  topP: 1,
  seed: 42,
  enableThinking: false as const,
})

// Both variants share the magnitude-icn executable and the installed target
// artifact; the only difference is whether the installed drafter is engaged.
const variants = (target: ModelArtifactDefinition, drafter: ModelArtifactDefinition) => [
  {
    id: "icn-standalone",
    artifact: target,
    engine: icn({ executable: "managed", speculativeDecoding: { kind: "none" } }),
  },
  {
    id: "icn-dflash",
    artifact: target,
    engine: icn({ executable: "managed", speculativeDecoding: { kind: "dflash", draftArtifact: drafter } }),
  },
]

// The comparison is defined by the context windows it tests.
export const defineIcnDflashComparison = (options: {
  readonly id: string
  readonly title: string
  readonly contexts: readonly number[]
  readonly target: ModelArtifactDefinition
  readonly drafter: ModelArtifactDefinition
}) => defineExperiment({
  id: options.id,
  title: options.title,
  suite: contextSweep({ checkpoints: options.contexts, charactersPerToken: 3.5, samplesPerCheckpoint: 1 }),
  requestPolicy: {
    ...requestPolicy(Math.max(...options.contexts)),
    requestTimeoutMs: LONG_CONTEXT_REQUEST_TIMEOUT_MS,
  },
  variants: variants(options.target, options.drafter),
  execution: { variantOrder: "balanced", blocks: 2 },
})

// One isolated request per variant, one block: qualifies loading and drafting.
export const defineIcnDflashSmoke = (options: {
  readonly id: string
  readonly title: string
  readonly target: ModelArtifactDefinition
  readonly drafter: ModelArtifactDefinition
}) => defineExperiment({
  id: options.id,
  title: options.title,
  suite: agentCore({ profile: "smoke" }),
  requestPolicy: requestPolicy(SMOKE_PROMPT_BUDGET),
  variants: variants(options.target, options.drafter),
  execution: { variantOrder: "declared", blocks: 1 },
})
