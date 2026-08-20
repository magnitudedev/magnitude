import type { ChatCompletionsRequestMessage, ChatTool } from "@magnitudedev/ai"
import type { JsonRecord } from "@magnitudedev/utils/schema"

export const CRITERIA = ["responsiveness", "prefill", "decode", "memory-usage", "distribution"] as const
export type Criterion = typeof CRITERIA[number]

export const WORKLOAD_PATTERNS = [
  "single-request",
  "sequential-session",
  "independent-concurrency",
  "forked-concurrency",
  "concurrency-pressure",
  "memory-pressure",
  "context-scaling",
] as const
export type WorkloadPattern = typeof WORKLOAD_PATTERNS[number]

export type ProfileName = "smoke" | "standard" | "full"
export type Outcome = "valid" | "invalid" | "rejected" | "protocol-error" | "error" | "timeout" | "cancelled"

export interface ExpectedToolCall {
  readonly name: string
  readonly arguments: Readonly<Record<string, readonly unknown[]>>
}

export interface BfclProvenance {
  readonly commit: string
  readonly questionFile: string
  readonly answerFile: string
  readonly recordId: string
}

export interface Fixture {
  readonly id: string
  readonly messages: readonly ChatCompletionsRequestMessage[]
  readonly tools: readonly ChatTool[]
  readonly expected: readonly ExpectedToolCall[]
  readonly canonicalAssistant: ChatCompletionsRequestMessage
  readonly canonicalToolMessages: readonly ChatCompletionsRequestMessage[]
  readonly provenance: BfclProvenance
}

export interface PlannedRequest {
  readonly id: string
  readonly fixtureId: string
  readonly messages: readonly ChatCompletionsRequestMessage[]
  readonly tools: readonly ChatTool[]
  readonly expected: readonly ExpectedToolCall[]
  readonly releaseOffsetMs: number
  readonly dependsOn: readonly string[]
  readonly maxOutputTokens: number
  readonly temperature?: number
  readonly topP?: number
  readonly seed?: number
  readonly enableThinking?: false
  readonly sessionId?: string
  readonly prefixGroup?: string
  readonly phase?: "setup" | "measure"
}

export interface TrialDefinition {
  readonly id: string
  readonly pattern: WorkloadPattern
  readonly criteria: readonly Criterion[]
  readonly checkpoint: string
  readonly repetition: number
  readonly state: "cache-disjoint" | "resident-prefix"
  readonly requests: readonly PlannedRequest[]
  readonly contextTarget?: {
    readonly tokens: number
    readonly characters: number
    readonly plannedCharacters: number
    readonly semanticDepth: number
  }
}

export interface TrialPlan {
  readonly profile: ProfileName | "context-sweep"
  readonly model: LogicalModelIdentity
  readonly servingPolicy: ServingPolicy
  readonly corpusDigest: string
  readonly createdAt: string
  readonly warmup: PlannedRequest
  readonly trials: readonly TrialDefinition[]
  readonly digest: string
}

export interface ServingPolicy {
  readonly contextTokensPerSequence: number
  readonly parallelSequences: number
  readonly temperature?: number
  readonly topP?: number
  readonly seed?: number
  readonly enableThinking?: false
}

export interface LogicalModelIdentity {
  readonly id: string
  readonly contextLimit: number
}

export interface ModelIdentity extends LogicalModelIdentity {
  readonly artifactPath: string
  readonly artifactSha256: string
  readonly chatTemplateDigest: string
}

export interface ExistingTarget {
  readonly kind: "existing"
  readonly id: string
  readonly endpoint: string
  readonly servedModel: string
  readonly apiKey?: string
  readonly requestBody?: JsonRecord
  readonly requestTimeoutMs?: number
  readonly parallelSequences: number
  readonly artifact?: {
    readonly kind: "gguf" | "mlx"
    readonly repository: string
    readonly revision: string
    readonly quantization: string
  }
}

export interface ManagedTarget {
  readonly kind: "managed"
  readonly engine: "icn" | "llama.cpp" | "mlx-lm" | "mlx-vlm" | "vllm" | "sglang" | "generic"
  readonly id: string
  readonly executable: string
  readonly args: readonly string[]
  readonly endpoint: string
  readonly servedModel: string
  readonly apiKey?: string
  readonly requestBody?: JsonRecord
  readonly requestTimeoutMs?: number
  readonly env?: Readonly<Record<string, string>>
  readonly readinessPath?: string
  readonly cwd?: string
  readonly modelLoad?: {
    readonly artifactSha256: string
    readonly contextLimit: number
    readonly instanceId: string
  }
  readonly parallelSequences: number
  readonly artifact?: {
    readonly kind: "gguf" | "mlx"
    readonly repository: string
    readonly revision: string
    readonly quantization: string
  }
  readonly logPath?: string
}

export type TargetConfiguration = ExistingTarget | ManagedTarget

export interface StreamEvent {
  readonly atMs: number
  readonly payload: unknown
}

export interface ToolCallObservation {
  readonly id: string
  readonly name: string
  readonly arguments: string
}

export interface RequestObservation {
  readonly requestId: string
  readonly outcome: Outcome
  readonly submittedAtMs?: number
  readonly status?: number
  readonly headersMs?: number
  readonly ttftMs?: number
  readonly completedMs: number
  readonly outputText: string
  readonly toolCalls: readonly ToolCallObservation[]
  readonly finishReason?: string
  readonly terminal?: TerminalEvidence
  readonly events: readonly StreamEvent[]
  readonly error?: string
}

export interface TokenUsage {
  readonly promptTokens: number
  readonly cachedPromptTokens: number
  readonly completionTokens: number
  readonly totalTokens: number
}

export interface NativeTimings {
  readonly cacheTokens: number
  readonly evaluatedPromptTokens: number
  readonly promptMs: number
  readonly generatedTokens: number
  readonly generationMs: number
  readonly draftTokens?: number
  readonly acceptedDraftTokens?: number
}

export interface TerminalEvidence {
  readonly usage: TokenUsage
  readonly timings: NativeTimings
}

export interface MemorySample {
  readonly atMs: number
  readonly hostBytes?: number
  readonly deviceBytes?: Readonly<Record<string, number>>
}

export interface MemoryObservation {
  readonly supported: boolean
  readonly source: string
  readonly scope: string
  readonly baselineBytes?: number
  readonly peakBytes?: number
  readonly retainedBytes?: number
  readonly baselineDeviceBytes?: Readonly<Record<string, number>>
  readonly peakDeviceBytes?: Readonly<Record<string, number>>
  readonly retainedDeviceBytes?: Readonly<Record<string, number>>
  readonly samples: readonly MemorySample[]
  readonly limitation?: string
}

export interface TrialObservation {
  readonly trial: TrialDefinition
  readonly startedAt: string
  readonly makespanMs: number
  readonly requests: readonly RequestObservation[]
  readonly setupRequests?: readonly RequestObservation[]
  readonly memory?: MemoryObservation
}

export interface MetricSummary {
  readonly count: number
  readonly min: number
  readonly max: number
  readonly mean: number
  readonly median: number
  readonly p90: number
  readonly p95: number
  readonly p99: number
  readonly medianAbsoluteDeviation: number
}

export interface TrialAnalysis {
  readonly trialId: string
  readonly pattern: WorkloadPattern
  readonly measuredRequests: number
  readonly validRequests: number
  readonly outcomes: Readonly<Record<Outcome, number>>
  readonly responsivenessMs?: MetricSummary
  readonly prefillTokensPerSecond?: MetricSummary
  readonly decodeTokensPerSecond?: MetricSummary
  readonly completionMs?: MetricSummary
  readonly achievedRequestsPerSecond?: number
  readonly achievedCompletionTokensPerSecond?: number
  readonly cacheReuseRatio?: MetricSummary
  readonly promptTokens?: MetricSummary
  readonly completionTokens?: MetricSummary
  readonly memory?: MemoryObservation
}

export interface BenchmarkResult {
  readonly target: TargetConfiguration
  readonly planDigest: string
  readonly startedAt: string
  readonly completedAt: string
  readonly trials: readonly TrialObservation[]
  readonly analyses: readonly TrialAnalysis[]
  readonly warnings: readonly string[]
}

export interface ComparisonResult {
  readonly comparisonKind: "strict" | "product"
  readonly differences: readonly string[]
  readonly plan: TrialPlan
  readonly results: readonly BenchmarkResult[]
}

export interface EvaluationBlock {
  readonly index: number
  readonly variantOrder: readonly string[]
  readonly comparison: ComparisonResult
}

export interface ExperimentRunResult {
  readonly runId: string
  readonly experimentId: string
  readonly startedAt: string
  readonly completedAt: string
  readonly blocks: readonly EvaluationBlock[]
}
