import type { ChatMessage, ChatTool } from "@magnitudedev/ai"
import { Data, Effect } from "effect"
import type { PreparedCorpus } from "./corpus"
import type {
  Criterion,
  Fixture,
  LogicalModelIdentity,
  PlannedRequest,
  ProfileName,
  TrialDefinition,
  TrialPlan,
} from "./domain"
import { digestObject, stableStringify } from "./hash"

export class PlanError extends Data.TaggedError("PlanError")<{
  readonly message: string
}> {}

interface ProfileShape {
  readonly repetitions: number
  readonly sequentialDepth: number
  readonly forkPrefixDepth: number
  readonly independentConcurrency: readonly number[]
  readonly concurrencyPressure: readonly number[]
  readonly memorySessions: number
  readonly memoryDepths: readonly number[]
}

const profiles: Record<ProfileName, ProfileShape> = {
  smoke: {
    repetitions: 1,
    sequentialDepth: 2,
    forkPrefixDepth: 2,
    independentConcurrency: [2],
    concurrencyPressure: [2, 4],
    memorySessions: 2,
    memoryDepths: [1, 2],
  },
  standard: {
    repetitions: 3,
    sequentialDepth: 128,
    forkPrefixDepth: 128,
    independentConcurrency: [2, 4, 8],
    concurrencyPressure: [1, 2, 4, 8, 16, 32],
    memorySessions: 4,
    memoryDepths: [4, 16, 64, 128],
  },
  full: {
    repetitions: 5,
    sequentialDepth: 399,
    forkPrefixDepth: 320,
    independentConcurrency: [2, 4, 8, 16],
    concurrencyPressure: [1, 2, 4, 8, 16, 32, 64],
    memorySessions: 8,
    memoryDepths: [4, 16, 32, 64, 128, 256, 399],
  },
}

const allCriteria: readonly Criterion[] = [
  "responsiveness",
  "prefill",
  "decode",
  "memory-usage",
  "distribution",
]

function mergeTools(fixtures: readonly Fixture[]): readonly ChatTool[] {
  const byName = new Map<string, ChatTool>()
  for (const fixture of fixtures) {
    for (const tool of fixture.tools) byName.set(tool.function.name, tool)
  }
  return [...byName.values()]
}

function completedUnit(fixture: Fixture): readonly ChatMessage[] {
  return [...fixture.messages, fixture.canonicalAssistant, ...fixture.canonicalToolMessages]
}

function cumulativeRequest(
  id: string,
  fixtures: readonly Fixture[],
  currentIndex: number,
  options: {
    readonly sessionId?: string
    readonly prefixGroup?: string
    readonly releaseOffsetMs?: number
    readonly dependsOn?: readonly string[]
    readonly maxOutputTokens?: number
    readonly temperature?: number
    readonly topP?: number
    readonly seed?: number
    readonly enableThinking?: false
  } = {},
): PlannedRequest {
  const current = fixtures[currentIndex]
  if (!current) throw new PlanError({ message: `Missing fixture at ${currentIndex}` })
  const namespace = options.prefixGroup ?? options.sessionId ?? id
  return {
    id,
    fixtureId: current.id,
    messages: [
      {
        role: "system",
        content: `Benchmark workload namespace: ${namespace}. Process each user request using the provided tools.`,
      },
      ...fixtures.slice(0, currentIndex).flatMap(completedUnit),
      ...current.messages,
    ],
    tools: mergeTools(fixtures.slice(0, currentIndex + 1)),
    expected: current.expected,
    releaseOffsetMs: options.releaseOffsetMs ?? 0,
    dependsOn: options.dependsOn ?? [],
    maxOutputTokens: options.maxOutputTokens ?? 128,
    temperature: options.temperature,
    topP: options.topP,
    seed: options.seed,
    enableThinking: options.enableThinking,
    sessionId: options.sessionId,
    prefixGroup: options.prefixGroup,
  }
}

function isolatedRequest(
  id: string,
  fixture: Fixture,
  releaseOffsetMs = 0,
  requestPolicy: Pick<PlannedRequest, "maxOutputTokens" | "temperature" | "topP" | "seed" | "enableThinking"> = { maxOutputTokens: 128 },
): PlannedRequest {
  return {
    id,
    fixtureId: fixture.id,
    messages: [
      {
        role: "system",
        content: `Benchmark workload namespace: ${id}. Process the user request using the provided tools.`,
      },
      ...fixture.messages,
    ],
    tools: fixture.tools,
    expected: fixture.expected,
    releaseOffsetMs,
    dependsOn: [],
    ...requestPolicy,
  }
}

export interface CompilePlanOptions {
  readonly profile?: ProfileName
  readonly contextSweep?: {
    readonly checkpoints: readonly number[]
    readonly charactersPerToken: number
    readonly samplesPerCheckpoint: number
  }
  readonly maxContextTokens?: number
  readonly parallelSequences?: number
  readonly maxOutputTokens?: number
  readonly temperature?: number
  readonly topP?: number
  readonly seed?: number
  readonly enableThinking?: false
}

type RequestPolicy = Pick<PlannedRequest, "maxOutputTokens" | "temperature" | "topP" | "seed" | "enableThinking">

function contextRequestCharacters(request: PlannedRequest): number {
  return [...stableStringify({ messages: request.messages, tools: request.tools })].length
}

function contextRequestAtTarget(
  id: string,
  fixtures: readonly Fixture[],
  targetCharacters: number,
  requestPolicy: RequestPolicy,
): { readonly request: PlannedRequest; readonly characters: number; readonly depth: number } {
  let previous: { readonly request: PlannedRequest; readonly characters: number; readonly depth: number } | undefined
  for (let index = 0; index < fixtures.length; index++) {
    const request = cumulativeRequest(id, fixtures, index, { prefixGroup: id, ...requestPolicy })
    const candidate = { request, characters: contextRequestCharacters(request), depth: index + 1 }
    if (candidate.characters >= targetCharacters) {
      if (!previous || candidate.characters - targetCharacters < targetCharacters - previous.characters) return candidate
      return previous
    }
    previous = candidate
  }
  throw new PlanError({
    message: `BFCL corpus reaches only ${previous?.characters ?? 0} canonical characters; cannot satisfy context target ${targetCharacters}`,
  })
}

function compileContextSweepTrials(
  fixtures: readonly Fixture[],
  sweep: NonNullable<CompilePlanOptions["contextSweep"]>,
  requestPolicy: RequestPolicy,
): readonly TrialDefinition[] {
  return sweep.checkpoints.flatMap((tokens) => {
    const characters = Math.round(tokens * sweep.charactersPerToken)
    return Array.from({ length: sweep.samplesPerCheckpoint }, (_, repetition) => {
      const id = `context-t${tokens}-r${repetition}`
      const selected = contextRequestAtTarget(`${id}-q0`, fixtures, characters, requestPolicy)
      return {
        id,
        pattern: "context-scaling" as const,
        criteria: allCriteria,
        checkpoint: `~${tokens}-tokens`,
        repetition,
        state: "cache-disjoint" as const,
        requests: [selected.request],
        contextTarget: {
          tokens,
          characters,
          plannedCharacters: selected.characters,
          semanticDepth: selected.depth,
        },
      }
    })
  })
}

export function compileTrialPlanSync(
  corpus: PreparedCorpus,
  modelInput: LogicalModelIdentity,
  options: CompilePlanOptions = {},
): TrialPlan {
  if (options.profile !== undefined && options.contextSweep !== undefined) {
    throw new PlanError({ message: "choose either an agent-core profile or a context sweep" })
  }
  const profileName = options.contextSweep ? "context-sweep" as const : options.profile ?? "standard"
  const requestPolicy = {
    maxOutputTokens: options.maxOutputTokens ?? 128,
    temperature: options.temperature ?? 0,
    topP: options.topP ?? 1,
    seed: options.seed ?? 42,
    enableThinking: options.enableThinking ?? false,
  }
  const model = {
    id: modelInput.id,
    contextLimit: Math.min(
      modelInput.contextLimit,
      options.maxContextTokens ?? modelInput.contextLimit,
    ),
  }
  if (corpus.fixtures.length === 0) {
    throw new PlanError({ message: "Corpus has no qualified fixtures" })
  }

  const fixtures = corpus.fixtures
  const warmup = isolatedRequest("warmup", fixtures[0]!, 0, requestPolicy)
  const trials: TrialDefinition[] = []
  let cursor = 0

  if (options.contextSweep) {
    trials.push(...compileContextSweepTrials(fixtures, options.contextSweep, requestPolicy))
  } else {
    const profile = profiles[options.profile ?? "standard"]
    for (let repetition = 0; repetition < profile.repetitions; repetition++) {
    const single = fixtures[cursor++ % fixtures.length]!
    trials.push({
      id: `single-r${repetition}`,
      pattern: "single-request",
      criteria: allCriteria,
      checkpoint: "one-turn",
      repetition,
      state: "cache-disjoint",
      requests: [isolatedRequest(`single-r${repetition}-q0`, single, 0, requestPolicy)],
    })

    const sequentialFixtures = Array.from(
      { length: profile.sequentialDepth },
      (_, index) => fixtures[(cursor + index) % fixtures.length]!,
    )
    cursor += profile.sequentialDepth
    trials.push({
      id: `sequential-r${repetition}`,
      pattern: "sequential-session",
      criteria: allCriteria,
      checkpoint: `${profile.sequentialDepth}-turns`,
      repetition,
      state: "resident-prefix",
      requests: sequentialFixtures.map((_, depth) =>
        cumulativeRequest(
          `sequential-r${repetition}-q${depth}`,
          sequentialFixtures,
          depth,
          {
            sessionId: `sequential-r${repetition}`,
            dependsOn:
              depth === 0 ? [] : [`sequential-r${repetition}-q${depth - 1}`],
            ...requestPolicy,
          },
        )),
    })

    for (const concurrency of profile.independentConcurrency) {
      const selected = Array.from(
        { length: concurrency },
        (_, index) => fixtures[(cursor + index) % fixtures.length]!,
      )
      cursor += concurrency
      trials.push({
        id: `independent-c${concurrency}-r${repetition}`,
        pattern: "independent-concurrency",
        criteria: allCriteria,
        checkpoint: `c${concurrency}`,
        repetition,
        state: "cache-disjoint",
        requests: selected.map((fixture, index) =>
          isolatedRequest(`independent-c${concurrency}-r${repetition}-q${index}`, fixture, 0, requestPolicy)),
      })
    }

    const forkFixtures = Array.from(
      { length: profile.forkPrefixDepth + 8 },
      (_, index) => fixtures[(cursor + index) % fixtures.length]!,
    )
    cursor += profile.forkPrefixDepth + 8
    const setupId = `forked-r${repetition}-setup`
    const prefix = forkFixtures.slice(0, profile.forkPrefixDepth)
    const setup = {
      ...cumulativeRequest(
        setupId,
        prefix,
        prefix.length - 1,
        {
          sessionId: `forked-r${repetition}-parent`,
          prefixGroup: `forked-r${repetition}`,
          ...requestPolicy,
        },
      ),
      phase: "setup" as const,
    }
    const branches = forkFixtures.slice(profile.forkPrefixDepth).map((fixture, index) => ({
      ...cumulativeRequest(
        `forked-r${repetition}-q${index}`,
        [...prefix, fixture],
        prefix.length,
        { prefixGroup: `forked-r${repetition}`, dependsOn: [setupId], ...requestPolicy },
      ),
      phase: "measure" as const,
    }))
    trials.push({
      id: `forked-r${repetition}`,
      pattern: "forked-concurrency",
      criteria: allCriteria,
      checkpoint: `${profile.forkPrefixDepth}-turn-prefix-c${branches.length}`,
      repetition,
      state: "resident-prefix",
      requests: [setup, ...branches],
    })

    for (const concurrency of profile.concurrencyPressure) {
      const selected = Array.from(
        { length: concurrency },
        (_, index) => fixtures[(cursor + index) % fixtures.length]!,
      )
      cursor += concurrency
      trials.push({
        id: `concurrency-pressure-c${concurrency}-r${repetition}`,
        pattern: "concurrency-pressure",
        criteria: allCriteria,
        checkpoint: `offered-c${concurrency}`,
        repetition,
        state: "cache-disjoint",
        requests: selected.map((fixture, index) =>
          isolatedRequest(`concurrency-pressure-c${concurrency}-r${repetition}-q${index}`, fixture, 0, requestPolicy)),
      })
    }

    const memoryBase = cursor
    const maximumDepth = Math.max(...profile.memoryDepths)
    for (const depth of profile.memoryDepths) {
      const requests = Array.from({ length: profile.memorySessions }, (_, session) => {
        const sessionFixtures = Array.from(
          { length: depth },
          (_, index) => fixtures[(memoryBase + session * maximumDepth + index) % fixtures.length]!,
        )
        return cumulativeRequest(
          `memory-pressure-d${depth}-s${session}-r${repetition}`,
          sessionFixtures,
          depth - 1,
          {
            sessionId: `memory-s${session}-r${repetition}`,
            releaseOffsetMs: session * 25,
            ...requestPolicy,
          },
        )
      })
      trials.push({
        id: `memory-pressure-d${depth}-r${repetition}`,
        pattern: "memory-pressure",
        criteria: allCriteria,
        checkpoint: `${depth}-turns-c${profile.memorySessions}`,
        repetition,
        state: "resident-prefix",
        requests,
      })
    }
      cursor += profile.memorySessions * maximumDepth
    }
  }

  const servingPolicy = {
    contextTokensPerSequence: model.contextLimit,
    parallelSequences: options.parallelSequences ?? 1,
    temperature: requestPolicy.temperature,
    topP: requestPolicy.topP,
    seed: requestPolicy.seed,
    enableThinking: requestPolicy.enableThinking,
  }
  const identity = { profile: profileName, model, servingPolicy, corpusDigest: corpus.digest, warmup, trials }
  return {
    ...identity,
    createdAt: new Date().toISOString(),
    digest: digestObject(identity),
  }
}

export const compileTrialPlan = (
  corpus: PreparedCorpus,
  model: LogicalModelIdentity,
  options: CompilePlanOptions = {},
): Effect.Effect<TrialPlan, PlanError> =>
  Effect.try({
    try: () => compileTrialPlanSync(corpus, model, options),
    catch: (error) =>
      error instanceof PlanError
        ? error
        : new PlanError({ message: error instanceof Error ? error.message : String(error) }),
  })

export function explainPlan(plan: TrialPlan): string {
  const counts = new Map<string, number>()
  for (const trial of plan.trials) {
    counts.set(trial.pattern, (counts.get(trial.pattern) ?? 0) + 1)
  }
  return [
    `plan: ${plan.digest}`,
    `profile: ${plan.profile}`,
    `model: ${plan.model.id}`,
    `context limit: ${plan.model.contextLimit}`,
    `parallel sequences: ${plan.servingPolicy.parallelSequences}`,
    `corpus: ${plan.corpusDigest}`,
    `trials: ${plan.trials.length}`,
    ...[...counts].map(([pattern, count]) => `  ${pattern}: ${count}`),
    ...plan.trials.flatMap((trial) => trial.contextTarget ? [
      `  ${trial.id}: target=${trial.contextTarget.tokens} tokens (${trial.contextTarget.characters} chars), planned=${trial.contextTarget.plannedCharacters} chars, depth=${trial.contextTarget.semanticDepth}`,
    ] : []),
  ].join("\n")
}
