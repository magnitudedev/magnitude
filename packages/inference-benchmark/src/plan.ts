import type { ChatMessage, ChatTool } from "@magnitudedev/ai"
import { Data, Effect } from "effect"
import type { PreparedCorpus } from "./corpus"
import type {
  Criterion,
  Fixture,
  ModelIdentity,
  PlannedRequest,
  ProfileName,
  TrialDefinition,
  TrialPlan,
} from "./domain"
import { digestObject } from "./hash"

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
    maxOutputTokens: 128,
    sessionId: options.sessionId,
    prefixGroup: options.prefixGroup,
  }
}

function isolatedRequest(id: string, fixture: Fixture, releaseOffsetMs = 0): PlannedRequest {
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
    maxOutputTokens: 128,
  }
}

export interface CompilePlanOptions {
  readonly profile?: ProfileName
  readonly maxContextTokens?: number
  readonly parallelSequences?: number
}

export function compileTrialPlanSync(
  corpus: PreparedCorpus,
  modelInput: ModelIdentity,
  options: CompilePlanOptions = {},
): TrialPlan {
  const profileName = options.profile ?? "standard"
  const profile = profiles[profileName]
  const model = {
    ...modelInput,
    contextLimit: Math.min(
      modelInput.contextLimit,
      options.maxContextTokens ?? modelInput.contextLimit,
    ),
  }
  if (corpus.fixtures.length === 0) {
    throw new PlanError({ message: "Corpus has no qualified fixtures" })
  }

  const fixtures = corpus.fixtures
  const warmup = isolatedRequest("warmup", fixtures[0]!)
  const trials: TrialDefinition[] = []
  let cursor = 0

  for (let repetition = 0; repetition < profile.repetitions; repetition++) {
    const single = fixtures[cursor++ % fixtures.length]!
    trials.push({
      id: `single-r${repetition}`,
      pattern: "single-request",
      criteria: allCriteria,
      checkpoint: "one-turn",
      repetition,
      state: "cache-disjoint",
      requests: [isolatedRequest(`single-r${repetition}-q0`, single)],
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
          isolatedRequest(`independent-c${concurrency}-r${repetition}-q${index}`, fixture)),
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
        },
      ),
      phase: "setup" as const,
    }
    const branches = forkFixtures.slice(profile.forkPrefixDepth).map((fixture, index) => ({
      ...cumulativeRequest(
        `forked-r${repetition}-q${index}`,
        [...prefix, fixture],
        prefix.length,
        { prefixGroup: `forked-r${repetition}`, dependsOn: [setupId] },
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
          isolatedRequest(`concurrency-pressure-c${concurrency}-r${repetition}-q${index}`, fixture)),
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

  const servingPolicy = {
    contextTokensPerSequence: model.contextLimit,
    parallelSequences: options.parallelSequences ?? 1,
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
  model: ModelIdentity,
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
  ].join("\n")
}
