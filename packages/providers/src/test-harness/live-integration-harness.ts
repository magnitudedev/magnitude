import { Effect, Layer, ManagedRuntime, Stream } from 'effect'
import type { StreamResult, CompleteResult, ExecutableDriver } from '../drivers/types'
import { BamlDriver, ResponsesDriver } from '../drivers'
import {
  bootstrapProviderRuntime,
  CodingAgentChat,
  GenerateChatTitle,
  makeModelResolver,
  makeProviderRuntimeLive,
  ModelResolver,
  ProviderAuth,
  ProviderState,
  TraceEmitter,
} from '../index'
import type { AuthInfo } from '../types'
import type { ModelSlot, CallUsage } from '../src/state/provider-state'
import type { Model } from '../src/model/model'
import type { BoundModel } from '../src/model/bound-model'
import type { TraceInput } from '../src/resolver/tracing'

export type ExpectedDriver = 'baml' | 'openai-responses'

export interface LiveTraceStore {
  readonly traces: TraceInput[]
  clear(): void
}

export interface LiveTestTarget {
  readonly slot: ModelSlot
  readonly model: Model
  readonly auth: AuthInfo | null
  readonly expectedDriver: ExpectedDriver
}

export interface ResolvedLiveTarget extends LiveTestTarget {
  readonly bound: BoundModel
}

export interface LiveHarness {
  readonly runtime: ManagedRuntime.ManagedRuntime<TraceEmitter | ModelResolver | ProviderAuth | ProviderState, never>
  readonly traces: LiveTraceStore
  getLiveTargets(): Promise<LiveTestTarget[]>
  resolveSlot(slot: ModelSlot): Promise<ResolvedLiveTarget>
  resolveTarget(target: LiveTestTarget): Promise<ResolvedLiveTarget>
  runBoundGenerateChatTitle(bound: BoundModel): Promise<{ title: string } | null>
  runBoundCodingAgentChat(bound: BoundModel): Promise<{ text: string; usage: CallUsage | null }>
  connectExpectedDriver(target: LiveTestTarget): Promise<{ driver: ExecutableDriver; connection: ResolvedLiveTarget['bound']['connection'] }>
  runDriverGenerateChatTitle(target: LiveTestTarget): Promise<CompleteResult<{ title: string } | null>>
  runDriverCodingAgentChat(target: LiveTestTarget): Promise<{ text: string; usage: CallUsage | null; collectorData: StreamResult['getCollectorData'] extends () => infer T ? T : never }>
}

export const LIVE_TEST_SLOTS: ModelSlot[] = ['primary', 'secondary', 'browser']

export function shouldRunLiveProviderTests(): boolean {
  return process.env.MAGNITUDE_RUN_LIVE_PROVIDER_TESTS === '1'
}

export function getLiveProviderTestSkipReason(): string | null {
  return shouldRunLiveProviderTests()
    ? null
    : 'Set MAGNITUDE_RUN_LIVE_PROVIDER_TESTS=1 to run live provider integration tests'
}

export function expectedDriverForSelection(model: Model, auth: AuthInfo | null): ExpectedDriver {
  const isCodex = model.providerId === 'openai' && auth?.type === 'oauth'
  const isCopilotCodex = model.providerId === 'github-copilot' && model.id.includes('codex')
  return (isCodex || isCopilotCodex) ? 'openai-responses' : 'baml'
}

export function makeGenerateChatTitleInput() {
  return {
    conversation: 'User: hello\nAssistant: hi\nUser: summarize tracing bug',
    defaultName: 'New Chat',
  }
}

export function makeCodingAgentChatInput() {
  return {
    systemPrompt: 'You are concise. Reply with 3-5 words.',
    messages: [{ role: 'user', content: 'Say hello.' }] as const,
    ackTurn: 'ack-test',
  }
}

function driverForExpected(expectedDriver: ExpectedDriver): ExecutableDriver {
  return expectedDriver === 'openai-responses' ? ResponsesDriver : BamlDriver
}

function makeTraceStore(): LiveTraceStore {
  const traces: TraceInput[] = []
  return {
    traces,
    clear() {
      traces.length = 0
    },
  }
}

export async function createLiveIntegrationHarness(): Promise<LiveHarness> {
  const traces = makeTraceStore()

  const traceEmitterLayer = Layer.succeed(TraceEmitter, {
    emit: (trace: TraceInput) =>
      Effect.sync(() => {
        traces.traces.push(trace)
      }),
  })

  const providerLayer = makeProviderRuntimeLive()
  const resolverLayer = makeModelResolver().pipe(Layer.provide(providerLayer))
  const appLayer = Layer.mergeAll(providerLayer, resolverLayer, traceEmitterLayer)
  const runtime = ManagedRuntime.make(appLayer)

  await runtime.runPromise(bootstrapProviderRuntime)

  async function getLiveTargets(): Promise<LiveTestTarget[]> {
    return runtime.runPromise(
      Effect.gen(function* () {
        const state = yield* ProviderState
        const auth = yield* ProviderAuth
        const connected = yield* auth.connectedProviderIds()

        const targets: LiveTestTarget[] = []

        for (const slot of LIVE_TEST_SLOTS) {
          const selection = yield* state.peek(slot)
          if (!selection) continue
          if (!connected.has(selection.model.providerId)) continue

          targets.push({
            slot,
            model: selection.model,
            auth: selection.auth ?? null,
            expectedDriver: expectedDriverForSelection(selection.model, selection.auth ?? null),
          })
        }

        return targets
      }),
    )
  }

  async function resolveSlot(slot: ModelSlot): Promise<ResolvedLiveTarget> {
    return runtime.runPromise(
      Effect.gen(function* () {
        const resolver = yield* ModelResolver
        const bound = yield* resolver.resolve(slot)
        return {
          slot,
          model: bound.model,
          auth: bound.connection.auth ?? null,
          expectedDriver: expectedDriverForSelection(bound.model, bound.connection.auth ?? null),
          bound,
        }
      }),
    )
  }

  async function resolveTarget(target: LiveTestTarget): Promise<ResolvedLiveTarget> {
    return resolveSlot(target.slot)
  }

  async function runBoundGenerateChatTitle(bound: BoundModel) {
    return runtime.runPromise(bound.invoke(GenerateChatTitle, makeGenerateChatTitleInput()))
  }

  async function runBoundCodingAgentChat(bound: BoundModel): Promise<{ text: string; usage: CallUsage | null }> {
    const chatStream = await runtime.runPromise(bound.invoke(CodingAgentChat, makeCodingAgentChatInput()))
    const text = await runtime.runPromise(Stream.runFold(chatStream.stream, '', (acc, chunk) => acc + chunk))
    return { text, usage: chatStream.getUsage() }
  }

  async function connectExpectedDriver(target: LiveTestTarget) {
    const driver = driverForExpected(target.expectedDriver)
    const connection = await runtime.runPromise(driver.connect(target.model, target.auth, {}))
    return { driver, connection }
  }

  async function runDriverGenerateChatTitle(target: LiveTestTarget) {
    const { driver, connection } = await connectExpectedDriver(target)
    const input = makeGenerateChatTitleInput()
    return runtime.runPromise(
      driver.complete<{ title: string } | null>({
        slot: target.slot,
        functionName: 'GenerateChatTitle',
        args: [input.conversation, input.defaultName, false],
        connection,
        model: target.model,
        inference: {},
      }),
    )
  }

  async function runDriverCodingAgentChat(target: LiveTestTarget): Promise<{ text: string; usage: CallUsage | null; collectorData: ReturnType<StreamResult['getCollectorData']> }> {
    const { driver, connection } = await connectExpectedDriver(target)
    const input = makeCodingAgentChatInput()
    const streamResult = await runtime.runPromise(
      driver.stream({
        slot: target.slot,
        functionName: 'CodingAgentChat',
        args: [input.systemPrompt, input.messages, input.ackTurn, false],
        connection,
        model: target.model,
        inference: {},
      }),
    )
    const text = await runtime.runPromise(Stream.runFold(streamResult.stream, '', (acc, chunk) => acc + chunk))
    return { text, usage: streamResult.getUsage(), collectorData: streamResult.getCollectorData() }
  }

  return {
    runtime,
    traces,
    getLiveTargets,
    resolveSlot,
    resolveTarget,
    runBoundGenerateChatTitle,
    runBoundCodingAgentChat,
    connectExpectedDriver,
    runDriverGenerateChatTitle,
    runDriverCodingAgentChat,
  }
}