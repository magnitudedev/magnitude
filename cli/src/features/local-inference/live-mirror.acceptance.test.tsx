import { act } from 'react'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { testRender } from '@opentui/react/test-utils'
import { RegistryProvider, Result } from '@effect-atom/atom-react'
import { Cause, Effect, Fiber, Option } from 'effect'
import { expect, test, vi } from 'vitest'
import {
  AgentClientProvider,
  createAgentClient,
  useLocalInferenceQuery,
} from '@magnitudedev/client-common'
import {
  connect,
  protocolLayer,
  ProviderIdSchema,
  ProviderModelIdSchema,
  type AcnClient,
} from '@magnitudedev/sdk'
import { LocalRuntimeStatusBar } from './status-bar'

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => ({
    primary: 'blue', secondary: 'gray', info: 'cyan', link: 'blue',
    foreground: 'white', muted: 'gray', border: 'gray', warning: 'magenta',
  }),
}))

const acnUrl = Option.fromNullable(process.env.LIVE_ACN_URL)
const modelId = 'mdl_5918543201b32042f2b7587e9753227aaac7fd9a1ba58406cb42f4bbb80cf5ef'
const requireAcnUrl = (): string => Option.getOrThrowWith(
  acnUrl,
  () => new Error('LIVE_ACN_URL is required for live acceptance testing'),
)

const call = <A, E>(effect: (client: AcnClient) => Effect.Effect<A, E>) =>
  Effect.runPromise(Effect.scoped(Effect.flatMap(connect(requireAcnUrl()), effect)))

const waitUntil = async (
  predicate: () => boolean | Promise<boolean>,
  timeoutMs = 60_000,
  describe: string | (() => string) = 'acceptance condition',
): Promise<void> => {
  const deadline = Date.now() + timeoutMs
  while (!(await predicate())) {
    if (Date.now() >= deadline) {
      const condition = typeof describe === 'string' ? describe : describe()
      throw new Error(`${condition} timed out`)
    }
    await act(async () => {
      await new Promise<void>((resolve) => setTimeout(resolve, 25))
    })
  }
}

test.skipIf(Option.isNone(acnUrl))('live mirrored state renders loading and loaded model state', async () => {
  await call((client) => client.DisableLocalInference({}))
  await waitUntil(async () => {
    const snapshot = await call((client) => client.GetIcnInventory({}))
    return snapshot.state.data.some((model) => model.id === modelId && model.residency.type === 'not_resident')
  })

  const renderedStates: string[] = []
  const Probe = () => {
    const result = useLocalInferenceQuery()
    if (!Result.isSuccess(result)) {
      renderedStates.push(Result.isFailure(result) ? `Failure:${Cause.pretty(result.cause)}` : result._tag)
      return <text>mirror:{result._tag}</text>
    }
    const model = Option.fromNullable(result.value.choices.find((choice) => choice.choiceId === modelId))
    const operation = Option.fromNullable(
      result.value.operations.find((candidate) => candidate.providerModelId === modelId),
    )
    const residency = Option.match(model, { onNone: () => 'absent', onSome: (choice) => choice.residency })
    const operationStatus = Option.match(operation, { onNone: () => 'idle', onSome: (current) => current.status })
    const state = `${residency}:${operationStatus}`
    renderedStates.push(state)
    return (
      <box style={{ flexDirection: 'column' }}>
        <text>mirror:{state}</text>
        <LocalRuntimeStatusBar state={result.value} width={100} onOpenHardware={() => {}} />
      </box>
    )
  }

  const agentClient = createAgentClient(protocolLayer(requireAcnUrl()))
  const view = await testRender(
    <RegistryProvider defaultIdleTTL={5_000}>
      <AgentClientProvider tag={agentClient}>
        <Probe />
      </AgentClientProvider>
    </RegistryProvider>,
    { width: 110, height: 8 },
  )

  try {
    await act(view.renderOnce)
    await waitUntil(
      () => renderedStates.includes('unloaded:idle'),
      15_000,
      () => `initial mirror; observed ${renderedStates.join(',')}`,
    )

    await call((client) => client.UpdateModelSlots({
      slots: {
        primary: {
          providerId: ProviderIdSchema.make('local'),
          providerModelId: ProviderModelIdSchema.make(modelId),
        },
      },
    }))
    await waitUntil(async () => {
      const slots = await call((client) => client.GetModelSlots({}))
      if (slots.state._tag === 'loading') return false
      return Option.exists(Option.fromNullable(slots.state.slots.primary), (primary) =>
        primary._tag === 'Ready'
        && primary.selection.providerId === 'local'
        && primary.selection.providerModelId === modelId)
    }, 30_000, 'primary local-model slot')

    const created = await call((client) => client.CreateSession({
      cwd: process.cwd(),
      sessionId: Option.none<string>(),
      initial: Option.none(),
      options: Option.some({
        disableShellSafeguards: false,
        disableCwdSafeguards: false,
        solo: true,
        headless: true,
      }),
      draftOwnerId: Option.none<string>(),
    }))
    expect(created._tag).toBe('created')
    if (created._tag !== 'created') throw new Error(`Session creation failed: ${created._tag}`)
    const sendFiber = Effect.runFork(Effect.scoped(Effect.flatMap(
      connect(requireAcnUrl()),
      (client) => client.SendMessage({
        sessionId: created.metadata.sessionId,
        messageId: Option.none<string>(),
        content: 'Reply with exactly: OK',
        visibleMessage: Option.some('Reply with exactly: OK'),
        taskMode: false,
        imageAttachments: [],
        mentions: [],
      }),
    )))
    try {
      await waitUntil(
        () => renderedStates.some((state) => state.startsWith('loading:running')),
        15_000,
        () => `loading mirror; observed ${renderedStates.join(',')}`,
      )
      await waitUntil(
        () => renderedStates.includes('loaded:idle'),
        120_000,
        () => `loaded mirror; observed ${renderedStates.join(',')}`,
      )
      await waitUntil(async () => {
        await act(view.renderOnce)
        return view.captureCharFrame().includes('Memory')
      }, 10_000, 'resident hardware memory')

      const frame = view.captureCharFrame()
      expect(frame).toContain('mirror:loaded:idle')
      expect(frame).toContain('Qwen')
      expect(frame).toContain('Ready')
      expect(frame).toContain('Memory')
      expect(frame).toContain(' / 64 GiB')
      await waitUntil(async () => {
        const eventsPath = join(homedir(), '.magnitude', 'sessions', created.metadata.sessionId, 'events.jsonl')
        const events = (await Bun.file(eventsPath).text()).trim().split('\n').map((line) => JSON.parse(line) as { type: string })
        return events.some((event) => event.type === 'message_end')
          && events.some((event) => event.type === 'turn_outcome')
      }, 60_000, 'completed assistant response')
    } finally {
      await Effect.runPromise(Fiber.interrupt(sendFiber))
    }
  } finally {
    await act(async () => view.renderer.destroy())
  }
}, 180_000)
