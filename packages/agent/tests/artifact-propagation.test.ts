import { describe, test, expect } from 'bun:test'
import { withHarness } from '../src/test-harness'
import { ArtifactAwarenessProjection } from '../src/projections/artifact-awareness'
import { ArtifactProjection } from '../src/projections/artifact'
import { MemoryProjection } from '../src/projections/memory'

describe('artifact awareness propagation', () => {
  test('Subagent writes artifact and refs it in comms → parent becomes aware', () =>
    withHarness(async (h) => {
      await h.turns()
        .user('create subagent that writes artifact')
        .orchestrator(
          h.response().createAgent('test-explorer', 'explorer', 'artifact test', 'create artifact').yield(),
        )
        .agent(
          'test-explorer',
          h.response().writeArtifact('my-artifact', 'hello from subagent').message('parent', 'wrote [[my-artifact]]').yield(),
        )
        .run()

      const artifactState = await h.projection(ArtifactProjection.Tag)
      expect(artifactState.artifacts.get('my-artifact')?.content).toBe('hello from subagent')

      const parentAwareness = await h.projectionFork(ArtifactAwarenessProjection.Tag, null)
      expect(parentAwareness.awareArtifactIds.has('my-artifact')).toBe(true)
    }),
  )

  test('Subagent writes artifact but does NOT ref it → parent should NOT be aware', () =>
    withHarness(async (h) => {
      await h.turns()
        .user('create subagent that writes artifact')
        .orchestrator(
          h.response().createAgent('test-explorer', 'explorer', 'artifact test', 'create artifact').yield(),
        )
        .agent(
          'test-explorer',
          h.response().writeArtifact('my-artifact', 'hello from subagent').message('parent', 'artifact written').yield(),
        )
        .run()

      const artifactState = await h.projection(ArtifactProjection.Tag)
      expect(artifactState.artifacts.get('my-artifact')?.content).toBe('hello from subagent')

      const parentAwareness = await h.projectionFork(ArtifactAwarenessProjection.Tag, null)
      expect(parentAwareness.awareArtifactIds.has('my-artifact')).toBe(false)
    }),
  )

  test('Artifact content injection into parent context on next turn', () =>
    withHarness(async (h) => {
      await h.turns()
        .user('create subagent that writes artifact and references it')
        .orchestrator(
          h.response().createAgent('test-explorer', 'explorer', 'artifact test', 'create artifact').yield(),
        )
        .agent(
          'test-explorer',
          h.response().writeArtifact('my-artifact', 'injected artifact body').message('parent', 'see [[my-artifact]]').yield(),
        )
        .run()

      await h.script.next({ xml: '<yield/>' })
      await h.user('trigger next parent turn')
      await h.wait.turnCompleted(null)

      const parentMemory = await h.projectionFork(MemoryProjection.Tag, null)

      const queuedSystemReminders = parentMemory.queuedMessages.filter(
        (q) => q.kind === 'system' && q.entry.kind === 'reminder',
      )
      const queuedHasArtifact = queuedSystemReminders.some(
        (q) =>
          q.entry.kind === 'reminder' &&
          q.entry.text.includes('<artifact id="my-artifact">') &&
          q.entry.text.includes('injected artifact body'),
      )

      const systemInboxMessages = parentMemory.messages.filter((m) => m.type === 'system_inbox')
      const flattened = systemInboxMessages.flatMap((m) => m.entries)
      const hasArtifactReminder = flattened.some((entry) =>
        entry.kind === 'reminder' &&
        entry.text.includes('<artifact id="my-artifact">') &&
        entry.text.includes('injected artifact body'),
      )

      expect(queuedHasArtifact || hasArtifactReminder).toBe(true)
    }),
  )

  test('Multiple subagents sharing an artifact become aware', () =>
    withHarness(async (h) => {
      const result = await h.turns()
        .user('share artifact across subagents')
        .orchestrator(
          h.response()
            .writeArtifact('shared-artifact', 'shared body')
            .createAgent('agent-one', 'explorer', 'one', 'first')
            .createAgent('agent-two', 'explorer', 'two', 'second')
            .yield(),
        )
        .agents({
          'agent-one': h.response().message('parent', 'shared [[shared-artifact]]').yield(),
          'agent-two': h.response().message('parent', 'shared [[shared-artifact]]').yield(),
        })
        .run()

      const awarenessOne = await h.projectionFork(ArtifactAwarenessProjection.Tag, result.forks.get('agent-one')!)
      const awarenessTwo = await h.projectionFork(ArtifactAwarenessProjection.Tag, result.forks.get('agent-two')!)

      expect(awarenessOne.awareArtifactIds.has('shared-artifact')).toBe(true)
      expect(awarenessTwo.awareArtifactIds.has('shared-artifact')).toBe(true)
    }),
  )
})