import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { textOf } from '../../src/content'
import { MemoryProjection, getView } from '../../src/projections/memory'
import { SessionContextProjection } from '../../src/projections/session-context'
import { getRootMemory, inboxMessages, lastInboxMessage, snapshotMessageRefs, assertPrefixUnchanged, sendUserMessage } from './helpers'

function userContextText(ctx: readonly { role: 'user' | 'assistant'; content: readonly unknown[] }[]): string {
  return ctx
    .filter(m => m.role === 'user')
    .map(m => textOf(m.content as any))
    .join('\n\n')
}

function renderedUserText(
  h: Effect.Effect.Success<typeof TestHarness>,
  forkId: string | null,
): Effect.Effect<string> {
  return Effect.gen(function* () {
    const memory = yield* h.projectionFork(MemoryProjection.Tag, forkId)
    const timezone = yield* h.runEffect(
      Effect.flatMap(SessionContextProjection.Tag, p =>
        Effect.map(p.get, s => s.context?.timezone ?? null),
      ),
    )
    const view = getView(memory.messages, timezone, 'agent')
    return userContextText(view as any)
  })
}

describe('memory/user-messages', () => {
  it.live('simple user message (no mentions) appears in rendered context', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1_710_000_000_000,
        text: 'hello from user',
      })

      const memory = yield* getRootMemory(h)
      expect(memory.queuedEntries.some(q => q.lane === 'timeline' && q.entry.kind === 'user_message')).toBe(true)

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-user-simple-flush', chainId: 'c-user-simple-flush' })

      const rendered = yield* renderedUserText(h, null)
      expect(rendered).toContain('<message from="user">hello from user</message>')
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('user message with file mentions includes resolved content inline in attachments', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1_710_000_000_001,
        text: 'please inspect these files',
        attachments: [
          { type: 'mention', path: 'src/a.ts', contentType: 'text' },
          { type: 'mention', path: 'src/dir', contentType: 'directory' },
        ],
      })

      const memory = yield* getRootMemory(h)
      const queuedUser = memory.queuedEntries.find(q => q.lane === 'timeline' && q.entry.kind === 'user_message')
      expect(queuedUser).toBeDefined()
      if (!queuedUser || queuedUser.lane !== 'timeline' || queuedUser.entry.kind !== 'user_message') return

      expect(queuedUser.entry.attachments).toHaveLength(2)
      const mentions = queuedUser.entry.attachments.filter(
        (a): a is Extract<typeof a, { kind: 'mention' }> => a.kind === 'mention',
      )
      const fileMention = mentions.find(m => m.path === 'src/a.ts')
      const dirMention = mentions.find(m => m.path === 'src/dir')

      expect(fileMention).toBeDefined()
      expect(dirMention).toBeDefined()

      if (fileMention) {
        expect(fileMention.contentType).toBe('text')
        expect(fileMention.error).toBeUndefined()
        expect(fileMention.content).toContain('export const a = 1')
        expect(fileMention.truncated).toBeUndefined()
        expect(fileMention.originalBytes).toBe('export const a = 1'.length)
      }

      if (dirMention) {
        expect(dirMention.contentType).toBe('directory')
        expect(dirMention.error).toBeUndefined()
        expect(dirMention.content).toContain('file1')
      }

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-user-mentions-flush', chainId: 'c-user-mentions-flush' })

      const rendered = yield* renderedUserText(h, null)
      expect(rendered).toContain('please inspect these files')
      expect(rendered).not.toContain('&lt;')
    }).pipe(Effect.provide(TestHarnessLive({
      sessionContext: { cwd: '/' },
      files: {
        '/src/a.ts': 'export const a = 1',
        '/src/dir/file1': '',
        '/src/dir/file2': '',
      },
      workers: { turnController: false },
    })))
  )

  it.live('user message during active turn is queued and appears after flush', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't1', chainId: 'c1' })
      const before = yield* getRootMemory(h)
      const beforeCount = before.messages.length

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1_710_000_000_002,
        text: 'queued while active',
      })

      const queued = yield* getRootMemory(h)
      expect(queued.messages.length).toBe(beforeCount)
      expect(queued.queuedEntries.length).toBeGreaterThan(0)
      expect(queued.queuedEntries.some(q => q.lane === 'timeline' && q.entry.kind === 'user_message')).toBe(true)

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't2', chainId: 'c1' })
      const flushed = yield* getRootMemory(h)
      expect(flushed.queuedEntries).toHaveLength(0)
      const last = lastInboxMessage(flushed)
      expect(last?.type).toBe('inbox')
      if (last?.type === 'inbox') {
        expect(last.timeline.some(t => t.kind === 'user_message')).toBe(true)
      }

      const rendered = yield* renderedUserText(h, null)
      expect(rendered).toContain('queued while active')
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('user message to subagent creates user_to_agent entry', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subforkId = 'fork-sub-1'

      yield* h.send({
        type: 'agent_created',
        forkId: subforkId,
        parentForkId: null,
        agentId: 'builder-auth',
        name: 'builder-auth',
        role: 'builder',
        context: '',
        mode: 'spawn',
        taskId: 'task-1',
        message: 'work on auth',
      })

      yield* sendUserMessage(h, {
        forkId: subforkId,
        timestamp: 1_710_000_000_003,
        text: 'please handle auth',
      })

      yield* h.send({ type: 'turn_started', forkId: subforkId, turnId: 'sub-t1', chainId: 'sub-c1' })

      const memory = yield* h.projectionFork(MemoryProjection.Tag, subforkId)
      const last = lastInboxMessage(memory)
      expect(last?.type).toBe('inbox')
      if (last?.type === 'inbox') {
        expect(last.timeline.some(t => t.kind === 'user_message')).toBe(true)
        expect(last.timeline.some(t => t.kind === 'user_to_agent')).toBe(true)
      }

      const rendered = yield* renderedUserText(h, subforkId)
      expect(rendered).toContain('<user-to-agent agent="builder-auth">please handle auth</user-to-agent>')
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('multiple user messages preserve chronological ordering', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1_710_000_000_010,
        text: 'first message',
      })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1_710_000_000_011,
        text: 'second message',
      })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1_710_000_000_012,
        text: 'third message',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-order-flush', chainId: 'c-order-flush' })

      const memory = yield* getRootMemory(h)
      const texts = inboxMessages(memory)
        .flatMap(m => m.type === 'inbox' ? m.timeline : [])
        .filter(t => t.kind === 'user_message')
        .map(t => t.text)

      expect(texts).toEqual(expect.arrayContaining(['first message', 'second message', 'third message']))

      const rendered = yield* renderedUserText(h, null)
      const i1 = rendered.indexOf('first message')
      const i2 = rendered.indexOf('second message')
      const i3 = rendered.indexOf('third message')
      expect(i1).toBeGreaterThanOrEqual(0)
      expect(i2).toBeGreaterThan(i1)
      expect(i3).toBeGreaterThan(i2)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('append-only: each user message creates a new inbox message, no merge', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1_710_000_000_020,
        text: 'alpha',
      })

      const before = yield* getRootMemory(h)
      const snapshot = snapshotMessageRefs(before)

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1_710_000_000_021,
        text: 'beta',
      })

      const after = yield* getRootMemory(h)
      expect(after.messages.length).toBeGreaterThanOrEqual(snapshot.refs.length)
      assertPrefixUnchanged(snapshot, after)

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-append-flush', chainId: 'c-append-flush' })

      const rendered = yield* renderedUserText(h, null)
      expect(rendered).toContain('alpha')
      expect(rendered).toContain('beta')
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )
})
