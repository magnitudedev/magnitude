import { Effect } from 'effect'
import { SessionContextProjection } from '../../src/projections/session-context'
import { getView, MemoryProjection, type ForkMemoryState, type Message } from '../../src/projections/memory'
import { TestHarness } from '../../src/test-harness/harness'
import { createId } from '../../src/util/id'

export function getRootMemory(h: Effect.Effect.Success<typeof TestHarness>) {
  return h.projectionFork(MemoryProjection.Tag, null)
}

export function sendUserMessage(h: Effect.Effect.Success<typeof TestHarness>, options: {
  text: string
  timestamp: number
  forkId?: string | null
  attachments?: any[]
}) {
  const forkId = options.forkId ?? null
  const timestamp = options.timestamp
  return Effect.gen(function* () {
    const messageId = createId()
    yield* h.send({
      type: 'user_message',
      messageId,
      forkId,
      timestamp,
      content: [{ type: 'text', text: options.text }],
      attachments: options.attachments ?? [],
      mode: 'text',
      synthetic: false,
      taskMode: false,
    })
    yield* h.wait.event('user_message_ready', (e) => e.messageId === messageId)
  })
}

export function lastInboxMessage(memory: ForkMemoryState): Message | undefined {
  for (let i = memory.messages.length - 1; i >= 0; i--) {
    if (memory.messages[i].type === 'inbox') return memory.messages[i]
  }
  return undefined
}

export function inboxMessages(memory: ForkMemoryState): Message[] {
  return memory.messages.filter(m => m.type === 'inbox')
}

export function snapshotMessageRefs(memory: ForkMemoryState): { refs: readonly Message[], json: string } {
  return {
    refs: memory.messages,
    json: JSON.stringify(memory.messages),
  }
}

export function assertPrefixUnchanged(
  before: { refs: readonly Message[], json: string },
  after: ForkMemoryState,
) {
  const beforeCount = before.refs.length
  for (let i = 0; i < beforeCount; i++) {
    if (after.messages[i] !== before.refs[i]) {
      throw new Error(`Message at index ${i} was mutated (reference changed)`)
    }
  }
  const afterPrefixJson = JSON.stringify(after.messages.slice(0, beforeCount))
  if (afterPrefixJson !== before.json) {
    throw new Error('Message prefix content was mutated')
  }
}

export function getRenderedUserText(h: Effect.Effect.Success<typeof TestHarness>) {
  return Effect.gen(function* () {
    const memory = yield* h.projectionFork(MemoryProjection.Tag, null)
    const session = yield* h.runEffect(Effect.flatMap(SessionContextProjection.Tag, p => p.get))
    const timezone = session.context?.timezone ?? null
    const rendered = getView(memory.messages, timezone, 'agent')
    return rendered
      .filter(m => m.role === 'user')
      .map(m => m.content.map(part => part.type === 'text' ? part.text : '[image]').join(''))
      .join('\n')
  })
}
