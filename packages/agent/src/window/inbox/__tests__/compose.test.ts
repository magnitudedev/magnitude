import { describe, expect, test } from 'vitest'
import { Option } from 'effect'
import {
  toTimelineAgentBlock,
  toTimelineObservation,
  toTimelineTaskUpdate,
  toTimelineLifecycleHook,
  toTimelineSubagentUserKilled,
  toTimelineUserMessage,
  toTimelineUserToAgent,
} from '../compose'
import type { UserPart } from '@magnitudedev/ai'
import type { AgentAtom, TimelineAttachment } from '../types'

const TS = 1711641600000

describe('inbox compose', () => {
  test('toTimeline* constructors set correct kinds', () => {
    const attachments: readonly TimelineAttachment[] = [{
      kind: 'mention',
      attachment: { type: 'mention_file', path: 'a.ts' },
      resolution: { status: 'resolved', content: '', truncated: false, originalBytes: 0 },
    }]
    const atoms: readonly AgentAtom[] = [{ kind: 'thought', timestamp: TS, text: 'thinking' }]
    const parts: readonly UserPart[] = [{ _tag: 'TextPart', text: 'obs' }]

    expect(toTimelineUserMessage({ timestamp: TS, text: 'u', attachments, synthetic: Option.none() }).kind).toBe('user_message')
    expect(toTimelineUserToAgent({ timestamp: TS, agentId: 'a1', text: 'u2a' }).kind).toBe('user_to_agent')
    expect(toTimelineAgentBlock({
      timestamp: TS,
      firstAtomTimestamp: TS,
      lastAtomTimestamp: TS,
      agentId: 'a1',
      role: 'engineer',
      status: 'working',
      atoms,
    }).kind).toBe('agent_block')
    expect(toTimelineSubagentUserKilled({ timestamp: TS, agentId: 'a1', agentType: 'builder' }).kind).toBe('worker_user_killed')
    expect(
      toTimelineLifecycleHook({
        timestamp: TS,
        agentId: 'a1',
        role: 'engineer',
        hookType: 'spawn',
        taskId: Option.none(),
        taskTitle: Option.none(),
      }).kind,
    ).toBe('lifecycle_hook')
    expect(
      toTimelineTaskUpdate({
        timestamp: TS,
        action: 'status_changed',
        taskId: 't1',
        title: Option.none(),
        previousStatus: Option.some('pending'),
        nextStatus: Option.some('completed'),
        cancelledCount: Option.none(),
      }).kind,
    ).toBe('task_update')
    expect(toTimelineObservation({ timestamp: TS, parts }).kind).toBe('observation')
  })

  test('readonly arrays are preserved by reference', () => {
    const attachments: readonly TimelineAttachment[] = [{
      kind: 'mention',
      attachment: { type: 'mention_file', path: 'x' },
      resolution: { status: 'resolved', content: '', truncated: false, originalBytes: 0 },
    }]
    const atoms: readonly AgentAtom[] = [{ kind: 'thought', timestamp: TS, text: 't' }]
    const parts: readonly UserPart[] = [{
      _tag: 'ImagePart',
      data: 'abc',
      mediaType: 'image/png',
    }]

    const msg = toTimelineUserMessage({ timestamp: TS, text: 'hello', attachments, synthetic: Option.none() })
    const block = toTimelineAgentBlock({
      timestamp: TS,
      firstAtomTimestamp: TS,
      lastAtomTimestamp: TS,
      agentId: 'a',
      role: 'engineer',
      status: 'working',
      atoms,
    })
    const obs = toTimelineObservation({ timestamp: TS, parts })

    if (msg.kind !== 'user_message') throw new Error('expected user_message')
    if (block.kind !== 'agent_block') throw new Error('expected agent_block')
    if (obs.kind !== 'observation') throw new Error('expected observation')

    expect(msg.attachments).toBe(attachments)
    expect(block.atoms).toBe(atoms)
    expect(obs.parts).toBe(parts)
  })

  test('handles edge cases: empty atoms and undefined optional fields', () => {
    const block = toTimelineAgentBlock({
      timestamp: TS,
      firstAtomTimestamp: TS,
      lastAtomTimestamp: TS,
      agentId: 'a',
      role: 'engineer',
      status: 'working',
      atoms: [],
    })
    const update = toTimelineTaskUpdate({
      timestamp: TS,
      action: 'created',
      taskId: 't1',
      title: Option.none(),
      previousStatus: Option.none(),
      nextStatus: Option.none(),
      cancelledCount: Option.none(),
    })

    if (block.kind !== 'agent_block') throw new Error('expected agent_block')
    if (update.kind !== 'task_update') throw new Error('expected task_update')

    expect(block.atoms).toEqual([])
    expect(Option.isNone(update.title)).toBe(true)
    expect(Option.isNone(update.previousStatus)).toBe(true)
    expect(Option.isNone(update.nextStatus)).toBe(true)
    expect(Option.isNone(update.cancelledCount)).toBe(true)
  })
})
