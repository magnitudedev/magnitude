import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getView } from '../../src/projections/memory'
import { getRootMemory, inboxMessages, lastInboxMessage, getRenderedUserText } from './helpers'

describe('memory/timeline-events', () => {
  it.live('subagent activity lifecycle (created → activity → idle) appears in timeline', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subforkId = 'fork-sub-activity'

      yield* h.send({
        type: 'agent_created',
        forkId: subforkId,
        parentForkId: null,
        agentId: 'builder-auth',
        name: 'builder-auth',
        role: 'builder',
        context: '',
        mode: 'spawn',
        taskId: 'task-sub-1',
        message: 'do work',
      })

      yield* h.send({ type: 'turn_started', forkId: subforkId, turnId: 'sub-turn-1', chainId: 'sub-chain-1' })
      yield* h.send({ type: 'message_start', forkId: subforkId, turnId: 'sub-turn-1', id: 'm1', destination: { kind: 'parent' } })
      yield* h.send({ type: 'message_chunk', forkId: subforkId, turnId: 'sub-turn-1', id: 'm1', text: 'working on auth flow' })
      yield* h.send({ type: 'message_end', forkId: subforkId, turnId: 'sub-turn-1', id: 'm1' })
      yield* h.send({
        type: 'turn_completed',
        forkId: subforkId,
        turnId: 'sub-turn-1',
        chainId: 'sub-chain-1',
        strategyId: 'xml-act',
        responseParts: [{ type: 'text', content: 'done' }],
        toolCalls: [],
        observedResults: [],
        result: { success: true, turnDecision: 'yield' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })

      yield* h.send({
        type: 'agent_killed',
        forkId: subforkId,
        parentForkId: null,
        agentId: 'builder-auth',
        reason: 'test cleanup',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 'root-turn-1', chainId: 'root-chain-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        const blocks = inbox.timeline.filter(t => t.kind === 'agent_block')
        expect(blocks.length).toBeGreaterThan(0)
        expect(blocks.some(b => b.atoms.some(a => a.kind === 'idle'))).toBe(true)
      }

      const rendered = yield* getRenderedUserText(h)
      expect(rendered).toContain('<agent id="builder-auth"')
      expect(rendered).toContain('status="idle"')
      expect(rendered).toContain('<idle')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('user presence changes are deferred then injected at flush', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'window_focus_changed', forkId: null, focused: false })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-presence-1', chainId: 'c-presence-1' })

      const after = yield* getRootMemory(h)
      const timeline = inboxMessages(after).flatMap(m => m.type === 'inbox' ? m.timeline : [])
      expect(timeline.some(t => t.kind === 'user_presence')).toBe(true)

      const rendered = yield* getRenderedUserText(h)
      expect(rendered).toContain('<user-presence>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('workflow and skill events map to timeline entries', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({
        type: 'skill_started',
        forkId: null,
        source: 'assistant',
        skill: { name: 'deploy', description: 'deploy workflow', preamble: '', phases: [{ name: 'phase1', prompt: 'do phase 1' }] },
      })
      yield* h.send({
        type: 'phase_criteria_verdict',
        forkId: null,
        parentForkId: null,
        criteriaIndex: 0,
        criteriaName: 'tests',
        criteriaType: 'shell',
        status: 'passed',
        command: 'npm test',
      })
      yield* h.send({
        type: 'phase_verdict',
        forkId: null,
        passed: true,
        verdicts: [{ criteriaIndex: 0, criteriaName: 'tests', passed: true, reason: 'ok' }],
        nextPhasePrompt: 'ship it',
        workflowCompleted: false,
      })
      yield* h.send({ type: 'skill_completed', forkId: null, skillName: 'deploy' })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-workflow-1', chainId: 'c-workflow-1' })

      const memory = yield* getRootMemory(h)
      const timeline = inboxMessages(memory).flatMap(m => m.type === 'inbox' ? m.timeline : [])
      expect(timeline.some(t => t.kind === 'phase_criteria')).toBe(true)
      expect(timeline.some(t => t.kind === 'phase_verdict')).toBe(true)
      expect(timeline.some(t => t.kind === 'skill_completed')).toBe(true)

      const rendered = yield* getRenderedUserText(h)
      expect(rendered).toContain('<phase_criteria')
      expect(rendered).toContain('<phase_verdict')
      expect(rendered).toContain('<workflow_phase')
      expect(rendered).toContain('<skill_completed')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('turn error entries are injected and rendered in results', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-rem-1', chainId: 'c-rem-1' })
      yield* h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't-rem-1',
        chainId: 'c-rem-1',
        strategyId: 'xml-act',
        responseParts: [{ type: 'text', content: 'ok' }],
        toolCalls: [],
        observedResults: [],
        result: {
          success: true,
          turnDecision: 'yield',
          errors: [{ code: 'nonexistent_agent_destination', message: 'follow up on deployment checks' }],
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-rem-2', chainId: 'c-rem-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        expect(inbox.results.some(r => r.kind === 'error' && r.message.includes('follow up on deployment checks'))).toBe(true)
      }

      const rendered = yield* getRenderedUserText(h)
      expect(rendered).toContain('<error>follow up on deployment checks</error>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('observations during turn append a new inbox message (append-only)', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-obs-1', chainId: 'c-obs-1' })
      yield* h.send({
        type: 'observations_captured',
        forkId: null,
        turnId: 't-obs-1',
        parts: [
          { type: 'text', text: 'saw important output' },
          { type: 'image', base64: 'aGVsbG8=', mediaType: 'image/png', width: 10, height: 10 },
        ],
      })

      const memory = yield* getRootMemory(h)
      const allInbox = inboxMessages(memory)
      expect(allInbox.length).toBeGreaterThan(0)
      const last = allInbox[allInbox.length - 1]
      const prev = allInbox.length > 1 ? allInbox[allInbox.length - 2] : undefined

      expect(last?.type).toBe('inbox')
      if (last?.type === 'inbox') {
        expect(last.timeline.some(t => t.kind === 'observation')).toBe(true)
      }
      if (prev && prev.type === 'inbox') {
        expect(prev.timeline.some(t => t.kind === 'observation')).toBe(false)
      }

      const rendered = getView(memory.messages, 'UTC', 'agent')
        .filter(m => m.role === 'user')
        .flatMap(m => m.content)
      const hasObservationText = rendered.some(p => p.type === 'text' && p.text.includes('saw important output'))
      const hasImage = rendered.some(p => p.type === 'image')
      expect(hasObservationText).toBe(true)
      expect(hasImage).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
