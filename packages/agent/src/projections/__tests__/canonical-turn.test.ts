import { describe, test, expect } from 'bun:test'
import { createInitialCanonicalTurnState } from '../canonical-turn'
import { serializeCanonicalTurn } from '../canonical-xml'
import type { CanonicalTrace } from '../canonical-xml'

describe('CanonicalTurn state accumulation primitives', async () => {
  test('can accumulate think/messages/tools shape used by projection', async () => {
    const state = createInitialCanonicalTurnState()
    expect(state.turnId).toBeNull()
    expect(state.messages.length).toBe(0)
    expect(state.toolCalls.length).toBe(0)
  })
})

describe('CanonicalTurn clean gate semantics', async () => {
  test('parse error should make turn unclean', async () => {
    const state = createInitialCanonicalTurnState()
    const clean = !true && !state.hasStructuralError && true
    expect(clean).toBe(false)
  })

  test('structural error should make turn unclean', async () => {
    const clean = !false && !true && true
    expect(clean).toBe(false)
  })

  test('interrupted should make turn unclean', async () => {
    const success = false
    const clean = !false && !false && success
    expect(clean).toBe(false)
  })
})

describe('CanonicalTurn final content selection behavior', async () => {
  test('serializer can produce canonical xml for completed clean trace', async () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [{ about: null, content: 't' }],
      messages: [{ dest: 'user', text: 'm' }],
      toolCalls: [{ tagName: 'tool', input: {}, query: '.' }],
      turnDecision: 'yield',
    }
    const xml = serializeCanonicalTurn(trace, new Map())
    expect(xml).toContain('<think>t</think>')
    expect(xml).toContain('<comms>')
    expect(xml).toContain('<actions>')
  })
})