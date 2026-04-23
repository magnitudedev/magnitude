/**
 * Tests for newline-based close tag confirmation.
 *
 * After a top-level close tag (</reason>, </message>), EITHER a newline OR
 * a left-angle-bracket should confirm the close. The bug manifests during
 * streaming: if the '\n' after a close tag arrives as a separate chunk, it
 * gets absorbed into wsBuffer by isAllWs(). The next non-whitespace content
 * token (e.g. "some prose text") then fails the '<' check in resolvePendingClose
 * and the close tag is rejected — swallowing everything that follows into the
 * (incorrectly still-open) reason/message body.
 *
 * These tests FAIL until the parser (and grammar) fix is applied.
 * They use multiple tokenizer.push() calls to simulate streaming, so that '\n'
 * arrives as a separate whitespace-only Content token rather than being
 * prepended to the next content chunk.
 */

import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'
import { Effect } from 'effect'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import type { RegisteredTool, TurnEngineEvent } from '../types'

// ---------------------------------------------------------------------------
// Test tool setup
// ---------------------------------------------------------------------------

const shellTool = defineTool({
  name: 'shell',
  label: 'Shell',
  description: 'Run a shell command',
  inputSchema: Schema.Struct({
    command: Schema.String,
    timeout: Schema.optional(Schema.Number),
  }),
  outputSchema: Schema.String,
  execute: (_input) => Effect.succeed('ok'),
})

const shellRegistered: RegisteredTool = {
  tool: shellTool,
  tagName: 'shell',
  groupName: 'default',
}

const tools = new Map<string, RegisteredTool>([
  ['shell', shellRegistered],
])

/**
 * Parse by pushing each chunk separately to simulate streaming.
 * This ensures '\n' tokens arrive as distinct Content tokens rather than
 * being concatenated with subsequent content.
 */
function parseStreaming(chunks: string[], customTools = tools): TurnEngineEvent[] {
  const p = createParser({ tools: customTools })
  const knownToolTags = new Set(customTools.keys())
  const events: TurnEngineEvent[] = []
  const tokenizer = createTokenizer(
    (token) => p.pushToken(token),
    knownToolTags,
  )
  for (const chunk of chunks) {
    tokenizer.push(chunk)
    events.push(...p.drain())
  }
  tokenizer.end()
  p.end()
  events.push(...p.drain())
  return events
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('newline close confirmation (streaming)', () => {
  /**
   * Test 1: </reason> then '\n' as separate chunk then prose text then invoke then yield
   *
   * The '\n' arrives as a separate whitespace-only Content token → goes into wsBuffer.
   * "some prose text\n" arrives next → 's' !== '<' and 's' !== '\n' → rejectAllPendingCloses.
   * Result: reason block never closes, invoke is swallowed as body content.
   *
   * Expected: LensEnd fires, ToolInputStarted fires, TurnEnd fires.
   * Currently FAILS: ToolInputStarted is never emitted.
   */
  it('reason closed with \\n (separate chunk) then prose then invoke then yield', () => {
    const chunks = [
      '<reason about="planning">',
      '\nsome reasoning here\n',
      '</reason>',
      '\n',                                          // newline as its own chunk
      'some prose text\n',
      '<invoke tool="shell">\n',
      '<parameter name="command">echo hi</parameter>\n',
      '</invoke>\n',
      '<yield_invoke/>',
    ]

    const events = parseStreaming(chunks)

    // Reason block must close
    expect(events.find(e => e._tag === 'LensStart')).toMatchObject({
      _tag: 'LensStart',
      name: 'planning',
    })
    expect(events.find(e => e._tag === 'LensEnd')).toMatchObject({
      _tag: 'LensEnd',
      name: 'planning',
    })

    // The invoke after the prose must be recognized
    expect(events.find(e => e._tag === 'ToolInputStarted')).toMatchObject({
      _tag: 'ToolInputStarted',
      tagName: 'shell',
    })

    // TurnEnd from yield
    const turnEnd = events.find(e => e._tag === 'TurnEnd')
    expect(turnEnd).toBeDefined()
    expect(turnEnd).toMatchObject({
      _tag: 'TurnEnd',
      result: { _tag: 'Success' },
    })
  })

  /**
   * Test 2: Two consecutive reason blocks where '\n' between them arrives separately
   *
   * The '\n' between </reason> and <reason> arrives as its own chunk → wsBuffer.
   * Then '<reason' arrives → '<' IS the check, so this might already work.
   * But testing it explicitly to guard against regressions.
   *
   * This test may already pass (< is the current confirmation trigger).
   * If it passes, good. If not, it must also be fixed.
   */
  it('two consecutive reason blocks with \\n (separate chunk) between them', () => {
    const chunks = [
      '<reason about="first">',
      '\nfirst reasoning\n',
      '</reason>',
      '\n',                      // separate newline chunk
      '<reason about="second">',
      '\nsecond reasoning\n',
      '</reason>',
      '\n',
      '<yield_user/>',
    ]

    const events = parseStreaming(chunks)

    const lensStarts = events.filter(e => e._tag === 'LensStart')
    const lensEnds = events.filter(e => e._tag === 'LensEnd')

    expect(lensStarts).toHaveLength(2)
    expect(lensEnds).toHaveLength(2)

    expect(lensStarts[0]).toMatchObject({ _tag: 'LensStart', name: 'first' })
    expect(lensStarts[1]).toMatchObject({ _tag: 'LensStart', name: 'second' })
    expect(lensEnds[0]).toMatchObject({ _tag: 'LensEnd', name: 'first' })
    expect(lensEnds[1]).toMatchObject({ _tag: 'LensEnd', name: 'second' })

    expect(events.find(e => e._tag === 'TurnEnd')).toBeDefined()
  })

  /**
   * Test 3: Reason then message, '\n' between them as separate chunk
   *
   * </reason>\n (separate) <message to="user"> ...
   * The '\n' goes to wsBuffer, then '<' arrives → should confirm.
   * Similar to test 2 but with different tag types.
   */
  it('reason then message with \\n (separate chunk) between them', () => {
    const chunks = [
      '<reason about="turn">',
      '\nsome reasoning\n',
      '</reason>',
      '\n',
      '<message to="user">',
      '\nhello world\n',
      '</message>',
      '\n',
      '<yield_user/>',
    ]

    const events = parseStreaming(chunks)

    expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()
    expect(events.find(e => e._tag === 'MessageStart')).toMatchObject({
      _tag: 'MessageStart',
      to: 'user',
    })
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
    expect(events.find(e => e._tag === 'TurnEnd')).toBeDefined()
  })

  /**
   * Test 4: Message closed with '\n' (separate chunk) then non-'<' prose then yield
   *
   * Same core bug as test 1 but for message blocks. The '\n' after </message>
   * goes into wsBuffer, then plain text starts with non-'<' → close rejected.
   */
  it('message closed with \\n (separate chunk) then prose then yield', () => {
    const chunks = [
      '<message to="user">',
      '\nhello world\n',
      '</message>',
      '\n',                      // separate newline chunk
      'This is some trailing prose.\n',
      '<yield_user/>',
    ]

    const events = parseStreaming(chunks)

    expect(events.find(e => e._tag === 'MessageStart')).toMatchObject({
      _tag: 'MessageStart',
      to: 'user',
    })
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()

    const turnEnd = events.find(e => e._tag === 'TurnEnd')
    expect(turnEnd).toBeDefined()
    expect(turnEnd).toMatchObject({
      _tag: 'TurnEnd',
      result: { _tag: 'Success', termination: 'natural' },
    })
  })

  /**
   * Test 5: Multiple '\n' chunks after close tag (blank line) then prose then invoke
   *
   * Multiple separate '\n' chunks go into wsBuffer one by one.
   * Then "some prose" arrives → 's' fails the check → reject.
   */
  it('reason closed with multiple separate \\n chunks then prose then invoke', () => {
    const chunks = [
      '<reason about="analysis">',
      '\nreasoning content\n',
      '</reason>',
      '\n',   // first newline
      '\n',   // second newline (blank line)
      'prose content here\n',
      '<invoke tool="shell">\n',
      '<parameter name="command">ls</parameter>\n',
      '</invoke>\n',
      '<yield_invoke/>',
    ]

    const events = parseStreaming(chunks)

    expect(events.find(e => e._tag === 'LensEnd')).toMatchObject({
      _tag: 'LensEnd',
      name: 'analysis',
    })

    expect(events.find(e => e._tag === 'ToolInputStarted')).toMatchObject({
      _tag: 'ToolInputStarted',
      tagName: 'shell',
    })

    expect(events.find(e => e._tag === 'TurnEnd')).toBeDefined()
  })
})