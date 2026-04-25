/**
 * Repro for runon bug: char-by-char streaming of the exact user scenario.
 */

import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'
import { Effect } from 'effect'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import type { RegisteredTool, TurnEngineEvent } from '../types'
import {
  TAG_REASON,
  TAG_INVOKE,
  TAG_MESSAGE,
  TAG_PARAMETER,
  TAG_FILTER,
  YIELD_USER,
  YIELD_INVOKE,
  YIELD_WORKER,
} from '../constants'

const readTool = defineTool({
  name: 'read',
  label: 'Read File',
  description: 'Read a file',
  inputSchema: Schema.Struct({
    path: Schema.String,
  }),
  outputSchema: Schema.String,
  execute: (_input) => Effect.succeed('content'),
})

const readRegistered: RegisteredTool = {
  tool: readTool,
  tagName: 'read',
  groupName: 'default',
}

const tools = new Map<string, RegisteredTool>([
  ['read', readRegistered],
])

// All-at-once parse
function parse(input: string): TurnEngineEvent[] {
  const p = createParser({ tools })
  const knownToolTags = new Set(tools.keys())
  const tokenizer = createTokenizer(
    (token) => p.pushToken(token),
    knownToolTags,
  )
  tokenizer.push(input + '\n')
  tokenizer.end()
  p.end()
  const fromPush = p.drain()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd]
}

// Char-by-char streaming parse
function parseStreaming(input: string): TurnEngineEvent[] {
  const p = createParser({ tools })
  const knownToolTags = new Set(tools.keys())
  const tokenizer = createTokenizer(
    (token) => p.pushToken(token),
    knownToolTags,
  )
  // Feed one char at a time
  for (let i = 0; i < input.length; i++) {
    tokenizer.push(input[i])
  }
  tokenizer.push('\n')
  tokenizer.end()
  p.end()
  const fromPush = p.drain()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd]
}

// Various chunk sizes
function parseChunked(input: string, chunkSize: number): TurnEngineEvent[] {
  const p = createParser({ tools })
  const knownToolTags = new Set(tools.keys())
  const tokenizer = createTokenizer(
    (token) => p.pushToken(token),
    knownToolTags,
  )
  for (let i = 0; i < input.length; i += chunkSize) {
    tokenizer.push(input.slice(i, i + chunkSize))
  }
  tokenizer.push('\n')
  tokenizer.end()
  p.end()
  const fromPush = p.drain()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd]
}

const BUG_INPUT =
  `<${TAG_REASON} about="alignment">User added a requirement: show the date the trial expired on in the billing card.</${TAG_REASON}>\n` +
  `\n` +
  `Got it — I'll include the trial expiration date in the card as well. Let me read the billing page code first. <${TAG_INVOKE} tool="read">\n` +
  `<${TAG_PARAMETER} name="path">app/(console)/billing/billing-client.tsx</${TAG_PARAMETER}>\n` +
  `</${TAG_INVOKE}>\n` +
  `\n` +
  `<${TAG_INVOKE} tool="read">\n` +
  `<${TAG_PARAMETER} name="path">lib/billing/plan-resolution.ts</${TAG_PARAMETER}>\n` +
  `</${TAG_INVOKE}>\n` +
  `\n` +
  YIELD_INVOKE

describe('runon bug: yield not terminating', () => {
  it('all-at-once: yield should be recognized', () => {
    const events = parse(BUG_INPUT)
    console.log('ALL-AT-ONCE:', JSON.stringify(events.map(e => e._tag)))

    const turnEnd = events.find(e => e._tag === 'TurnEnd')
    expect(turnEnd).toBeDefined()
    if (turnEnd && turnEnd._tag === 'TurnEnd' && turnEnd.outcome._tag === 'Success') {
      expect(turnEnd.outcome.turnControl.target).toBe('invoke')
    }
  })

  it('char-by-char streaming: yield should be recognized', () => {
    const events = parseStreaming(BUG_INPUT)
    console.log('CHAR-BY-CHAR:', JSON.stringify(events.map(e => e._tag)))

    const turnEnd = events.find(e => e._tag === 'TurnEnd')
    expect(turnEnd).toBeDefined()
    if (turnEnd && turnEnd._tag === 'TurnEnd' && turnEnd.outcome._tag === 'Success') {
      expect(turnEnd.outcome.turnControl.target).toBe('invoke')
    }
  })

  for (const size of [2, 3, 5, 7, 11, 13]) {
    it(`chunked (${size}): yield should be recognized`, () => {
      const events = parseChunked(BUG_INPUT, size)
      const turnEnd = events.find(e => e._tag === 'TurnEnd')
      expect(turnEnd).toBeDefined()
      if (turnEnd && turnEnd._tag === 'TurnEnd' && turnEnd.outcome._tag === 'Success') {
        expect(turnEnd.outcome.turnControl.target).toBe('invoke')
      }
    })
  }
})