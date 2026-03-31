import { describe, expect, it } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'
import {
  ACTIONS_CLOSE,
  ACTIONS_OPEN,
  AGENT_CREATE_TAG,
  MESSAGE_OPEN,
  MESSAGE_TAG,
  TITLE_CLOSE,
  TITLE_OPEN,
  TITLE_TAG,
  agentCreateOpen,
} from '../constants'

const knownTags = new Set([AGENT_CREATE_TAG])
const childTagMap = new Map<string, Set<string>>([[AGENT_CREATE_TAG, new Set([TITLE_TAG, MESSAGE_TAG])]])

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  return [...parser.processChunk(xml), ...parser.flush()]
}

describe('behavior: child context takes precedence over structural close', () => {
  it('treats ACTIONS_CLOSE token inside open MESSAGE_TAG as literal child text', () => {
    const xml = [
      ACTIONS_OPEN,
      agentCreateOpen({ id: 'nested-1', type: 'builder', observe: '.' }),
      `${TITLE_OPEN}t${TITLE_CLOSE}`,
      `${MESSAGE_OPEN}hello`,
      ACTIONS_CLOSE,
    ].join('\n')

    const events = parse(xml)

    const parseErrors = events.filter(e => e._tag === 'ParseError').map(e => e.error._tag)

    // desired behavior in truncated output:
    // while MESSAGE_TAG is open, ACTIONS_CLOSE is child content and flush should report UnclosedChild
    expect(parseErrors.includes('UnclosedChild')).toBe(true)
    expect(parseErrors.includes('IncompleteTag')).toBe(false)
    expect(parseErrors.includes('UnclosedContainer')).toBe(false)
  })
})
