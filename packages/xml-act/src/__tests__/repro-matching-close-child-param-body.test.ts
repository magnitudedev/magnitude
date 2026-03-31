import { describe, expect, it } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'
import {
  ACTIONS_CLOSE,
  ACTIONS_OPEN,
  AGENT_CREATE_OPEN_PREFIX,
  AGENT_CREATE_TAG,
  COMMS_CLOSE,
  COMMS_OPEN,
  LENSES_CLOSE,
  LENSES_OPEN,
  TURN_CONTROL_FINISH,
  TURN_CONTROL_NEXT,
  TURN_CONTROL_YIELD,
  xmlClose,
  xmlOpen,
} from '../constants'

type ToolCase = {
  toolTag: string
  attrs: Record<string, string>
  children: [string, string]
}

const TOOL_CASES: ToolCase[] = [
  {
    toolTag: AGENT_CREATE_TAG,
    attrs: { id: 'mc-agent', type: 'builder', observe: '.' },
    children: ['title', 'message'],
  },
  {
    toolTag: 'task-create',
    attrs: { id: 'mc-task', kind: 'plan', observe: '.' },
    children: ['summary', 'details'],
  },
]

const knownTags = new Set(TOOL_CASES.map(t => t.toolTag))
const childTagMap = new Map<string, Set<string>>(TOOL_CASES.map(t => [t.toolTag, new Set(t.children)]))

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  return [...parser.processChunk(xml), ...parser.flush()]
}

function childBodyText(events: ParseEvent[], childTagName: string) {
  return events
    .filter(
      (e): e is Extract<ParseEvent, { _tag: 'ChildBodyChunk' }> =>
        e._tag === 'ChildBodyChunk' && e.childTagName === childTagName,
    )
    .map(e => e.text)
    .join('')
}

function childOpens(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'ChildOpened' }> => e._tag === 'ChildOpened')
}

function childCloses(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'ChildComplete' }> => e._tag === 'ChildComplete')
}

function parseErrors(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'ParseError' }> => e._tag === 'ParseError')
}

function multilineToolOpen(tag: string, attrs: Record<string, string>) {
  const lines = Object.entries(attrs)
    .map(([k, v]) => `${k}="${v}"`)
    .join('\n')
  return `<${tag}\n${lines}\n>`
}

function runActiveChildCase(tc: ToolCase, activeChild: string, interferingText: string) {
  const xml = [
    ACTIONS_OPEN,
    xmlOpen(tc.toolTag, tc.attrs),
    xmlOpen(activeChild),
    'prefix',
    interferingText,
    'suffix',
    xmlClose(activeChild),
    xmlClose(tc.toolTag),
    ACTIONS_CLOSE,
    TURN_CONTROL_NEXT,
  ].join('\n')

  const events = parse(xml)
  expect(childOpens(events).map(e => e.childTagName)).toEqual([activeChild])
  expect(childCloses(events).map(e => e.childTagName)).toEqual([activeChild])
  expect(childBodyText(events, activeChild).includes(interferingText)).toBe(true)
  expect(parseErrors(events)).toHaveLength(0)
}

describe('repro matrix: active child-body passthrough (multi-tool, multi-child, multi-token)', () => {
  for (const tc of TOOL_CASES) {
    const [childA, childB] = tc.children
    for (const activeChild of [childA, childB]) {
      const siblingChild = activeChild === childA ? childB : childA
      const otherTool = tc.toolTag === AGENT_CREATE_TAG ? 'task-create' : AGENT_CREATE_TAG

      const tokenRows: Array<{ name: string; text: string }> = [
        { name: 'ACTIONS_OPEN raw', text: ACTIONS_OPEN },
        { name: 'ACTIONS_CLOSE raw', text: ACTIONS_CLOSE },
        { name: 'COMMS_OPEN raw', text: COMMS_OPEN },
        { name: 'COMMS_CLOSE raw', text: COMMS_CLOSE },
        { name: 'LENSES_OPEN raw', text: LENSES_OPEN },
        { name: 'LENSES_CLOSE raw', text: LENSES_CLOSE },
        { name: 'TURN_CONTROL_NEXT raw', text: TURN_CONTROL_NEXT },
        { name: 'TURN_CONTROL_YIELD raw', text: TURN_CONTROL_YIELD },
        { name: 'TURN_CONTROL_FINISH raw', text: TURN_CONTROL_FINISH },
        { name: 'same-child open text raw', text: xmlOpen(activeChild, { nested: '1' }) },
        { name: 'sibling-child open text raw', text: xmlOpen(siblingChild, { s: '1' }) },
        { name: 'sibling-child close text raw', text: xmlClose(siblingChild) },
        { name: 'parent/tool close text raw', text: xmlClose(tc.toolTag) },
        { name: 'other known tool open text raw', text: xmlOpen(otherTool, { id: 'inside' }) },
        { name: 'other known tool close text raw', text: xmlClose(otherTool) },
        { name: 'malformed tag-like text raw', text: '</not-closed' },
        { name: 'unknown tag-like text raw', text: '<totally-unknown attr="1">' },
        {
          name: 'multiline open-prefix-like text raw',
          text:
            tc.toolTag === AGENT_CREATE_TAG
              ? `${AGENT_CREATE_OPEN_PREFIX}
id="raw-through"
type="explorer"
observe="."
>`
              : `<${tc.toolTag}
id="raw-through"
kind="plan"
observe="."
>`,
        },
      ]

      for (const row of tokenRows) {
        it(`${tc.toolTag}/${activeChild}: ${row.name}`, () => {
          runActiveChildCase(tc, activeChild, row.text)
        })
      }
    }
  }
})

describe('RED matrix: multiline tool opens should parse structurally across tools', () => {
  for (const tc of TOOL_CASES) {
    const [childA, childB] = tc.children
    it(`${tc.toolTag}: multiline open should still open tool and children`, () => {
      const xml = [
        ACTIONS_OPEN,
        multilineToolOpen(tc.toolTag, tc.attrs),
        `${xmlOpen(childA)}a${xmlClose(childA)}`,
        `${xmlOpen(childB)}b${xmlClose(childB)}`,
        xmlClose(tc.toolTag),
        ACTIONS_CLOSE,
        TURN_CONTROL_NEXT,
      ].join('\n')

      const events = parse(xml)
      expect(childOpens(events).map(e => e.childTagName)).toEqual([childA, childB])
      expect(childCloses(events).map(e => e.childTagName)).toEqual([childA, childB])
      expect(parseErrors(events)).toHaveLength(0)
    })
  }
})

describe('RED matrix: missing matching child close emits only child-root-cause error', () => {
  for (const tc of TOOL_CASES) {
    const [childA, childB] = tc.children
    for (const activeChild of [childA, childB]) {
      const siblingChild = activeChild === childA ? childB : childA
      it(`${tc.toolTag}/${activeChild}: only UnclosedChild when ${activeChild} close missing`, () => {
        const xml = [
          ACTIONS_OPEN,
          xmlOpen(tc.toolTag, tc.attrs),
          xmlOpen(activeChild),
          'still-open',
          xmlOpen(siblingChild),
          xmlClose(siblingChild),
          ACTIONS_CLOSE,
          COMMS_OPEN,
          LENSES_OPEN,
          TURN_CONTROL_NEXT,
        ].join('\n')

        const errs = parseErrors(parse(xml)).map(e => e.error._tag)
        expect(errs.includes('UnclosedChild')).toBe(true)
        expect(errs.includes('IncompleteTag')).toBe(false)
        expect(errs.includes('UnclosedContainer')).toBe(false)
      })
    }
  }
})
