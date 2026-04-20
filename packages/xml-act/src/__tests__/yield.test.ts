import { describe, test, expect } from 'vitest'
import { Effect, Stream } from 'effect'
import { createXmlRuntime } from '../execution/xml-runtime'
import { createStreamingXmlParser } from '../parser'
import type { XmlRuntimeConfig, XmlTurnEngineEvent } from '../types'
import { YIELD_USER, YIELD_TOOL, YIELD_WORKER, YIELD_PARENT } from '../constants'

function cfg(): XmlRuntimeConfig {
  return { tools: new Map() }
}

function ofType<T extends XmlTurnEngineEvent['_tag']>(events: XmlTurnEngineEvent[], tag: T) {
  return events.filter(e => e._tag === tag) as Extract<XmlTurnEngineEvent, { _tag: T }>[]
}

async function run(chunks: string[]): Promise<XmlTurnEngineEvent[]> {
  const runtime = createXmlRuntime(cfg())
  const stream = Stream.fromIterable(chunks)
  const events = await Effect.runPromise(Stream.runCollect(runtime.streamWith(stream)))
  return Array.from(events)
}

describe('yield tags', () => {
  test('basic yield-user', async () => {
    const events = await run(['<message to="user">hello</message>\n' + YIELD_USER])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: { target: 'user' }, termination: 'natural' })
  })

  test('basic yield-tool', async () => {
    const events = await run(['<message to="user">hello</message>\n' + YIELD_TOOL])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: { target: 'tool' }, termination: 'natural' })
  })

  test('basic yield-worker', async () => {
    const events = await run(['<message to="user">hello</message>\n' + YIELD_WORKER])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: { target: 'worker' }, termination: 'natural' })
  })

  test('basic yield-parent (with subagent yield tags)', async () => {
    // Test parser directly with subagent yield tags
    const parser = createStreamingXmlParser(new Set(), new Map(), new Map(), undefined, undefined, ['yield-parent', 'yield-tool'])
    const events = [...parser.processChunk('<message to="parent">hello</message>\n' + YIELD_PARENT), ...parser.flush()]
    const tcs = events.filter((e): e is Extract<typeof e, { _tag: 'TurnControl' }> => e._tag === 'TurnControl')
    expect(tcs).toHaveLength(1)
    expect(tcs[0].target).toBe('parent')
  })

  test('multi-chunk streaming yield-user', async () => {
    const events = await run([
      '<message to="user">hello</message>\n<yield-',
      'user/>',
    ])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: { target: 'user' }, termination: 'natural' })
  })

  test('multi-chunk streaming yield-tool', async () => {
    const events = await run([
      '<message to="user">hello</message>\n<yield-',
      'tool/>',
    ])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: { target: 'tool' }, termination: 'natural' })
  })

  test('yield tag mentioned inside message is not structural', async () => {
    const events = await run([
      '<message to="user">Use ' + YIELD_USER + ' to yield to user</message>\n' + YIELD_USER,
    ])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: { target: 'user' }, termination: 'natural' })
  })

  test('no content after yield tag', async () => {
    const events = await run([
      '<message to="user">hello</message>\n' + YIELD_USER + '\ntrailing content',
    ])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    // Should still have exactly one TurnEnd - trailing content ignored after done
  })

  test('runtime: runaway termination after yield-user', async () => {
    const events = await run([YIELD_USER, '<lens name="x">more stuff'])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: { target: 'user' }, termination: 'runaway' })
  })

  test('runtime: runaway termination after yield-tool', async () => {
    const events = await run([YIELD_TOOL, '<lens name="x">more stuff'])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: { target: 'tool' }, termination: 'runaway' })
  })
})

describe('termination classification', () => {
  const { createStreamingXmlParser } = require('../parser')

  function collectTurnControls(parser: any, chunks: string[], flush = true) {
    const allEvents: any[] = []
    for (const chunk of chunks) {
      allEvents.push(...parser.processChunk(chunk))
    }
    if (flush) {
      allEvents.push(...parser.flush())
    }
    return allEvents.filter((e: any) => e._tag === 'TurnControl')
  }

  test('stop sequence fired (EOF after yield)', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, [YIELD_USER])
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
    expect(tc[0].termination).toBe('natural')
  })

  test('runaway content after yield-user', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, [
      YIELD_USER,
      '<lens name="x">',
    ], false)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
    expect(tc[0].termination).toBe('runaway')
  })

  test('runaway content after yield-tool', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, [
      YIELD_TOOL,
      '<shell>rm -rf</shell>',
    ], false)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('tool')
    expect(tc[0].termination).toBe('runaway')
  })

  test('runaway content after yield-worker', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, [
      YIELD_WORKER,
      '<message to="user">extra</message>',
    ], false)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('worker')
    expect(tc[0].termination).toBe('runaway')
  })

  test('immediate EOF after yield', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, [YIELD_USER])
    expect(tc).toHaveLength(1)
    expect(tc[0].termination).toBe('natural')
  })
})
