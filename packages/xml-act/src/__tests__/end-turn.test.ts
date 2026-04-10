import { describe, test, expect } from 'vitest'
import { Effect, Stream } from 'effect'
import { createXmlRuntime } from '../execution/xml-runtime'
import type { XmlRuntimeConfig, XmlRuntimeEvent } from '../types'

function cfg(): XmlRuntimeConfig {
  return { tools: new Map() }
}

function ofType<T extends XmlRuntimeEvent['_tag']>(events: XmlRuntimeEvent[], tag: T) {
  return events.filter(e => e._tag === tag) as Extract<XmlRuntimeEvent, { _tag: T }>[]
}

async function run(chunks: string[]): Promise<XmlRuntimeEvent[]> {
  const runtime = createXmlRuntime(cfg())
  const stream = Stream.fromIterable(chunks)
  const events = await Effect.runPromise(Stream.runCollect(runtime.streamWith(stream)))
  return Array.from(events)
}

describe('end-turn block', () => {
  test('basic idle', async () => {
    const events = await run(['<message to="user">hello</message>\n<end-turn>\n<idle/>\n</end-turn>'])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: 'idle', termination: 'natural' })
  })

  test('basic continue', async () => {
    const events = await run(['<message to="user">hello</message>\n<end-turn>\n<continue/>\n</end-turn>'])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: 'continue', termination: 'natural' })
  })

  test('self-closing end-turn defaults to idle', async () => {
    const events = await run(['<message to="user">hello</message>\n<end-turn/>'])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: 'idle', termination: 'natural' })
  })

  test('flush on EOF (stop sequence scenario) - idle captured', async () => {
    const events = await run(['<message to="user">hello</message>\n<end-turn>\n<idle/>'])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: 'idle', termination: 'natural' })
  })

  test('flush on EOF (stop sequence scenario) - continue captured', async () => {
    const events = await run(['<message to="user">hello</message>\n<end-turn>\n<continue/>'])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: 'continue', termination: 'natural' })
  })

  test('flush on EOF - no decision defaults to idle', async () => {
    const events = await run(['<message to="user">hello</message>\n<end-turn>'])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: 'idle', termination: 'natural' })
  })

  test('multi-chunk streaming', async () => {
    const events = await run([
      '<message to="user">hello</message>\n<end-',
      'turn>\n<idle',
      '/>\n</end-turn>',
    ])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: 'idle', termination: 'natural' })
  })

  test('end-turn mentioned inside message is not structural', async () => {
    const events = await run([
      '<message to="user">Use <end-turn><idle/></end-turn> to end your turn</message>\n<end-turn>\n<idle/>\n</end-turn>',
    ])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: 'idle', termination: 'natural' })
  })

  test('no content after end-turn close', async () => {
    const events = await run([
      '<message to="user">hello</message>\n<end-turn>\n<idle/>\n</end-turn>\ntrailing content',
    ])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    // Should still have exactly one TurnEnd - trailing content ignored after done
  })

  test('standalone idle no longer works at top level', async () => {
    const events = await run(['<message to="user">hello</message>\n<idle/>'])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: null, termination: 'natural' })
  })

  test('runtime: runaway termination', async () => {
    const events = await run(['<end-turn>\n<idle/>\n</end-turn>', '<lens name="x">more stuff'])
    const ends = ofType(events, 'TurnEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].result).toEqual({ _tag: 'Success', turnControl: 'idle', termination: 'runaway' })
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

  test('case 1: stop sequence fired (EOF after close)', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, ['<end-turn>\n<idle/>\n</end-turn>'])
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
    expect(tc[0].termination).toBe('natural')
  })

  test('case 2: extra close then EOF', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, [
      '<end-turn>\n<idle/>\n</end-turn>',
      '</end-turn>',
    ])
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
    expect(tc[0].termination).toBe('natural')
  })

  test('case 3: runaway content', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, [
      '<end-turn>\n<idle/>\n</end-turn>',
      '<lens name="x">',
    ], false)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
    expect(tc[0].termination).toBe('runaway')
  })

  test('chunk-split extra close', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, [
      '<end-turn>\n<idle/>\n</end-turn>',
      '</end-tu',
      'rn>',
    ])
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
    expect(tc[0].termination).toBe('natural')
  })

  test('immediate EOF after end-turn', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, ['<end-turn>\n<idle/>\n</end-turn>'])
    expect(tc).toHaveLength(1)
    expect(tc[0].termination).toBe('natural')
  })

  test('runaway with continue decision', () => {
    const parser = createStreamingXmlParser()
    const tc = collectTurnControls(parser, [
      '<end-turn>\n<continue/>\n</end-turn>',
      '<shell>rm -rf</shell>',
    ], false)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
    expect(tc[0].termination).toBe('runaway')
  })
})
