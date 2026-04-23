import { describe, it, expect } from 'vitest'
import { Effect, Stream, Chunk } from 'effect'
import { createTurnEngine } from '../engine/turn-engine'
import type { TurnEngineEvent } from '../types'
import os from 'os'
import path from 'path'

function collectEvents(input: string, defaultProseDest: string = 'user'): Promise<TurnEngineEvent[]> {
  const engine = createTurnEngine({
    tools: new Map(),
    defaultProseDest,
    resultsDir: path.join(os.tmpdir(), 'xml-act-test-' + Date.now()),
  })

  const textStream = Stream.succeed(input)
  const eventStream = engine.streamWith(textStream)

  return Effect.runPromise(
    Effect.scoped(
      eventStream.pipe(
        Stream.runCollect,
        Effect.map(chunk => Array.from(Chunk.toReadonlyArray(chunk))),
      )
    )
  )
}

describe('prose-to-message conversion', () => {
  it('converts free prose to message events with defaultProseDest=user', async () => {
    const events = await collectEvents('Hello world' + String.fromCharCode(10))

    console.log('Events:', events.map(e => e._tag))

    const msgStart = events.find(e => e._tag === 'MessageStart')
    expect(msgStart).toBeDefined()
    expect((msgStart as any).to).toBe('user')

    const msgChunks = events.filter(e => e._tag === 'MessageChunk')
    expect(msgChunks.length).toBeGreaterThan(0)

    const msgEnd = events.find(e => e._tag === 'MessageEnd')
    expect(msgEnd).toBeDefined()

    const proseEvents = events.filter(e => e._tag === 'ProseChunk' || e._tag === 'ProseEnd')
    expect(proseEvents.length).toBe(0)
  })

  it('uses defaultProseDest=parent when configured', async () => {
    const events = await collectEvents('Hello world' + String.fromCharCode(10), 'parent')

    const msgStart = events.find(e => e._tag === 'MessageStart')
    expect(msgStart).toBeDefined()
    expect((msgStart as any).to).toBe('parent')
  })

  it('converts prose after reason block to message', async () => {
    const NL = String.fromCharCode(10)
    const input = '<reason about="test">' + NL +
      'Some thinking.' + NL +
      '</reason>' + NL + NL +
      'Let me help you with that.' + NL + NL +
      '<yield_user/>'

    const events = await collectEvents(input)
    console.log('Events:', events.map(e => e._tag))

    expect(events.find(e => e._tag === 'LensStart')).toBeDefined()
    expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()

    const msgStart = events.find(e => e._tag === 'MessageStart')
    expect(msgStart).toBeDefined()
    expect((msgStart as any).to).toBe('user')

    const msgChunks = events.filter(e => e._tag === 'MessageChunk')
    expect(msgChunks.length).toBeGreaterThan(0)
    const fullText = msgChunks.map((e: any) => e.text).join('')
    expect(fullText).toContain('Let me help you with that.')

    const turnEnd = events.find(e => e._tag === 'TurnEnd') as any
    expect(turnEnd).toBeDefined()
    expect(turnEnd.result.turnControl?.target).toBe('user')
  })
})
