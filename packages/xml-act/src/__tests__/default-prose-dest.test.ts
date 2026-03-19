import { describe, expect, test } from 'bun:test'
import { Effect, Stream } from 'effect'
import { createXmlRuntime, type XmlRuntimeEvent } from '../index'

async function collect(xml: string, defaultProseDest?: 'user' | 'parent') {
  const runtime = createXmlRuntime({
    tools: new Map(),
    ...(defaultProseDest ? { defaultProseDest } : {}),
  })
  const events = await Effect.runPromise(
    Stream.runCollect(runtime.streamWith(Stream.make(xml)))
  )
  return Array.from(events) as XmlRuntimeEvent[]
}

describe('xml-act default prose recipient', () => {
  test('message without to defaults to user by default', async () => {
    const events = await collect('<comms><message>hello</message></comms><yield/>')
    const start = events.find(e => e._tag === 'MessageStart')
    expect(start).toBeDefined()
    expect((start as any).dest).toBe('user')
  })

  test('message without to uses configured defaultProseDest=parent', async () => {
    const events = await collect('<comms><message>hello</message></comms><yield/>', 'parent')
    const start = events.find(e => e._tag === 'MessageStart')
    expect(start).toBeDefined()
    expect((start as any).dest).toBe('parent')
  })

  test('explicit to="parent" overrides defaultProseDest=user', async () => {
    const events = await collect('<comms><message to="parent">hello</message></comms><yield/>', 'user')
    const start = events.find(e => e._tag === 'MessageStart')
    expect(start).toBeDefined()
    expect((start as any).dest).toBe('parent')
  })
})