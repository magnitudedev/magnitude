import { describe, test, expect } from 'bun:test'
import { Effect, Stream } from 'effect'
import { guardStream, guardEffectStream } from './stream-guard'

const OPEN = '<actions>'
const CLOSE = '</actions>'

async function collectGuarded(chunks: string[]): Promise<string> {
  async function* gen() {
    for (const c of chunks) {
      yield c
    }
  }

  const out: string[] = []
  for await (const c of guardStream(gen(), CLOSE, OPEN)) {
    out.push(c)
  }
  return out.join('')
}

describe('stream-guard', () => {
  test('does not inject close for inline prose mention of <actions>', async () => {
    const out = await collectGuarded(['hello <actions> world'])
    expect(out).toBe('hello <actions> world')
  })

  test('injects close when structural open exists without structural close', async () => {
    const out = await collectGuarded(['preface\n<actions>\n<tool />\n'])
    expect(out).toBe('preface\n<actions>\n<tool />\n</actions>')
  })

  test('truncates on structural close and ignores trailing content', async () => {
    const out = await collectGuarded(['x\n<actions>\n', '</actions>\nextra'])
    expect(out).toBe('x\n<actions>\n</actions>')
  })

  test('does not treat inline </actions> as structural close', async () => {
    const out = await collectGuarded(['x\n<actions>\nmessage with inline </actions> text'])
    expect(out).toBe('x\n<actions>\nmessage with inline </actions> text</actions>')
  })

  test('trailing-newline boundary alone is enough for open-tag detection', async () => {
    const out = await collectGuarded(['prefix <actions>\n<tool />\n'])
    expect(out).toBe('prefix <actions>\n<tool />\n</actions>')
  })

  test('trailing-newline boundary alone is enough for close-tag truncation', async () => {
    const out = await collectGuarded(['prefix <actions>\nbody ', '</actions>\nextra'])
    expect(out).toBe('prefix <actions>\nbody </actions>')
  })

  test('effect variant follows same boundary behavior', async () => {
    const stream = Stream.fromIterable(['hello <actions> world'])
    const out = await Effect.runPromise(Stream.runCollect(guardEffectStream(stream, CLOSE, OPEN)))
    expect(Array.from(out).join('')).toBe('hello <actions> world')
  })
})