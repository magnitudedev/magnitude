import { describe, test, expect } from 'vitest'
import { Stream, Effect } from 'effect'
import { guardEffectStream } from '../stream-guard'
const taskTagOpen = () => '<task id="t1">'
const taskTagClose = () => '</task>'

/**
 * Regression test: the stream guard's injected closing tag must be captured
 * by a downstream tap — simulating the cortex pipeline where rawCodeChunks
 * is accumulated via tap AFTER the guard is applied.
 *
 * Bug: the tap ran before the guard (inside execManager), so the injected
 * closing tag never made it into rawCodeChunks / responseParts / LLM history.
 * Fix: guard is now applied in cortex before the tap.
 */
describe('stream-guard tap captures injected closing tag', () => {
  async function runGuardedTap(chunks: string[]): Promise<string> {
    const captured: string[] = []
    const source = Stream.fromIterable(chunks)
    const closingTag = '\n' + taskTagClose()
    const openingTag = taskTagOpen()
    const guarded = guardEffectStream(source, closingTag, openingTag)
    const tapped = guarded.pipe(
      Stream.tap(chunk => Effect.sync(() => { captured.push(chunk) }))
    )
    await Effect.runPromise(tapped.pipe(Stream.runForEach(() => Effect.void)))
    return captured.join('')
  }

  test('tap captures injected closing tag when LLM omits it', async () => {
    const open = taskTagOpen()
    const close = taskTagClose()
    // Simulate truncated LLM output: has opening tag but no closing tag
    const chunks = ['preface\n' + open + '\n<tool id="t1" />\n']
    const tapped = await runGuardedTap(chunks)
    expect(tapped).toContain(open)
    expect(tapped).toContain(close)
  })

  test('tap captures closing tag when LLM emits it normally', async () => {
    const open = taskTagOpen()
    const close = taskTagClose()
    const chunks = ['preface\n' + open + '\n<tool id="t1" />\n' + close]
    const tapped = await runGuardedTap(chunks)
    expect(tapped).toContain(close)
  })

  test('no closing tag injected for prose-only turn', async () => {
    const tapped = await runGuardedTap(['Just some prose with no actions block'])
    expect(tapped).not.toContain(taskTagClose())
  })
})