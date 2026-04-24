import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getRenderedUserText, getRootMemory, inboxMessages, lastInboxMessage, sendUserMessage } from './helpers'

describe('memory rendering', () => {
  it.live('progressive activation renders simple user message without marker', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'hello there',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-render-simple-flush', chainId: 'c-render-simple-flush' })

      const userTexts = yield* getRenderedUserText(h)
      expect(userTexts).toContain('<magnitude:message from="user">hello there</magnitude:message>')
      expect(userTexts).toContain('--- ')
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('user message with attachments renders mention and image content', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641599000,
        text: 'see attachments',
        attachments: [
          { type: 'mention', path: 'src/a.ts', contentType: 'text' },
          {
            type: 'image',
            base64: 'ZmFrZQ==',
            mediaType: 'image/png',
            width: 1,
            height: 1,
            filename: 'pixel.png',
          },
        ],
      })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'follow up',
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-render-attachments', chainId: 'c-render-attachments' })

      const userTexts = yield* getRenderedUserText(h)
      expect(userTexts).toContain('see attachments')
      expect(userTexts).toContain('follow up')
    }).pipe(Effect.provide(TestHarnessLive({
      sessionContext: { cwd: '/' },
      files: {
        '/src/a.ts': 'export const a = 1',
      },
      workers: { turnController: false },
    })))
  )

  it.live('results render before timeline', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({
        type: 'turn_started',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
      })
      yield* h.send({
        type: 'turn_outcome',

        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',


        outcome: {
          _tag: 'Completed',
          completion: {
            yieldTarget: 'user',
            feedback: [{ _tag: 'InvalidMessageDestination', destination: 'unknown', message: 'check this' }],
          },
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({
        type: 'turn_started',
        forkId: null,
        turnId: 't-2',
        chainId: 'c-1',
      })

      const userTexts = yield* getRenderedUserText(h)

      expect(userTexts).toContain('<error>check this</error>')
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('time markers use full date first then time-only in same day', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({
        type: 'turn_started',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
      })

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'first marker',
      })

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641660000,
        text: 'second marker',
      })

      yield* h.send({
        type: 'turn_started',
        forkId: null,
        turnId: 't-2',
        chainId: 'c-1',
      })

      const text = yield* getRenderedUserText(h)

      expect(text).toContain('--- ')
      expect(text).toContain('first marker')
      expect(text).toContain('second marker')
    }).pipe(Effect.provide(TestHarnessLive({ sessionContext: { timezone: 'UTC' }, workers: { turnController: false } })))
  )

  it.live('date transitions render full date again', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711670340000,
        text: 'before midnight',
      })

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711670520000,
        text: 'after midnight',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const text = yield* getRenderedUserText(h)

      expect(text).toContain('--- ')
      expect(text).toContain('before midnight')
      expect(text).toContain('after midnight')
    }).pipe(Effect.provide(TestHarnessLive({ sessionContext: { timezone: 'UTC' }, workers: { turnController: false } })))
  )

  it.live('attention is not shown when only user messages are present', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'please review',
      })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641660000,
        text: 'later event',
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const text = yield* getRenderedUserText(h)

      expect(text).not.toContain('<attention>')
    }).pipe(Effect.provide(TestHarnessLive({ sessionContext: { timezone: 'UTC' }, workers: { turnController: false } })))
  )

  it.live('multiple inbox messages stay in order', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'first',
      })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641660000,
        text: 'second',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-render-order-flush', chainId: 'c-render-order-flush' })

      const text = yield* getRenderedUserText(h)
      expect(text.indexOf('first')).toBeLessThan(text.indexOf('second'))
    }).pipe(Effect.provide(TestHarnessLive({ sessionContext: { timezone: 'UTC' }, workers: { turnController: false } })))
  )

  it.live('multi-inbox with results and timeline both render', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'start',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'turn_outcome',

        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',


        outcome: {
          _tag: 'Completed',
          completion: {
            yieldTarget: 'user',
            feedback: [{ _tag: 'InvalidMessageDestination', destination: 'unknown', message: 'after turn' }],
          },
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const text = yield* getRenderedUserText(h)
      expect(text).toContain('start')
      expect(text).toContain('<error>after turn</error>')
    }).pipe(Effect.provide(TestHarnessLive({ sessionContext: { timezone: 'UTC' }, workers: { turnController: false } })))
  )
})
