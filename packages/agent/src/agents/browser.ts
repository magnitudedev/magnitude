/**
 * Browser Agent Definition
 *
 * Visual browser interaction agent. Has browser tools + think.
 * Receives automatic screenshots before each turn.
 */

import { defineRole, observe, idle, finish, defineThinkingLens } from '@magnitudedev/roles'
import browserPromptRaw from './prompts/browser.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { BrowserHarnessTag } from '../tools/browser-tools'
import { BrowserService } from '../services/browser-service'
import { Effect, Layer } from 'effect'
import { catalog } from '../catalog'

import { browserObservable } from '../observables/browser-observable'
import type { PolicyContext } from './types'
import { allowAll } from './policy'

const strategyLens = defineThinkingLens({
  name: 'strategy',
  trigger: 'When deciding how to interact with the page',
  description: 'How should you use the browser tools to accomplish the task? What sequence of interactions is needed? Consider page state, loading, and what you need to observe.',
})

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When planning your next actions',
  description: 'Plan what browser actions to take this turn. What to click, type, or scroll? What can you do reliably before needing to observe the screen state again?',
})

const systemPrompt = compilePromptTemplate(browserPromptRaw)

const tools = catalog.pick(
  'click',
  'doubleClick',
  'rightClick',
  'type',
  'scroll',
  'drag',
  'navigate',
  'goBack',
  'switchTab',
  'newTab',
  'screenshot',
  'evaluate',
)

export const browserRole = defineRole<typeof tools, 'browser', PolicyContext, BrowserHarnessTag, BrowserService>({
  tools,
  id: 'browser',
  slot: 'browser',
  systemPrompt,
  lenses: [strategyLens, turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,

  setup: ({ forkId }) => Effect.gen(function* () {
    const browserService = yield* BrowserService
    return Layer.succeed(BrowserHarnessTag, {
      get: () => browserService.get(forkId)
    })
  }),

  teardown: ({ forkId }) => Effect.gen(function* () {
    const browserService = yield* BrowserService
    yield* browserService.release(forkId)
  }),

  policy: [allowAll()],

  turn: {
    decide(turnCtx) {
      if (turnCtx.cancelled) return finish()
      if (turnCtx.error) return observe()
      if (turnCtx.toolsCalled.length === 0 && turnCtx.messagesSent.some(m => m.taskId === null)) return idle()
      return observe()
    },
  },

  observables: [browserObservable],
})
