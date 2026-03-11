/**
 * Browser Agent Definition
 *
 * Visual browser interaction agent. Has browser tools + think.
 * Receives automatic screenshots before each turn.
 */

import { toolSet, defineAgent, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/agent-definition'
import {
  clickTool, doubleClickTool, rightClickTool, typeTool,
  scrollTool, dragTool, navigateTool, goBackTool,
  switchTabTool, newTabTool, screenshotTool, evaluateTool
} from '../tools/browser-tools'

import { thinkTool } from '../tools/globals'
import { browserObservable } from '../observables/browser-observable'
import type { PolicyContext } from './types'

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

const tools = toolSet({
  click:        clickTool,
  doubleClick:  doubleClickTool,
  rightClick:   rightClickTool,
  type:         typeTool,
  scroll:       scrollTool,
  drag:         dragTool,
  navigate:     navigateTool,
  goBack:       goBackTool,
  switchTab:    switchTabTool,
  newTab:       newTabTool,
  screenshot:   screenshotTool,
  evaluate:     evaluateTool,

  think:        thinkTool,
})

export const createBrowser = (systemPrompt: string) => defineAgent<typeof tools, PolicyContext>(tools, {
  id: 'browser',
  model: 'browser',
  systemPrompt,
  thinkingLenses: [strategyLens, turnLens],

  permission: (p) => ({
    _default() { return p.allow() },
  }),

  turn: {
    decide(turnCtx) {
      if (turnCtx.cancelled) return finish()
      if (turnCtx.error) return continue_()
      if (turnCtx.toolsCalled.length === 0 && turnCtx.messagesSent.some(m => m.dest === 'parent')) return yield_()
      return continue_()
    },
  },


  display: (d) => ({
    think() { return d.hidden() },
    screenshot() { return d.hidden() },
    _default() { return d.visible() },
  }),

  observables: [browserObservable],
})
