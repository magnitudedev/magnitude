/**
 * Browser Agent Definition
 *
 * Visual browser interaction agent. Has browser tools + think.
 * Receives automatic screenshots before each turn.
 */

import { toolSet, defineAgent, continue_, yield_, finish, taskThinkingLens, turnThinkingLens } from '@magnitudedev/agent-definition'
import {
  clickTool, doubleClickTool, rightClickTool, typeTool,
  scrollTool, dragTool, navigateTool, goBackTool,
  switchTabTool, newTabTool, screenshotTool, evaluateTool
} from '../tools/browser-tools'

import { thinkTool } from '../tools/globals'
import { browserObservable } from '../observables/browser-observable'
import type { PolicyContext } from './types'

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
  thinkingLenses: [taskThinkingLens, turnThinkingLens],

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
