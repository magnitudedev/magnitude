/**
 * Browser Tools
 *
 * Tools for visual browser interaction via WebHarness.
 * All tools are in the 'browser' group and access the harness via BrowserHarnessTag.
 */

import { Context, Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import type { WebHarness } from '@magnitudedev/browser-harness'
import { getBrowserActionBaseLabel } from './browser-action-visuals'

export interface BrowserHarnessAccessor {
  readonly get: () => Effect.Effect<WebHarness>
}

export class BrowserHarnessTag extends Context.Tag('BrowserHarness')<
  BrowserHarnessTag,
  BrowserHarnessAccessor
>() {}

const BrowserError = ToolErrorSchema('BrowserError', {})
type BrowserError = { readonly _tag: 'BrowserError'; readonly message: string }

function browserError(e: unknown): BrowserError {
  return { _tag: 'BrowserError', message: e instanceof Error ? e.message : String(e) }
}

// =============================================================================
// click
// =============================================================================

export const clickTool = defineTool({
  name: 'click',
  group: 'browser',
  description: 'Click at coordinates (x, y) on the page',
  inputSchema: Schema.Struct({
    x: Schema.Number.annotations({ description: 'X coordinate' }),
    y: Schema.Number.annotations({ description: 'Y coordinate' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('click'),
  execute: ({ x, y }) =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      yield* Effect.tryPromise({ try: () => harness.click({ x, y }), catch: browserError })
      return 'clicked'
    }),
})

// =============================================================================
// doubleClick
// =============================================================================

export const doubleClickTool = defineTool({
  name: 'doubleClick',
  group: 'browser',
  description: 'Double-click at coordinates (x, y) on the page',
  inputSchema: Schema.Struct({
    x: Schema.Number.annotations({ description: 'X coordinate' }),
    y: Schema.Number.annotations({ description: 'Y coordinate' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('doubleClick'),
  execute: ({ x, y }) =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      yield* Effect.tryPromise({ try: () => harness.doubleClick({ x, y }), catch: browserError })
      return 'double-clicked'
    }),
})

// =============================================================================
// rightClick
// =============================================================================

export const rightClickTool = defineTool({
  name: 'rightClick',
  group: 'browser',
  description: 'Right-click at coordinates (x, y) on the page',
  inputSchema: Schema.Struct({
    x: Schema.Number.annotations({ description: 'X coordinate' }),
    y: Schema.Number.annotations({ description: 'Y coordinate' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('rightClick'),
  execute: ({ x, y }) =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      yield* Effect.tryPromise({ try: () => harness.rightClick({ x, y }), catch: browserError })
      return 'right-clicked'
    }),
})

// =============================================================================
// type
// =============================================================================

export const typeTool = defineTool({
  name: 'type',
  group: 'browser',
  description: 'Type text content. Use after clicking on an input field. Supports <enter> and <tab> special keys.',
  inputSchema: Schema.Struct({
    content: Schema.String.annotations({ description: 'Text to type (supports <enter> and <tab>)' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('type'),
  execute: ({ content }) =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      yield* Effect.tryPromise({ try: () => harness.type({ content }), catch: browserError })
      return 'typed'
    }),
})

// =============================================================================
// scroll
// =============================================================================

export const scrollTool = defineTool({
  name: 'scroll',
  group: 'browser',
  description: 'Scroll at coordinates (x, y) by delta amounts. Use positive deltaY to scroll down, negative to scroll up.',
  inputSchema: Schema.Struct({
    x: Schema.Number.annotations({ description: 'X coordinate to scroll at' }),
    y: Schema.Number.annotations({ description: 'Y coordinate to scroll at' }),
    deltaX: Schema.optionalWith(Schema.Number.annotations({ description: 'Horizontal scroll amount' }), { default: () => 0 }),
    deltaY: Schema.Number.annotations({ description: 'Vertical scroll amount (positive = down)' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('scroll'),
  execute: ({ x, y, deltaX, deltaY }) =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      yield* Effect.tryPromise({ try: () => harness.scroll({ x, y, deltaX: deltaX ?? 0, deltaY }), catch: browserError })
      return 'scrolled'
    }),
})

// =============================================================================
// drag
// =============================================================================

export const dragTool = defineTool({
  name: 'drag',
  group: 'browser',
  description: 'Drag from (x1, y1) to (x2, y2)',
  inputSchema: Schema.Struct({
    x1: Schema.Number.annotations({ description: 'Start X' }),
    y1: Schema.Number.annotations({ description: 'Start Y' }),
    x2: Schema.Number.annotations({ description: 'End X' }),
    y2: Schema.Number.annotations({ description: 'End Y' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('drag'),
  execute: ({ x1, y1, x2, y2 }) =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      yield* Effect.tryPromise({ try: () => harness.drag({ x1, y1, x2, y2 }), catch: browserError })
      return 'dragged'
    }),
})

// =============================================================================
// navigate
// =============================================================================

export const navigateTool = defineTool({
  name: 'navigate',
  group: 'browser',
  description: 'Navigate to a URL',
  inputSchema: Schema.Struct({
    url: Schema.String.annotations({ description: 'URL to navigate to' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('navigate'),
  execute: ({ url }) =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      yield* Effect.tryPromise({ try: () => harness.navigate(url), catch: browserError })
      return `navigated to ${url}`
    }),
})

// =============================================================================
// goBack
// =============================================================================

export const goBackTool = defineTool({
  name: 'goBack',
  group: 'browser',
  description: 'Go back to the previous page',
  inputSchema: Schema.Struct({}),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('goBack'),
  execute: () =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      yield* Effect.tryPromise({ try: () => harness.goBack(), catch: browserError })
      return 'went back'
    }),
})

// =============================================================================
// switchTab
// =============================================================================

export const switchTabTool = defineTool({
  name: 'switchTab',
  group: 'browser',
  description: 'Switch to a browser tab by index',
  inputSchema: Schema.Struct({
    index: Schema.Number.annotations({ description: 'Tab index to switch to' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('switchTab'),
  execute: ({ index }) =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      yield* Effect.tryPromise({ try: () => harness.switchTab({ index }), catch: browserError })
      return `switched to tab ${index}`
    }),
})

// =============================================================================
// newTab
// =============================================================================

export const newTabTool = defineTool({
  name: 'newTab',
  group: 'browser',
  description: 'Open a new browser tab',
  inputSchema: Schema.Struct({}),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('newTab'),
  execute: () =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      yield* Effect.tryPromise({ try: () => harness.newTab(), catch: browserError })
      return 'new tab opened'
    }),
})

// =============================================================================
// screenshot
// =============================================================================

export const screenshotTool = defineTool({
  name: 'screenshot',
  group: 'browser',
  description: 'Take a screenshot of the current page. You automatically receive screenshots, but use this to see the result of your actions.',
  inputSchema: Schema.Struct({}),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('screenshot'),
  execute: () =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      const tabState = yield* Effect.tryPromise({ try: () => harness.retrieveTabState(), catch: browserError })
      const tabLines = tabState.tabs.map((t, i) =>
        `${i === tabState.activeTab ? '[ACTIVE] ' : ''}${i}: ${t.title} (${t.url})`
      )
      const activeUrl = tabState.tabs[tabState.activeTab]?.url ?? 'unknown'
      return `Screenshot captured. Current page: ${activeUrl}\nTabs:\n${tabLines.join('\n')}`
    }),
})

// =============================================================================
// evaluate
// =============================================================================

export const evaluateTool = defineTool({
  name: 'evaluate',
  group: 'browser',
  description: 'Execute JavaScript code in the browser page context. Returns the stringified result.',
  inputSchema: Schema.Struct({
    code: Schema.String.annotations({ description: 'JavaScript code to execute in the browser' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  label: () => getBrowserActionBaseLabel('evaluate'),
  execute: ({ code }) =>
    Effect.gen(function* () {
      const { get } = yield* BrowserHarnessTag
      const harness = yield* get()
      const result = yield* Effect.tryPromise({ try: () => harness.page.evaluate(code), catch: browserError })
      try {
        return JSON.stringify(result, null, 2)
      } catch {
        return String(result)
      }
    }),
})

// =============================================================================
// Tool Group
// =============================================================================

export const browserTools = [
  clickTool,
  doubleClickTool,
  rightClickTool,
  typeTool,
  scrollTool,
  dragTool,
  navigateTool,
  goBackTool,
  switchTabTool,
  newTabTool,
  screenshotTool,
  evaluateTool,
]
