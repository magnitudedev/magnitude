/**
 * Browser Tools
 *
 * Tools for visual browser interaction via WebHarness.
 * All tools are in the 'browser' group and access the harness via BrowserHarnessTag.
 */

import { Context, Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import type { WebHarness } from '@magnitudedev/browser-harness'
import { getBrowserActionBaseLabel } from './browser-action-visuals'

export class BrowserHarnessTag extends Context.Tag('BrowserHarness')<
  BrowserHarnessTag,
  WebHarness
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
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.click({ x, y }), catch: browserError })
      return 'clicked'
    }),
})

export const clickXmlBinding = defineXmlBinding(clickTool, {
  input: { attributes: [{ field: 'x', attr: 'x' }, { field: 'y', attr: 'y' }] },
  output: {},
} as const)

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
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.doubleClick({ x, y }), catch: browserError })
      return 'double-clicked'
    }),
})

export const doubleClickXmlBinding = defineXmlBinding(doubleClickTool, {
  input: { attributes: [{ field: 'x', attr: 'x' }, { field: 'y', attr: 'y' }] },
  output: {},
} as const)

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
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.rightClick({ x, y }), catch: browserError })
      return 'right-clicked'
    }),
})

export const rightClickXmlBinding = defineXmlBinding(rightClickTool, {
  input: { attributes: [{ field: 'x', attr: 'x' }, { field: 'y', attr: 'y' }] },
  output: {},
} as const)

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
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.type({ content }), catch: browserError })
      return 'typed'
    }),
})

export const typeXmlBinding = defineXmlBinding(typeTool, {
  input: { body: 'content' },
  output: {},
} as const)

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
    deltaX: Schema.optionalWith(Schema.Number, { default: () => 0 }).annotations({ description: 'Horizontal scroll amount' }),
    deltaY: Schema.Number.annotations({ description: 'Vertical scroll amount (positive = down)' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,


  label: () => getBrowserActionBaseLabel('scroll'),
  execute: ({ x, y, deltaX, deltaY }) =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.scroll({ x, y, deltaX: deltaX ?? 0, deltaY }), catch: browserError })
      return 'scrolled'
    }),
})

export const scrollXmlBinding = defineXmlBinding(scrollTool, {
  input: {
    attributes: [
      { field: 'x', attr: 'x' },
      { field: 'y', attr: 'y' },
      { field: 'deltaX', attr: 'deltaX' },
      { field: 'deltaY', attr: 'deltaY' },
    ],
  },
  output: {},
} as const)

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
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.drag({ x1, y1, x2, y2 }), catch: browserError })
      return 'dragged'
    }),
})

export const dragXmlBinding = defineXmlBinding(dragTool, {
  input: {
    attributes: [
      { field: 'x1', attr: 'x1' },
      { field: 'y1', attr: 'y1' },
      { field: 'x2', attr: 'x2' },
      { field: 'y2', attr: 'y2' },
    ],
  },
  output: {},
} as const)

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
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.navigate(url), catch: browserError })
      return `navigated to ${url}`
    }),
})

export const navigateXmlBinding = defineXmlBinding(navigateTool, {
  input: { attributes: [{ field: 'url', attr: 'url' }] },
  output: {},
} as const)

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
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.goBack(), catch: browserError })
      return 'went back'
    }),
})

export const goBackXmlBinding = defineXmlBinding(goBackTool, {
  input: {},
  output: {},
} as const)

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
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.switchTab({ index }), catch: browserError })
      return `switched to tab ${index}`
    }),
})

export const switchTabXmlBinding = defineXmlBinding(switchTabTool, {
  input: { attributes: [{ field: 'index', attr: 'index' }] },
  output: {},
} as const)

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
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.newTab(), catch: browserError })
      return 'new tab opened'
    }),
})

export const newTabXmlBinding = defineXmlBinding(newTabTool, {
  input: {},
  output: {},
} as const)

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
      const harness = yield* BrowserHarnessTag
      const tabState = yield* Effect.tryPromise({ try: () => harness.retrieveTabState(), catch: browserError })
      const tabLines = tabState.tabs.map((t, i) =>
        `${i === tabState.activeTab ? '[ACTIVE] ' : ''}${i}: ${t.title} (${t.url})`
      )
      const activeUrl = tabState.tabs[tabState.activeTab]?.url ?? 'unknown'
      return `Screenshot captured. Current page: ${activeUrl}\nTabs:\n${tabLines.join('\n')}`
    }),
})

export const screenshotXmlBinding = defineXmlBinding(screenshotTool, {
  input: {},
  output: {},
} as const)

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
      const harness = yield* BrowserHarnessTag
      const result = yield* Effect.tryPromise({ try: () => harness.page.evaluate(code), catch: browserError })
      try {
        return JSON.stringify(result, null, 2)
      } catch {
        return String(result)
      }
    }),
})

export const evaluateXmlBinding = defineXmlBinding(evaluateTool, {
  input: { body: 'code' },
  output: {},
} as const)

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

export const browserXmlBindings = [
  clickXmlBinding,
  doubleClickXmlBinding,
  rightClickXmlBinding,
  typeXmlBinding,
  scrollXmlBinding,
  dragXmlBinding,
  navigateXmlBinding,
  goBackXmlBinding,
  switchTabXmlBinding,
  newTabXmlBinding,
  screenshotXmlBinding,
  evaluateXmlBinding,
]
