/**
 * Browser Tools
 *
 * Tools for visual browser interaction via WebHarness.
 * All tools are in the 'browser' group and access the harness via BrowserHarnessTag.
 */

import { Context, Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
import type { WebHarness } from '@magnitudedev/browser-harness'

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

export const clickTool = createTool({
  name: 'click',
  group: 'browser',
  description: 'Click at coordinates (x, y) on the page',
  inputSchema: Schema.Struct({
    x: Schema.Number.annotations({ description: 'X coordinate' }),
    y: Schema.Number.annotations({ description: 'Y coordinate' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  argMapping: ['x', 'y'],
  bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'x', attr: 'x' }, { field: 'y', attr: 'y' }], selfClosing: true }, xmlOutput: { type: 'tag' as const } } as const,
  execute: ({ x, y }) =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.click({ x, y }), catch: browserError })
      return 'clicked'
    }),
})

// =============================================================================
// doubleClick
// =============================================================================

export const doubleClickTool = createTool({
  name: 'doubleClick',
  group: 'browser',
  description: 'Double-click at coordinates (x, y) on the page',
  inputSchema: Schema.Struct({
    x: Schema.Number.annotations({ description: 'X coordinate' }),
    y: Schema.Number.annotations({ description: 'Y coordinate' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  argMapping: ['x', 'y'],
  bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'x', attr: 'x' }, { field: 'y', attr: 'y' }], selfClosing: true }, xmlOutput: { type: 'tag' as const } } as const,
  execute: ({ x, y }) =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.doubleClick({ x, y }), catch: browserError })
      return 'double-clicked'
    }),
})

// =============================================================================
// rightClick
// =============================================================================

export const rightClickTool = createTool({
  name: 'rightClick',
  group: 'browser',
  description: 'Right-click at coordinates (x, y) on the page',
  inputSchema: Schema.Struct({
    x: Schema.Number.annotations({ description: 'X coordinate' }),
    y: Schema.Number.annotations({ description: 'Y coordinate' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  argMapping: ['x', 'y'],
  bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'x', attr: 'x' }, { field: 'y', attr: 'y' }], selfClosing: true }, xmlOutput: { type: 'tag' as const } } as const,
  execute: ({ x, y }) =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.rightClick({ x, y }), catch: browserError })
      return 'right-clicked'
    }),
})

// =============================================================================
// type
// =============================================================================

export const typeTool = createTool({
  name: 'type',
  group: 'browser',
  description: 'Type text content. Use after clicking on an input field. Supports <enter> and <tab> special keys.',
  inputSchema: Schema.Struct({
    content: Schema.String.annotations({ description: 'Text to type (supports <enter> and <tab>)' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  argMapping: ['content'],
  bindings: { xmlInput: { type: 'tag', body: 'content' }, xmlOutput: { type: 'tag' as const } } as const,
  execute: ({ content }) =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.type({ content }), catch: browserError })
      return 'typed'
    }),
})

// =============================================================================
// scroll
// =============================================================================

export const scrollTool = createTool({
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
  argMapping: ['x', 'y', 'deltaX', 'deltaY'],
  bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'x', attr: 'x' }, { field: 'y', attr: 'y' }, { field: 'deltaX', attr: 'deltaX' }, { field: 'deltaY', attr: 'deltaY' }], selfClosing: true }, xmlOutput: { type: 'tag' as const } } as const,
  execute: ({ x, y, deltaX, deltaY }) =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.scroll({ x, y, deltaX: deltaX ?? 0, deltaY }), catch: browserError })
      return 'scrolled'
    }),
})

// =============================================================================
// drag
// =============================================================================

export const dragTool = createTool({
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
  argMapping: ['x1', 'y1', 'x2', 'y2'],
  bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'x1', attr: 'x1' }, { field: 'y1', attr: 'y1' }, { field: 'x2', attr: 'x2' }, { field: 'y2', attr: 'y2' }], selfClosing: true }, xmlOutput: { type: 'tag' as const } } as const,
  execute: ({ x1, y1, x2, y2 }) =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.drag({ x1, y1, x2, y2 }), catch: browserError })
      return 'dragged'
    }),
})

// =============================================================================
// navigate
// =============================================================================

export const navigateTool = createTool({
  name: 'navigate',
  group: 'browser',
  description: 'Navigate to a URL',
  inputSchema: Schema.Struct({
    url: Schema.String.annotations({ description: 'URL to navigate to' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  argMapping: ['url'],
  bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'url', attr: 'url' }], selfClosing: true }, xmlOutput: { type: 'tag' as const } } as const,
  execute: ({ url }) =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.navigate(url), catch: browserError })
      return `navigated to ${url}`
    }),
})

// =============================================================================
// goBack
// =============================================================================

export const goBackTool = createTool({
  name: 'goBack',
  group: 'browser',
  description: 'Go back to the previous page',
  inputSchema: Schema.Struct({}),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  bindings: { xmlInput: { type: 'tag', selfClosing: true }, xmlOutput: { type: 'tag' as const } } as const,
  execute: () =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.goBack(), catch: browserError })
      return 'went back'
    }),
})

// =============================================================================
// switchTab
// =============================================================================

export const switchTabTool = createTool({
  name: 'switchTab',
  group: 'browser',
  description: 'Switch to a browser tab by index',
  inputSchema: Schema.Struct({
    index: Schema.Number.annotations({ description: 'Tab index to switch to' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  argMapping: ['index'],
  bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'index', attr: 'index' }], selfClosing: true }, xmlOutput: { type: 'tag' as const } } as const,
  execute: ({ index }) =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.switchTab({ index }), catch: browserError })
      return `switched to tab ${index}`
    }),
})

// =============================================================================
// newTab
// =============================================================================

export const newTabTool = createTool({
  name: 'newTab',
  group: 'browser',
  description: 'Open a new browser tab',
  inputSchema: Schema.Struct({}),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  bindings: { xmlInput: { type: 'tag', selfClosing: true }, xmlOutput: { type: 'tag' as const } } as const,
  execute: () =>
    Effect.gen(function* () {
      const harness = yield* BrowserHarnessTag
      yield* Effect.tryPromise({ try: () => harness.newTab(), catch: browserError })
      return 'new tab opened'
    }),
})

// =============================================================================
// screenshot
// =============================================================================

export const screenshotTool = createTool({
  name: 'screenshot',
  group: 'browser',
  description: 'Take a screenshot of the current page. You automatically receive screenshots, but use this to see the result of your actions.',
  inputSchema: Schema.Struct({}),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  bindings: { xmlInput: { type: 'tag', selfClosing: true }, xmlOutput: { type: 'tag' as const } } as const,
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

// =============================================================================
// evaluate
// =============================================================================

export const evaluateTool = createTool({
  name: 'evaluate',
  group: 'browser',
  description: 'Execute JavaScript code in the browser page context. Returns the stringified result.',
  inputSchema: Schema.Struct({
    code: Schema.String.annotations({ description: 'JavaScript code to execute in the browser' }),
  }),
  outputSchema: Schema.String,
  errorSchema: BrowserError,
  argMapping: ['code'],
  bindings: { xmlInput: { type: 'tag', body: 'code' }, xmlOutput: { type: 'tag' as const } } as const,
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
