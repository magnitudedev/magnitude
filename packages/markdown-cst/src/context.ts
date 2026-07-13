/**
 * Compile Context Implementation
 *
 * Type-safe context for building AST from micromark events.
 */

import type { Token } from './tokenizer'
import type {
  Builder,
  BuilderType,
  CompileContext,
  FragmentBuilder,
  InlineCodeBuilder,
  MutableTextNode,
} from './types'
import type { InlineNode, SoftBreakNode, HardBreakNode, SourcePoint, SourcePosition } from './schema'

const ZERO_POINT: SourcePoint = { line: 1, column: 1, offset: 0 }
const ZERO_POSITION: SourcePosition = { start: ZERO_POINT, end: ZERO_POINT }

function point(point: SourcePoint): SourcePoint {
  return { line: point.line, column: point.column, offset: point.offset }
}

function positionFromPoints(start: SourcePoint | null, end: SourcePoint | null): SourcePosition {
  return {
    start: start ? point(start) : ZERO_POINT,
    end: end ? point(end) : ZERO_POINT,
  }
}

/** Builder with inline content trait */
type InlineContentBuilder = Builder & {
  content: InlineNode[]
  currentText: MutableTextNode | null
  pendingSoftBreak: { continuation: string; position: SourcePosition } | null
  pendingHardBreak: {
    style: 'space' | 'backslash'
    spaces: number
    continuation: string
    position: SourcePosition
  } | null
}

/**
 * Check if a builder has inline content (content array + currentText + pendingSoftBreak)
 */
function hasInlineContent(builder: Builder): builder is InlineContentBuilder {
  return (
    builder.builderType === 'paragraph' ||
    builder.builderType === 'heading' ||
    builder.builderType === 'link' ||
    builder.builderType === 'emphasis' ||
    builder.builderType === 'strong' ||
    builder.builderType === 'strikethrough' ||
    builder.builderType === 'tableCell'
  )
}

/**
 * Flush any pending soft break on a builder, adding it to content
 */
function flushPendingSoftBreak(builder: InlineContentBuilder): void {
  if (builder.pendingSoftBreak) {
    const softBreak: SoftBreakNode = {
      type: 'softBreak',
      meta: { continuation: builder.pendingSoftBreak.continuation },
      position: builder.pendingSoftBreak.position,
    }
    builder.content.push(softBreak)
    builder.pendingSoftBreak = null
  }
}

/**
 * Flush any pending hard break on a builder, adding it to content
 */
function flushPendingHardBreak(builder: InlineContentBuilder): void {
  if (builder.pendingHardBreak) {
    const hardBreak: HardBreakNode = {
      type: 'hardBreak',
      meta: {
        style: builder.pendingHardBreak.style,
        spaces: builder.pendingHardBreak.spaces,
        continuation: builder.pendingHardBreak.continuation,
      },
      position: builder.pendingHardBreak.position,
    }
    builder.content.push(hardBreak)
    builder.pendingHardBreak = null
  }
}

/**
 * Create a compile context
 */
export function createContext(source: string): CompileContext {
  const stack: Builder[] = []
  const tokenStack: Array<{ token: Token; builderType: BuilderType }> = []

  const ctx: CompileContext = {
    stack,
    tokenStack,
    source,
    currentToken: null,

    push<T extends Builder>(builder: T): T {
      if (builder.positionStart === undefined) builder.positionStart = null
      if (builder.positionEnd === undefined) builder.positionEnd = null
      stack.push(builder)
      return builder
    },

    pop<T extends BuilderType>(expectedType: T): Extract<Builder, { builderType: T }> {
      const builder = stack.pop()
      if (!builder) {
        throw new Error(`Cannot pop: stack is empty, expected ${expectedType}`)
      }
      if (builder.builderType !== expectedType) {
        throw new Error(
          `Builder type mismatch: expected ${expectedType}, got ${builder.builderType}`
        )
      }
      return builder as Extract<Builder, { builderType: T }>
    },

    current<T extends BuilderType>(expectedType: T): Extract<Builder, { builderType: T }> | null {
      const builder = stack[stack.length - 1]
      if (!builder || builder.builderType !== expectedType) {
        return null
      }
      return builder as Extract<Builder, { builderType: T }>
    },

    require<T extends BuilderType>(expectedType: T): Extract<Builder, { builderType: T }> {
      const builder = stack[stack.length - 1]
      if (!builder) {
        throw new Error(`Cannot require ${expectedType}: stack is empty`)
      }
      if (builder.builderType !== expectedType) {
        throw new Error(
          `Builder type mismatch: expected ${expectedType}, got ${builder.builderType}`
        )
      }
      return builder as Extract<Builder, { builderType: T }>
    },

    find<T extends BuilderType>(type: T): Extract<Builder, { builderType: T }> | null {
      for (let i = stack.length - 1; i >= 0; i--) {
        if (stack[i].builderType === type) {
          return stack[i] as Extract<Builder, { builderType: T }>
        }
      }
      return null
    },

    enterToken(token: Token, builderType: BuilderType): void {
      tokenStack.push({ token, builderType })
      const builder = stack[stack.length - 1]
      if (builder && builder.builderType === builderType && builder.positionStart === null) {
        builder.positionStart = point(token.start)
      }
    },

    exitToken(token: Token): void {
      const entry = tokenStack.pop()
      if (!entry) {
        throw new Error(`Cannot exit token ${token.type}: token stack is empty`)
      }
      if (entry.token.type !== token.type) {
        throw new Error(
          `Token mismatch: expected ${entry.token.type}, got ${token.type}`
        )
      }
      // Update the builder's end position from the exit token
      const builder = stack.find(b => b.builderType === entry.builderType)
      if (builder) {
        builder.positionEnd = point(token.end)
      }
    },

    slice(token: Token): string {
      return source.slice(token.start.offset, token.end.offset)
    },

    buffer(): void {
      const fragment: FragmentBuilder = {
        builderType: 'fragment',
        positionStart: null,
        positionEnd: null,
        text: '',
      }
      stack.push(fragment)
    },

    resume(): string {
      const builder = stack.pop()
      if (!builder || builder.builderType !== 'fragment') {
        throw new Error('Expected fragment builder for resume()')
      }
      return (builder as FragmentBuilder).text
    },

    appendText(text: string): void {
      const top = stack[stack.length - 1]
      if (!top) return

      // If we're in a fragment (buffering), append to fragment text
      if (top.builderType === 'fragment') {
        ;(top as FragmentBuilder).text += text
        return
      }

      // If we're in inline code builder, append to content
      if (top.builderType === 'inlineCode') {
        ;(top as InlineCodeBuilder).content += text
        return
      }

      // For inline content builders, use currentText
      if (hasInlineContent(top)) {
        // Flush any pending breaks before adding text
        flushPendingSoftBreak(top)
        flushPendingHardBreak(top)

        if (!top.currentText) {
          const currentToken = ctx.currentToken
          const textPosition = currentToken
            ? positionFromPoints(currentToken.start, currentToken.end)
            : ZERO_POSITION
          top.currentText = { type: 'text', text: '', position: textPosition }
          top.content.push(top.currentText)
        }
        top.currentText.text += text
        if (ctx.currentToken) {
          top.currentText.position.end = point(ctx.currentToken.end)
        }
      }
    },

    flushText(): void {
      const top = stack[stack.length - 1]
      if (top && hasInlineContent(top)) {
        top.currentText = null
      }
    },

    addInline(node: InlineNode): void {
      // Flush any pending text first
      ctx.flushText()

      const top = stack[stack.length - 1]
      if (top && hasInlineContent(top)) {
        // Flush any pending breaks before adding inline node
        flushPendingSoftBreak(top)
        flushPendingHardBreak(top)
        top.content.push(node)
      }
    },

    startSoftBreak(): void {
      // Flush current text first
      ctx.flushText()

      const top = stack[stack.length - 1]
      if (top && hasInlineContent(top)) {
        // Start a new pending soft break
        top.pendingSoftBreak = {
          continuation: '',
          position: ctx.currentToken
            ? positionFromPoints(ctx.currentToken.start, ctx.currentToken.end)
            : ZERO_POSITION,
        }
      }
    },

    startHardBreak(style: 'space' | 'backslash', spaces: number): void {
      // Flush current text first
      ctx.flushText()

      const top = stack[stack.length - 1]
      if (top && hasInlineContent(top)) {
        // Flush any existing pending hard break before starting a new one
        flushPendingHardBreak(top)
        // Start a new pending hard break
        top.pendingHardBreak = {
          style,
          spaces,
          continuation: '',
          position: ctx.currentToken
            ? positionFromPoints(ctx.currentToken.start, ctx.currentToken.end)
            : ZERO_POSITION,
        }
      }
    },

    appendContinuation(text: string): boolean {
      const top = stack[stack.length - 1]
      if (top && hasInlineContent(top)) {
        // Try soft break first, then hard break
        if (top.pendingSoftBreak) {
          top.pendingSoftBreak.continuation += text
          if (ctx.currentToken) {
            top.pendingSoftBreak.position.end = point(ctx.currentToken.end)
          }
          return true
        }
        if (top.pendingHardBreak) {
          top.pendingHardBreak.continuation += text
          if (ctx.currentToken) {
            top.pendingHardBreak.position.end = point(ctx.currentToken.end)
          }
          return true
        }
      }
      return false
    },

    hasPendingSoftBreak(): boolean {
      const top = stack[stack.length - 1]
      return !!(top && hasInlineContent(top) && top.pendingSoftBreak)
    },

    hasPendingHardBreak(): boolean {
      const top = stack[stack.length - 1]
      return !!(top && hasInlineContent(top) && top.pendingHardBreak)
    },
  }

  return ctx
}
