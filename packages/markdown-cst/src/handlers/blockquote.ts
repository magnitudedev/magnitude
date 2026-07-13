/**
 * Blockquote Handlers
 */

import type { BlockquoteBuilder, RawBlockBuilder } from '../types'
import type { ParagraphNode, BlockquoteItemBreakNode, SourcePoint, SourcePosition } from '../schema'
import { finalizeBlockquote } from '../finalize'
import { definePartialHandlers } from './define'
import { isUnsupportedInCurrentContext, addBlockToParent } from './helpers'

function tokenPosition(token: { start: SourcePoint; end: SourcePoint }): SourcePosition {
  return { start: token.start, end: token.end }
}

// =============================================================================
// ENTER HANDLERS
// =============================================================================

export const enter = definePartialHandlers({
  blockQuote: (ctx, token) => {
    if (isUnsupportedInCurrentContext(ctx, 'blockquote')) {
      const builder: RawBlockBuilder = {
        builderType: 'rawBlock',
        positionStart: null,
        positionEnd: null,
        startOffset: token.start.offset,
        originalType: 'blockQuote',
      }
      ctx.push(builder)
      ctx.enterToken(token, 'rawBlock')
      return
    }

    const builder: BlockquoteBuilder = {
      builderType: 'blockquote',
      positionStart: null,
      positionEnd: null,
      content: [],
      currentLeadingIndent: '',
      currentPrefixWhitespace: '',
      pendingBreak: null,
      currentPrefixChain: '',
      seenPrefixOnLine: false,
      contentAddedAfterPrefix: false,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'blockquote')
  },

  blockQuotePrefix: (ctx) => {
    const paragraph = ctx.find('paragraph')
    if (paragraph) {
      return
    }

    const blockquote = ctx.find('blockquote')
    const bulletList = ctx.find('bulletList')
    const orderedList = ctx.find('orderedList')
    const list = bulletList || orderedList

    if (list && blockquote) {
      if (!list.pendingBreak) {
        list.pendingBreak = { blankLines: [], continuation: '' }
      }
      list.pendingBreak.continuation += '>'
      return
    }

    if (blockquote) {
      blockquote.currentPrefixChain += '>'
      blockquote.currentPrefixWhitespace = ''
      blockquote.seenPrefixOnLine = true
      blockquote.contentAddedAfterPrefix = false
    }
  },

  // blockQuoteMarker: the '>' character - tracked via blockQuotePrefix which fires for full prefix
  blockQuoteMarker: () => {},
  // blockQuotePrefixWhitespace: whitespace after '>' - captured in exit handler
  blockQuotePrefixWhitespace: () => {},
})

// =============================================================================
// EXIT HANDLERS
// =============================================================================

export const exit = definePartialHandlers({
  blockQuote: (ctx, token) => {
    const rawBuilder = ctx.current('rawBlock')
    if (rawBuilder && rawBuilder.originalType === 'blockQuote') {
      ctx.exitToken(token)
      const builder = ctx.pop('rawBlock')
      const text = ctx.source.slice(builder.startOffset, token.end.offset)
      const para: ParagraphNode = {
        type: 'paragraph',
        content: text ? [{ type: 'text', text, position: tokenPosition(token) }] : undefined,
        position: tokenPosition(token),
      }
      addBlockToParent(ctx, para)
      return
    }

    ctx.exitToken(token)
    const builder = ctx.pop('blockquote')

    if (builder.seenPrefixOnLine && !builder.contentAddedAfterPrefix) {
      const trailingPrefix = builder.currentPrefixChain
      if (trailingPrefix) {
        if (builder.pendingBreak) {
          builder.pendingBreak.continuation = trailingPrefix
        } else {
          builder.pendingBreak = { blankLines: [], continuation: trailingPrefix }
        }
      }
    }

    if (builder.pendingBreak && (builder.pendingBreak.blankLines.length > 0 || builder.pendingBreak.continuation)) {
      const breakNode: BlockquoteItemBreakNode = {
        type: 'blockquoteItemBreak',
        meta: {
          blankLines: builder.pendingBreak.blankLines,
          continuation: builder.pendingBreak.continuation,
        },
        position: tokenPosition(token),
      }
      builder.content.push(breakNode)
    }

    const node = finalizeBlockquote(builder)
    addBlockToParent(ctx, node)
  },

  blockQuotePrefix: (ctx, token) => {
    const text = ctx.slice(token)

    if (ctx.appendContinuation(text)) {
      return
    }

    const listItem = ctx.find('listItem')
    if (listItem) {
      // Track blockquote prefix separately from list indent.
      // We accumulate within a line (for nested blockquotes like "> > ")
      // but pendingBlockquotePrefix gets reset on lineEnding (handled in flow.ts)
      listItem.pendingBlockquotePrefix += text
      return
    }
  },

  blockQuotePrefixWhitespace: (ctx, token) => {
    const whitespace = ctx.slice(token)

    if (ctx.hasPendingSoftBreak()) {
      return
    }

    const bulletList = ctx.find('bulletList')
    const orderedList = ctx.find('orderedList')
    const list = bulletList || orderedList
    if (list && list.pendingBreak) {
      list.pendingBreak.continuation += whitespace
      return
    }

    const builder = ctx.find('blockquote')
    if (builder) {
      builder.currentPrefixWhitespace = whitespace
      builder.currentPrefixChain += whitespace
    }
  },

  // blockQuoteMarker: the '>' character - parent blockQuotePrefix handles full prefix tracking
  blockQuoteMarker: () => {},
})
