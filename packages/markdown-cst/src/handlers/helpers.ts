/**
 * Shared helpers for handlers
 */

import type { CompileContext, Builder, ParagraphBuilder, HeadingBuilder, LinkBuilder, ImageBuilder, EmphasisBuilder, StrongBuilder, StrikethroughBuilder } from '../types'
import type {
  DocumentContentNode,
  ListItemContentNode,
  BlockquoteContentNode,
  BlankLinesNode,
  BlockquoteItemBreakNode,
} from '../schema'

// =============================================================================
// RESOURCE BUILDER HELPER
// =============================================================================

export function getResourceBuilder(ctx: CompileContext): LinkBuilder | ImageBuilder | null {
  // Check image first since it might be nested inside link
  const image = ctx.find('image')
  if (image) return image
  const link = ctx.find('link')
  if (link) return link
  return null
}

// =============================================================================
// BLOCK TO PARENT HELPER
// =============================================================================

/** Types allowed in list items */
const LIST_ITEM_TYPES = new Set(['paragraph', 'bulletList', 'orderedList', 'taskList', 'blankLines'])

/** Types allowed in blockquotes */
const BLOCKQUOTE_TYPES = new Set(['paragraph', 'heading', 'bulletList', 'orderedList', 'taskList', 'blockquote', 'blankLines'])

function isListItemContent(block: DocumentContentNode): block is ListItemContentNode {
  return LIST_ITEM_TYPES.has(block.type)
}

function isBlockquoteContent(block: DocumentContentNode): block is BlockquoteContentNode {
  return BLOCKQUOTE_TYPES.has(block.type)
}

export function isUnsupportedInCurrentContext(ctx: CompileContext, blockType: string): boolean {
  const listItem = ctx.find('listItem')
  if (listItem && !LIST_ITEM_TYPES.has(blockType)) {
    return true
  }
  const blockquote = ctx.find('blockquote')
  if (blockquote && !BLOCKQUOTE_TYPES.has(blockType)) {
    return true
  }
  return false
}

export function addBlockToParent(ctx: CompileContext, block: DocumentContentNode): void {
  // Try list item first
  const listItem = ctx.find('listItem')
  if (listItem) {
    // Combine blockquote prefix and list indent for the full indent
    const fullIndent = listItem.pendingBlockquotePrefix + listItem.pendingIndent
    listItem.pendingBlockquotePrefix = ''
    listItem.pendingIndent = ''

    const bulletList = ctx.find('bulletList')
    const orderedList = ctx.find('orderedList')
    const list = bulletList || orderedList
    if (list && list.pendingBreak) {
      list.pendingBreak.continuation = ''
    }

    if (listItem.pendingBlankLines > 0) {
      const blankLines: BlankLinesNode = {
        type: 'blankLines',
        count: listItem.pendingBlankLines,
        meta: { lines: Array(listItem.pendingBlankLines).fill('') },
        position: {
          start: listItem.positionStart ?? block.position.start,
          end: listItem.positionStart ?? block.position.start,
        },
      }
      listItem.content.push({ block: blankLines, indent: '' })
      listItem.pendingBlankLines = 0
    }

    if (!isListItemContent(block)) {
      throw new Error(`Bug: unsupported block type "${block.type}" in list item. Handler should have converted to raw text.`)
    }
    // Preserve the indent for all block types including nested lists
    listItem.content.push({ block, indent: fullIndent })
    return
  }

  // Try blockquote
  const blockquote = ctx.find('blockquote')
  if (blockquote) {
    if (!isBlockquoteContent(block)) {
      throw new Error(`Bug: unsupported block type "${block.type}" in blockquote. Handler should have converted to raw text.`)
    }

    if (blockquote.pendingBreak) {
      if (!blockquote.pendingBreak.continuation && blockquote.currentPrefixChain) {
        blockquote.pendingBreak.continuation = blockquote.currentPrefixChain
      }
      const breakNode: BlockquoteItemBreakNode = {
        type: 'blockquoteItemBreak',
        meta: {
          blankLines: blockquote.pendingBreak.blankLines,
          continuation: blockquote.pendingBreak.continuation,
        },
        position: {
          start: blockquote.positionStart ?? block.position.start,
          end: blockquote.positionStart ?? block.position.start,
        },
      }
      blockquote.content.push(breakNode)
      blockquote.pendingBreak = null
    }

    blockquote.content.push({
      block,
      leadingIndent: blockquote.currentLeadingIndent,
      prefixWhitespace: blockquote.currentPrefixWhitespace,
    })
    blockquote.contentAddedAfterPrefix = true
    return
  }

  // Add to container
  const container = ctx.find('container')
  if (container) {
    container.content.push({ block, leadingIndent: '' })
  }
}

// =============================================================================
// INLINE HELPERS
// =============================================================================

type InlineContentBuilder =
  | ParagraphBuilder
  | HeadingBuilder
  | LinkBuilder
  | EmphasisBuilder
  | StrongBuilder
  | StrikethroughBuilder

export function hasInlineContent(builder: Builder): builder is InlineContentBuilder {
  return (
    builder.builderType === 'paragraph' ||
    builder.builderType === 'heading' ||
    builder.builderType === 'link' ||
    builder.builderType === 'emphasis' ||
    builder.builderType === 'strong' ||
    builder.builderType === 'strikethrough'
  )
}

export function hasPendingHardBreak(ctx: CompileContext): boolean {
  return ctx.hasPendingHardBreak()
}
