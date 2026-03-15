/**
 * List Handlers
 *
 * Handlers for list elements: ordered lists, unordered lists, list items.
 */

import type {
  BulletListBuilder,
  OrderedListBuilder,
  ListItemBuilder,
} from '../types'
import type { BlankLinesNode, ListItemBreakNode, SourcePoint, SourcePosition } from '../schema'
import { finalizeBulletList, finalizeOrderedList } from '../finalize'
import { definePartialHandlers } from './define'
import { addBlockToParent } from './helpers'

function tokenPosition(token: { start: SourcePoint; end: SourcePoint }): SourcePosition {
  return { start: token.start, end: token.end }
}

// =============================================================================
// ENTER HANDLERS
// =============================================================================

export const enter = definePartialHandlers({
  listUnordered: (ctx, token) => {
    const builder: BulletListBuilder = {
      builderType: 'bulletList',
      positionStart: null,
      positionEnd: null,
      marker: '-',
      content: [],
      pendingBreak: null,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'bulletList')
  },

  listOrdered: (ctx, token) => {
    const builder: OrderedListBuilder = {
      builderType: 'orderedList',
      positionStart: null,
      positionEnd: null,
      start: 1,
      delimiter: '.',
      numbers: [],
      content: [],
      expectingFirstValue: true,
      pendingBreak: null,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'orderedList')
  },

  listItem: (ctx, token) => {
    const bulletList = ctx.current('bulletList')
    const orderedList = ctx.current('orderedList')
    const list = bulletList || orderedList

    if (list && list.pendingBreak) {
      // Create a break node if there's a continuation OR blank lines
      // continuation is needed for tight lists in blockquotes (e.g., "> - a\n> - b")
      // blankLines is needed for loose lists
      if (list.pendingBreak.blankLines.length > 0 || list.pendingBreak.continuation) {
        const breakNode: ListItemBreakNode = {
          type: 'listItemBreak',
          meta: {
            blankLines: list.pendingBreak.blankLines,
            continuation: list.pendingBreak.continuation,
          },
          position: tokenPosition(token),
        }
        list.content.push(breakNode)
      }
      list.pendingBreak = null
    }

    const builder: ListItemBuilder = {
      builderType: 'listItem',
      positionStart: null,
      positionEnd: null,
      prefixWhitespace: token._prefixWhitespace,
      indent: token._indent || '',
      content: [],
      pendingIndent: '',
      pendingBlockquotePrefix: '',
      pendingBlankLines: 0,
      seenMarkerLineEnding: false,
      taskCheckbox: null,
    }

    const marker = token._marker
    if (marker && bulletList && bulletList.marker === '-') {
      if (marker === '-' || marker === '*' || marker === '+') {
        bulletList.marker = marker
      }
    }

    const numberString = token._numberString
    if (numberString && orderedList) {
      orderedList.numbers.push(numberString)
    }

    if (marker && orderedList && orderedList.delimiter === '.') {
      if (marker.endsWith(')')) {
        orderedList.delimiter = ')'
      }
    }

    ctx.push(builder)
    ctx.enterToken(token, 'listItem')
  },

  listItemValue: (ctx, token) => {
    const orderedList = ctx.find('orderedList')
    if (orderedList && orderedList.expectingFirstValue) {
      orderedList.start = parseInt(ctx.slice(token), 10)
      orderedList.expectingFirstValue = false
    }
  },

  lineEndingBlank: (ctx, token) => {
    const listItem = ctx.find('listItem')
    const bulletList = ctx.find('bulletList')
    const orderedList = ctx.find('orderedList')
    const blockquote = ctx.find('blockquote')
    const list = bulletList || orderedList

    if (listItem) {
      const listItemIndex = ctx.stack.indexOf(listItem)
      const bulletIndex = bulletList ? ctx.stack.indexOf(bulletList) : -1
      const orderedIndex = orderedList ? ctx.stack.indexOf(orderedList) : -1
      const topListIndex = Math.max(bulletIndex, orderedIndex)

      if (topListIndex > listItemIndex) {
        const targetList = bulletIndex > orderedIndex ? bulletList : orderedList
        if (targetList) {
          const blankContent = blockquote
            ? blockquote.currentLeadingIndent + '>' + blockquote.currentPrefixWhitespace
            : ''

          if (!targetList.pendingBreak) {
            targetList.pendingBreak = { blankLines: [], continuation: '' }
          }
          targetList.pendingBreak.blankLines.push(blankContent)
        }
        return
      }

      if (listItem.content.length > 0) {
        let blankContent = listItem.pendingBlockquotePrefix + listItem.pendingIndent
        if (list && list.pendingBreak && list.pendingBreak.continuation) {
          blankContent = list.pendingBreak.continuation
          list.pendingBreak.continuation = ''
        }
        if (blankContent) {
          const blankLines: BlankLinesNode = {
            type: 'blankLines',
            count: 1,
            meta: { lines: [blankContent] },
            position: tokenPosition(token),
          }
          listItem.content.push({ block: blankLines, indent: '' })
          listItem.pendingBlockquotePrefix = ''
          listItem.pendingIndent = ''
        } else {
          listItem.pendingBlankLines++
        }
      } else if (!listItem.seenMarkerLineEnding) {
        listItem.seenMarkerLineEnding = true
        // Clear pending prefix for blank line after marker
        listItem.pendingBlockquotePrefix = ''
      } else {
        const blankContent = listItem.pendingBlockquotePrefix + listItem.pendingIndent
        if (blankContent) {
          const blankLines: BlankLinesNode = {
            type: 'blankLines',
            count: 1,
            meta: { lines: [blankContent] },
            position: tokenPosition(token),
          }
          listItem.content.push({ block: blankLines, indent: '' })
          listItem.pendingBlockquotePrefix = ''
          listItem.pendingIndent = ''
        } else {
          listItem.pendingBlankLines++
        }
      }
      return
    }

    if (list) {
      if (!list.pendingBreak) {
        list.pendingBreak = { blankLines: [], continuation: '' }
      }

      if (list.pendingBreak.continuation) {
        list.pendingBreak.blankLines.push(list.pendingBreak.continuation)
        list.pendingBreak.continuation = ''
      } else {
        const blankContent = blockquote
          ? blockquote.currentLeadingIndent + '>' + blockquote.currentPrefixWhitespace
          : ''
        list.pendingBreak.blankLines.push(blankContent)
      }
      return
    }

    if (blockquote) {
      if (!blockquote.pendingBreak) {
        blockquote.pendingBreak = { blankLines: [], continuation: '' }
      }
      if (blockquote.pendingBreak.continuation) {
        blockquote.pendingBreak.blankLines.push(blockquote.pendingBreak.continuation)
        blockquote.pendingBreak.continuation = ''
      } else {
        blockquote.pendingBreak.blankLines.push(blockquote.currentPrefixChain)
      }
      blockquote.currentLeadingIndent = ''
      blockquote.currentPrefixChain = ''
      blockquote.seenPrefixOnLine = false
    }
  },

  // listItemMarker - marker is captured via token._marker from tokenizer preprocessing
  listItemMarker: () => {},
  // listItemPrefix - structural token, item prefix is captured via token metadata
  listItemPrefix: () => {},
  // listItemPrefixWhitespace - captured via token._prefixWhitespace from tokenizer
  listItemPrefixWhitespace: () => {},
  // listItemIndent - handled in exit
  listItemIndent: () => {},
})

// =============================================================================
// EXIT HANDLERS
// =============================================================================

export const exit = definePartialHandlers({
  listUnordered: (ctx, token) => {
    ctx.exitToken(token)
    const b = ctx.pop('bulletList')

    // Clear parent listItem's pendingIndent (list indent portion) before adding nested list.
    // The indent before nested list items is already captured in each item's own indent
    // field (from tokenizer). pendingBlockquotePrefix is preserved - it's NOT in the token.
    const parentListItem = ctx.find('listItem')
    if (parentListItem) {
      parentListItem.pendingIndent = ''
    }

    // finalizeBulletList returns array of lists (split at checkbox boundaries)
    const nodes = finalizeBulletList(b)
    for (const node of nodes) {
      addBlockToParent(ctx, node)
    }

    if (b.pendingBreak) {
      const blockquote = ctx.find('blockquote')
      if (blockquote) {
        if (!blockquote.pendingBreak) {
          blockquote.pendingBreak = { blankLines: [], continuation: '' }
        }
        for (const blankContent of b.pendingBreak.blankLines) {
          blockquote.pendingBreak.blankLines.push(blankContent)
        }
        if (b.pendingBreak.continuation) {
          blockquote.pendingBreak.blankLines.push(b.pendingBreak.continuation)
        }
      }
    }
  },

  listOrdered: (ctx, token) => {
    ctx.exitToken(token)
    const b = ctx.pop('orderedList')

    // Clear parent listItem's pendingIndent (list indent portion) before adding nested list.
    // See comment in listUnordered handler for explanation.
    const parentListItem = ctx.find('listItem')
    if (parentListItem) {
      parentListItem.pendingIndent = ''
    }

    // finalizeOrderedList returns array of lists (split at checkbox boundaries)
    const nodes = finalizeOrderedList(b)
    for (const node of nodes) {
      addBlockToParent(ctx, node)
    }

    if (b.pendingBreak) {
      const blockquote = ctx.find('blockquote')
      if (blockquote) {
        if (!blockquote.pendingBreak) {
          blockquote.pendingBreak = { blankLines: [], continuation: '' }
        }
        for (const blankContent of b.pendingBreak.blankLines) {
          blockquote.pendingBreak.blankLines.push(blankContent)
        }
        if (b.pendingBreak.continuation) {
          blockquote.pendingBreak.blankLines.push(b.pendingBreak.continuation)
        }
      }
    }
  },

  listItem: (ctx, token) => {
    ctx.exitToken(token)
    const builder = ctx.pop('listItem')

    const bulletList = ctx.current('bulletList')
    if (bulletList) {
      bulletList.content.push(builder)
      return
    }

    const orderedList = ctx.current('orderedList')
    if (orderedList) {
      orderedList.content.push(builder)
    }
  },

  listItemIndent: (ctx, token) => {
    const indent = ctx.slice(token)

    if (ctx.appendContinuation(indent)) {
      return
    }

    const bulletList = ctx.current('bulletList')
    const orderedList = ctx.current('orderedList')
    if (bulletList || orderedList) {
      return
    }

    const listItem = ctx.find('listItem')
    if (listItem) {
      listItem.pendingIndent += indent
    }
  },

  // listItemValue: start number captured in enter handler
  listItemValue: () => {},
  // lineEndingBlank: all logic handled in enter handler
  lineEndingBlank: () => {},

  // List item sub-tokens: all metadata captured via tokenizer preprocessing on ListItemToken
  listItemMarker: () => {},           // - or * or + or 1. - captured in token._marker
  listItemPrefix: () => {},           // container for marker line - metadata on token
  listItemPrefixWhitespace: () => {}, // whitespace after marker - captured in token._prefixWhitespace
})
