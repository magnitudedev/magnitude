/**
 * Flow Handlers
 *
 * Handlers for structural/flow tokens: whitespace, line endings, line prefix, etc.
 */

import { definePartialHandlers } from './define'
import { getResourceBuilder, hasPendingHardBreak } from './helpers'

// =============================================================================
// ENTER HANDLERS
// =============================================================================

export const enter = definePartialHandlers({
  lineEnding: (ctx) => {
    const blockquote = ctx.find('blockquote')
    if (blockquote) {
      const paragraph = ctx.find('paragraph')
      if (!paragraph) {
        blockquote.seenPrefixOnLine = false
        blockquote.currentPrefixChain = ''
      }
    }

    // Reset listItem's pendingBlockquotePrefix on new line (when not in paragraph).
    // This ensures each content block only gets its line's prefix, not accumulated.
    const listItem = ctx.find('listItem')
    if (listItem) {
      const paragraph = ctx.find('paragraph')
      if (!paragraph) {
        listItem.pendingBlockquotePrefix = ''
      }
    }
  },

  // whitespace: spacing between elements - captured in exit for context-specific handling
  whitespace: () => {},

  // linePrefix/lineSuffix: indentation and trailing whitespace - captured in exit
  linePrefix: () => {},               // leading whitespace on lines - exit handles context
  lineSuffix: () => {},               // trailing whitespace on lines - exit handles context

  // Chunk tokens: internal micromark document structure, not needed for AST
  content: () => {},                  // content container - internal structure
  chunkDocument: () => {},            // document chunk - internal structure
  chunkContent: () => {},             // content chunk - internal structure
  chunkFlow: () => {},                // flow chunk - internal structure
  chunkText: () => {},                // text chunk - internal structure
  chunkString: () => {},              // string chunk - internal structure

  // Attention/space: micromark internal tokens that get compiled away
  attentionSequence: () => {},        // potential emphasis markers - resolved to emphasis/strong
  space: () => {},                    // space character - internal

  // htmlText: inline HTML container - content handled by htmlTextData
  htmlText: () => {},                 // container for <tag> - htmlTextData captures content
})

// =============================================================================
// EXIT HANDLERS
// =============================================================================

export const exit = definePartialHandlers({
  whitespace: (ctx, token) => {
    const ws = ctx.slice(token)

    // Check for table cell - capture leading/trailing whitespace
    const tableCell = ctx.find('tableCell')
    if (tableCell) {
      if (!tableCell.sawContent) {
        tableCell.leadingWhitespace += ws
      } else {
        tableCell.trailingWhitespace += ws
      }
      return
    }

    // Check for code block - whitespace in opening fence
    const codeBlock = ctx.find('codeBlock')
    if (codeBlock && !codeBlock.insideCode) {
      // Whitespace before language is seen → infoWhitespace
      // Whitespace after language is seen → infoMetaWhitespace
      if (codeBlock.language === null) {
        codeBlock.infoWhitespace = ws
      } else {
        // Accumulate all whitespace after language into infoMetaWhitespace
        codeBlock.infoMetaWhitespace += ws
      }
      return
    }

    // Check for heading - opening whitespace or pending
    const heading = ctx.find('heading')
    if (heading) {
      if (heading.content.length === 0 && heading.currentText === null) {
        heading.openingWhitespace = ws
      } else {
        heading.pendingWhitespace = ws
      }
      return
    }
  },

  linePrefix: (ctx, token) => {
    const text = ctx.slice(token)

    if (ctx.appendContinuation(text)) {
      return
    }

    const resource = getResourceBuilder(ctx)
    if (resource && resource.seenUrl && !resource.seenTitle) {
      resource.midWhitespace += text
      return
    }

    const setextHeading = ctx.find('setextHeading')
    if (setextHeading) {
      if (!setextHeading.text) {
        setextHeading.leadingIndent = text
      } else {
        setextHeading.underlineIndent = text
      }
      return
    }

    const codeBlock = ctx.find('codeBlock')
    if (codeBlock) {
      if (codeBlock.inClosingFence) {
        codeBlock.closingFenceIndent = text
      } else if (codeBlock.insideCode) {
        const fragment = ctx.find('fragment')
        if (fragment) {
          fragment.text += text
        }
      }
      return
    }

    const listItem = ctx.find('listItem')
    if (listItem) {
      if (listItem.content.length === 0 && !listItem.seenMarkerLineEnding) {
        listItem.prefixWhitespace += text
      } else {
        listItem.pendingIndent += text
      }
      return
    }

    const blockquote = ctx.find('blockquote')
    if (blockquote) {
      if (blockquote.seenPrefixOnLine) {
        blockquote.currentPrefixWhitespace += text
        blockquote.currentPrefixChain += text
      } else {
        blockquote.currentLeadingIndent = text
        blockquote.currentPrefixChain += text
      }
      return
    }
  },

  lineEnding: (ctx, token) => {
    if (hasPendingHardBreak(ctx)) {
      return
    }

    const heading = ctx.find('heading')
    if (heading && heading.slurpLineEnding) {
      return
    }

    const text = ctx.slice(token)

    const resource = getResourceBuilder(ctx)
    if (resource && resource.seenUrl && !resource.seenTitle) {
      resource.midWhitespace += text
      return
    }

    const top = ctx.stack[ctx.stack.length - 1]
    if (top && (top.builderType === 'fragment' || top.builderType === 'inlineCode')) {
      ctx.appendText(text)
      return
    }

    ctx.startSoftBreak()
  },

  lineSuffix: (ctx, token) => {
    const ws = ctx.slice(token)

    const resource = getResourceBuilder(ctx)
    if (resource) {
      if (!resource.seenUrl) {
        resource.preUrlWhitespace = ws
      } else if (!resource.seenTitle) {
        resource.midWhitespace += ws
      } else {
        resource.postTitleWhitespace = ws
      }
      return
    }

    ctx.appendText(ws)
  },

  // Chunk tokens: internal micromark document structure, not needed for AST
  content: () => {},                  // content container - internal structure
  chunkDocument: () => {},            // document chunk - internal structure
  chunkContent: () => {},             // content chunk - internal structure
  chunkFlow: () => {},                // flow chunk - internal structure
  chunkText: () => {},                // text chunk - internal structure
  chunkString: () => {},              // string chunk - internal structure

  // Attention/space: micromark internal tokens that get compiled away
  attentionSequence: () => {},        // potential emphasis markers - already resolved
  space: () => {},                    // space character - internal

  // htmlText: inline HTML container - content already captured by htmlTextData
  htmlText: () => {},                 // container for <tag> - content via htmlTextData
})
