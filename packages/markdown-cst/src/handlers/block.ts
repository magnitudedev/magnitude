/**
 * Block Handlers
 *
 * Handlers for block-level elements: paragraph, headings, code blocks,
 * thematic break, definition, html flow.
 */

import type {
  ParagraphBuilder,
  HeadingBuilder,
  SetextHeadingBuilder,
  CodeBlockBuilder,
  RawBlockBuilder,
} from '../types'
import type { ParagraphNode, HeadingLevel, SourcePoint, SourcePosition } from '../schema'
import {
  finalizeParagraph,
  finalizeHeading,
  finalizeCodeBlock,
  finalizeHorizontalRule,
} from '../finalize'
import { definePartialHandlers } from './define'
import { isUnsupportedInCurrentContext, addBlockToParent } from './helpers'

function tokenPosition(token: { start: SourcePoint; end: SourcePoint }): SourcePosition {
  return { start: token.start, end: token.end }
}

// =============================================================================
// ENTER HANDLERS
// =============================================================================

export const enter = definePartialHandlers({
  paragraph: (ctx, token) => {
    const builder: ParagraphBuilder = {
      builderType: 'paragraph',
      positionStart: null,
      positionEnd: null,
      content: [],
      currentText: null,
      pendingSoftBreak: null,
      pendingHardBreak: null,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'paragraph')
  },

  atxHeading: (ctx, token) => {
    // If inside unsupported context (list item), capture as raw text
    if (isUnsupportedInCurrentContext(ctx, 'heading')) {
      const text = ctx.slice(token)
      const node: ParagraphNode = {
        type: 'paragraph',
        content: text ? [{ type: 'text', text, position: tokenPosition(token) }] : undefined,
        position: tokenPosition(token),
      }
      addBlockToParent(ctx, node)

      const builder: RawBlockBuilder = {
        builderType: 'rawBlock',
        positionStart: null,
        positionEnd: null,
        startOffset: token.start.offset,
        originalType: 'atxHeading',
      }
      ctx.push(builder)
      ctx.enterToken(token, 'rawBlock')
      return
    }

    const builder: HeadingBuilder = {
      builderType: 'heading',
      positionStart: null,
      positionEnd: null,
      content: [],
      currentText: null,
      pendingSoftBreak: null,
      pendingHardBreak: null,
      level: null,
      openingWhitespace: '',
      closingHashes: '',
      trailingWhitespace: '',
      pendingWhitespace: '',
      slurpLineEnding: false,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'heading')
  },

  setextHeading: (ctx, token) => {
    if (isUnsupportedInCurrentContext(ctx, 'heading')) {
      const text = ctx.slice(token)
      const node: ParagraphNode = {
        type: 'paragraph',
        content: text ? [{ type: 'text', text, position: tokenPosition(token) }] : undefined,
        position: tokenPosition(token),
      }
      addBlockToParent(ctx, node)

      const builder: RawBlockBuilder = {
        builderType: 'rawBlock',
        positionStart: null,
        positionEnd: null,
        startOffset: token.start.offset,
        originalType: 'setextHeading',
      }
      ctx.push(builder)
      ctx.enterToken(token, 'rawBlock')
      return
    }

    const builder: SetextHeadingBuilder = {
      builderType: 'setextHeading',
      positionStart: null,
      positionEnd: null,
      leadingIndent: '',
      text: '',
      underlineIndent: '',
      underline: '',
    }
    ctx.push(builder)
    ctx.enterToken(token, 'setextHeading')
  },

  codeFenced: (ctx, token) => {
    if (isUnsupportedInCurrentContext(ctx, 'codeBlock')) {
      const builder: RawBlockBuilder = {
        builderType: 'rawBlock',
        positionStart: null,
        positionEnd: null,
        startOffset: token.start.offset,
        originalType: 'codeFenced',
      }
      ctx.push(builder)
      ctx.enterToken(token, 'rawBlock')
      return
    }

    const builder: CodeBlockBuilder = {
      builderType: 'codeBlock',
      positionStart: null,
      positionEnd: null,
      language: null,
      fence: '`',
      fenceLength: 3,
      closingFenceLength: 3,
      closingFenceIndent: '',
      fenceMeta: '',
      infoWhitespace: '',
      infoMetaWhitespace: '',
      lines: [],
      insideCode: false,
      inClosingFence: false,
      closed: false,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'codeBlock')
  },

  codeFencedFence: (ctx) => {
    const builder = ctx.find('codeBlock')
    if (!builder) return
    if (builder.insideCode) {
      builder.inClosingFence = true
    }
  },

  codeFencedFenceInfo: (ctx) => {
    ctx.buffer()
  },

  codeFencedFenceMeta: (ctx) => {
    ctx.buffer()
  },

  codeIndented: (ctx, token) => {
    const text = ctx.slice(token)
    const node: ParagraphNode = {
      type: 'paragraph',
      content: text ? [{ type: 'text', text, position: tokenPosition(token) }] : undefined,
      position: tokenPosition(token),
    }
    addBlockToParent(ctx, node)

    const builder: RawBlockBuilder = {
      builderType: 'rawBlock',
      positionStart: null,
      positionEnd: null,
      startOffset: token.start.offset,
      originalType: 'codeIndented',
    }
    ctx.push(builder)
    ctx.enterToken(token, 'rawBlock')
  },

  thematicBreak: (ctx, token) => {
    if (isUnsupportedInCurrentContext(ctx, 'horizontalRule')) {
      const text = ctx.slice(token)
      const para: ParagraphNode = {
        type: 'paragraph',
        content: text ? [{ type: 'text', text, position: tokenPosition(token) }] : undefined,
      position: tokenPosition(token),
      }
      addBlockToParent(ctx, para)
      return
    }

    const original = ctx.slice(token)
    const node = finalizeHorizontalRule({
      builderType: 'horizontalRule',
      positionStart: token.start,
      positionEnd: token.end,
      original,
    })
    addBlockToParent(ctx, node)
  },

  htmlFlow: (ctx, token) => {
    let text = ctx.slice(token)
    if (text.endsWith('\n')) {
      text = text.slice(0, -1)
    }
    const node: ParagraphNode = {
      type: 'paragraph',
      content: text ? [{ type: 'text', text, position: tokenPosition(token) }] : undefined,
      position: tokenPosition(token),
    }
    addBlockToParent(ctx, node)

    const builder: RawBlockBuilder = {
      builderType: 'rawBlock',
      positionStart: null,
      positionEnd: null,
      startOffset: token.start.offset,
      originalType: 'htmlFlow',
    }
    ctx.push(builder)
    ctx.enterToken(token, 'rawBlock')
  },

  htmlFlowData: () => {
    // Text appended via appendText in exit handler
  },

  definition: (ctx, token) => {
    const text = ctx.slice(token)
    const node: ParagraphNode = {
      type: 'paragraph',
      content: text ? [{ type: 'text', text, position: tokenPosition(token) }] : undefined,
      position: tokenPosition(token),
    }
    addBlockToParent(ctx, node)

    const builder: RawBlockBuilder = {
      builderType: 'rawBlock',
      positionStart: null,
      positionEnd: null,
      startOffset: token.start.offset,
      originalType: 'definition',
    }
    ctx.push(builder)
    ctx.enterToken(token, 'rawBlock')
  },

  // Definition sub-tokens: definition is compiled as raw paragraph text,
  // so all internal structure tokens are skipped
  definitionLabelString: () => {},    // label text inside []
  definitionDestinationString: () => {}, // URL string
  definitionTitleString: () => {},    // title string
  definitionDestination: () => {},    // destination container
  definitionDestinationLiteral: () => {}, // <url> style destination
  definitionDestinationLiteralMarker: () => {}, // the < and > markers
  definitionDestinationRaw: () => {}, // bare url style destination
  definitionLabel: () => {},          // [label] container
  definitionLabelMarker: () => {},    // the [ and ] markers
  definitionMarker: () => {},         // the : after label
  definitionTitle: () => {},          // title container
  definitionTitleMarker: () => {},    // the " or ' or ( markers

  // Setext heading sub-tokens: text captured in exit, underline captured in exit
  setextHeadingText: () => {},        // heading text content - captured in exit
  setextHeadingLine: () => {},        // underline - captured in exit
  setextHeadingLineSequence: () => {}, // the === or --- characters

  // ATX heading sub-tokens: level set in exit via sequence, text via inline handlers
  atxHeadingText: () => {},           // heading text - handled by inline content handlers
  atxHeadingSequence: () => {},       // ### markers - level set in exit handler

  // Thematic break: full original captured on enter, sequence not needed separately
  thematicBreakSequence: () => {},    // the --- or *** or ___ characters

  // Code fence sub-tokens: sequence info captured in exit handler
  codeFencedFenceSequence: () => {},  // the ``` or ~~~ characters - captured in exit
})

// =============================================================================
// EXIT HANDLERS
// =============================================================================

export const exit = definePartialHandlers({
  paragraph: (ctx, token) => {
    ctx.exitToken(token)
    ctx.flushText()
    const builder = ctx.pop('paragraph')
    const node = finalizeParagraph(builder)
    addBlockToParent(ctx, node)
  },

  atxHeading: (ctx, token) => {
    const rawBlock = ctx.current('rawBlock')
    if (rawBlock && rawBlock.originalType === 'atxHeading') {
      ctx.exitToken(token)
      ctx.pop('rawBlock')
      return
    }

    ctx.exitToken(token)
    ctx.flushText()
    const builder = ctx.pop('heading')
    builder.trailingWhitespace = builder.pendingWhitespace
    const node = finalizeHeading(builder)
    addBlockToParent(ctx, node)
  },

  atxHeadingSequence: (ctx, token) => {
    const builder = ctx.find('heading')
    if (!builder) return

    const seq = ctx.slice(token)
    if (builder.level === null) {
      builder.level = seq.length as HeadingLevel
    } else {
      builder.closingHashes = builder.pendingWhitespace + seq
      builder.pendingWhitespace = ''
    }
  },

  setextHeading: (ctx, token) => {
    const rawBlock = ctx.current('rawBlock')
    if (rawBlock && rawBlock.originalType === 'setextHeading') {
      ctx.exitToken(token)
      ctx.pop('rawBlock')
      return
    }

    ctx.exitToken(token)
    const builder = ctx.pop('setextHeading')

    const rawText = builder.leadingIndent + builder.text + '\n' + builder.underlineIndent + builder.underline
    const node: ParagraphNode = {
      type: 'paragraph',
      content: rawText ? [{ type: 'text', text: rawText, position: tokenPosition(token) }] : undefined,
      position: tokenPosition(token),
    }
    addBlockToParent(ctx, node)
  },

  setextHeadingText: (ctx, token) => {
    const builder = ctx.find('setextHeading')
    if (builder) {
      builder.text = ctx.slice(token)
    }
  },

  setextHeadingLine: (ctx, token) => {
    const builder = ctx.find('setextHeading')
    if (builder) {
      builder.underline = ctx.slice(token)
    }
  },

  codeFenced: (ctx, token) => {
    const rawBuilder = ctx.current('rawBlock')
    if (rawBuilder && rawBuilder.originalType === 'codeFenced') {
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

    const content = ctx.resume()
    const builder = ctx.find('codeBlock')
    if (builder) {
      const value = content.replace(/^(\r?\n|\r)|(\r?\n|\r)$/g, '')
      if (value) {
        builder.lines = [value]
      }
    }
    ctx.exitToken(token)
    const b = ctx.pop('codeBlock')
    const node = finalizeCodeBlock(b)
    addBlockToParent(ctx, node)
  },

  codeFencedFence: (ctx) => {
    const builder = ctx.find('codeBlock')
    if (!builder) return
    if (!builder.insideCode) {
      ctx.buffer()
      builder.insideCode = true
    } else {
      builder.inClosingFence = false
    }
  },

  codeFencedFenceInfo: (ctx) => {
    const info = ctx.resume()
    const builder = ctx.find('codeBlock')
    if (builder) {
      builder.language = info || null
    }
  },

  codeFencedFenceMeta: (ctx) => {
    const meta = ctx.resume()
    const builder = ctx.find('codeBlock')
    if (builder) {
      builder.fenceMeta = meta
    }
  },

  codeFencedFenceSequence: (ctx, token) => {
    const builder = ctx.find('codeBlock')
    if (!builder) return

    const seq = ctx.slice(token)
    if (!builder.insideCode) {
      builder.fence = seq[0] as '`' | '~'
      builder.fenceLength = seq.length
      builder.closingFenceLength = seq.length
    } else {
      builder.closingFenceLength = seq.length
      builder.closed = true
    }
  },

  codeIndented: (ctx, token) => {
    ctx.exitToken(token)
    ctx.pop('rawBlock')
  },

  thematicBreak: () => {
    // Node already added on enter
  },

  htmlFlow: (ctx, token) => {
    ctx.exitToken(token)
    ctx.pop('rawBlock')
  },

  htmlFlowData: (ctx, token) => {
    const text = ctx.slice(token)
    ctx.appendText(text)
  },

  definition: (ctx, token) => {
    ctx.exitToken(token)
    ctx.pop('rawBlock')
  },

  // Definition sub-tokens: definition compiled as raw text, sub-tokens skipped
  definitionLabelString: () => {},    // label text - parent captures raw
  definitionDestinationString: () => {}, // URL - parent captures raw
  definitionTitleString: () => {},    // title - parent captures raw
  definitionDestination: () => {},    // container - parent captures raw
  definitionDestinationLiteral: () => {}, // <url> container - parent captures raw
  definitionDestinationLiteralMarker: () => {}, // < > markers - parent captures raw
  definitionDestinationRaw: () => {}, // bare url container - parent captures raw
  definitionLabel: () => {},          // [label] container - parent captures raw
  definitionLabelMarker: () => {},    // [ ] markers - parent captures raw
  definitionMarker: () => {},         // : marker - parent captures raw
  definitionTitle: () => {},          // title container - parent captures raw
  definitionTitleMarker: () => {},    // " ' ( markers - parent captures raw

  // Setext heading: underline sequence not needed, full line captured elsewhere
  setextHeadingLineSequence: () => {}, // === or --- chars - line captured in setextHeadingLine

  // ATX heading: text content handled by inline handlers (data, emphasis, etc.)
  atxHeadingText: () => {},           // container for heading text - inline handlers process content

  // Thematic break: full original captured on enter, sequence redundant
  thematicBreakSequence: () => {},    // --- or *** chars - full string captured on thematicBreak enter
})
