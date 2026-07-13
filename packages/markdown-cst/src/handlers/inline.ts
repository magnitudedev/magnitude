/**
 * Inline Handlers
 *
 * Handlers for inline elements: emphasis, strong, code, links, images, etc.
 */

import type {
  EmphasisBuilder,
  StrongBuilder,
  StrikethroughBuilder,
  InlineCodeBuilder,
  LinkBuilder,
  ImageBuilder,
} from '../types'
import {
  finalizeEmphasis,
  finalizeStrong,
  finalizeStrikethrough,
  finalizeInlineCode,
  finalizeLink,
  finalizeImage,
} from '../finalize'
import { definePartialHandlers } from './define'
import { getResourceBuilder } from './helpers'

// Track spaces for hard break trailing (needs to persist between enter and exit)
let hardBreakSpaces = 0

// =============================================================================
// ENTER HANDLERS
// =============================================================================

export const enter = definePartialHandlers({
  emphasis: (ctx, token) => {
    const builder: EmphasisBuilder = {
      builderType: 'emphasis',
      positionStart: null,
      positionEnd: null,
      content: [],
      currentText: null,
      pendingSoftBreak: null,
      pendingHardBreak: null,
      delimiter: '*',
    }
    ctx.push(builder)
    ctx.enterToken(token, 'emphasis')
  },

  strong: (ctx, token) => {
    const builder: StrongBuilder = {
      builderType: 'strong',
      positionStart: null,
      positionEnd: null,
      content: [],
      currentText: null,
      pendingSoftBreak: null,
      pendingHardBreak: null,
      delimiter: '**',
    }
    ctx.push(builder)
    ctx.enterToken(token, 'strong')
  },

  strikethrough: (ctx, token) => {
    const builder: StrikethroughBuilder = {
      builderType: 'strikethrough',
      positionStart: null,
      positionEnd: null,
      content: [],
      currentText: null,
      pendingSoftBreak: null,
      pendingHardBreak: null,
      delimiter: '~~',
    }
    ctx.push(builder)
    ctx.enterToken(token, 'strikethrough')
  },

  codeText: (ctx, token) => {
    const builder: InlineCodeBuilder = {
      builderType: 'inlineCode',
      positionStart: null,
      positionEnd: null,
      backticks: 1,
      content: '',
      padding: '',
    }
    ctx.push(builder)
    ctx.enterToken(token, 'inlineCode')
  },

  link: (ctx, token) => {
    const builder: LinkBuilder = {
      builderType: 'link',
      positionStart: null,
      positionEnd: null,
      content: [],
      currentText: null,
      pendingSoftBreak: null,
      pendingHardBreak: null,
      href: '',
      title: null,
      hasAngleBrackets: false,
      titleQuote: null,
      preUrlWhitespace: '',
      midWhitespace: '',
      postTitleWhitespace: '',
      seenUrl: false,
      seenTitle: false,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'link')
  },

  image: (ctx, token) => {
    const builder: ImageBuilder = {
      builderType: 'image',
      positionStart: null,
      positionEnd: null,
      src: '',
      alt: null,
      title: null,
      hasAngleBrackets: false,
      titleQuote: null,
      preUrlWhitespace: '',
      midWhitespace: '',
      postTitleWhitespace: '',
      seenUrl: false,
      seenTitle: false,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'image')
  },

  resource: (ctx) => {
    const resource = getResourceBuilder(ctx)
    if (resource) {
      resource.seenUrl = false
      resource.seenTitle = false
    }
  },

  resourceDestinationString: (ctx) => {
    ctx.buffer()
  },

  resourceTitleString: (ctx) => {
    ctx.buffer()
  },

  hardBreakEscape: (ctx) => {
    ctx.startHardBreak('backslash', 0)
  },

  hardBreakTrailing: (ctx, token) => {
    ctx.flushText()
    hardBreakSpaces = ctx.slice(token).length
  },

  // Text content tokens: actual text appended in exit handlers
  data: () => {},                     // plain text - appended in exit
  codeFlowValue: () => {},            // code block content - appended in exit
  codeTextData: () => {},             // inline code content - appended in exit
  htmlTextData: () => {},             // inline HTML content - appended in exit

  // Character escape/reference: full escape sequence captured in exit
  characterEscape: () => {},          // e.g. \! - full sequence captured in exit
  characterReference: () => {},       // e.g. &amp; - full reference captured in exit

  // Emphasis/strong/strikethrough sub-tokens: delimiter style captured in exit
  emphasisSequence: () => {},         // * or _ - delimiter captured in exit
  strongSequence: () => {},           // ** or __ - delimiter captured in exit
  strikethroughSequence: () => {},    // ~ or ~~ - delimiter captured in exit
  strikethroughSequenceTemporary: () => {}, // temporary sequence during parsing
  emphasisText: () => {},             // container - content via data/inline handlers
  strongText: () => {},               // container - content via data/inline handlers
  strikethroughText: () => {},        // container - content via data/inline handlers

  // Inline code sub-tokens: backtick count and padding captured in exit
  codeTextSequence: () => {},         // ` or `` etc - count captured in exit
  codeTextPadding: () => {},          // space padding - captured in exit

  // Label tokens: for links/images, text captured in labelText exit
  label: () => {},                    // container for [text]
  labelText: () => {},                // the text inside [] - captured in exit for images
  labelLink: () => {},                // marks this as a link label
  labelImage: () => {},               // marks this as an image label
  labelMarker: () => {},              // the [ and ] characters
  labelImageMarker: () => {},         // the ! before image
  labelEnd: () => {},                 // marks end of label

  // Resource sub-tokens: URL/title captured via resourceDestinationString/resourceTitleString
  resourceDestination: () => {},      // container for URL
  resourceDestinationLiteral: () => {}, // <url> style container
  resourceDestinationLiteralMarker: () => {}, // < > markers - hasAngleBrackets set in exit
  resourceDestinationRaw: () => {},   // bare url style container
  resourceMarker: () => {},           // the ( and ) around resource
  resourceTitle: () => {},            // container for title
  resourceTitleMarker: () => {},      // " or ' or ( - captured in exit

  // Reference tokens: reference links compiled as raw text for lossless round-trip
  reference: () => {},                // [ref] part of reference link
  referenceMarker: () => {},          // the [ and ] around reference
  referenceString: () => {},          // the reference identifier

  // Autolink tokens: <url> style captured as raw text in exit
  autolink: () => {},                 // container for <url>
  autolinkEmail: () => {},            // email autolink
  autolinkMarker: () => {},           // < > markers
  autolinkProtocol: () => {},         // http:// etc
  literalAutolink: () => {},          // GFM bare URL - captured in exit

  // Character sub-tokens: parent captures full escape/reference
  characterEscapeValue: () => {},     // the escaped character - parent captures full
  characterReferenceMarker: () => {}, // & and ; markers - parent captures full
  characterReferenceMarkerNumeric: () => {}, // # in &#123; - parent captures full
  characterReferenceMarkerHexadecimal: () => {}, // x in &#x1F; - parent captures full
  characterReferenceValue: () => {},  // the reference value - parent captures full

  // escapeMarker: the \ in hard break escape - hardBreakEscape handler tracks this
  escapeMarker: () => {},
})

// =============================================================================
// EXIT HANDLERS
// =============================================================================

export const exit = definePartialHandlers({
  emphasis: (ctx, token) => {
    ctx.exitToken(token)
    ctx.flushText()
    const builder = ctx.pop('emphasis')
    const node = finalizeEmphasis(builder)
    ctx.addInline(node)
  },

  emphasisSequence: (ctx, token) => {
    const builder = ctx.find('emphasis')
    if (builder) {
      const seq = ctx.slice(token)
      builder.delimiter = seq[0] as '*' | '_'
    }
  },

  strong: (ctx, token) => {
    ctx.exitToken(token)
    ctx.flushText()
    const builder = ctx.pop('strong')
    const node = finalizeStrong(builder)
    ctx.addInline(node)
  },

  strongSequence: (ctx, token) => {
    const builder = ctx.find('strong')
    if (builder) {
      const seq = ctx.slice(token)
      builder.delimiter = seq.slice(0, 2) as '**' | '__'
    }
  },

  strikethrough: (ctx, token) => {
    ctx.exitToken(token)
    ctx.flushText()
    const builder = ctx.pop('strikethrough')
    const node = finalizeStrikethrough(builder)
    ctx.addInline(node)
  },

  strikethroughSequence: (ctx, token) => {
    const builder = ctx.find('strikethrough')
    if (builder) {
      const seq = ctx.slice(token)
      builder.delimiter = seq.length === 1 ? '~' : '~~'
    }
  },

  codeText: (ctx, token) => {
    ctx.exitToken(token)
    const builder = ctx.pop('inlineCode')

    const content = builder.content
    if (
      content.length >= 2 &&
      (content[0] === ' ' || content[0] === '\n') &&
      (content[content.length - 1] === ' ' || content[content.length - 1] === '\n')
    ) {
      builder.padding = content[0]
      builder.content = content.slice(1, -1)
    }

    const node = finalizeInlineCode(builder)
    ctx.addInline(node)
  },

  codeTextSequence: (ctx, token) => {
    const builder = ctx.find('inlineCode')
    if (builder) {
      const seq = ctx.slice(token)
      builder.backticks = seq.length
    }
  },

  codeTextPadding: (ctx, token) => {
    const text = ctx.slice(token)
    ctx.appendText(text)
  },

  link: (ctx, token) => {
    ctx.exitToken(token)
    ctx.flushText()
    const builder = ctx.pop('link')

    if (!builder.href && !builder.seenUrl) {
      const text = ctx.slice(token)
      ctx.appendText(text)
      return
    }

    const node = finalizeLink(builder)
    ctx.addInline(node)
  },

  autolink: (ctx, token) => {
    const text = ctx.slice(token)
    ctx.appendText(text)
  },

  literalAutolink: (ctx, token) => {
    const text = ctx.slice(token)
    ctx.appendText(text)
  },

  image: (ctx, token) => {
    ctx.exitToken(token)
    const builder = ctx.pop('image')

    if (!builder.src && !builder.seenUrl) {
      const text = ctx.slice(token)
      ctx.appendText(text)
      return
    }

    const node = finalizeImage(builder)
    ctx.addInline(node)
  },

  resourceDestinationString: (ctx) => {
    const url = ctx.resume()

    const resource = getResourceBuilder(ctx)
    if (resource) {
      if (resource.builderType === 'image') {
        resource.src = url
      } else {
        resource.href = url
      }
      resource.seenUrl = true
    }
  },

  resourceDestinationLiteralMarker: (ctx) => {
    const resource = getResourceBuilder(ctx)
    if (resource) {
      resource.hasAngleBrackets = true
    }
  },

  resourceTitleString: (ctx) => {
    const title = ctx.resume()

    const resource = getResourceBuilder(ctx)
    if (resource) {
      resource.title = title
      resource.seenTitle = true
    }
  },

  resourceTitleMarker: (ctx, token) => {
    const marker = ctx.slice(token)

    const resource = getResourceBuilder(ctx)
    if (resource && resource.titleQuote === null) {
      resource.titleQuote = marker as '"' | "'" | '('
    }
  },

  labelText: (ctx, token) => {
    const text = ctx.slice(token)

    const image = ctx.find('image')
    if (image) {
      image.alt = text
    }
  },

  hardBreakTrailing: (ctx) => {
    ctx.startHardBreak('space', hardBreakSpaces)
    hardBreakSpaces = 0
  },

  data: (ctx, token) => {
    const text = ctx.slice(token)
    ctx.appendText(text)
  },

  codeFlowValue: (ctx, token) => {
    const text = ctx.slice(token)
    ctx.appendText(text)
  },

  codeTextData: (ctx, token) => {
    const text = ctx.slice(token)
    ctx.appendText(text)
  },

  htmlTextData: (ctx, token) => {
    const text = ctx.slice(token)
    ctx.appendText(text)
  },

  characterEscape: (ctx, token) => {
    // Only append the escaped character, not the backslash
    // The token spans the full \x sequence, but we render just x
    const text = ctx.slice(token)
    ctx.appendText(text.length > 1 ? text.slice(1) : text)
  },

  characterReference: (ctx, token) => {
    const text = ctx.slice(token)
    ctx.appendText(text)
  },

  // Emphasis/strong/strikethrough text containers: content handled by data/inline handlers
  emphasisText: () => {},             // container - content via nested handlers
  strongText: () => {},               // container - content via nested handlers
  strikethroughText: () => {},        // container - content via nested handlers
  strikethroughSequenceTemporary: () => {}, // temporary sequence during parsing

  // Resource containers: URL/title already captured via string handlers
  resource: () => {},                 // container for (url "title") - state reset in enter
  resourceDestination: () => {},      // container - URL captured via resourceDestinationString
  resourceDestinationLiteral: () => {}, // container - URL captured via resourceDestinationString
  resourceDestinationRaw: () => {},   // container - URL captured via resourceDestinationString
  resourceMarker: () => {},           // ( ) markers - structural
  resourceTitle: () => {},            // container - title captured via resourceTitleString

  // Label containers: alt text captured in labelText, rest structural
  label: () => {},                    // container for [text] - content via nested handlers
  labelLink: () => {},                // marker - indicates link
  labelImage: () => {},               // marker - indicates image
  labelMarker: () => {},              // [ ] markers - structural
  labelImageMarker: () => {},         // ! marker - structural
  labelEnd: () => {},                 // marker - indicates label end

  // Reference containers: reference links compiled as raw text
  reference: () => {},                // container - full link captured in link exit
  referenceMarker: () => {},          // [ ] markers - full link captured in link exit
  referenceString: () => {},          // ref text - full link captured in link exit

  // Hard break escape: state tracked in enter, line ending handles the break
  hardBreakEscape: () => {},          // \ at end of line - break started in enter

  // Autolink sub-tokens: full autolink captured in autolink exit
  autolinkEmail: () => {},            // email content - parent captures full <email>
  autolinkMarker: () => {},           // < > markers - parent captures full <url>
  autolinkProtocol: () => {},         // http:// - parent captures full <url>

  // Character escape/reference sub-tokens: parent captures full sequence
  characterEscapeValue: () => {},     // escaped char - parent captures \x
  characterReferenceMarker: () => {}, // & ; markers - parent captures &ref;
  characterReferenceMarkerNumeric: () => {}, // # marker - parent captures &#123;
  characterReferenceMarkerHexadecimal: () => {}, // x marker - parent captures &#x1F;
  characterReferenceValue: () => {},  // value - parent captures full reference

  // Escape marker: part of hard break handling
  escapeMarker: () => {},             // \ character - hardBreakEscape tracks this
})
