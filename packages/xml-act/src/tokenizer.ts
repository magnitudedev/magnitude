import { KNOWN_CLOSE_TAG_NAMES } from './constants'

/**
 * Streaming XML tokenizer for the new response format.
 *
 * Handles standard XML tags:
 *   Open:      <tag attr="val">
 *   Close:     </tag>
 *   SelfClose: <tag attr="val"/>
 *   Content:   raw text between tags
 *   CDATA:     <![CDATA[...]]>>  (emitted as Content)
 *
 * Close tags for known structural names (reason, message, invoke, parameter, filter)
 * are emitted immediately as Close tokens. The parser handles confirmation logic
 * (greedy last-match, tentative close state). Unknown close tags become Content.
 */

export type Token =
  | { readonly _tag: 'Open';      readonly tagName: string; readonly attrs: ReadonlyMap<string, string>; readonly afterNewline: boolean; readonly raw?: string }
  | { readonly _tag: 'Close';     readonly tagName: string; readonly afterNewline: boolean; readonly raw?: string }
  | { readonly _tag: 'SelfClose'; readonly tagName: string; readonly attrs: ReadonlyMap<string, string>; readonly afterNewline: boolean; readonly raw?: string }
  | { readonly _tag: 'Content';   readonly text: string }

export interface Tokenizer {
  push(chunk: string): void
  end(): void
}

type TagPhase =
  | 'name'
  | 'attrs'
  | 'attrKey'
  | 'attrAfterKey'
  | 'attrBeforeValue'
  | 'attrValueQuoted'
  | 'attrValueUnquoted'
  | 'malformed'

type ActiveTag = {
  raw: string
  savedAfterNewline: boolean
  isClose: boolean
  name: string
  attrs: Map<string, string>
  phase: TagPhase
  pendingSelfClose: boolean
  attrKey: string
  attrValue: string
  attrQuote: '"' | "'" | null
  attrEscaping: boolean
}

const CDATA_OPEN = '<![CDATA['

function isWhitespace(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r'
}

function isNameStart(ch: string): boolean {
  return /[a-zA-Z_]/.test(ch)
}

function isNameContinue(ch: string): boolean {
  return /[a-zA-Z0-9_.:-]/.test(ch)
}

export function createTokenizer(
  onToken: (token: Token) => void,
  knownToolTags?: ReadonlySet<string>,
): Tokenizer {
  let contentBuffer = ''
  let afterNewline = true
  let activeTag: ActiveTag | null = null
  let pendingLt = false
  let cdataBuffer: string | null = null
  let cdataCloseProgress = 0

  function flushContent(): void {
    if (contentBuffer.length === 0) return
    const text = contentBuffer
    contentBuffer = ''
    onToken({ _tag: 'Content', text })
    for (const ch of text) {
      if (ch === '\n') afterNewline = true
      else if (ch !== ' ' && ch !== '\t') afterNewline = false
    }
  }

  function failTagAsContent(): void {
    if (!activeTag) return
    const tag = activeTag
    if (tag.name.length > 0 && !tag.isClose && (knownToolTags?.has(tag.name) || tag.name === 'magnitude:invoke')) {
      tag.phase = 'malformed'
      return
    }
    activeTag = null
    contentBuffer += tag.raw
  }

  function emitTag(tag: ActiveTag): void {
    flushContent()
    if (tag.isClose) {
      if (!KNOWN_CLOSE_TAG_NAMES.has(tag.name)) {
        // Unknown close tag — treat as content
        contentBuffer += tag.raw
        return
      }
      // Emit Close immediately — parser handles confirmation
      onToken({ _tag: 'Close', tagName: tag.name, afterNewline: tag.savedAfterNewline, raw: tag.raw })
      afterNewline = false
    } else if (tag.pendingSelfClose) {
      onToken({
        _tag: 'SelfClose',
        tagName: tag.name,
        attrs: new Map(tag.attrs),
        afterNewline: tag.savedAfterNewline,
        raw: tag.raw,
      })
      afterNewline = false
    } else {
      onToken({
        _tag: 'Open',
        tagName: tag.name,
        attrs: new Map(tag.attrs),
        afterNewline: tag.savedAfterNewline,
        raw: tag.raw,
      })
      afterNewline = false
    }
  }

  function startTag(): void {
    activeTag = {
      raw: '<',
      savedAfterNewline: afterNewline,
      isClose: false,
      name: '',
      attrs: new Map(),
      phase: 'name',
      pendingSelfClose: false,
      attrKey: '',
      attrValue: '',
      attrQuote: null,
      attrEscaping: false,
    }
  }

  function finalizeBooleanAttr(tag: ActiveTag): void {
    if (tag.attrKey.length > 0) {
      tag.attrs.set(tag.attrKey, '')
      tag.attrKey = ''
    }
  }

  function finalizeAttrValue(tag: ActiveTag): void {
    tag.attrs.set(tag.attrKey, tag.attrValue)
    tag.attrKey = ''
    tag.attrValue = ''
    tag.attrQuote = null
  }


  function processTagChar(ch: string): void {
    const tag = activeTag!

    if (tag.phase === 'malformed') {
      tag.raw += ch
      if (ch === '>') {
        flushContent()
        onToken({
          _tag: 'Open',
          tagName: tag.name,
          attrs: new Map(tag.attrs),
          afterNewline: tag.savedAfterNewline,
          raw: tag.raw,
        })
        activeTag = null
        afterNewline = false
      }
      return
    }

    if (ch === '<' && tag.phase !== 'attrValueQuoted') {
      failTagAsContent()
      if (activeTag) return
      startTag()
      return
    }

    tag.raw += ch

    if (tag.phase === 'name') {
      if (tag.raw.startsWith('<!')) {
        if (CDATA_OPEN.startsWith(tag.raw)) {
          if (tag.raw === CDATA_OPEN) {
            activeTag = null
            cdataBuffer = ''
            cdataCloseProgress = 0
          }
          return
        }
        failTagAsContent()
        return
      }

      if (tag.raw.length === 2 && ch === '/') {
        tag.isClose = true
        return
      }

      if (tag.raw.length === 2 && ch === '!') {
        return
      }

      const firstNamePos = tag.isClose ? 2 : 1
      const namePos = tag.raw.length - 1 - firstNamePos
      if (namePos < 0) return

      if (namePos === 0) {
        if (!isNameStart(ch)) {
          failTagAsContent()
        } else {
          tag.name += ch
        }
        return
      }

      if (isNameContinue(ch)) {
        tag.name += ch
        return
      }

      if (isWhitespace(ch)) {
        tag.phase = 'attrs'
        return
      }

      if (ch === '>') {
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        tag.pendingSelfClose = true
        tag.phase = 'attrs'
        return
      }

      failTagAsContent()
      return
    }

    if (tag.phase === 'attrs') {
      if (isWhitespace(ch)) return

      if (ch === '>') {
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        if (tag.pendingSelfClose) {
          failTagAsContent()
          return
        }
        tag.pendingSelfClose = true
        return
      }

      if (!isNameStart(ch)) {
        failTagAsContent()
        return
      }

      tag.attrKey = ch
      tag.phase = 'attrKey'
      return
    }

    if (tag.phase === 'attrKey') {
      if (isNameContinue(ch)) {
        tag.attrKey += ch
        return
      }

      if (isWhitespace(ch)) {
        tag.phase = 'attrAfterKey'
        return
      }

      if (ch === '=') {
        tag.phase = 'attrBeforeValue'
        return
      }

      if (ch === '>') {
        finalizeBooleanAttr(tag)
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        finalizeBooleanAttr(tag)
        tag.pendingSelfClose = true
        tag.phase = 'attrs'
        return
      }

      failTagAsContent()
      return
    }

    if (tag.phase === 'attrAfterKey') {
      if (isWhitespace(ch)) return

      if (ch === '=') {
        tag.phase = 'attrBeforeValue'
        return
      }

      if (ch === '>') {
        finalizeBooleanAttr(tag)
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        finalizeBooleanAttr(tag)
        tag.pendingSelfClose = true
        tag.phase = 'attrs'
        return
      }

      if (isNameStart(ch)) {
        finalizeBooleanAttr(tag)
        tag.attrKey = ch
        tag.phase = 'attrKey'
        return
      }

      failTagAsContent()
      return
    }

    if (tag.phase === 'attrBeforeValue') {
      if (isWhitespace(ch)) return

      if (ch === '"' || ch === "'") {
        tag.attrQuote = ch
        tag.attrValue = ''
        tag.phase = 'attrValueQuoted'
        return
      }

      if (ch === '>') {
        finalizeAttrValue(tag)
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        finalizeAttrValue(tag)
        tag.pendingSelfClose = true
        tag.phase = 'attrs'
        return
      }

      tag.attrValue = ch
      tag.phase = 'attrValueUnquoted'
      return
    }

    if (tag.phase === 'attrValueQuoted') {
      if (tag.attrEscaping) {
        tag.attrEscaping = false
        tag.attrValue += ch === '"' ? '"' : '\\' + ch
        return
      }

      if (ch === '\\' && tag.attrQuote === '"') {
        tag.attrEscaping = true
        return
      }

      if (ch === tag.attrQuote) {
        finalizeAttrValue(tag)
        tag.phase = 'attrs'
        return
      }

      if (ch === '<') {
        failTagAsContent()
        return
      }

      tag.attrValue += ch
      return
    }

    if (tag.phase === 'attrValueUnquoted') {
      if (isWhitespace(ch)) {
        finalizeAttrValue(tag)
        tag.phase = 'attrs'
        return
      }

      if (ch === '>') {
        finalizeAttrValue(tag)
        emitTag(tag)
        activeTag = null
        return
      }

      if (ch === '/' && !tag.isClose) {
        finalizeAttrValue(tag)
        tag.pendingSelfClose = true
        tag.phase = 'attrs'
        return
      }

      tag.attrValue += ch
      return
    }
  }

  return {
    push(chunk: string): void {
      let i = 0

      // Resolve pending < from previous chunk boundary
      if (pendingLt) {
        pendingLt = false
        if (chunk.length === 0) {
          contentBuffer += '<'
          return
        }
        const ch0 = chunk[0]
        if (ch0 === '/' || isNameStart(ch0) || ch0 === '!') {
          flushContent()
          startTag()
          // i stays 0 — main loop feeds chunk[0] into processTagChar
        } else {
          contentBuffer += '<'
          // i stays 0, process ch0 normally
        }
      }

      for (; i < chunk.length; i++) {
        const ch = chunk[i]

        // CDATA mode
        if (cdataBuffer !== null) {
          if (ch === ']') {
            if (cdataCloseProgress === 0) cdataCloseProgress = 1
            else if (cdataCloseProgress === 1) cdataCloseProgress = 2
            else cdataBuffer += ']'
          } else if (ch === '>' && cdataCloseProgress === 2) {
            contentBuffer += cdataBuffer
            cdataBuffer = null
            cdataCloseProgress = 0
          } else {
            if (cdataCloseProgress > 0) {
              cdataBuffer += ']'.repeat(cdataCloseProgress)
              cdataCloseProgress = 0
            }
            cdataBuffer += ch
          }
          continue
        }

        // Active tag parsing
        if (activeTag) {
          processTagChar(ch)
          continue
        }

        // Content / tag-start
        if (ch === '<') {
          if (i + 1 < chunk.length) {
            const next = chunk[i + 1]
            if (next === '/' || isNameStart(next) || next === '!') {
              flushContent()
              startTag()
              // Skip < — startTag sets raw='<', main loop increments i
              // so chunk[i+1] is processed next as first char of tag
            } else {
              contentBuffer += '<'
              afterNewline = false
            }
          } else {
            pendingLt = true
          }
        } else {
          contentBuffer += ch
          if (ch === '\n') afterNewline = true
          else if (ch !== ' ' && ch !== '\t') afterNewline = false
        }
      }

      if (!activeTag && !pendingLt && cdataBuffer === null) {
        flushContent()
      }
    },

    end(): void {
      if (pendingLt) {
        contentBuffer += '<'
        pendingLt = false
      }

      if (activeTag) {
        const tag = activeTag
        if (!tag.isClose && tag.name.length > 0 && (knownToolTags?.has(tag.name) || tag.name === 'magnitude:invoke')) {
          flushContent()
          onToken({
            _tag: 'Open',
            tagName: tag.name,
            attrs: new Map(tag.attrs),
            afterNewline: tag.savedAfterNewline,
            raw: tag.raw,
          })
          activeTag = null
        } else if (!tag.isClose && tag.name.length > 0 && tag.phase !== 'attrValueQuoted') {
          flushContent()
          onToken({
            _tag: tag.pendingSelfClose ? 'SelfClose' : 'Open',
            tagName: tag.name,
            attrs: new Map(tag.attrs),
            afterNewline: tag.savedAfterNewline,
            raw: tag.raw,
          })
          activeTag = null
        } else {
          failTagAsContent()
        }
      }

      if (cdataBuffer !== null) {
        contentBuffer += CDATA_OPEN + cdataBuffer + ']'.repeat(cdataCloseProgress)
        cdataBuffer = null
        cdataCloseProgress = 0
      }

      flushContent()
    },
  }
}
