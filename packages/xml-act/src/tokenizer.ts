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
 * Key feature: close-tag lookahead confirmation.
 * After reading </tag>, the tokenizer enters pendingClose state and waits
 * for a confirming character (\n or <) before emitting the Close token.
 * This mirrors the grammar tw-state mechanism exactly.
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

type PendingClose = {
  tagName: string
  raw: string
  afterNewline: boolean
  wsBuffer: string
  /** For parameter/filter: continuation prefix matching state */
  continuationBuffer: string
  /** Whether we're in deep confirmation mode (parameter/filter close tags) */
  deepConfirm: boolean
  /** Whether we've seen '<' and are now matching the continuation prefix */
  matchingContinuation: boolean
}

const MAX_TRAILING_WS = 4

const CDATA_OPEN = '<![CDATA['

function isWhitespace(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r'
}

function isHorizontalWs(ch: string): boolean {
  return ch === ' ' || ch === '\t'
}

function isNameStart(ch: string): boolean {
  return /[a-zA-Z_]/.test(ch)
}

function isNameContinue(ch: string): boolean {
  return /[a-zA-Z0-9_.-]/.test(ch)
}

export function createTokenizer(
  onToken: (token: Token) => void,
  knownToolTags?: ReadonlySet<string>,
): Tokenizer {
  let contentBuffer = ''
  let afterNewline = true
  let activeTag: ActiveTag | null = null
  let pendingLt = false
  let pendingClose: PendingClose | null = null
  let cdataBuffer: string | null = null
  let cdataCloseProgress = 0
  let replayBuffer = ''

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
    if (tag.name.length > 0 && !tag.isClose && (knownToolTags?.has(tag.name) || tag.name === 'invoke')) {
      tag.phase = 'malformed'
      return
    }
    activeTag = null
    contentBuffer += tag.raw
  }

  function emitTag(tag: ActiveTag): void {
    flushContent()
    if (tag.isClose) {
      pendingClose = {
        tagName: tag.name,
        raw: tag.raw,
        afterNewline: tag.savedAfterNewline,
        wsBuffer: '',
        continuationBuffer: '',
        deepConfirm: tag.name === 'parameter' || tag.name === 'filter',
        matchingContinuation: false,
      }
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

  // Valid continuation prefixes after parameter/filter close (inside invoke)
  const PARAM_FILTER_CONTINUATIONS = ['parameter ', 'filter>', '/invoke>']

  /**
   * Process a character while in pendingClose state.
   * Returns true if consumed, false if caller should process ch normally.
   *
   * For reason/message close tags: original behavior (confirm on \n or <).
   * For parameter/filter close tags: deep confirmation — buffer whitespace,
   * then match a full continuation prefix before confirming.
   */
  function processPendingClose(ch: string): boolean {
    const pc = pendingClose!

    // --- Standard confirmation for reason/message ---
    if (!pc.deepConfirm) {
      if (isHorizontalWs(ch)) {
        if (pc.wsBuffer.length < MAX_TRAILING_WS) {
          pc.wsBuffer += ch
          return true
        } else {
          contentBuffer += pc.raw + pc.wsBuffer
          pendingClose = null
          return false
        }
      }

      if (ch === '\n') {
        flushContent()
        onToken({ _tag: 'Close', tagName: pc.tagName, afterNewline: pc.afterNewline, raw: pc.raw })
        afterNewline = true
        contentBuffer += '\n'
        pendingClose = null
        return true
      }

      if (ch === '<') {
        flushContent()
        onToken({ _tag: 'Close', tagName: pc.tagName, afterNewline: pc.afterNewline, raw: pc.raw })
        afterNewline = false
        pendingClose = null
        pendingLt = true
        return true
      }

      contentBuffer += pc.raw + pc.wsBuffer
      pendingClose = null
      return false
    }

    // --- Deep confirmation for parameter/filter ---

    if (pc.matchingContinuation) {
      // We're matching a continuation prefix after seeing '<'
      pc.continuationBuffer += ch

      // Check if any continuation still matches
      let anyMatch = false
      let fullMatch = false
      for (const cont of PARAM_FILTER_CONTINUATIONS) {
        if (cont.startsWith(pc.continuationBuffer)) {
          anyMatch = true
          if (cont === pc.continuationBuffer) {
            fullMatch = true
          }
        }
      }

      if (fullMatch) {
        // Full continuation prefix matched — CONFIRM the close tag
        flushContent()
        onToken({ _tag: 'Close', tagName: pc.tagName, afterNewline: pc.afterNewline, raw: pc.raw })
        afterNewline = false

        // Feed back the continuation characters (< + continuationBuffer) as new input
        // The '<' + continuation prefix is the start of the next structural element
        // We need to re-process these characters through the tokenizer
        const replay = '<' + pc.continuationBuffer
        pendingClose = null
        // Push the continuation back through — start a new tag
        pendingLt = false
        flushContent()
        startTag()
        for (let j = 1; j < replay.length; j++) {
          processTagChar(replay[j])
        }
        return true
      }

      if (anyMatch) {
        // Partial match — keep buffering
        return true
      }

      // No continuation matches — REJECT
      // Dump close tag raw + wsBuffer as content, replay '<' + continuationBuffer
      contentBuffer += pc.raw + pc.wsBuffer
      replayBuffer = '<' + pc.continuationBuffer
      pendingClose = null
      return true
    }

    // Not yet matching continuation — buffering whitespace
    if (isWhitespace(ch)) {
      // Buffer all whitespace (unbounded for parameter/filter)
      pc.wsBuffer += ch
      return true
    }

    if (ch === '<') {
      // Start matching continuation prefix
      pc.matchingContinuation = true
      pc.continuationBuffer = ''
      return true
    }

    // Non-whitespace, non-< — REJECT
    contentBuffer += pc.raw + pc.wsBuffer
    pendingClose = null
    return false
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

        // Drain replay buffer (from deep confirmation rejection)
        if (replayBuffer.length > 0) {
          const rb = replayBuffer
          replayBuffer = ''
          // Prepend replay chars to unprocessed input and restart loop
          chunk = rb + chunk.slice(i)
          i = -1 // will be incremented to 0 by for-loop
          continue
        }

        // Pending close confirmation
        if (pendingClose) {
          const consumed = processPendingClose(ch)
          if (consumed) {
            // processPendingClose may have set pendingLt (when ch was <)
            // Resolve inline since we are mid-loop
            if (pendingLt) {
              pendingLt = false
              if (i + 1 < chunk.length) {
                const next = chunk[i + 1]
                if (next === '/' || isNameStart(next) || next === '!') {
                  flushContent()
                  startTag()
                } else {
                  contentBuffer += '<'
                }
              } else {
                pendingLt = true
              }
            }
            continue
          }
          // Not consumed — pendingClose rejected, fall through to process ch normally
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

      if (pendingClose) {
        if (pendingClose.deepConfirm) {
          // EOF confirms parameter/filter close tags
          flushContent()
          onToken({ _tag: 'Close', tagName: pendingClose.tagName, afterNewline: pendingClose.afterNewline, raw: pendingClose.raw })
          afterNewline = false
          if (pendingClose.wsBuffer) contentBuffer += pendingClose.wsBuffer
          if (pendingClose.matchingContinuation) contentBuffer += '<' + pendingClose.continuationBuffer
        } else {
          contentBuffer += pendingClose.raw + pendingClose.wsBuffer
        }
        pendingClose = null
      }

      if (activeTag) {
        const tag = activeTag
        if (!tag.isClose && tag.name.length > 0 && (knownToolTags?.has(tag.name) || tag.name === 'invoke')) {
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
