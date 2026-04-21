
/**
 * Streaming tokenizer for the Mact format.
 * 
 * Uses asymmetric delimiters:
 * - Open: <|tag> or <|tag:variant>
 * - Close: <tag|>
 * - Self-close: <|tag|> or <|tag:variant|>
 * - Parameter open: <|parameter:name>
 * - Parameter close: <parameter|>
 * 
 * Architecture follows the old XML tokenizer:
 * - Persistent state across chunks
 * - Char-by-char state machine
 * - No peek-ahead (enter pending state on `<`)
 * - Known tags commit even if malformed
 */

import type { Token } from './types'

export interface Tokenizer {
  push(chunk: string): void
  end(): void
}

// State machine states for tag parsing
type TagPhase =
  | 'open_name'      // After <|, reading name
  | 'open_colon'     // After <|name, saw :, waiting for variant
  | 'open_variant'   // After <|name:, reading variant
  | 'open_pipe'      // After <|name or <|name:variant, saw |, waiting for >
  | 'close_name'     // After <, reading name for close tag
  | 'close_pipe'     // After <name|, reading optional pipe or >
  | 'malformed'      // Known tool tag with invalid syntax — consume to > then emit

type ActiveTag = {
  raw: string
  savedAfterNewline: boolean
  phase: TagPhase
  name: string
  variant: string  // Used for variant in open and pipe in close
}

function isNameStart(ch: string): boolean {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch === '_'
}

function isNameContinue(ch: string): boolean {
  return isNameStart(ch) || (ch >= '0' && ch <= '9') || ch === '-'
}

function isWhitespace(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r'
}

const TOP_LEVEL_TAGS = new Set(['think', 'message', 'invoke', 'yield'])

const DEFAULT_TOKENIZER_OPTIONS = { toolKeyword: 'invoke' } as const

export function createTokenizer(
  onToken: (token: Token) => void,
  knownToolTags: ReadonlySet<string> = new Set(),
  options: { toolKeyword: string } = DEFAULT_TOKENIZER_OPTIONS,
): Tokenizer {
  const toolKeyword = options.toolKeyword
  let contentBuffer = ''
  let afterNewline = true
  let activeTag: ActiveTag | null = null
  let pendingLt: boolean = false  // true if we have a pending < at chunk boundary

  function flushContent(): void {
    if (contentBuffer.length === 0) return
    const text = contentBuffer
    contentBuffer = ''
    onToken({ _tag: 'Content', text })
    // Update afterNewline based on content
    for (const ch of text) {
      if (ch === '\n') afterNewline = true
      else if (ch !== ' ' && ch !== '\t') afterNewline = false
    }
  }

  function emitOpen(name: string, variant: string | undefined): void {
    // Unit B: invoke-without-keyword leniency
    let emitName = name
    let emitVariant = variant
    if (variant === undefined && knownToolTags?.has(name)) {
      emitName = toolKeyword
      emitVariant = name
    }
    // Unit C: newline enforcement for top-level tags
    if (TOP_LEVEL_TAGS.has(emitName) && !activeTag!.savedAfterNewline) {
      failAsContent()
      return
    }
    flushContent()
    onToken({ _tag: 'Open', name: emitName, variant: emitVariant })
    afterNewline = false
  }

  function emitClose(name: string, pipe: string | undefined): void {
    // Unit C: newline enforcement for top-level tags
    if (TOP_LEVEL_TAGS.has(name) && !activeTag!.savedAfterNewline) {
      failAsContent()
      return
    }
    flushContent()
    onToken({ _tag: 'Close', name, pipe })
    afterNewline = false
  }

  function emitSelfClose(name: string, variant: string | undefined): void {
    // Unit C: newline enforcement for top-level tags (yield is self-close)
    if (TOP_LEVEL_TAGS.has(name) && !activeTag!.savedAfterNewline) {
      failAsContent()
      return
    }
    flushContent()
    onToken({ _tag: 'SelfClose', name, variant })
    afterNewline = false
  }

  function emitParameterOpen(name: string): void {
    flushContent()
    onToken({ _tag: 'Parameter', name })
    afterNewline = false
  }

  function failAsContent(): void {
    if (!activeTag) return
    const tag = activeTag
    // If this is a known tool tag (invoke keyword) being parsed as an open tag,
    // don't abandon — enter malformed phase so the parser can produce structured
    // error feedback instead of silently losing the call.
    // Only applies to open tag phases, not close tag phases.
    const isOpenPhase = tag.phase === 'open_name' || tag.phase === 'open_colon' ||
      tag.phase === 'open_variant' || tag.phase === 'open_pipe' || tag.phase === 'malformed'
    if (tag.name === toolKeyword && isOpenPhase && tag.phase !== 'malformed') {
      tag.phase = 'malformed'
      return
    }
    activeTag = null
    contentBuffer += tag.raw
  }

  function startOpenTag(): void {
    activeTag = {
      raw: '<|',
      savedAfterNewline: afterNewline,
      phase: 'open_name',
      name: '',
      variant: '',
    }
  }

  function startCloseTag(): void {
    activeTag = {
      raw: '<',
      savedAfterNewline: afterNewline,
      phase: 'close_name',
      name: '',
      variant: '',
    }
  }

  function emitMalformedInvoke(tag: ActiveTag): void {
    flushContent()
    onToken({ _tag: 'Open', name: toolKeyword, variant: tag.variant || undefined })
    activeTag = null
    afterNewline = false
  }

  function processTagChar(ch: string): void {
    const tag = activeTag!
    tag.raw += ch

    switch (tag.phase) {
      case 'open_name': {
        // Reading name after <|
        if (tag.name.length === 0) {
          if (isNameStart(ch)) {
            tag.name += ch
            return
          }
          // Invalid first character
          failAsContent()
          return
        }

        // Continue reading name
        if (isNameContinue(ch)) {
          tag.name += ch
          return
        }

        if (ch === ':') {
          // Transition to variant
          tag.phase = 'open_colon'
          return
        }

        if (ch === '|') {
          // Potential self-close - need to see if next is >
          tag.phase = 'open_pipe'
          return
        }

        if (ch === '>') {
          // End of open tag: <|name>
          if (tag.name === 'parameter') {
            // <|parameter> without variant is invalid
            failAsContent()
            return
          }
          emitOpen(tag.name, undefined)
          activeTag = null
          return
        }

        if (isWhitespace(ch)) {
          // Whitespace terminates tag name - invalid in strict Mact
          failAsContent()
          return
        }

        // Invalid character
        failAsContent()
        return
      }

      case 'open_colon': {
        // After <|name:, waiting for variant start
        if (isNameStart(ch)) {
          tag.variant = ch
          tag.phase = 'open_variant'
          return
        }
        // Invalid after colon
        failAsContent()
        return
      }

      case 'open_variant': {
        // Reading variant after <|name:
        if (isNameContinue(ch)) {
          tag.variant += ch
          return
        }

        if (ch === '|') {
          // Potential self-close
          tag.phase = 'open_pipe'
          return
        }

        if (ch === '>') {
          // End of open tag: <|name:variant>
          if (tag.name === 'parameter') {
            emitParameterOpen(tag.variant)
          } else {
            emitOpen(tag.name, tag.variant)
          }
          activeTag = null
          return
        }

        if (isWhitespace(ch)) {
          // Whitespace terminates variant - invalid in strict Mact
          failAsContent()
          return
        }

        // Invalid character in variant
        failAsContent()
        return
      }

      case 'open_pipe': {
        // After | in open tag, must see > for self-close
        if (ch === '>') {
          // Self-close: <|name|> or <|name:variant|>
          if (tag.name === 'parameter') {
            // <|parameter|> is invalid, but <|parameter:name|> would have variant set
            if (tag.variant) {
              emitParameterOpen(tag.variant)
            } else {
              failAsContent()
              return
            }
          } else {
            emitSelfClose(tag.name, tag.variant || undefined)
          }
          activeTag = null
          return
        }
        // Anything other than > after | is invalid
        failAsContent()
        return
      }

      case 'close_name': {
        // Reading name after < for close tag
        if (tag.name.length === 0) {
          if (isNameStart(ch)) {
            tag.name += ch
            return
          }
          // Invalid first character for close tag name
          failAsContent()
          return
        }

        // Continue reading name
        if (isNameContinue(ch)) {
          tag.name += ch
          return
        }

        if (ch === '|') {
          // Found pipe, could be <name|> or <name|pipe>
          tag.phase = 'close_pipe'
          return
        }

        if (ch === '>') {
          // Lenient: close without pipe <name>
          emitClose(tag.name, undefined)
          activeTag = null
          return
        }

        if (isWhitespace(ch)) {
          // Whitespace - lenient close
          emitClose(tag.name, undefined)
          activeTag = null
          contentBuffer += ch
          if (ch === '\n') afterNewline = true
      else if (ch !== ' ' && ch !== '\t') afterNewline = false
          return
        }

        // Invalid character
        failAsContent()
        return
      }

      case 'malformed': {
        // Consume chars until > then emit the Open token so parser can produce an error
        if (ch === '>') {
          emitMalformedInvoke(tag)
        }
        // Otherwise just accumulate (raw already updated at top of processTagChar)
        return
      }

      case 'close_pipe': {
        // After <name|, reading optional pipe name or >
        if (ch === '>') {
          // Simple close: <name|>
          emitClose(tag.name, undefined)
          activeTag = null
          return
        }

        if (tag.variant.length === 0) {
          if (isNameStart(ch)) {
            tag.variant = ch
            return
          }
          if (isWhitespace(ch)) {
            // <name| > - lenient, treat as close without pipe
            emitClose(tag.name, undefined)
            activeTag = null
            contentBuffer += ch
            if (ch === '\n') afterNewline = true
      else if (ch !== ' ' && ch !== '\t') afterNewline = false
            return
          }
          // Invalid after pipe
          failAsContent()
          return
        }

        // Continue reading pipe name
        if (isNameContinue(ch)) {
          tag.variant += ch
          return
        }

        if (ch === '>') {
          // Piped close: <name|pipe>
          emitClose(tag.name, tag.variant)
          activeTag = null
          return
        }

        if (isWhitespace(ch)) {
          // Whitespace terminates pipe
          emitClose(tag.name, tag.variant)
          activeTag = null
          contentBuffer += ch
          if (ch === '\n') afterNewline = true
      else if (ch !== ' ' && ch !== '\t') afterNewline = false
          return
        }

        // Invalid character in pipe name
        failAsContent()
        return
      }
    }
  }

  return {
    push(chunk: string): void {
      let i = 0
      
      // Handle pending < from previous chunk
      if (pendingLt) {
        pendingLt = false
        if (chunk.length === 0) {
          // Empty chunk, just treat pending < as content
          contentBuffer += '<'
          return
        }
        const ch = chunk[0]
        if (ch === '|') {
          // Open tag: <|...
          flushContent()
          startOpenTag()
          i = 1  // Skip the |, we've recorded it in startOpenTag
        } else if (isNameStart(ch)) {
          // Close tag: <name...
          flushContent()
          startCloseTag()
          // Don't skip, process this char as part of close tag
          // i stays 0 to process this char
        } else if (ch === '/') {
          // Unit A: lenient close tag - skip the / and start close tag
          flushContent()
          startCloseTag()
          activeTag!.raw += '/'
          i = 1  // Skip /, main loop processes chunk[1] as first name char
        } else {
          // Not a tag, treat pending < as content
          contentBuffer += '<'
          afterNewline = false
          // i stays 0 to process current char normally
        }
      }
      
      for (; i < chunk.length; i++) {
        const ch = chunk[i]

        if (activeTag) {
          processTagChar(ch)
          continue
        }

        // Not in a tag - look for tag starts
        if (ch === '<') {
          // Check if we can determine tag type from what we've seen
          // We need at least one more char to decide between <| and <name
          
          if (i + 1 < chunk.length) {
            const next = chunk[i + 1]
            if (next === '|') {
              // Open tag: <|
              flushContent()
              startOpenTag()
              i++ // Skip the |, we've recorded it in startOpenTag
              continue
            } else if (isNameStart(next)) {
              // Close tag: <name
              flushContent()
              startCloseTag()
              // Don't skip, let processTagChar handle the name char
              continue
            } else if (next === '/') {
              // Unit A: lenient close tag - skip < and /
              flushContent()
              startCloseTag()
              activeTag!.raw += '/'
              i += 1  // Point at /, loop increments to i+2 (first name char)
              continue
            } else {
              // < followed by non-tag char - treat as content
              contentBuffer += ch
              afterNewline = false
            }
          } else {
            // Can't determine yet - < at end of chunk
            // Remember we saw it and wait for next chunk
            pendingLt = true
            // Don't add to content yet
          }
        } else {
          contentBuffer += ch
          if (ch === '\n') afterNewline = true
      else if (ch !== ' ' && ch !== '\t') afterNewline = false
        }
      }

      // Flush content at end of each push() for incremental streaming,
      // but only when not mid-tag-parse and no pending < at chunk boundary
      if (!activeTag && !pendingLt) {
        flushContent()
      }
    },

    end(): void {
      // Handle any pending < from chunk boundary
      if (pendingLt) {
        contentBuffer += '<'
        pendingLt = false
      }
      
      if (activeTag) {
        const tag = activeTag
        // If we were parsing an invoke tag (or already in malformed phase),
        // emit the Open token so the parser can produce structured error feedback
        if (tag.name === toolKeyword || tag.phase === 'malformed') {
          emitMalformedInvoke(tag)
        } else {
          // At end of stream, incomplete tags are treated as content
          failAsContent()
        }
      }
      flushContent()
    },
  }
}
