import type { JsonToken, JsonTokenizer, PendingToken } from './types'

// ---------------------------------------------------------------------------
// Tokenizer mode (internal state)
// ---------------------------------------------------------------------------

type TokenizerMode =
  | { readonly _tag: "default" }
  | { readonly _tag: "inString"; content: string; pendingEscape: boolean; pendingUnicodeHex: string | null }
  | { readonly _tag: "inNumber"; content: string }
  | { readonly _tag: "inKeyword"; content: string; candidates: string[] }
  | { readonly _tag: "inUnquoted"; content: string }
const WHITESPACE = new Set([' ', '\t', '\n', '\r'])
const NUMBER_CHARS = new Set(['0','1','2','3','4','5','6','7','8','9','.','+','-','e','E'])
const NUMBER_START = new Set(['0','1','2','3','4','5','6','7','8','9','-'])
const DELIMITERS = new Set(['{', '}', '[', ']', ':', ',', '"', ' ', '\t', '\n', '\r'])

const COMPLETE_NUMBER_RE = /^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$/

function isCompleteNumber(s: string): boolean {
  return COMPLETE_NUMBER_RE.test(s)
}

const ESCAPE_MAP: Record<string, string> = {
  'n': '\n',
  't': '\t',
  'r': '\r',
  'b': '\b',
  'f': '\f',
  '\\': '\\',
  '"': '"',
  '/': '/',
}

export function createJsonTokenizer(onToken: (token: JsonToken) => void): JsonTokenizer {
  let mode: TokenizerMode = { _tag: "default" }

  function emitKeywordOrUnquoted(content: string): void {
    switch (content) {
      case 'true': onToken({ _tag: 'true' }); break
      case 'false': onToken({ _tag: 'false' }); break
      case 'null': onToken({ _tag: 'null' }); break
      default: onToken({ _tag: 'unquotedString', value: content, complete: true }); break
    }
  }

  function processChar(ch: string): boolean {
    switch (mode._tag) {
      case "default": {
        switch (ch) {
          case '{': onToken({ _tag: 'objectOpen' }); return true
          case '}': onToken({ _tag: 'objectClose' }); return true
          case '[': onToken({ _tag: 'arrayOpen' }); return true
          case ']': onToken({ _tag: 'arrayClose' }); return true
          case ':': onToken({ _tag: 'colon' }); return true
          case ',': onToken({ _tag: 'comma' }); return true
          case '"':
            mode = { _tag: 'inString', content: '', pendingEscape: false, pendingUnicodeHex: null }
            return true
          default:
            if (WHITESPACE.has(ch)) return true
            if (NUMBER_START.has(ch)) {
              mode = { _tag: 'inNumber', content: ch }
              return true
            }
            if (ch === 't') {
              mode = { _tag: 'inKeyword', content: 't', candidates: ['true'] }
              return true
            }
            if (ch === 'f') {
              mode = { _tag: 'inKeyword', content: 'f', candidates: ['false'] }
              return true
            }
            if (ch === 'n') {
              mode = { _tag: 'inKeyword', content: 'n', candidates: ['null'] }
              return true
            }
            // Permissive: unquoted string
            mode = { _tag: 'inUnquoted', content: ch }
            return true
        }
      }

      case "inString": {
        if (mode.pendingUnicodeHex !== null) {
          mode.pendingUnicodeHex += ch
          if (mode.pendingUnicodeHex.length === 4) {
            const hex = mode.pendingUnicodeHex
            const code = parseInt(hex, 16)
            if (isNaN(code)) {
              mode.content += '\\u' + hex
            } else {
              mode.content += String.fromCharCode(code)
            }
            mode.pendingUnicodeHex = null
          }
          return true
        }
        if (mode.pendingEscape) {
          mode.pendingEscape = false
          if (ch === 'u') {
            mode.pendingUnicodeHex = ''
            return true
          }
          const mapped = ESCAPE_MAP[ch]
          mode.content += mapped !== undefined ? mapped : '\\' + ch
          return true
        }
        if (ch === '\\') {
          mode.pendingEscape = true
          return true
        }
        if (ch === '"') {
          onToken({ _tag: 'string', value: mode.content, complete: true })
          mode = { _tag: 'default' }
          return true
        }
        mode.content += ch
        return true
      }

      case "inNumber": {
        if (NUMBER_CHARS.has(ch)) {
          mode.content += ch
          return true
        }
        // Terminate number, reprocess char
        onToken({ _tag: 'number', value: mode.content, complete: isCompleteNumber(mode.content) })
        mode = { _tag: 'default' }
        return false // reprocess
      }

      case "inKeyword": {
        const pos = mode.content.length
        // Check if any candidate matches at this position
        const newCandidates = mode.candidates.filter(c => pos < c.length && c[pos] === ch)

        if (newCandidates.length === 0) {
          // Check if we had a full match already
          const fullMatch = mode.candidates.find(c => c.length === pos)
          if (fullMatch) {
            // We have a completed keyword, and current char doesn't extend it
            if (DELIMITERS.has(ch)) {
              emitKeywordOrUnquoted(fullMatch)
              mode = { _tag: 'default' }
              return false // reprocess
            } else {
              // Continuation char — transition to unquoted
              mode = { _tag: 'inUnquoted', content: mode.content + ch }
              return true
            }
          }
          // No candidates and no full match — transition to unquoted
          mode = { _tag: 'inUnquoted', content: mode.content + ch }
          return true
        }

        mode.content += ch
        mode.candidates = newCandidates

        // Check if we now have a full match — but don't emit yet, wait for delimiter
        return true
      }

      case "inUnquoted": {
        if (DELIMITERS.has(ch)) {
          onToken({ _tag: 'unquotedString', value: mode.content, complete: true })
          mode = { _tag: 'default' }
          return false // reprocess
        }
        mode.content += ch
        return true
      }
    }
  }

  return {
    push(chunk: string): void {
      for (let i = 0; i < chunk.length; i++) {
        const consumed = processChar(chunk[i])
        if (!consumed) {
          i-- // reprocess
        }
      }
    },

    end(): void {
      switch (mode._tag) {
        case 'default':
          break
        case 'inString':
          onToken({ _tag: 'string', value: mode.content, complete: false })
          break
        case 'inNumber':
          onToken({ _tag: 'number', value: mode.content, complete: isCompleteNumber(mode.content) })
          break
        case 'inKeyword': {
          const currentContent = mode.content
          const fullMatch = mode.candidates.find(c => c === currentContent)
          if (fullMatch) {
            emitKeywordOrUnquoted(fullMatch)
          } else {
            onToken({ _tag: 'unquotedString', value: currentContent, complete: false })
          }
          break
        }
        case 'inUnquoted':
          onToken({ _tag: 'unquotedString', value: mode.content, complete: true })
          break
      }
      mode = { _tag: 'default' }
    },

    get pending(): PendingToken | null {
      switch (mode._tag) {
        case 'default': return null
        case 'inString': return { _tag: 'string', content: mode.content }
        case 'inNumber': return { _tag: 'number', content: mode.content }
        case 'inKeyword': return { _tag: 'keyword', content: mode.content }
        case 'inUnquoted': return { _tag: 'unquoted', content: mode.content }
      }
    },
  }
}
