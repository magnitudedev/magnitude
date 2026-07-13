/**
 * Wiring layer — creates the full incremental JSON parser pipeline.
 */

import type { IncrementalJsonParser } from '../types'
import type { ParsedValue } from '../types'
import { createJsonTokenizer } from './tokenizer'
import { createJsonParser } from './parser'

export function createIncrementalJsonParser(): IncrementalJsonParser {
  let isDone = false
  const tokenizer = createJsonTokenizer((token) => parser.feed(token))
  const parser = createJsonParser(tokenizer)

  return {
    push(chunk: string): void {
      tokenizer.push(chunk)
    },
    end(): void {
      tokenizer.end()
      parser.end()
      isDone = true
    },
    get partial(): ParsedValue | undefined {
      return parser.partial
    },
    get done(): boolean {
      return isDone
    },
    get currentPath(): readonly string[] {
      return parser.currentPath
    },
  }
}
