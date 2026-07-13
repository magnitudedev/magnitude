/**
 * Markdown Parser
 *
 * Uses micromark for tokenization with a handler-based compiler that converts
 * events to DocumentNode AST with lossless metadata.
 *
 * Architecture:
 * 1. Tokenizer: micromark tokenization + preprocessing (synthetic tokens)
 * 2. Compile: Document-level structure, handlers build AST nodes
 */

import type { DocumentNode } from './schema'
import { tokenize } from './tokenizer'
import { compile } from './compile'
import {
  findDivergence,
  findStablePrefixCount,
  rebaseDocumentPositions,
} from './incremental'

function parseMarkdownFull(source: string): DocumentNode {
  const events = tokenize(source)
  const doc = compile(events, source)
  doc.source = source
  return doc
}

/**
 * Parse markdown string to DocumentNode AST.
 */
export function parseMarkdown(
  source: string,
  options?: { previous?: DocumentNode }
): DocumentNode {
  if (!options?.previous) {
    return parseMarkdownFull(source)
  }

  const previous = options.previous
  const divergeAt = findDivergence(previous.source, source)

  if (divergeAt === null) {
    return previous
  }

  const { stableCount, cutPoint } = findStablePrefixCount(previous, divergeAt)

  if (stableCount === 0) {
    return parseMarkdownFull(source)
  }

  const tailSource = source.slice(cutPoint)
  const tailDoc = parseMarkdownFull(tailSource)
  const rebasedTailItems = rebaseDocumentPositions(tailDoc, cutPoint, source)
  const stableItems = previous.content.slice(0, stableCount)

  // Merge adjacent blankLines nodes at the splice boundary.
  // The stable prefix ends at a blankLines node (by construction),
  // and the tail reparse may produce a blankLines node at its start.
  // A fresh parse would produce a single blankLines node, so we merge
  // them into one with the combined position and count.
  let combinedContent: typeof stableItems
  if (
    stableItems.length > 0 &&
    rebasedTailItems.length > 0 &&
    stableItems[stableItems.length - 1].content.type === 'blankLines' &&
    rebasedTailItems[0].content.type === 'blankLines'
  ) {
    const lastStable = stableItems[stableItems.length - 1]
    const firstTail = rebasedTailItems[0]
    const mergedBlankLines = {
      ...lastStable,
      content: {
        ...firstTail.content,
        position: {
          start: lastStable.content.position.start,
          end: firstTail.content.position.end,
        },
        count: (lastStable.content as any).count + (firstTail.content as any).count,
      },
      position: {
        start: lastStable.position.start,
        end: firstTail.position.end,
      },
    }
    combinedContent = [...stableItems.slice(0, -1), mergedBlankLines, ...rebasedTailItems.slice(1)]
  } else {
    combinedContent = [...stableItems, ...rebasedTailItems]
  }
  const fallbackEnd = { line: 1, column: 1, offset: source.length }

  const result: DocumentNode = {
    type: 'doc',
    content: combinedContent,
    meta: { trailingNewline: source.endsWith('\n') },
    position: {
      start: { line: 1, column: 1, offset: 0 },
      end:
        rebasedTailItems[rebasedTailItems.length - 1]?.content.position.end ??
        stableItems[stableItems.length - 1]?.content.position.end ??
        fallbackEnd,
    },
    source,
  }

  return result
}

export { compile } from './compile'
export { tokenize } from './tokenizer'
export type { Event, Token, Point, TokenType } from './tokenizer'
export type { CompileContext, Handler, HandlerConfig, Builder, BuilderType } from './types'