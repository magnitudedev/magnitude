/**
 * Document Compilation
 *
 * Main compilation function that:
 * 1. Preprocesses events (injects synthetic tokens)
 * 2. Handles document-level structure (DocumentItemNode wrapping)
 * 3. Delegates to handlers for content building
 *
 * Key insight: The DocumentItemNode wrapper is handled OUTSIDE the handler stack.
 * Handlers build blocks directly onto a temporary container. After each top-level
 * block completes, we wrap it in DocumentItemNode and add to the document.
 */

import type { DocumentNode, DocumentItemNode, RootBlockNode, BlankLinesNode, SourcePoint, SourcePosition } from './schema'
import type { ContainerBuilder, CompileContext } from './types'
import type { Event, Token } from './tokenizer'
import { createContext } from './context'
import { config } from './handlers'

const ZERO_POINT: SourcePoint = { line: 1, column: 1, offset: 0 }
const ZERO_POSITION: SourcePosition = { start: ZERO_POINT, end: ZERO_POINT }

/** Block token types that start a new block at document level */
const BLOCK_TOKENS = new Set([
  'paragraph',
  'atxHeading',
  'setextHeading',
  'codeFenced',
  'codeIndented',
  'thematicBreak',
  'blockQuote',
  'listUnordered',
  'listOrdered',
  'table',
  'html',
  'htmlFlow',
  'definition',
])

/**
 * Check if a token type is a block-level token.
 */
function isBlockToken(type: string): boolean {
  return BLOCK_TOKENS.has(type)
}

/**
 * Create a blank lines node.
 */
function createBlankLines(
  count: number,
  lines: string[],
  position: SourcePosition
): BlankLinesNode {
  return {
    type: 'blankLines',
    count,
    meta: { lines },
    position,
  }
}

/**
 * Create a document item node.
 */
function createDocumentItem(
  leadingIndent: string,
  content: RootBlockNode | BlankLinesNode
): DocumentItemNode {
  return {
    type: 'documentItem',
    content,
    meta: { leadingIndent },
    position: content.position ?? ZERO_POSITION,
  }
}

/**
 * Compile preprocessed events into a DocumentNode.
 */
export function compile(events: Event[], source: string): DocumentNode {
  // Check if source ends with newline (for lossless round-trip)
  const endsWithNewline = source.endsWith('\n')

  // Create container builder for top-level blocks
  const container: ContainerBuilder = {
    builderType: 'container',
    positionStart: null,
    positionEnd: null,
    content: [],
  }

  // Create compile context with container as root
  const ctx = createContext(source)
  ctx.push(container)

  // Document items to collect
  const documentItems: DocumentItemNode[] = []

  // Document-level state
  let depth = 0
  let pendingIndent = ''
  let pendingBlankLines: { count: number; lines: string[]; start: SourcePoint; end: SourcePoint } | null = null

  // Process events
  for (const event of events) {
    const [type, token] = event

    // Capture document-level linePrefix
    if (token.type === 'linePrefix' && type === 'exit' && depth === 0) {
      pendingIndent = source.slice(token.start.offset, token.end.offset)
      continue
    }

    // Handle blank lines at document level
    if (token.type === 'lineEndingBlank' && type === 'enter' && depth === 0) {
      if (!pendingBlankLines) {
        pendingBlankLines = { count: 0, lines: [], start: token.start, end: token.end }
      }
      pendingBlankLines.count++
      pendingBlankLines.end = token.end
      // Capture the blank line content (whitespace from pendingIndent)
      pendingBlankLines.lines.push(pendingIndent)
      pendingIndent = '' // Reset after capturing
      continue
    }

    // Block enter at document level - flush blank lines and remember indent
    if (type === 'enter' && isBlockToken(token.type) && depth === 0) {
      // Flush pending blank lines as a document item
      if (pendingBlankLines) {
        const blankNode = createBlankLines(
          pendingBlankLines.count,
          pendingBlankLines.lines,
          { start: pendingBlankLines.start, end: pendingBlankLines.end }
        )
        documentItems.push(createDocumentItem('', blankNode))
        pendingBlankLines = null
      }
    }

    // Check if we're in raw block mode (capturing unsupported block as raw text)
    // If so, skip all handlers except the exit handler for the original token type
    const rawBuilder = ctx.stack.find(b => b.builderType === 'rawBlock')
    if (rawBuilder && 'originalType' in rawBuilder) {
      // Only allow the exit handler for the original type to fire
      const isExitForOriginal = type === 'exit' && token.type === rawBuilder.originalType
      if (!isExitForOriginal) {
        // Track depth for raw block tracking
        if (type === 'enter' && isBlockToken(token.type)) {
          depth++
        }
        if (type === 'exit' && isBlockToken(token.type)) {
          depth--
        }
        continue
      }
    }

    // Delegate to handler
    const handlers = type === 'enter' ? config.enter : config.exit
    ctx.currentToken = token
    const handler = handlers[token.type] as ((ctx: CompileContext, token: Token) => void) | undefined
    if (handler) {
      handler(ctx, token)
    }
    ctx.currentToken = null

    // Track depth
    if (type === 'enter' && isBlockToken(token.type)) {
      depth++
    }
    if (type === 'exit' && isBlockToken(token.type)) {
      depth--
      // When we return to depth 0, wrap the completed block(s)
      if (depth === 0) {
        // The block(s) should now be in container.content
        // Note: Some exits may produce multiple blocks (e.g., list splitting at checkbox boundaries)
        while (container.content.length > 0) {
          const item = container.content.shift()
          if (item) {
            documentItems.push(createDocumentItem(pendingIndent, item.block))
            pendingIndent = ''
          }
        }
      }
    }
  }

  // Handle trailing blank lines
  if (pendingBlankLines) {
    const blankNode = createBlankLines(
      pendingBlankLines.count,
      pendingBlankLines.lines,
      { start: pendingBlankLines.start, end: pendingBlankLines.end }
    )
    documentItems.push(createDocumentItem('', blankNode))
  }

  // Handle unclosed tokens
  if (ctx.tokenStack.length > 0) {
    const tail = ctx.tokenStack[ctx.tokenStack.length - 1]
    console.warn(`Warning: unclosed token ${tail.token.type}`)
  }

  // Determine trailing newline:
  // - If source ends with newline AND last item is NOT blankLines, set to true
  // - If last item is blankLines, the trailing newline is already accounted for
  const lastItem = documentItems[documentItems.length - 1]
  const endsWithBlankLines = lastItem && lastItem.content.type === 'blankLines'
  const trailingNewline = endsWithNewline && !endsWithBlankLines

  // Build final document
  const doc: DocumentNode = {
    type: 'doc',
    content: documentItems,
    meta: { trailingNewline },
    position: source.length > 0
      ? {
          start: { line: 1, column: 1, offset: 0 },
          end: documentItems[documentItems.length - 1]?.position.end ?? { line: 1, column: 1, offset: 0 },
        }
      : ZERO_POSITION,
    source,
  }

  return doc
}
