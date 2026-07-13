/**
 * Tokenizer
 *
 * Wraps micromark tokenization and preprocessing with our own type system.
 * Defines token types that include synthetic tokens like 'listItem'.
 */

import { parse, preprocess, postprocess } from 'micromark'
import { gfm } from 'micromark-extension-gfm'
import { gfmTaskListItem } from 'micromark-extension-gfm-task-list-item'
// DISABLED: Math support temporarily disabled
// import { math } from 'micromark-extension-math'
import type {
  Event as MicromarkEvent,
  TokenType as MicromarkTokenType,
} from 'micromark-util-types'

// Import extension types to get TokenTypeMap augmentations
import 'micromark-extension-gfm-strikethrough'
import 'micromark-extension-gfm-table'
import 'micromark-extension-gfm-task-list-item'
import 'micromark-extension-gfm-autolink-literal'
import 'micromark-extension-gfm-footnote'

// =============================================================================
// OUR TOKEN TYPES
// =============================================================================

/** Position in source */
export interface Point {
  line: number
  column: number
  offset: number
}

/** All token types - micromark's plus our synthetic listItem */
export type TokenType = MicromarkTokenType | 'listItem'

/** Base token structure */
interface TokenBase<T extends TokenType> {
  type: T
  start: Point
  end: Point
}

/** Synthetic listItem token created during preprocessing */
export interface ListItemToken extends TokenBase<'listItem'> {
  _spread: boolean
  _indent: string
  _marker: string
  _prefixWhitespace: string
  _numberString: string
}

/** List token with spread metadata added during preprocessing */
export interface ListOrderedToken extends TokenBase<'listOrdered'> {
  _spread: boolean
}

export interface ListUnorderedToken extends TokenBase<'listUnordered'> {
  _spread: boolean
}

/** Token types that are handled as regular tokens (no extra metadata) */
type RegularTokenType = Exclude<TokenType, 'listItem' | 'listOrdered' | 'listUnordered'>

/** Helper to distribute over union - creates TokenBase<T> for each T in the union */
type DistributeTokenBase<T> = T extends TokenType ? TokenBase<T> : never

/** Regular token without extra metadata - distributed union for each token type */
export type RegularToken = DistributeTokenBase<RegularTokenType>

/** All token types */
export type Token = RegularToken | ListItemToken | ListOrderedToken | ListUnorderedToken

/** Extract token type by its type field */
export type TokenForType<T extends string> = Extract<Token, { type: T }>

/** Event tuple */
export type Event = ['enter' | 'exit', Token]

// =============================================================================
// TOKENIZATION
// =============================================================================

/**
 * Tokenize markdown and return preprocessed events.
 */
export function tokenize(source: string): Event[] {
  // Tokenize with micromark
  const micromarkEvents = postprocess(
    parse({
      // DISABLED: Math support temporarily disabled - removed math()
      extensions: [gfm(), gfmTaskListItem()],
    })
      .document()
      .write(preprocess()(source, 'utf-8', true))
  )

  // Convert to our event format and preprocess
  const events = convertEvents(micromarkEvents)
  preprocessLists(events, source)
  preprocessEmptyTaskItems(events, source)

  return events
}

/**
 * Convert micromark events to our event format.
 * At this stage all tokens are RegularTokens - list tokens get metadata during preprocessing.
 */
function convertEvents(micromarkEvents: MicromarkEvent[]): Event[] {
  return micromarkEvents.map(([type, token]): Event => [
    type,
    {
      type: token.type,
      start: {
        line: token.start.line,
        column: token.start.column,
        offset: token.start.offset,
      },
      end: {
        line: token.end.line,
        column: token.end.column,
        offset: token.end.offset,
      },
    } as RegularToken,
  ])
}

// =============================================================================
// LIST PREPROCESSING
// =============================================================================

/**
 * Preprocess lists to inject synthetic listItem tokens.
 * Modifies events array in place.
 */
function preprocessLists(events: Event[], source: string): void {
  const listStack: Array<{ index: number; initialIndent: string }> = []
  let index = -1
  let pendingIndentForNestedList = ''

  while (++index < events.length) {
    const [eventType, token] = events[index]

    // Track listItemIndent, linePrefix, and blockQuotePrefix for potential list's first item
    // Accumulate multiple consecutive indent events (for deeply nested lists)
    if (token.type === 'listItemIndent') {
      if (eventType === 'exit') {
        pendingIndentForNestedList += source.slice(token.start.offset, token.end.offset)
      }
      // Don't clear on enter - we need to accumulate consecutive indents
    } else if (token.type === 'linePrefix') {
      // linePrefix can also contribute to nested list indent (comes after listItemIndent)
      // But only when we're inside a list context (listStack is non-empty)
      if (eventType === 'exit' && listStack.length > 0) {
        pendingIndentForNestedList += source.slice(token.start.offset, token.end.offset)
      }
    } else if (token.type === 'blockQuotePrefix') {
      // Capture blockquote prefix as part of the line prefix for list items.
      // Only capture when we're inside a list (listStack not empty) - this handles
      // nested lists in blockquotes. For top-level lists in blockquotes, the prefix
      // is captured when we see it before listUnordered/listOrdered enter.
      if (eventType === 'exit') {
        pendingIndentForNestedList += source.slice(token.start.offset, token.end.offset)
      }
    } else if (token.type === 'blockQuoteMarker' || token.type === 'blockQuotePrefixWhitespace') {
      // Sub-tokens of blockQuotePrefix - don't clear pending, we capture via blockQuotePrefix exit
      continue
    } else if (token.type === 'blockQuote') {
      // blockQuote enter: don't clear pending indent - we're accumulating the full prefix chain
      // blockQuote exit: clear pending indent - the prefix belonged to content inside, not after
      if (eventType === 'exit') {
        pendingIndentForNestedList = ''
      }
      // Skip the else branch that would clear on enter
      continue
    } else if (token.type === 'listOrdered' || token.type === 'listUnordered') {
      if (eventType === 'enter') {
        listStack.push({ index, initialIndent: pendingIndentForNestedList })
        pendingIndentForNestedList = '' // Consumed
      } else {
        const { index: start, initialIndent } = listStack.pop()!
        index = prepareList(events, source, start, index, initialIndent)
      }
    } else if (eventType === 'enter') {
      // Any other enter event clears the pending indent
      pendingIndentForNestedList = ''
    }
  }
}

/**
 * Preprocess events for a single list, injecting listItem tokens.
 * Returns new index to continue from.
 */
function prepareList(
  events: Event[],
  source: string,
  start: number,
  length: number,
  initialIndent: string = ''
): number {
  let index = start - 1
  let containerBalance = -1
  let listSpread = false
  let listItem: ListItemToken | undefined
  let lineIndex: number | undefined
  let firstBlankLineIndex: number | undefined
  let atMarker: boolean | undefined

  // Track pending metadata - initialize with indent from before the list
  let pendingIndent = initialIndent
  let pendingMarker = ''
  let pendingPrefixWhitespace = ''
  let pendingNumberString = ''
  // Track if we've seen a line ending since last marker (to distinguish leading indent from trailing whitespace)
  let seenLineEndingSinceMarker = true

  while (++index <= length) {
    const event = events[index]
    const [type, token] = event

    // Track container depth
    if (
      token.type === 'listUnordered' ||
      token.type === 'listOrdered' ||
      token.type === 'blockQuote'
    ) {
      if (type === 'enter') {
        containerBalance++
      } else {
        containerBalance--
        // Clear pending indent when blockquote exits - the prefix belonged to content
        // inside the blockquote, not to items after it
        if (token.type === 'blockQuote') {
          pendingIndent = ''
        }
      }
      atMarker = undefined
    } else if (token.type === 'lineEndingBlank') {
      if (type === 'enter') {
        if (listItem && !atMarker && !containerBalance && !firstBlankLineIndex) {
          firstBlankLineIndex = index
        }
        atMarker = undefined
        seenLineEndingSinceMarker = true
      }
    } else if (token.type === 'lineEnding') {
      if (type === 'enter') {
        seenLineEndingSinceMarker = true
      }
    } else if (token.type === 'blockQuotePrefix') {
      // Capture blockquote prefix as part of the line prefix for list items
      // This ensures bulletItem.meta.indent includes the full line prefix (e.g., "> ")
      if (type === 'exit') {
        pendingIndent += source.slice(token.start.offset, token.end.offset)
      }
    } else if (token.type === 'blockQuoteMarker' || token.type === 'blockQuotePrefixWhitespace') {
      // Sub-tokens of blockQuotePrefix - don't clear pending, we capture via blockQuotePrefix exit
    } else if (token.type === 'linePrefix') {
      // Capture line prefix for potential NEXT item's leading indent
      // Only capture if we've seen a line ending since the last marker
      // (i.e., this is at the start of a line, not trailing whitespace after marker)
      if (type === 'exit' && !containerBalance && seenLineEndingSinceMarker) {
        pendingIndent = source.slice(token.start.offset, token.end.offset)
      }
    } else if (token.type === 'listItemIndent') {
      if (type === 'exit') {
        // Accumulate consecutive listItemIndent events (for deeply nested lists)
        pendingIndent += source.slice(token.start.offset, token.end.offset)
      }
    } else if (token.type === 'listItemMarker') {
      if (type === 'exit') {
        pendingMarker = source.slice(token.start.offset, token.end.offset)
      }
    } else if (token.type === 'listItemValue') {
      if (type === 'exit') {
        pendingNumberString = source.slice(token.start.offset, token.end.offset)
      }
    } else if (token.type === 'listItemPrefixWhitespace') {
      if (type === 'exit') {
        pendingPrefixWhitespace = source.slice(token.start.offset, token.end.offset)
      }
    } else if (token.type === 'listItemPrefix') {
      // Handled specially below
    } else {
      // Any other token type - clear pendingIndent when we see content
      // This ensures only the linePrefix immediately before a listItemPrefix is used
      // e.g., linePrefix for content continuation inside list item shouldn't become
      // the indent for the next list item
      if (type === 'enter' && pendingIndent) {
        pendingIndent = ''
      }
      atMarker = undefined
    }

    // Check if we need to close/open list items
    if (
      (!containerBalance && type === 'enter' && token.type === 'listItemPrefix') ||
      (containerBalance === -1 &&
        type === 'exit' &&
        (token.type === 'listUnordered' || token.type === 'listOrdered'))
    ) {
      // Close previous listItem
      if (listItem) {
        let tailIndex = index
        lineIndex = undefined

        while (tailIndex--) {
          const tailEvent = events[tailIndex]
          if (
            tailEvent[1].type === 'lineEnding' ||
            tailEvent[1].type === 'lineEndingBlank'
          ) {
            if (tailEvent[0] === 'exit') continue
            if (lineIndex) {
              events[lineIndex][1].type = 'lineEndingBlank'
              listSpread = true
            }
            tailEvent[1].type = 'lineEnding'
            lineIndex = tailIndex
          } else if (
            tailEvent[1].type === 'linePrefix' ||
            tailEvent[1].type === 'blockQuotePrefix' ||
            tailEvent[1].type === 'blockQuotePrefixWhitespace' ||
            tailEvent[1].type === 'blockQuoteMarker' ||
            tailEvent[1].type === 'listItemIndent'
          ) {
            // Skip
          } else {
            break
          }
        }

        if (firstBlankLineIndex && (!lineIndex || firstBlankLineIndex < lineIndex)) {
          listItem._spread = true
        }

        listItem.end = lineIndex
          ? { ...events[lineIndex][1].start }
          : { ...event[1].end }

        events.splice(lineIndex || index, 0, ['exit', listItem])
        index++
        length++
      }

      // Create new listItem
      if (token.type === 'listItemPrefix') {
        listItem = {
          type: 'listItem',
          start: { ...token.start },
          end: { line: 0, column: 0, offset: 0 },
          _spread: false,
          _indent: pendingIndent,
          _marker: pendingMarker,
          _prefixWhitespace: pendingPrefixWhitespace,
          _numberString: pendingNumberString,
        }

        pendingIndent = ''
        pendingMarker = ''
        pendingPrefixWhitespace = ''
        pendingNumberString = ''
        seenLineEndingSinceMarker = false // Reset - we're now after the marker

        events.splice(index, 0, ['enter', listItem])
        index++
        length++
        firstBlankLineIndex = undefined
        atMarker = true
      }
    }

    // Update metadata on listItemPrefix exit (only at current list level, not nested)
    if (type === 'exit' && token.type === 'listItemPrefix' && listItem && !containerBalance) {
      listItem._marker = pendingMarker || listItem._marker
      listItem._prefixWhitespace = pendingPrefixWhitespace || listItem._prefixWhitespace
      listItem._numberString = pendingNumberString || listItem._numberString
      pendingMarker = ''
      pendingPrefixWhitespace = ''
      pendingNumberString = ''
    }
  }

  // Replace the list token with a properly typed one that includes _spread
  const originalToken = events[start][1]
  if (originalToken.type === 'listOrdered') {
    const listToken: ListOrderedToken = {
      type: 'listOrdered',
      start: originalToken.start,
      end: originalToken.end,
      _spread: listSpread,
    }
    events[start][1] = listToken
  } else if (originalToken.type === 'listUnordered') {
    const listToken: ListUnorderedToken = {
      type: 'listUnordered',
      start: originalToken.start,
      end: originalToken.end,
      _spread: listSpread,
    }
    events[start][1] = listToken
  }

  return length
}

// =============================================================================
// EMPTY TASK ITEM PREPROCESSING
// =============================================================================

/**
 * Detect and fix empty task items that micromark doesn't recognize.
 * 
 * When a list item contains only a checkbox followed by whitespace (e.g., `- [ ] `),
 * micromark doesn't emit taskListCheck tokens - it treats the `[ ]` as regular data.
 * This function detects that pattern and injects synthetic taskListCheck tokens.
 * 
 * This goes against CommonMark/GFM spec which requires content after checkboxes,
 * but provides better UX for our use case where empty task items should roundtrip.
 */
function preprocessEmptyTaskItems(events: Event[], source: string): void {
  let index = -1
  let inListItem = false
  let listItemContentStart = -1
  
  while (++index < events.length) {
    const [eventType, token] = events[index]
    
    // Track when we enter/exit list items
    if (token.type === 'listItem') {
      if (eventType === 'enter') {
        inListItem = true
        listItemContentStart = -1
      } else {
        inListItem = false
      }
      continue
    }
    
    // Track when we enter paragraph inside a list item
    // This is where checkbox content would start
    if (token.type === 'paragraph' && eventType === 'enter' && inListItem) {
      listItemContentStart = index
      continue
    }
    
    // Look for data tokens at the start of paragraph content in list items
    if (token.type === 'data' && eventType === 'enter' && listItemContentStart >= 0) {
      // Check if this is right after paragraph enter (possibly with other data tokens)
      // We need to check if the combined data tokens form a checkbox pattern
      
      const contentStartOffset = events[listItemContentStart][1].start.offset
      
      // Only process if this data token is at the start of the paragraph content
      if (token.start.offset !== contentStartOffset) {
        // Reset - we're past the start, no checkbox here
        listItemContentStart = -1
        continue
      }
      
      // Look at the source content starting from this position
      // Check if it matches checkbox pattern: [ ] or [x] or [X]
      const remaining = source.slice(token.start.offset)
      const checkboxMatch = remaining.match(/^\[([ xX])\]/)
      
      if (!checkboxMatch) {
        // Not a checkbox pattern
        listItemContentStart = -1
        continue
      }
      
      // Found a checkbox pattern! Now check if there's already a taskListCheck
      // token (meaning micromark recognized it). If so, skip.
      let hasTaskListCheck = false
      for (let j = listItemContentStart; j < events.length && j < index + 10; j++) {
        if (events[j][1].type === 'taskListCheck') {
          hasTaskListCheck = true
          break
        }
      }
      
      if (hasTaskListCheck) {
        listItemContentStart = -1
        continue
      }
      
      // We need to inject synthetic taskListCheck tokens
      // First, find all the data tokens that make up the checkbox
      const checkboxStr = checkboxMatch[0] // "[ ]" or "[x]" or "[X]"
      const checkboxValue = checkboxMatch[1] // " " or "x" or "X"
      const isChecked = checkboxValue.toLowerCase() === 'x'
      
      const checkboxStartOffset = token.start.offset
      const checkboxEndOffset = checkboxStartOffset + checkboxStr.length
      
      // Find and remove data tokens that are part of the checkbox
      let dataTokensToRemove: number[] = []
      let j = index
      while (j < events.length) {
        const [evtType, evtToken] = events[j]
        if (evtToken.type === 'data') {
          // Check if this data token overlaps with our checkbox
          if (evtToken.start.offset < checkboxEndOffset) {
            dataTokensToRemove.push(j)
            // If this is an exit, we've found the pair
            if (evtType === 'exit') {
              // Check if there's more data after
              if (j + 1 < events.length && 
                  events[j + 1][1].type === 'data' && 
                  events[j + 1][1].start.offset < checkboxEndOffset) {
                j++
                continue
              }
            }
          }
        } else if (evtType === 'enter') {
          // Hit a non-data token entering, stop looking
          break
        }
        j++
      }
      
      // Calculate positions for the synthetic tokens
      const bracketOpenStart = checkboxStartOffset
      const bracketOpenEnd = bracketOpenStart + 1
      const valueStart = bracketOpenEnd
      const valueEnd = valueStart + 1
      const bracketCloseStart = valueEnd
      const bracketCloseEnd = bracketCloseStart + 1
      
      // Get line/column info from the first data token
      const startLine = token.start.line
      const startColumn = token.start.column
      
      // Create synthetic tokens
      const syntheticTokens: Event[] = [
        ['enter', {
          type: 'taskListCheck',
          start: { line: startLine, column: startColumn, offset: bracketOpenStart },
          end: { line: startLine, column: startColumn + 3, offset: bracketCloseEnd },
        } as RegularToken],
        ['enter', {
          type: 'taskListCheckMarker',
          start: { line: startLine, column: startColumn, offset: bracketOpenStart },
          end: { line: startLine, column: startColumn + 1, offset: bracketOpenEnd },
        } as RegularToken],
        ['exit', {
          type: 'taskListCheckMarker',
          start: { line: startLine, column: startColumn, offset: bracketOpenStart },
          end: { line: startLine, column: startColumn + 1, offset: bracketOpenEnd },
        } as RegularToken],
        ['enter', {
          type: isChecked ? 'taskListCheckValueChecked' : 'taskListCheckValueUnchecked',
          start: { line: startLine, column: startColumn + 1, offset: valueStart },
          end: { line: startLine, column: startColumn + 2, offset: valueEnd },
        } as RegularToken],
        ['exit', {
          type: isChecked ? 'taskListCheckValueChecked' : 'taskListCheckValueUnchecked',
          start: { line: startLine, column: startColumn + 1, offset: valueStart },
          end: { line: startLine, column: startColumn + 2, offset: valueEnd },
        } as RegularToken],
        ['enter', {
          type: 'taskListCheckMarker',
          start: { line: startLine, column: startColumn + 2, offset: bracketCloseStart },
          end: { line: startLine, column: startColumn + 3, offset: bracketCloseEnd },
        } as RegularToken],
        ['exit', {
          type: 'taskListCheckMarker',
          start: { line: startLine, column: startColumn + 2, offset: bracketCloseStart },
          end: { line: startLine, column: startColumn + 3, offset: bracketCloseEnd },
        } as RegularToken],
        ['exit', {
          type: 'taskListCheck',
          start: { line: startLine, column: startColumn, offset: bracketOpenStart },
          end: { line: startLine, column: startColumn + 3, offset: bracketCloseEnd },
        } as RegularToken],
      ]
      
      // Check if there's remaining content after the checkbox in the last data token
      // that we need to preserve
      const lastDataTokenIdx = dataTokensToRemove[dataTokensToRemove.length - 1]
      if (lastDataTokenIdx !== undefined) {
        const lastDataToken = events[lastDataTokenIdx][1]
        if (lastDataToken.end.offset > checkboxEndOffset) {
          // There's content after the checkbox - create a new data token for it
          const remainingStart = checkboxEndOffset
          const remainingEnd = lastDataToken.end.offset
          const remainingContent = source.slice(remainingStart, remainingEnd)
          
          // Only add if there's actual non-whitespace content or we're preserving whitespace
          if (remainingContent.length > 0) {
            syntheticTokens.push(
              ['enter', {
                type: 'data',
                start: { 
                  line: startLine, 
                  column: startColumn + 3, 
                  offset: remainingStart 
                },
                end: lastDataToken.end,
              } as RegularToken],
              ['exit', {
                type: 'data',
                start: { 
                  line: startLine, 
                  column: startColumn + 3, 
                  offset: remainingStart 
                },
                end: lastDataToken.end,
              } as RegularToken]
            )
          }
        }
      }
      
      // Remove old data tokens and insert synthetic ones
      // Sort in reverse order to remove from end first
      dataTokensToRemove.sort((a, b) => b - a)
      for (const removeIdx of dataTokensToRemove) {
        events.splice(removeIdx, 1)
      }
      
      // Insert synthetic tokens at the position of the first removed token
      const insertAt = index
      events.splice(insertAt, 0, ...syntheticTokens)
      
      // Adjust index to skip past the synthetic tokens we just inserted
      index = insertAt + syntheticTokens.length - 1
      
      // Reset - we're done with this list item's checkbox
      listItemContentStart = -1
    }
  }
}
