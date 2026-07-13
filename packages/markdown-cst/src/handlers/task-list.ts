/**
 * Task List Handlers
 *
 * Handlers for GFM task list checkboxes ([ ] and [x]).
 *
 * Task list checkboxes appear inside list item paragraphs. When we see a taskListCheck,
 * we mark the list item as a task item by setting its taskCheckbox field.
 * During finalization, lists containing task items become TaskListNodes.
 *
 * Important: Checkbox must be at the very start of list item content. If content
 * already exists (e.g., a definition), the [ ] is NOT a checkbox - it's literal text
 * that we must preserve.
 */

import type { CheckboxMarker } from '../schema'
import { definePartialHandlers } from './define'

// =============================================================================
// ENTER HANDLERS
// =============================================================================

export const enter = definePartialHandlers({
  // taskListCheck: container for the checkbox ([ ] or [x])
  // Track that we're in a potential checkbox - we'll decide on exit whether to use it
  taskListCheck: (ctx) => {
    const listItem = ctx.find('listItem')
    if (listItem) {
      // Mark that we're processing a potential checkbox
      // Valid if:
      // 1. No content yet
      // 2. No checkbox already set
      // 3. Still on marker line (not a continuation line)
      listItem._pendingCheckboxValid =
        listItem.content.length === 0 &&
        listItem.taskCheckbox === null &&
        !listItem.seenMarkerLineEnding
    }
  },

  // taskListCheckMarker: the [ and ] characters - structural, not needed
  taskListCheckMarker: () => {},

  // taskListCheckValueUnchecked: the whitespace in [ ] - marks unchecked state
  // Can be space, tab, or newline - preserve exactly for lossless roundtrip
  taskListCheckValueUnchecked: (ctx, token) => {
    const listItem = ctx.find('listItem')
    if (listItem && listItem._pendingCheckboxValid) {
      const value = ctx.slice(token)
      listItem._pendingCheckboxMarker = '[' + value + ']'
    }
  },

  // taskListCheckValueChecked: the x in [x] - marks checked state
  taskListCheckValueChecked: (ctx, token) => {
    const listItem = ctx.find('listItem')
    if (listItem && listItem._pendingCheckboxValid) {
      // Preserve the case (x vs X)
      const value = ctx.slice(token)
      listItem._pendingCheckboxMarker = value === 'X' ? '[X]' : '[x]'
    }
  },
})

// =============================================================================
// EXIT HANDLERS
// =============================================================================

export const exit = definePartialHandlers({
  // taskListCheck: checkbox container exit - finalize the checkbox decision
  taskListCheck: (ctx, token) => {
    const listItem = ctx.find('listItem')
    if (!listItem) return

    if (listItem._pendingCheckboxValid && listItem._pendingCheckboxMarker) {
      // Valid checkbox at start of content - use it
      listItem.taskCheckbox = listItem._pendingCheckboxMarker as CheckboxMarker
    } else {
      // Not a valid checkbox position - add the raw text to preserve it
      // The checkbox syntax becomes literal text
      const checkboxText = ctx.slice(token)
      const paragraph = ctx.find('paragraph')
      if (paragraph) {
        // Prepend the checkbox text to the paragraph content
        paragraph.pendingCheckboxText = checkboxText
      }
    }

    // Clean up temporary fields
    listItem._pendingCheckboxValid = undefined
    listItem._pendingCheckboxMarker = undefined
  },

  // taskListCheckMarker: the [ and ] characters - structural, not needed
  taskListCheckMarker: () => {},

  // taskListCheckValueUnchecked: already handled in enter
  taskListCheckValueUnchecked: () => {},

  // taskListCheckValueChecked: already handled in enter
  taskListCheckValueChecked: () => {},
})
