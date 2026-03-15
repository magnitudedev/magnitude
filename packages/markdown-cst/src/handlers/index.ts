/**
 * Handler Index
 *
 * Merges all partial handlers into complete Handlers objects.
 * Uses type-level checks to ensure:
 * 1. All token types are covered (exhaustiveness)
 * 2. No token type is handled by multiple files (no duplicates)
 */

import type { HandlerConfig } from '../types'
import { mergeHandlers } from './merge'

import * as block from './block'
import * as list from './list'
import * as taskList from './task-list'
import * as blockquote from './blockquote'
import * as table from './table'
import * as inline from './inline'
import * as flow from './flow'
import * as noop from './noop'

const enterHandlers = mergeHandlers(
  block.enter,
  list.enter,
  taskList.enter,
  blockquote.enter,
  table.enter,
  inline.enter,
  flow.enter,
  noop.enter,
)

const exitHandlers = mergeHandlers(
  block.exit,
  list.exit,
  taskList.exit,
  blockquote.exit,
  table.exit,
  inline.exit,
  flow.exit,
  noop.exit,
)

export const config: HandlerConfig = {
  enter: enterHandlers,
  exit: exitHandlers,
}

// Re-export helpers for use in other modules
export { addBlockToParent, isUnsupportedInCurrentContext, getResourceBuilder } from './helpers'
