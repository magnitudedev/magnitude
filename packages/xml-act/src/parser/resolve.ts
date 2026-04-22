/**
 * Token resolution — returns typed bound handlers for structural tokens.
 *
 * resolveOpenHandler:     tagName + top → BoundOpenHandler | undefined
 * resolveCloseHandler:    tagName + top → BoundCloseHandler | undefined
 * resolveSelfCloseHandler: tagName + top → BoundSelfCloseHandler | undefined
 *
 * If undefined is returned, the token is treated as content (tokenRaw appended).
 *
 * Architecture:
 * - resolveOpenHandler switches on tagName, then checks top.type.
 *   TypeScript narrows top in the fall-through of each type guard.
 *   bindOpen(handler, narrowedTop) captures the narrowed frame in a closure.
 * - resolveCloseHandler switches on top.type first.
 *   TypeScript narrows top in every case branch.
 *   bindClose(handler, narrowedTop) captures the narrowed frame.
 * - No as-casts. No validTags sets. One layer, not two.
 *
 * Compile-time lockstep with grammar via nesting.ts:
 *   import type { _VerifyProseChildren, _VerifyInvokeChildren } from '../nesting'
 */

import type { Token } from '../types'
import type { Frame } from './types'
import type { BoundOpenHandler, BoundCloseHandler, BoundSelfCloseHandler } from './handler'
import { bindOpen, bindClose } from './handler'
import { KNOWN_STRUCTURAL_TAGS } from '../constants'

// Compile-time verification that this file stays in lockstep with nesting.ts
import type { _VerifyProseChildren, _VerifyInvokeChildren } from '../nesting'
type _AssertProseChildren = _VerifyProseChildren extends true ? true : never
type _AssertInvokeChildren = _VerifyInvokeChildren extends true ? true : never
declare const _assertProseChildren: _AssertProseChildren
declare const _assertInvokeChildren: _AssertInvokeChildren

import {
  reasonOpenHandler,
  reasonCloseHandler,
} from './handlers/reason'
import {
  messageOpenHandler,
  messageCloseHandler,
} from './handlers/message'
import {
  invokeOpenHandler,
  invokeCloseHandler,
  parameterOpenHandler,
  parameterCloseHandler,
  filterOpenHandler,
  filterCloseHandler,
} from './handlers/invoke'
import { makeYieldHandler } from './handlers/yield'

// =============================================================================
// resolveOpenHandler
// =============================================================================

export function resolveOpenHandler(tagName: string, top: Frame): BoundOpenHandler | undefined {
  switch (tagName) {
    case 'reason':
      if (top.type !== 'prose') return undefined
      // top is narrowed to ProseFrame — TypeScript verifies reasonOpenHandler: OpenHandler<ProseFrame, ...>
      return bindOpen(reasonOpenHandler, top)

    case 'message':
      if (top.type !== 'prose') return undefined
      // top is narrowed to ProseFrame
      return bindOpen(messageOpenHandler, top)

    case 'invoke':
      if (top.type !== 'prose') return undefined
      // top is narrowed to ProseFrame
      return bindOpen(invokeOpenHandler, top)

    case 'parameter':
      if (top.type !== 'invoke') return undefined
      // top is narrowed to InvokeFrame — TypeScript verifies parameterOpenHandler: OpenHandler<InvokeFrame, ...>
      return bindOpen(parameterOpenHandler, top)

    case 'filter':
      if (top.type !== 'invoke') return undefined
      // top is narrowed to InvokeFrame
      return bindOpen(filterOpenHandler, top)

    default:
      return undefined
  }
}

// =============================================================================
// resolveCloseHandler
// =============================================================================

export function resolveCloseHandler(tagName: string, top: Frame): BoundCloseHandler | undefined {
  // Switch on top.type first — TypeScript narrows top in every case
  switch (top.type) {
    case 'reason':
      // top is ReasonFrame — TypeScript verifies reasonCloseHandler: CloseHandler<ReasonFrame>
      return tagName === 'reason' ? bindClose(reasonCloseHandler, top) : undefined

    case 'message':
      // top is MessageFrame
      return tagName === 'message' ? bindClose(messageCloseHandler, top) : undefined

    case 'invoke':
      // top is InvokeFrame
      return tagName === 'invoke' ? bindClose(invokeCloseHandler, top) : undefined

    case 'parameter':
      // top is ParameterFrame
      return tagName === 'parameter' ? bindClose(parameterCloseHandler, top) : undefined

    case 'filter':
      // top is FilterFrame
      return tagName === 'filter' ? bindClose(filterCloseHandler, top) : undefined

    case 'prose':
      return undefined
  }
}

// =============================================================================
// resolveSelfCloseHandler
// =============================================================================

export function resolveSelfCloseHandler(tagName: string, top: Frame): BoundSelfCloseHandler | undefined {
  if (top.type === 'prose' && tagName.startsWith('yield_')) {
    return makeYieldHandler(tagName)
  }
  return undefined
}

// =============================================================================
// tokenRaw — reconstruct raw text for non-structural tokens
// =============================================================================

/**
 * Reconstruct the raw text of a token for appending as literal content.
 *
 * attrs is always ReadonlyMap<string, string> — the tokenizer always constructs it as a Map.
 * The previous instanceof Map defensive branch is removed. If a non-Map path is ever found
 * in the tokenizer, fix the tokenizer — not this function.
 */
export function tokenRaw(token: Token): string {
  switch (token._tag) {
    case 'Open': {
      const attrsStr = token.attrs
        ? Array.from(token.attrs.entries()).map(([k, v]) => ` ${k}="${v}"`).join('')
        : ''
      return `<${token.tagName}${attrsStr}>`
    }
    case 'Close':
      return token.raw ?? `</${token.tagName}>`
    case 'SelfClose': {
      const attrsStr = token.attrs
        ? Array.from(token.attrs.entries()).map(([k, v]) => ` ${k}="${v}"`).join('')
        : ''
      return `<${token.tagName}${attrsStr}/>`
    }
    case 'Content':
      return token.text
  }
}
