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
import type { HandlerContext } from './handler-context'
import { bindOpen, bindClose } from './handler'
import { KNOWN_STRUCTURAL_TAGS, MAGNITUDE_PREFIX } from '../constants'

// Compile-time verification that this file stays in lockstep with nesting.ts
import type { _VerifyProseChildren, _VerifyInvokeChildren } from '../nesting'
type _AssertProseChildren = _VerifyProseChildren extends true ? true : never
type _AssertInvokeChildren = _VerifyInvokeChildren extends true ? true : never
declare const _assertProseChildren: _AssertProseChildren
declare const _assertInvokeChildren: _AssertInvokeChildren

import {
  thinkOpenHandler,
  thinkCloseHandler,
} from './handlers/think'
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

export type OpenResolution =
  | { kind: 'handler'; handler: BoundOpenHandler }
  | { kind: 'heuristicInvoke'; toolTag: string }
  | { kind: 'heuristicParameter'; paramName: string }
  | { kind: 'invalidMagnitudeOpen' }
  | { kind: 'unresolved' }

// =============================================================================
// resolveOpenHandler
// =============================================================================

export function resolveOpenHandler(tagName: string, top: Frame): BoundOpenHandler | undefined {
  switch (tagName) {
    case 'magnitude:think':
      if (top.type !== 'prose') return undefined
      // top is narrowed to ProseFrame — TypeScript verifies thinkOpenHandler: OpenHandler<ProseFrame, ...>
      return bindOpen(thinkOpenHandler, top)

    case 'magnitude:message':
      if (top.type !== 'prose') return undefined
      // top is narrowed to ProseFrame
      return bindOpen(messageOpenHandler, top)

    case 'magnitude:invoke':
      if (top.type !== 'prose') return undefined
      // top is narrowed to ProseFrame
      return bindOpen(invokeOpenHandler, top)

    case 'magnitude:parameter':
      if (top.type !== 'invoke') return undefined
      // top is narrowed to InvokeFrame — TypeScript verifies parameterOpenHandler: OpenHandler<InvokeFrame, ...>
      return bindOpen(parameterOpenHandler, top)

    case 'magnitude:filter':
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

export function resolveOpen(
  tagName: string,
  top: Frame,
  isSelfClose: boolean,
  handlerCtx: HandlerContext,
): OpenResolution {
  const handler = resolveOpenHandler(tagName, top)
  if (handler) return { kind: 'handler', handler }

  if (!tagName.startsWith(MAGNITUDE_PREFIX)) {
    return { kind: 'unresolved' }
  }

  if (isSelfClose) {
    return { kind: 'invalidMagnitudeOpen' }
  }

  const suffix = tagName.slice(MAGNITUDE_PREFIX.length)

  if (top.type === 'prose') {
    return handlerCtx.invokeCtx.tools.has(suffix)
      ? { kind: 'heuristicInvoke', toolTag: suffix }
      : { kind: 'invalidMagnitudeOpen' }
  }

  if (top.type === 'invoke') {
    const schema = handlerCtx.invokeCtx.toolSchemas.get(top.toolTag)
    return schema && schema.parameters.has(suffix)
      ? { kind: 'heuristicParameter', paramName: suffix }
      : { kind: 'invalidMagnitudeOpen' }
  }

  return { kind: 'invalidMagnitudeOpen' }
}

export function resolveCloseHandler(tagName: string, top: Frame): BoundCloseHandler | undefined {
  // Switch on top.type first — TypeScript narrows top in every case
  switch (top.type) {
    case 'think':
      // top is ThinkFrame — TypeScript verifies thinkCloseHandler: CloseHandler<ThinkFrame>
      // 'think' is the short-form close alias for 'magnitude:think'
      return (tagName === 'magnitude:think' || tagName === 'think') ? bindClose(thinkCloseHandler, top) : undefined

    case 'message':
      // top is MessageFrame
      return tagName === 'magnitude:message' ? bindClose(messageCloseHandler, top) : undefined

    case 'invoke':
      // top is InvokeFrame
      if (tagName === 'magnitude:invoke') return bindClose(invokeCloseHandler, top)
      if (tagName === `magnitude:${top.toolTag}`) return bindClose(invokeCloseHandler, top)
      return undefined

    case 'parameter':
      // top is ParameterFrame
      if (tagName === 'magnitude:parameter') return bindClose(parameterCloseHandler, top)
      if (tagName === `magnitude:${top.paramName}`) return bindClose(parameterCloseHandler, top)
      return undefined

    case 'filter':
      // top is FilterFrame
      return tagName === 'magnitude:filter' ? bindClose(filterCloseHandler, top) : undefined

    case 'prose':
      return undefined
  }
}

// =============================================================================
// resolveSelfCloseHandler
// =============================================================================

export function resolveSelfCloseHandler(tagName: string, top: Frame): BoundSelfCloseHandler | undefined {
  if (top.type === 'prose' && tagName.startsWith('magnitude:yield_')) {
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
