/**
 * Token dispatch — routes tokens to bound handlers or content paths.
 *
 * pushToken is the main entry point. For each token:
 * - Open:      resolveOpenHandler → if found, end prose if needed, apply handler ops
 * - Close:     resolveCloseHandler → if found, apply handler ops; else content
 * - SelfClose: resolveSelfCloseHandler → if found, end prose if needed, apply handler ops
 * - Content:   onContent(top, text) → apply ops
 *
 * endCurrentProse is the loop's responsibility — handlers do NOT call it.
 * All effects go through ParserOp[] returned by handlers and applied by machine.apply().
 */

import type { Token } from '../types'
import type { Frame } from './types'
import type { ParserOp } from './ops'
import type { HandlerContext } from './handler-context'
import { resolveOpenHandler, resolveCloseHandler, resolveSelfCloseHandler, tokenRaw } from './resolve'
import { onContent, endTopProse, isAllWhitespace } from './content'
import { emitStructuralError } from './ops'
import { KNOWN_STRUCTURAL_TAGS } from '../constants'

// =============================================================================
// ParserLoopContext — all shared state needed for token dispatch
// =============================================================================

export interface ParserLoopContext {
  machine: {
    mode: string
    peek(): Frame | undefined
    apply(ops: ParserOp[]): void
    readonly stack: readonly Frame[]
  }
  handlerCtx: HandlerContext
  deferredYield: { target: 'user' | 'invoke' | 'worker' | 'parent' | null; postYieldHasContent: boolean }
}

// =============================================================================
// pushToken — main entry point
// =============================================================================

export function pushToken(token: Token, ctx: ParserLoopContext): void {
  if (ctx.machine.mode === 'observing') {
    if (token._tag !== 'Content' || !isAllWhitespace(token.text)) {
      ctx.deferredYield.postYieldHasContent = true
    }
    return
  }
  if (ctx.machine.mode !== 'active') return

  const top = ctx.machine.peek()
  if (!top) return

  switch (token._tag) {
    case 'Open': {
      const handler = resolveOpenHandler(token.tagName, top)
      if (handler) {
        // Loop responsibility: end prose before any structural open tag in prose context.
        // resolveOpenHandler returns a handler for reason/message/invoke only when top.type === 'prose'.
        // For parameter/filter, top.type === 'invoke' — no prose to end.
        if (top.type === 'prose') {
          ctx.machine.apply(endTopProse(top) as ParserOp[])
        }
        ctx.machine.apply(handler.open(token.attrs ?? new Map(), ctx.handlerCtx))
      } else {
        ctx.machine.apply(onContent(top, tokenRaw(token)))
      }
      break
    }

    case 'Close': {
      const handler = resolveCloseHandler(token.tagName, top)
      if (handler) {
        ctx.machine.apply(handler.close(ctx.handlerCtx))
      } else {
        if (KNOWN_STRUCTURAL_TAGS.has(token.tagName)) {
          ctx.machine.apply([emitStructuralError({
            _tag: 'StrayCloseTag',
            tagName: token.tagName,
            detail: `Unexpected close '</${token.tagName}>' with no matching open in current context`,
          })])
        }
        ctx.machine.apply(onContent(top, tokenRaw(token)))
      }
      break
    }

    case 'SelfClose': {
      const handler = resolveSelfCloseHandler(token.tagName, top)
      if (handler) {
        if (top.type === 'prose') {
          ctx.machine.apply(endTopProse(top) as ParserOp[])
        }
        const ops = handler.selfClose(token.attrs ?? new Map(), ctx.handlerCtx)
        // Extract yield target from observe op before applying
        for (const op of ops) {
          if (op.type === 'observe' && op.target) {
            ctx.deferredYield.target = op.target as 'user' | 'invoke' | 'worker' | 'parent'
            ctx.deferredYield.postYieldHasContent = false
          }
        }
        ctx.machine.apply(ops)
      } else {
        ctx.machine.apply(onContent(top, tokenRaw(token)))
      }
      break
    }

    case 'Content': {
      ctx.machine.apply(onContent(top, token.text))
      break
    }
  }
}
