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
import type { Frame, PendingClose } from './types'
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
  pendingCloseStack: PendingClose[]
}

// =============================================================================
// pushToken — main entry point
// =============================================================================

// =============================================================================
// Tentative close helpers (stack-based for cascade support)
// =============================================================================

const PROSE_LEVEL_FRAME_TYPES = new Set(['prose', 'reason', 'message'])

function isAllWs(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch !== ' ' && ch !== '\t' && ch !== '\n' && ch !== '\r') return false
  }
  return true
}

function isValidContinuation(token: Token, frameType: string, ctx: ParserLoopContext): boolean {
  if (token._tag === 'Open') {
    if (frameType === 'parameter') {
      if (token.tagName === 'filter') return true
      if (token.tagName === 'parameter') {
        // Validate param name against tool schema
        const paramName = (token.attrs ?? new Map()).get('name') ?? ''
        // Find the invoke frame — it's the parent of the parameter frame on the machine stack
        const stack = ctx.machine.stack
        for (let i = stack.length - 1; i >= 0; i--) {
          const frame = stack[i]
          if (frame.type === 'invoke') {
            const schema = ctx.handlerCtx.invokeCtx.toolSchemas.get(frame.toolTag)
            if (schema && !schema.parameters.has(paramName)) {
              return false  // Invalid param name — not a valid continuation
            }
            break
          }
        }
        return true
      }
      return false
    }
    return true
  }
  if (token._tag === 'SelfClose') {
    return true
  }
  return false
}

/** Build the raw text for all pending closes (for rejection) */
function pendingCloseRaw(stack: PendingClose[]): string {
  let raw = ''
  for (const pc of stack) {
    raw += `</${pc.tagName}>` + pc.wsBuffer
  }
  return raw
}

/** Confirm all pending closes — apply close handlers top-down */
function confirmAllPendingCloses(ctx: ParserLoopContext): void {
  for (let i = 0; i < ctx.pendingCloseStack.length; i++) {
    const pc = ctx.pendingCloseStack[i]
    const top = ctx.machine.peek()
    if (!top) break
    const handler = resolveCloseHandler(pc.tagName, top)
    if (!handler) {
      // Frame was already popped (e.g., filterCloseHandler pops both filter and invoke).
      // Skip this entry — it was handled by a previous close handler's cascade.
      continue
    }
    ctx.machine.apply(handler.close(ctx.handlerCtx))
  }
  ctx.pendingCloseStack = []
}

/** Reject all pending closes — dump as content */
function rejectAllPendingCloses(ctx: ParserLoopContext, extraContent?: string): void {
  const top = ctx.machine.peek()!
  const raw = pendingCloseRaw(ctx.pendingCloseStack) + (extraContent ?? '')
  ctx.machine.apply(onContent(top, raw))
  ctx.pendingCloseStack = []
}

/**
 * Resolve the tentative close stack against the incoming token.
 * Returns 'consumed' if the token was absorbed, or 'passthrough' if the
 * token should be processed normally (stack was cleared).
 */
function resolvePendingClose(token: Token, ctx: ParserLoopContext): 'consumed' | 'passthrough' {
  const stack = ctx.pendingCloseStack
  const lastPc = stack[stack.length - 1]
  const top = ctx.machine.peek()!

  // Determine the "effective frame" — the frame that would be on top if all pending closes were confirmed
  // For a single pending close, this is the current top.
  // For cascade (e.g., </parameter></invoke>), the effective frame is computed by looking
  // past the pending close count on the machine stack.
  const machineStack = ctx.machine.stack
  const effectiveFrameIdx = machineStack.length - 1 - stack.length
  const effectiveFrame = effectiveFrameIdx >= 0 ? machineStack[effectiveFrameIdx] : null

  if (token._tag === 'Content') {
    if (isAllWs(token.text)) {
      if (token.text.includes('\n')) {
        lastPc.sawNewline = true
      }
      lastPc.wsBuffer += token.text
      return 'consumed'
    }
    // Non-whitespace content
    // For top-level effective frame: confirm on '<', newline as first char, or if we saw a newline in buffered whitespace
    if (effectiveFrame && PROSE_LEVEL_FRAME_TYPES.has(effectiveFrame.type) && token.text.length > 0
        && (token.text[0] === '<' || token.text[0] === '\n' || lastPc.sawNewline)) {
      confirmAllPendingCloses(ctx)
      return 'passthrough'
    }
    // Reject entire stack
    rejectAllPendingCloses(ctx, token.text)
    return 'consumed'
  }

  if (token._tag === 'Close') {
    // Check if matches the frame that would be on top after all pending confirms
    // i.e., would this close extend the cascade?
    if (effectiveFrame) {
      const handler = resolveCloseHandler(token.tagName, effectiveFrame)
      if (handler) {
        // Extend cascade — push another pending close
        stack.push({ tagName: token.tagName, wsBuffer: '', sawNewline: false })
        return 'consumed'
      }
    }

    // Check if matches the SAME frame as the last pending close (greedy replace)
    // This only applies when stack has exactly 1 entry
    if (stack.length === 1) {
      const handler = resolveCloseHandler(token.tagName, top)
      if (handler) {
        // Same-frame close — replace (greedy last-match)
        ctx.machine.apply(onContent(top, `</${lastPc.tagName}>` + lastPc.wsBuffer))
        ctx.pendingCloseStack = [{ tagName: token.tagName, wsBuffer: '', sawNewline: false }]
        return 'consumed'
      }
    }

    // Matches neither — reject entire stack
    rejectAllPendingCloses(ctx, tokenRaw(token))
    return 'consumed'
  }

  // Open or SelfClose
  // Check if valid continuation for the frame being closed by the last pending close
  // The last pending close's tagName tells us the frame type (e.g., 'parameter' → parameter frame)
  const lastClosedFrameType = lastPc.tagName  // tagName matches frame type for structural tags
  if (isValidContinuation(token, lastClosedFrameType, ctx)) {
    confirmAllPendingCloses(ctx)
    return 'passthrough'
  } else {
    rejectAllPendingCloses(ctx)
    return 'passthrough'
  }
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

  // Resolve tentative close against incoming token
  if (ctx.pendingCloseStack.length > 0) {
    const result = resolvePendingClose(token, ctx)
    if (result === 'consumed') return
    // passthrough — pendingClose was cleared, process token normally
  }

  const top = ctx.machine.peek()
  if (!top) return

  switch (token._tag) {
    case 'Open': {
      const handler = resolveOpenHandler(token.tagName, top)
      if (handler) {
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
        // Enter tentative close — confirmed by next token
        ctx.pendingCloseStack.push({ tagName: token.tagName, wsBuffer: '', sawNewline: false })
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
