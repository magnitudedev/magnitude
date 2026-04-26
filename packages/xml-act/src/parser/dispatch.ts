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

import type { Token, SourceSpan } from '../types'
import type { Frame, InvokeFrame, PendingClose } from './types'
import type { ParserOp } from './ops'
import type { HandlerContext } from './handler-context'
import type { OpenResolution } from './resolve'
import { resolveOpen, resolveCloseHandler, resolveSelfCloseHandler, tokenRaw } from './resolve'
import { onContent, endTopProse, isAllWhitespace } from './content'
import { emitStructuralError } from './ops'
import { bindOpen } from './handler'
import { invokeOpenHandler, parameterOpenHandler } from './handlers/invoke'
import { KNOWN_STRUCTURAL_TAGS, MAGNITUDE_PREFIX } from '../constants'

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
  invalidSubtree: {
    tag: string
    depth: number
    invoke: boolean
  } | null
  pendingMismatch: PendingClose | null
}

// =============================================================================
// pushToken — main entry point
// =============================================================================

// =============================================================================
// Shared helpers
// =============================================================================

const PROSE_LEVEL_FRAME_TYPES = new Set(['prose', 'reason', 'message'])

function isValidContinuation(token: Token, frameType: string, ctx: ParserLoopContext): boolean {
  if (token._tag === 'Open') {
    if (frameType === 'parameter') {
      if (token.tagName === 'magnitude:filter') return true

      let invokeFrame: Frame | undefined
      const stack = ctx.machine.stack
      for (let i = stack.length - 1; i >= 0; i--) {
        const frame = stack[i]
        if (frame.type === 'invoke') {
          invokeFrame = frame
          break
        }
      }

      if (token.tagName === 'magnitude:parameter') {
        const paramName = (token.attrs ?? new Map()).get('name') ?? ''
        if (invokeFrame?.type === 'invoke') {
          const schema = ctx.handlerCtx.invokeCtx.toolSchemas.get(invokeFrame.toolTag)
          if (schema && !schema.parameters.has(paramName)) {
            return false
          }
        }
        return true
      }

      if (token.tagName.startsWith(MAGNITUDE_PREFIX) && invokeFrame?.type === 'invoke') {
        const suffix = token.tagName.slice(MAGNITUDE_PREFIX.length)
        const schema = ctx.handlerCtx.invokeCtx.toolSchemas.get(invokeFrame.toolTag)
        return schema ? schema.parameters.has(suffix) : false
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

// =============================================================================
// Mismatch recovery helpers
// =============================================================================

function isAllWs(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch !== ' ' && ch !== '\t' && ch !== '\n' && ch !== '\r') return false
  }
  return true
}

/** Confirm a pending mismatch — silently close the frame using the effective (canonical) close tag */
function confirmMismatch(ctx: ParserLoopContext): void {
  const pm = ctx.pendingMismatch!
  const top = ctx.machine.peek()
  if (!top) { ctx.pendingMismatch = null; return }

  const handler = resolveCloseHandler(pm.effectiveTagName, top)
  if (handler) {
    // Silent recovery — no structural error emitted
    ctx.machine.apply(handler.close(ctx.handlerCtx, pm.tokenSpan))
    // Apply buffered whitespace to the new top frame
    if (pm.wsBuffer.length > 0) {
      const newTop = ctx.machine.peek()
      if (newTop) {
        ctx.machine.apply(onContent(newTop, pm.wsBuffer, pm.tokenSpan))
      }
    }
  }
  ctx.pendingMismatch = null
}

/** Reject a pending mismatch — dump as content */
function rejectMismatch(ctx: ParserLoopContext, extraContent?: string): void {
  const pm = ctx.pendingMismatch!
  const top = ctx.machine.peek()!

  ctx.machine.apply([emitStructuralError({
    _tag: 'AmbiguousMagnitudeClose',
    tagName: pm.tagName,
    expectedTagName: pm.effectiveTagName,
    raw: `</${pm.tagName}>`,
    detail: `Close tag </${pm.tagName}> does not match the current ${getParentTagName(top)} block. Did you mean </${pm.effectiveTagName}>?`,
    primarySpan: pm.tokenSpan,
  })])

  const raw = `</${pm.tagName}>` + pm.wsBuffer + (extraContent ?? '')
  ctx.machine.apply(onContent(top, raw, pm.tokenSpan))
  ctx.pendingMismatch = null
}

/**
 * Resolve a pending mismatch against the incoming token.
 * Returns 'consumed' if the token was absorbed, 'passthrough' if the caller should process it.
 */
function resolvePendingMismatch(token: Token, ctx: ParserLoopContext): 'consumed' | 'passthrough' {
  const pm = ctx.pendingMismatch!

  if (token._tag === 'Content') {
    if (isAllWs(token.text)) {
      if (token.text.includes('\n')) {
        pm.sawNewline = true
      }
      pm.wsBuffer += token.text
      return 'consumed'
    }
    // Non-whitespace content after mismatch
    if (pm.sawNewline) {
      // Newline was seen — confirm the mismatch recovery
      confirmMismatch(ctx)
      return 'passthrough'
    }
    // No newline — reject, dump as content
    rejectMismatch(ctx, token.text)
    return 'consumed'
  }

  if (token._tag === 'Close') {
    // Check if this close matches the frame ABOVE the current one (cascade)
    // e.g., mismatched param close followed by invoke close
    const machineStack = ctx.machine.stack
    const parentIdx = machineStack.length - 2
    const parentFrame = parentIdx >= 0 ? machineStack[parentIdx] : null
    if (parentFrame) {
      const parentHandler = resolveCloseHandler(token.tagName, parentFrame)
      if (parentHandler) {
        // Cascade: confirm mismatch (close current frame), then passthrough for parent close
        confirmMismatch(ctx)
        return 'passthrough'
      }
    }

    // For Close tokens, only confirm via cascade (above). Don't use sawNewline alone —
    // a newline after a mismatch followed by another close that doesn't cascade should reject.
    rejectMismatch(ctx)
    return 'passthrough'
  }

  if (token._tag === 'Open' || token._tag === 'SelfClose') {
    // Check if this open is a valid structural continuation for the parent frame
    // e.g., mismatched filter close followed by <magnitude:command> in invoke
    const machineStack = ctx.machine.stack
    const parentIdx = machineStack.length - 2
    const parentFrame = parentIdx >= 0 ? machineStack[parentIdx] : null
    if (parentFrame) {
      if (parentFrame.type !== 'prose' && isValidContinuation(token, parentFrame.type, ctx)) {
        confirmMismatch(ctx)
        return 'passthrough'
      }
      // For prose-level parent: confirm if the open is a recognized structural tag
      if (parentFrame.type === 'prose' && token.tagName.startsWith(MAGNITUDE_PREFIX)) {
        confirmMismatch(ctx)
        return 'passthrough'
      }
    }

    if (pm.sawNewline) {
      // At prose level, only confirm if the open is a magnitude structural tag
      if (token.tagName.startsWith(MAGNITUDE_PREFIX)) {
        confirmMismatch(ctx)
        return 'passthrough'
      }
      confirmMismatch(ctx)
      return 'passthrough'
    }
    rejectMismatch(ctx)
    return 'passthrough'
  }

  rejectMismatch(ctx)
  return 'passthrough'
}

// =============================================================================
// Close-tag helpers
// =============================================================================

function getCanonicalClose(frame: Frame): string {
  switch (frame.type) {
    case 'invoke':
      return 'magnitude:invoke'
    case 'parameter':
      return 'magnitude:parameter'
    case 'filter':
      return 'magnitude:filter'
    case 'message':
      return 'magnitude:message'
    case 'reason':
      return 'magnitude:reason'
    case 'prose':
      return ''
  }
}

function getParentTagName(frame: Frame): string | undefined {
  switch (frame.type) {
    case 'prose':
      return undefined
    case 'invoke':
      return 'magnitude:invoke'
    case 'parameter':
      return 'magnitude:parameter'
    case 'filter':
      return 'magnitude:filter'
    case 'message':
      return 'magnitude:message'
    case 'reason':
      return 'magnitude:reason'
  }
}

function invalidMagnitudeOpenDetail(raw: string, top: Frame): string {
  if (top.type === 'prose') {
    return `Unknown tag ${raw} at top level. Did you mean to invoke a tool?`
  }
  if (top.type === 'invoke') {
    return `Invalid tag ${raw} inside magnitude:invoke. Did you mean <magnitude:parameter name="...">?`
  }
  return `Invalid tag ${raw} inside ${getParentTagName(top)}. Nested magnitude: tags are not allowed here.`
}

function absorbInvalidSubtreeToken(ctx: ParserLoopContext, token: Token): void {
  const top = ctx.machine.peek()
  if (!top || ctx.invalidSubtree?.invoke) return
  ctx.machine.apply(onContent(top, tokenRaw(token), token.span))
}

function handleInvalidMagnitudeOpen(
  ctx: ParserLoopContext,
  top: Frame,
  token: Extract<Token, { readonly _tag: 'Open' | 'SelfClose' }>,
): void {
  const raw = tokenRaw(token)
  ctx.machine.apply([emitStructuralError({
    _tag: 'InvalidMagnitudeOpen',
    tagName: token.tagName,
    parentTagName: getParentTagName(top),
    raw,
    detail: invalidMagnitudeOpenDetail(raw, top),
    primarySpan: token.span,
  })])

  if (top.type !== 'invoke' && top.type !== 'prose') {
    ctx.machine.apply(onContent(top, raw, token.span))
  }

  if (token._tag === 'Open' || token._tag === 'SelfClose') {
    ctx.invalidSubtree = {
      tag: token.tagName,
      depth: 1,
      invoke: top.type === 'invoke',
    }
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

  // Resolve pending mismatch against incoming token
  if (ctx.pendingMismatch !== null) {
    const result = resolvePendingMismatch(token, ctx)
    if (result === 'consumed') return
    // passthrough — mismatch was resolved, process token normally
  }

  if (ctx.invalidSubtree !== null) {
    const top = ctx.machine.peek()
    if (!top) {
      ctx.invalidSubtree = null
      return
    }

    const sub = ctx.invalidSubtree

    if (token._tag === 'Open' && token.tagName === sub.tag) {
      sub.depth++
      absorbInvalidSubtreeToken(ctx, token)
      return
    }

    if (token._tag === 'Close' && token.tagName === sub.tag) {
      sub.depth--
      if (sub.depth === 0) {
        const wasInvoke = sub.invoke
        ctx.invalidSubtree = null
        if (wasInvoke) {
          // Invoke subtree: silently discard the close
          return
        }
        // Body frame (parameter/filter): if close matches current frame,
        // let it fall through for greedy matching
        const handler = resolveCloseHandler(token.tagName, top)
        if (handler) {
          // Fall through to normal token processing below
        } else {
          absorbInvalidSubtreeToken(ctx, token)
          return
        }
      } else {
        absorbInvalidSubtreeToken(ctx, token)
        return
      }
    }

    if (token._tag === 'Close') {
      const handler = resolveCloseHandler(token.tagName, top)
      if (handler) {
        ctx.invalidSubtree = null
      } else {
        absorbInvalidSubtreeToken(ctx, token)
        return
      }
    } else {
      absorbInvalidSubtreeToken(ctx, token)
      return
    }
  }

  const top = ctx.machine.peek()
  if (!top) return

  switch (token._tag) {
    case 'Open': {
      const resolution: OpenResolution = resolveOpen(token.tagName, top, false, ctx.handlerCtx)
      switch (resolution.kind) {
        case 'handler': {
          if (top.type === 'prose') {
            ctx.machine.apply(endTopProse(top) as ParserOp[])
          }
          ctx.machine.apply(resolution.handler.open(token.attrs ?? new Map(), ctx.handlerCtx, token.span))
          break
        }
        case 'heuristicInvoke': {
          if (top.type === 'prose') {
            ctx.machine.apply(endTopProse(top) as ParserOp[])
          }
          const bound = bindOpen(invokeOpenHandler, top)
          ctx.machine.apply(bound.open(new Map([['tool', resolution.toolTag]]), ctx.handlerCtx, token.span))
          break
        }
        case 'heuristicParameter': {
          const bound = bindOpen(parameterOpenHandler, top as InvokeFrame)
          ctx.machine.apply(bound.open(new Map([['name', resolution.paramName]]), ctx.handlerCtx, token.span))
          break
        }
        case 'invalidMagnitudeOpen': {
          handleInvalidMagnitudeOpen(ctx, top, token)
          break
        }
        case 'unresolved': {
          ctx.machine.apply(onContent(top, tokenRaw(token), token.span))
          break
        }
      }
      break
    }

    case 'Close': {
      const handler = resolveCloseHandler(token.tagName, top)
      if (handler) {
        ctx.machine.apply(handler.close(ctx.handlerCtx, token.span))
      } else if (token.tagName.startsWith(MAGNITUDE_PREFIX) && top.type === 'prose') {
        // Stray magnitude close at prose level — no matching open
        ctx.machine.apply([emitStructuralError({
          _tag: 'StrayCloseTag',
          tagName: token.tagName,
          detail: `Unexpected close '</${token.tagName}>' with no matching open in current context`,
          primarySpan: token.span,
        })])
        ctx.machine.apply(onContent(top, tokenRaw(token), token.span))
      } else if (token.tagName.startsWith(MAGNITUDE_PREFIX) && top.type !== 'prose') {
        // Magnitude close mismatch inside a body frame — buffer for mismatch recovery
        ctx.pendingMismatch = {
          tagName: token.tagName,
          effectiveTagName: getCanonicalClose(top),
          wsBuffer: '',
          sawNewline: false,
          tokenSpan: token.span,
        }
      } else {
        if (KNOWN_STRUCTURAL_TAGS.has(token.tagName)) {
          ctx.machine.apply([emitStructuralError({
            _tag: 'StrayCloseTag',
            tagName: token.tagName,
            detail: `Unexpected close '</${token.tagName}>' with no matching open in current context`,
            primarySpan: token.span,
          })])
        }
        ctx.machine.apply(onContent(top, tokenRaw(token), token.span))
      }
      break
    }

    case 'SelfClose': {
      const handler = resolveSelfCloseHandler(token.tagName, top)
      if (handler) {
        if (top.type === 'prose') {
          ctx.machine.apply(endTopProse(top) as ParserOp[])
        }
        const ops = handler.selfClose(token.attrs ?? new Map(), ctx.handlerCtx, token.span)
        for (const op of ops) {
          if (op.type === 'observe' && op.target) {
            ctx.deferredYield.target = op.target as 'user' | 'invoke' | 'worker' | 'parent'
            ctx.deferredYield.postYieldHasContent = false
          }
        }
        ctx.machine.apply(ops)
      } else {
        const resolution: OpenResolution = resolveOpen(token.tagName, top, true, ctx.handlerCtx)
        if (resolution.kind === 'invalidMagnitudeOpen') {
          handleInvalidMagnitudeOpen(ctx, top, token)
        } else {
          ctx.machine.apply(onContent(top, tokenRaw(token), token.span))
        }
      }
      break
    }

    case 'Content': {
      ctx.machine.apply(onContent(top, token.text, token.span))
      break
    }
  }
}
