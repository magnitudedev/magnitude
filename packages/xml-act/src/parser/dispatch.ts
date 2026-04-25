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
import type { Frame, InvokeFrame, PendingClose } from './types'
import type { ParserOp } from './ops'
import type { HandlerContext } from './handler-context'
import type { OpenResolution } from './resolve'
import { resolveOpen, resolveCloseHandler, resolveSelfCloseHandler, tokenRaw } from './resolve'
import { onContent, endTopProse, isAllWhitespace } from './content'
import { emitStructuralError } from './ops'
import { bindOpen } from './handler'
import { invokeOpenHandler, parameterOpenHandler } from './handlers/invoke'
import { KNOWN_STRUCTURAL_TAGS, ESCAPE_TAG, MAGNITUDE_PREFIX } from '../constants'

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
  /** Escape nesting depth — when > 0, all tokens become raw content */
  escapeDepth: number
  invalidSubtree: {
    tag: string
    depth: number
    invoke: boolean
  } | null
}

// =============================================================================
// pushToken — main entry point
// =============================================================================

// =============================================================================
// Tentative close helpers (stack-based for cascade support)
// =============================================================================

const PROSE_LEVEL_FRAME_TYPES = new Set(['prose', 'reason', 'message'])

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

    const handler = resolveCloseHandler(pc.effectiveTagName, top)
    if (!handler) {
      if (pc.wsBuffer.length > 0) {
        ctx.machine.apply(onContent(top, pc.wsBuffer, pc.tokenSpan))
      }
      continue
    }
    ctx.machine.apply(handler.close(ctx.handlerCtx, pc.tokenSpan))

    // Apply buffered whitespace to the NEW top frame (after close)
    if (pc.wsBuffer.length > 0) {
      const newTop = ctx.machine.peek()
      if (newTop) {
        ctx.machine.apply(onContent(newTop, pc.wsBuffer, pc.tokenSpan))
      }
    }
  }
  ctx.pendingCloseStack = []
}

/** Reject all pending closes — dump as content */
function rejectAllPendingCloses(ctx: ParserLoopContext, extraContent?: string): void {
  for (const pc of ctx.pendingCloseStack) {
    if (pc.mismatchRecovery) {
      ctx.machine.apply([emitStructuralError({
        _tag: 'AmbiguousMagnitudeClose',
        tagName: pc.tagName,
        expectedTagName: pc.effectiveTagName,
        raw: `</${pc.tagName}>`,
        detail: `Close tag </${pc.tagName}> does not match the current ${pc.effectiveTagName} block. Did you mean </${pc.effectiveTagName}>?`,
        primarySpan: pc.tokenSpan,
      })])
    }
  }

  const top = ctx.machine.peek()!
  const raw = pendingCloseRaw(ctx.pendingCloseStack) + (extraContent ?? '')
  ctx.machine.apply(onContent(top, raw, ctx.pendingCloseStack[0].tokenSpan))
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
    // Check greedy replace FIRST — if the close matches the same frame as the
    // last pending close, it's a greedy last-match replacement
    if (stack.length === 1) {
      const handler = resolveCloseHandler(token.tagName, top)
      if (handler) {
        // Same-frame close — replace (greedy last-match)
        ctx.machine.apply(onContent(top, `</${lastPc.tagName}>` + lastPc.wsBuffer, lastPc.tokenSpan))
        ctx.pendingCloseStack = [{
          tagName: token.tagName,
          effectiveTagName: token.tagName,
          mismatchRecovery: false,
          frameType: top.type,
          wsBuffer: '',
          sawNewline: false,
          tokenSpan: token.span,
        }]
        return 'consumed'
      }
    }

    // Check if matches the frame that would be on top after all pending confirms
    // i.e., would this close extend the cascade?
    if (effectiveFrame) {
      const handler = resolveCloseHandler(token.tagName, effectiveFrame)
      if (handler) {
        // Extend cascade — push another pending close
        stack.push({
          tagName: token.tagName,
          effectiveTagName: token.tagName,
          mismatchRecovery: false,
          frameType: effectiveFrame.type,
          wsBuffer: '',
          sawNewline: false,
          tokenSpan: token.span,
        })
        return 'consumed'
      }
      if (token.tagName.startsWith(MAGNITUDE_PREFIX) && effectiveFrame.type !== 'prose') {
        stack.push({
          tagName: token.tagName,
          effectiveTagName: getCanonicalClose(effectiveFrame),
          mismatchRecovery: true,
          frameType: effectiveFrame.type,
          wsBuffer: '',
          sawNewline: false,
          tokenSpan: token.span,
        })
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
  if (isValidContinuation(token, lastPc.frameType, ctx)) {
    confirmAllPendingCloses(ctx)
    return 'passthrough'
  }
  if (effectiveFrame && PROSE_LEVEL_FRAME_TYPES.has(effectiveFrame.type)) {
    confirmAllPendingCloses(ctx)
    return 'passthrough'
  }
  rejectAllPendingCloses(ctx)
  return 'passthrough'
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

  // Escape mode: all tokens become raw content until escape close
  if (ctx.escapeDepth > 0) {
    if (token._tag === 'Close' && token.tagName === ESCAPE_TAG) {
      ctx.escapeDepth--
      // Silently consume the close tag — don't emit as content
    } else {
      // Everything else is raw content on the current top frame
      const top = ctx.machine.peek()
      if (top) {
        ctx.machine.apply(onContent(top, tokenRaw(token), token.span))
      }
    }
    return
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

  // Resolve tentative close against incoming token
  if (ctx.pendingCloseStack.length > 0) {
    const result = resolvePendingClose(token, ctx)
    if (result === 'consumed') return
    // passthrough — pendingClose was cleared, process token normally
  }

  const top = ctx.machine.peek()
  if (!top) return

  // Check for escape open — enter escape mode (after pending close resolution)
  // Valid everywhere except inside invoke frames (where only parameter/filter are valid children)
  if (token._tag === 'Open' && token.tagName === ESCAPE_TAG) {
    if (top.type !== 'invoke') {
      ctx.escapeDepth++
      // Silently consume the open tag — don't emit as content
      return
    }
    // Fall through to normal Open handling — will be resolved as invalidMagnitudeOpen
  }

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
        ctx.pendingCloseStack.push({
          tagName: token.tagName,
          effectiveTagName: token.tagName,
          mismatchRecovery: false,
          frameType: top.type,
          wsBuffer: '',
          sawNewline: false,
          tokenSpan: token.span,
        })
      } else if (token.tagName.startsWith(MAGNITUDE_PREFIX) && top.type !== 'prose') {
        ctx.pendingCloseStack.push({
          tagName: token.tagName,
          effectiveTagName: getCanonicalClose(top),
          mismatchRecovery: true,
          frameType: top.type,
          wsBuffer: '',
          sawNewline: false,
          tokenSpan: token.span,
        })
      } else if (token.tagName.startsWith(MAGNITUDE_PREFIX) && top.type === 'prose') {
        // Stray magnitude close at prose level — no matching open
        ctx.machine.apply([emitStructuralError({
          _tag: 'StrayCloseTag',
          tagName: token.tagName,
          detail: `Unexpected close '</${token.tagName}>' with no matching open in current context`,
          primarySpan: token.span,
        })])
        ctx.machine.apply(onContent(top, tokenRaw(token), token.span))
      } else if (token.tagName.startsWith(MAGNITUDE_PREFIX)) {
        // Magnitude close mismatch inside a body frame
        ctx.machine.apply([emitStructuralError({
          _tag: 'AmbiguousMagnitudeClose',
          tagName: token.tagName,
          expectedTagName: getCanonicalClose(top).replace('magnitude:', ''),
          raw: tokenRaw(token),
          detail: `Close tag ${tokenRaw(token)} does not match the current ${getParentTagName(top)} block. Did you mean </${getCanonicalClose(top)}>?`,
          primarySpan: token.span,
        })])
        ctx.machine.apply(onContent(top, tokenRaw(token), token.span))
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
