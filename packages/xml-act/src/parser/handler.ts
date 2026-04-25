/**
 * handler.ts — generic handler interfaces and binding utilities.
 *
 * OpenHandler<TParent, TChild>  — typed open handler; TypeScript enforces parent/child relationship.
 * CloseHandler<TFrame>          — typed close handler; receives narrowed frame directly.
 * SelfCloseHandler              — for self-closing tags (yield_* tags).
 *
 * BoundOpenHandler / BoundCloseHandler — parent/top frame already captured in closure.
 * These are what resolveOpenHandler / resolveCloseHandler return to the parser loop.
 * The loop calls handler.open(attrs, ctx) without needing to know TParent.
 *
 * bindOpen / bindClose — create bound handlers from typed handlers + narrowed frames.
 * The binding is explicit and localized to resolve.ts.
 */

import type { Frame } from './types'
import type { SourceSpan } from '../types'
import type { HandlerContext } from './handler-context'
import type { ParserOp } from './ops'

// =============================================================================
// Specific op type aliases (for constraining handler return types)
// =============================================================================

import type { Op } from '../machine'
import type { TurnEngineEvent } from '../types'

export type PushOp<F extends Frame> = { readonly type: 'push'; readonly frame: F }
export type PopOp = { readonly type: 'pop' }
export type ReplaceOp<F extends Frame> = { readonly type: 'replace'; readonly frame: F }
export type EmitOp = { readonly type: 'emit'; readonly event: TurnEngineEvent }
export type ObserveOp = { readonly type: 'observe' }
export type DoneOp = { readonly type: 'done' }

// Re-export for convenience
export type { Op }

// =============================================================================
// Handler interfaces
// =============================================================================

/**
 * OpenHandler<TParent, TChild> — handles an Open token in a specific parent context.
 *
 * TParent: the frame type that must be on top of the stack when this tag opens.
 * TChild:  the frame type this handler pushes onto the stack.
 *
 * TypeScript enforces the parent/child relationship at handler definition time.
 * If open() tries to push a frame of the wrong type, TypeScript errors.
 *
 * The handler is stateless — it receives the narrowed parent frame at call time.
 * One handler instance is created per tag type and reused across all invocations.
 */
export interface OpenHandler<TParent extends Frame, TChild extends Frame> {
  open(
    attrs: ReadonlyMap<string, string>,
    parent: TParent,
    ctx: HandlerContext,
    tokenSpan: SourceSpan,
  ): ParserOp[]
}

/**
 * CloseHandler<TFrame> — handles a Close token when the top frame is TFrame.
 *
 * Receives the narrowed frame directly — no cast needed.
 * TypeScript narrows `top` via the discriminant check in resolveCloseHandler.
 */
export interface CloseHandler<TFrame extends Frame> {
  close(top: TFrame, ctx: HandlerContext, tokenSpan: SourceSpan): ParserOp[]
}

/**
 * SelfCloseHandler — handles a SelfClose token.
 * Used for yield_* tags which have no parent/child relationship to enforce.
 */
export interface SelfCloseHandler {
  selfClose(attrs: ReadonlyMap<string, string>, ctx: HandlerContext, tokenSpan: SourceSpan): ParserOp[]
}

// =============================================================================
// Bound handler interfaces (parent/top frame already captured)
// =============================================================================

/**
 * BoundOpenHandler — open handler with parent frame already captured in closure.
 *
 * Returned by resolveOpenHandler. The parser loop calls open(attrs, ctx) without
 * needing to know TParent — the type safety was enforced at bindOpen() call time.
 */
export interface BoundOpenHandler {
  open(attrs: ReadonlyMap<string, string>, ctx: HandlerContext, tokenSpan: SourceSpan): ParserOp[]
}

/**
 * BoundCloseHandler — close handler with top frame already captured in closure.
 *
 * Returned by resolveCloseHandler. The parser loop calls close(ctx) without
 * needing to know TFrame.
 */
export interface BoundCloseHandler {
  close(ctx: HandlerContext, tokenSpan: SourceSpan): ParserOp[]
}

/** BoundSelfCloseHandler is identical to SelfCloseHandler (no frame to capture). */
export type BoundSelfCloseHandler = SelfCloseHandler

// =============================================================================
// Binding utilities
// =============================================================================

/**
 * bindOpen — creates a BoundOpenHandler from a typed handler + narrowed parent frame.
 *
 * The binding is explicit and localized to resolveOpenHandler.
 * TypeScript verifies that `parent` satisfies `TParent` at the call site.
 *
 */
export function bindOpen<TParent extends Frame, TChild extends Frame>(
  handler: OpenHandler<TParent, TChild>,
  parent: TParent,
): BoundOpenHandler {
  return {
    open: (attrs, ctx, tokenSpan) => handler.open(attrs, parent, ctx, tokenSpan),
  }
}

/**
 * bindClose — creates a BoundCloseHandler from a typed handler + narrowed top frame.
 *
 * TypeScript verifies that `top` satisfies `TFrame` at the call site.
 */
export function bindClose<TFrame extends Frame>(
  handler: CloseHandler<TFrame>,
  top: TFrame,
): BoundCloseHandler {
  return {
    close: (ctx, tokenSpan) => handler.close(top, ctx, tokenSpan),
  }
}
