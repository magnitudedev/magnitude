/**
 * Token dispatch — routes tokens to structural handlers or content paths.
 */

import type { Op } from '../machine'
import type {
  Token,
  TurnEngineEvent,
  StructuralParseError,
  ToolParseError,
  DeepPaths,
} from '../types'
import type {
  Frame,
  InvokeFrame,
  ParameterFrame,
  FilterFrame,
} from './types'
import { tokenRaw, resolveToken } from './resolve'
import { KNOWN_STRUCTURAL_TAGS } from '../constants'
import { appendProse, appendMessage, appendThink, isAllWhitespace } from './content'
import { openThink, closeThink } from './handlers/think'
import { openMessage, closeMessage } from './handlers/message'
import { handleYield } from './handlers/yield'
import {
  openInvoke,
  openFilter,
  closeFilter,
  openParameter,
  finalizeParameter,
  finalizeInvoke,
  type InvokeContext,
} from './handlers/invoke'

// =============================================================================
// DispatchContext — all shared state needed for token dispatch
// =============================================================================

export interface DispatchContext {
  machine: {
    mode: string
    peek(): Frame | undefined
    apply(ops: Op<Frame, TurnEngineEvent>[]): void
    readonly stack: readonly Frame[]
  }
  emit: (event: TurnEngineEvent) => void
  emitStructuralError: (error: StructuralParseError) => void
  emitToolError: (error: ToolParseError, context: { toolCallId: string; tagName: string; toolName: string; group: string; correctToolShape?: string }) => void
  invokeCtx: InvokeContext
  endCurrentProse: () => void
  generateId: () => string
  deferredYield: { target: 'user' | 'invoke' | 'worker' | 'parent' | null; postYieldHasContent: boolean }
}

// =============================================================================
// appendContentToFrame — route content to the correct per-frame handler
// =============================================================================

export function appendContentToFrame(top: Frame, text: string, ctx: DispatchContext): void {
  switch (top.type) {
    case 'prose':
      ctx.machine.apply(appendProse(top, text) as Op<Frame, TurnEngineEvent>[])
      break
    case 'think':
      ctx.machine.apply(appendThink(top, text) as Op<Frame, TurnEngineEvent>[])
      break
    case 'message':
      ctx.machine.apply(appendMessage(top, text) as Op<Frame, TurnEngineEvent>[])
      break
    case 'parameter': {
      const paramTop = top as ParameterFrame
      if (!paramTop.dead) {
        paramTop.rawValue += text
        if (paramTop.jsonishParser !== null) paramTop.jsonishParser.push(text)
        const jsonPath = paramTop.jsonishParser !== null ? paramTop.jsonishParser.currentPath : []
        const path = [paramTop.paramName, ...jsonPath] as unknown as DeepPaths<unknown>
        ctx.machine.apply([{
          type: 'emit',
          event: {
            _tag: 'ToolInputFieldChunk',
            toolCallId: paramTop.toolCallId,
            field: paramTop.paramName as string & keyof unknown,
            path,
            delta: text,
          },
        }])
      }
      break
    }
    case 'filter': {
      const filterTop = top as FilterFrame
      filterTop.query += text
      break
    }
    case 'invoke':
      if (!isAllWhitespace(text)) {
        ctx.emitStructuralError(
          { _tag: 'UnexpectedContent', context: 'invoke:' + top.toolTag, detail: `Unexpected content between parameters: "${text.slice(0, 40)}"` },
        )
      }
      break
  }
}

// =============================================================================
// handleOpen — dispatch Open tokens to tag-specific handlers
// =============================================================================

export function handleOpen(name: string, variant: string | undefined, ctx: DispatchContext): void {
  switch (name) {
    case 'think':
      openThink(variant, ctx.endCurrentProse, (ops) => ctx.machine.apply(ops))
      break
    case 'message':
      openMessage(variant, ctx.generateId, ctx.endCurrentProse, (ops) => ctx.machine.apply(ops))
      break
    case 'invoke':
      openInvoke(variant, ctx.invokeCtx)
      break
  }
}

// =============================================================================
// handleClose — dispatch Close tokens to tag-specific handlers
// =============================================================================

export function handleClose(name: string, pipe: string | undefined, top: Frame, ctx: DispatchContext): void {
  if (pipe) {
    openFilter(top as InvokeFrame, pipe, ctx.invokeCtx)
    return
  }

  // Use top.type (not name) to route — close tag name may differ from frame type
  // due to close-tag mismatch lenience (e.g. <message|> closing a think frame)
  switch (top.type) {
    case 'think':
      closeThink(top as import('./types').ThinkFrame, ctx.emitStructuralError, (ops) => ctx.machine.apply(ops), false)
      break
    case 'message':
      closeMessage(top as import('./types').MessageFrame, (ops) => ctx.machine.apply(ops))
      break
    case 'invoke':
      finalizeInvoke(top as InvokeFrame, ctx.invokeCtx)
      break
    case 'parameter':
      finalizeParameter(top as ParameterFrame, ctx.invokeCtx)
      break
    case 'filter':
      closeFilter(top as FilterFrame, ctx.invokeCtx)
      break
  }
}

// =============================================================================
// pushToken — main entry point for token dispatch
// =============================================================================

export function pushToken(token: Token, ctx: DispatchContext): void {
  if (ctx.machine.mode === 'observing') {
    if (token._tag === 'Content' && !isAllWhitespace(token.text)) {
      ctx.deferredYield.postYieldHasContent = true
    } else if (token._tag !== 'Content') {
      ctx.deferredYield.postYieldHasContent = true
    }
    return
  }
  if (ctx.machine.mode !== 'active') return

  const top = ctx.machine.peek()
  if (!top) return

  const resolution = resolveToken(token, top)

  if (resolution === 'content') {
    if (token._tag === 'Close' && !token.pipe && KNOWN_STRUCTURAL_TAGS.has(token.name)) {
      ctx.emitStructuralError({
        _tag: 'StrayCloseTag',
        tagName: token.name,
        detail: `Unexpected close '<${token.name}|>' with no matching open in current context`,
      })
    }
    appendContentToFrame(top, tokenRaw(token), ctx)
    return
  }

  switch (token._tag) {
    case 'Open':
      handleOpen(token.name, token.variant, ctx)
      break
    case 'Close':
      handleClose(token.name, token.pipe, top, ctx)
      break
    case 'SelfClose':
      if (token.name === 'yield') {
        handleYield(token.variant, ctx.endCurrentProse, (ops) => ctx.machine.apply(ops), (target) => {
          ctx.deferredYield.target = target
          ctx.deferredYield.postYieldHasContent = false
        })
      }
      break
    case 'Parameter':
      openParameter(token.name, top as InvokeFrame, ctx.invokeCtx)
      break
    case 'ParameterClose':
      finalizeParameter(top as ParameterFrame, ctx.invokeCtx)
      break
    case 'Content':
      appendContentToFrame(top, token.text, ctx)
      break
  }
}
