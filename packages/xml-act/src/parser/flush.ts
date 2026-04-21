/**
 * EOF handling — finalize all open frames top-to-bottom.
 */

import type { TurnEngineEvent } from '../types'
import type { Op } from '../machine'
import type { Frame, ThinkFrame, MessageFrame, InvokeFrame, ParameterFrame } from './types'
import { PROSE_VALID_TAGS } from './types'
import { stripTrailingWhitespace } from './content'
import type { StructuralParseError, ToolParseError } from '../types'
import type { InvokeContext } from './handlers/invoke'
import { finalizeParameter, finalizeInvoke } from './handlers/invoke'
import { generateToolInterface } from '@magnitudedev/tools'

interface FlushContext {
  peek: () => Frame | undefined
  apply: (ops: Op<Frame, TurnEngineEvent>[]) => void
  emit: (event: TurnEngineEvent) => void
  emitStructuralError: (error: StructuralParseError) => void
  emitToolError: (error: ToolParseError, context: { toolCallId: string; tagName: string; toolName: string; group: string; correctToolShape?: string }) => void
  invokeCtx: InvokeContext
  getCorrectToolShape: (toolTag: string) => string | undefined
}

export function flushAllFrames(ctx: FlushContext): void {
  let safety = 0
  while (safety++ < 100) {
    const top = ctx.peek()
    if (!top) break

    switch (top.type) {
      case 'parameter':
        finalizeParameter(top as ParameterFrame, ctx.invokeCtx)
        break

      case 'filter':
        // Lost filter at EOF — just pop silently
        ctx.apply([{ type: 'pop' }])
        break

      case 'invoke': {
        const invokeFrame = top as InvokeFrame
        if (!invokeFrame.dead) {
          ctx.emitToolError(
            {
              _tag: 'IncompleteTool',
              toolCallId: invokeFrame.toolCallId,
              tagName: invokeFrame.toolTag,
              detail: `Invoke for '${invokeFrame.toolTag}' was never closed`,
            },
            {
              toolCallId: invokeFrame.toolCallId,
              tagName: invokeFrame.toolTag,
              toolName: invokeFrame.toolName,
              group: invokeFrame.group,
              correctToolShape: ctx.getCorrectToolShape(invokeFrame.toolTag),
            },
          )
        }
        ctx.apply([{ type: 'pop' }])
        break
      }

      case 'message': {
        const msgFrame = top as MessageFrame
        ctx.apply([
          { type: 'emit', event: { _tag: 'MessageEnd', id: msgFrame.id } },
          { type: 'pop' },
        ])
        break
      }

      case 'think': {
        const thinkFrame = top as ThinkFrame
        const trimmed = stripTrailingWhitespace(thinkFrame.content)
        ctx.emitStructuralError({ _tag: 'UnclosedThink', message: `Unclosed think tag: ${thinkFrame.name}` })
        ctx.apply([
          { type: 'emit', event: { _tag: 'LensEnd', name: thinkFrame.name, content: trimmed } },
          { type: 'pop' },
        ])
        break
      }

      case 'prose': {
        const trimmed = stripTrailingWhitespace(top.body)
        if (trimmed.length > 0 || top.hasContent) {
          ctx.apply([
            { type: 'emit', event: { _tag: 'ProseEnd', content: trimmed } },
            { type: 'done' },
          ])
        } else {
          ctx.apply([{ type: 'done' }])
        }
        return
      }
    }
  }

  ctx.apply([{ type: 'done' }])
}
