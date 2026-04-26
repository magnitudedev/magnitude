/**
 * EOF handling — finalize all open frames top-to-bottom.
 *
 * Uses switch (top.type) narrowing — TypeScript narrows top in every case.
 * All effects via ParserOp[] — no direct emit/apply calls inside switch branches.
 * No as-casts needed: each case receives a narrowed frame type.
 */

import type { ParserOp } from './ops'
import type { Frame } from './types'
import type { InvokeContext } from './handler-context'
import { emitEvent, emitStructuralError, emitToolError } from './ops'
import { stripTrailingWhitespace } from './content'
import { finalizeParameterOps, finalizeInvokeOps } from './handlers/invoke'
import { closeThinkAtEof } from './handlers/think'
import { generateToolInterface } from '@magnitudedev/tools'

function getCorrectToolShape(toolTag: string, invokeCtx: InvokeContext): string | undefined {
  const registered = invokeCtx.tools.get(toolTag)
  if (!registered) return undefined
  try {
    const result = generateToolInterface(registered.tool, registered.groupName ?? 'tools', undefined, { extractCommon: false, showErrors: false })
    return result.signature
  } catch {
    return undefined
  }
}

export function flushAllFrames(
  peek: () => Frame | undefined,
  apply: (ops: ParserOp[]) => void,
  invokeCtx: InvokeContext,
): void {
  let safety = 0
  while (safety++ < 100) {
    const top = peek()
    if (!top) break

    switch (top.type) {
      case 'parameter':
        // top is ParameterFrame — TypeScript narrows
        apply(finalizeParameterOps(top, invokeCtx))
        break

      case 'filter':
        // top is FilterFrame — TypeScript narrows. Lost filter at EOF — pop silently.
        apply([{ type: 'pop' }])
        break

      case 'invoke':
        // top is InvokeFrame — TypeScript narrows
        if (!top.dead) {
          apply([emitToolError(
            {
              _tag: 'IncompleteTool',
              toolCallId: top.toolCallId,
              tagName: top.toolTag,
              detail: `Invoke for '${top.toolTag}' was never closed`,
              primarySpan: top.openSpan,
            },
            {
              toolCallId: top.toolCallId,
              tagName: top.toolTag,
              toolName: top.toolName,
              group: top.group,
              correctToolShape: getCorrectToolShape(top.toolTag, invokeCtx),
            },
          )])
        }
        apply([{ type: 'pop' }])
        break

      case 'message':
        // top is MessageFrame — TypeScript narrows
        apply([
          emitEvent({ _tag: 'MessageEnd', id: top.id }),
          { type: 'pop' },
        ])
        break

      case 'think':
        // top is ThinkFrame — TypeScript narrows
        apply(closeThinkAtEof(top))
        break

      case 'prose': {
        // top is ProseFrame — TypeScript narrows
        const trimmed = stripTrailingWhitespace(top.body)
        if (trimmed.length > 0 || top.hasContent) {
          apply([
            emitEvent({ _tag: 'ProseEnd', content: trimmed }),
            { type: 'done' },
          ])
        } else {
          apply([{ type: 'done' }])
        }
        return
      }
    }
  }

  apply([{ type: 'done' }])
}
