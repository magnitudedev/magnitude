import type { RuntimeEvent, ToolResult } from '@magnitudedev/xml-act'
import type { StreamingPartial, ToolStateEvent } from '@magnitudedev/tools'

export function normalizeToolEvent<TInput, TOutput, TEmission>(
  raw: RuntimeEvent,
  streaming: StreamingPartial<TInput>,
): ToolStateEvent<TInput, TOutput, TEmission> | undefined {
  switch (raw._tag) {
    case 'ToolInputStarted':
      return { type: 'started' }
    case 'ToolInputFieldValue':
      return { type: 'inputUpdated', streaming, changed: 'field', name: String(raw.field) }
    case 'ToolInputReady':
      return { type: 'inputReady', input: raw.input as TInput, streaming }
    case 'ToolInputParseError': {
      const err = raw.error
      const detail = typeof err === 'object' && err !== null && 'detail' in err
        ? (err as { detail?: string }).detail
        : undefined
      const message = typeof detail === 'string'
        ? detail
        : typeof err === 'string'
          ? err
          : 'Tool input parse error'
      return { type: 'parseError', error: message }
    }
    case 'ToolExecutionStarted':
      return { type: 'executionStarted' }
    case 'ToolEmission':
      return { type: 'emission', value: raw.value as TEmission }
    case 'ToolExecutionEnded':
      return normalizeToolResult<TInput, TOutput, TEmission>(raw.result)
    case 'ToolObservation':
      return undefined
    // Structural events don't produce tool state events
    case 'ProseChunk':
    case 'ProseEnd':
    case 'LensStart':
    case 'LensChunk':
    case 'LensEnd':
    case 'MessageStart':
    case 'MessageChunk':
    case 'MessageEnd':
    case 'StructuralParseError':
    case 'TurnEnd':
      return undefined
  }
}

function normalizeToolResult<TInput, TOutput, TEmission>(
  result: ToolResult
): ToolStateEvent<TInput, TOutput, TEmission> {
  switch (result._tag) {
    case 'Success':
      return { type: 'completed', output: result.output as TOutput }
    case 'Error':
      return { type: 'error', error: new Error(result.error) }
    case 'Rejected':
      return { type: 'rejected' }
    case 'Interrupted':
      return { type: 'interrupted' }
  }
}
