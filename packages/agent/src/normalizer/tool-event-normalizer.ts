import { SchemaAccumulator } from '@magnitudedev/xml-act'
import type { ToolCallEvent, XmlToolResult } from '@magnitudedev/xml-act'
import type { ToolStateEvent } from '@magnitudedev/tools'
import { emptyStreamingInput } from '@magnitudedev/tools'
import type { StreamingInput } from '@magnitudedev/tools'

interface CallEntry {
  accumulator: SchemaAccumulator
}

export class ToolEventNormalizer {
  private readonly calls = new Map<string, CallEntry>()

  /**
   * Translate a raw tool event into a normalized ToolStateEvent.
   * Returns undefined for events that don't produce a state event (e.g. ToolObservation).
   */
  normalize(toolKey: string, callId: string, rawEvent: unknown): ToolStateEvent<any, any, any, any> | undefined {
    void toolKey
    if (!isTaggedEvent(rawEvent)) return undefined

    switch (rawEvent._tag) {
      case 'ToolInputStarted': {
        const acc = new SchemaAccumulator()
        this.calls.set(callId, { accumulator: acc })
        acc.ingest(asToolCallEvent(rawEvent))
        return { type: 'started' }
      }

      case 'ToolInputFieldValue':
      case 'ToolInputBodyChunk':
      case 'ToolInputChildStarted':
      case 'ToolInputChildComplete': {
        const call = this.calls.get(callId)
        if (!call) return undefined
        call.accumulator.ingest(asToolCallEvent(rawEvent))
        return {
          type: 'inputUpdated',
          streaming: call.accumulator.current,
          changed: rawEvent._tag === 'ToolInputFieldValue'
            ? 'field'
            : rawEvent._tag === 'ToolInputBodyChunk'
              ? 'body'
              : 'child',
          name: rawEvent._tag === 'ToolInputFieldValue' && 'field' in rawEvent
            ? String(rawEvent.field)
            : undefined,
        }
      }

      case 'ToolInputReady': {
        const call = this.calls.get(callId)
        if (call) call.accumulator.ingest(asToolCallEvent(rawEvent))
        const input = 'input' in rawEvent ? rawEvent.input : undefined
        return {
          type: 'inputReady',
          input,
          streaming: call?.accumulator.current ?? emptyStreamingInput(),
        }
      }

      case 'ToolInputParseError': {
        const err = 'error' in rawEvent ? rawEvent.error : undefined
        const detail = isRecord(err) && 'detail' in err ? err.detail : undefined
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
        return { type: 'emission', value: ('value' in rawEvent ? rawEvent.value : undefined) }

      case 'ToolExecutionEnded': {
        const result = 'result' in rawEvent ? rawEvent.result : undefined
        if (!isXmlToolResult(result)) return undefined
        return mapXmlToolResult(result)
      }

      case 'ToolObservation':
        return undefined

      default:
        return undefined
    }
  }

  /** Get current streaming input for a call */
  getStreaming(callId: string): StreamingInput<any, any> | undefined {
    return this.calls.get(callId)?.accumulator.current
  }

  /** Get parsed input for a call (from accumulator) */
  getInput(callId: string): unknown | undefined {
    const call = this.calls.get(callId)
    if (!call) return undefined
    // The accumulator tracks fields; return them as the input
    return call.accumulator.current.fields
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isTaggedEvent(value: unknown): value is { _tag: string } {
  return isRecord(value) && typeof value._tag === 'string'
}

function isToolCallEvent(value: unknown): value is ToolCallEvent {
  if (!isRecord(value) || typeof value._tag !== 'string') return false
  return value._tag.startsWith('Tool')
}

function asToolCallEvent(value: unknown): ToolCallEvent {
  if (!isToolCallEvent(value)) throw new Error('Expected ToolCallEvent')
  return value
}

function isXmlToolResult(value: unknown): value is XmlToolResult {
  return isRecord(value) && typeof value._tag === 'string'
}

function mapXmlToolResult(result: XmlToolResult): ToolStateEvent<any, any, any, any> {
  switch (result._tag) {
    case 'Success':
      return { type: 'completed', output: result.output }
    case 'Error':
      return { type: 'error', error: new Error(result.error) }
    case 'Rejected':
      return { type: 'rejected' }
    case 'Interrupted':
      return { type: 'interrupted' }
  }
}
