// Types
export type {
  ModelSlot, CallUsage, CollectorData,
  TraceInput, TraceData, AgentTrace, AgentTraceMeta,
  TraceSessionMeta,
} from './types'

// Emission
export {
  onTrace, isTracing,
  extractCollectorData,
  emitTrace, wrapStreamForTrace,
} from './emit'
export type { TraceContext, TracedStream } from './emit'

// Writer
export { initTraceSession, writeTrace, updateTraceMeta, getTraceSessionId } from './trace-writer'
