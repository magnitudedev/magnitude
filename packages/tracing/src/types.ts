import type { ModelCallTrace, AssembledToolCall, TokenLogprob, RawInputToken, RawOutputToken, RawLogprobEntry } from "@magnitudedev/ai"

export type { ModelCallTrace, AssembledToolCall, TokenLogprob, RawInputToken, RawOutputToken, RawLogprobEntry } from "@magnitudedev/ai"

export interface AgentCallTrace extends ModelCallTrace {
  readonly traceId: string
  readonly sessionId: string
  readonly actor: AgentTraceActor
  readonly callType: AgentCallType
  readonly scope: AgentTraceScope
}

export interface AgentTraceActor {
  readonly agentId: string
  readonly forkId: string | null
  readonly roleId: string | null
}

export type AgentCallType =
  | 'chat'
  | 'compact'
  | 'advisor'
  | 'observer'
  | 'image'
  | 'title'
  | 'autopilot'
  | 'extract-memory-diff'

export type AgentTraceOperationKind =
  | 'compact'
  | 'observer'
  | 'advisor'
  | 'autopilot'
  | 'title'
  | 'image'
  | 'background'

export type AgentTraceScope =
  | {
      readonly kind: 'turn'
      readonly turnId: string
      readonly chainId: string
    }
  | {
      readonly kind: 'operation'
      readonly operationId: string
      readonly operationKind: AgentTraceOperationKind
      readonly chainId?: string
      readonly relatedTurnId?: string
      readonly relatedMessageId?: string
      readonly parentOperationId?: string
      readonly forkId?: string | null
    }

export interface TraceSessionMeta {
  readonly sessionId: string
  readonly created: string
  readonly cwd: string | null
  readonly platform: string | null
  readonly gitBranch: string | null
  readonly chatName: string | null
}
