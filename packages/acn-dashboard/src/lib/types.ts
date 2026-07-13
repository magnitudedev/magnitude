import type { AcnRegistration } from '@magnitudedev/protocol'

export type {
  AcnDisplayViewIntrospection,
  AcnIntrospectionSession as AcnSession,
  AcnSessionIntrospection,
} from '@magnitudedev/acn'

export type {
  AddressedAtlasGroup,
  AddressedAtlasMetrics,
  AddressedAtlasNode,
  AddressedAtlasResident,
  AddressedAtlasSegment,
  AddressedPin,
  ProjectionIntrospection,
} from '@magnitudedev/agent'

export interface AcnInfo {
  readonly version: string
  readonly registration: AcnRegistration
  readonly registryPath: string
  readonly health: {
    readonly ok: boolean
    readonly service?: string
    readonly version?: string
    readonly pid?: number
    readonly schedulerElapsedMs?: number
    readonly error?: string
  }
  readonly introspection: {
    readonly ok: boolean
    readonly error?: string
  }
}

export interface KillAllAcnResult {
  readonly version: string
  readonly pid: number
  readonly status: 'killed' | 'stale' | 'skipped_self' | 'failed'
  readonly error?: string
}

export interface RpcTraceSummary {
  readonly traceId: string
  readonly serviceName: string
  readonly rootOperationName: string
  readonly startedAt: string
  readonly isRunning: boolean
  readonly durationMs: number
  readonly spanCount: number
  readonly errorCount: number
  readonly warnings: readonly string[]
  readonly rpcName: string
  readonly kind: 'command' | 'stream'
}
