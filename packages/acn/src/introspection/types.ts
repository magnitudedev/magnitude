import type { AgentIntrospection, SessionWorkStatus } from "@magnitudedev/agent"
import type { DirectoryPath, DisplayViewShape } from "@magnitudedev/acn-protocol"

export interface AcnDisplayViewIntrospection {
  readonly sessionId: string
  readonly viewId: string
  readonly shape: DisplayViewShape
  readonly subscriberCount: number
  readonly openedAt: number
  readonly lastActivityAt: number
}

export interface AcnIntrospectionSession {
  readonly sessionId: string
  readonly generation: number
  readonly title: string
  readonly cwd: DirectoryPath
  readonly scratchpadPath: string
  readonly createdAt: number
  readonly updatedAt: number
  readonly loadedAt: number
  readonly workStatus: SessionWorkStatus
  readonly phase: "open" | "retirement-claimed" | "retired"
  readonly scopedUseCount: number
  readonly scopedUseLabels: ReadonlyArray<string>
  readonly idleSince: number | null
  readonly revision: number
  readonly retirement: {
    readonly stage: "notifying-observers" | "closing-runtime" | "removing-generation"
    readonly startedAt: number
    readonly stageStartedAt: number
  } | null
}

export interface AcnIntrospectionOverview {
  readonly schemaVersion: 3
  readonly timestamp: number
  readonly sessions: readonly AcnIntrospectionSession[]
}

export interface AcnSessionIntrospection {
  readonly schemaVersion: 3
  readonly timestamp: number
  readonly session: AcnIntrospectionSession
  readonly displayViews: readonly AcnDisplayViewIntrospection[]
  readonly introspection: AgentIntrospection | null
}
