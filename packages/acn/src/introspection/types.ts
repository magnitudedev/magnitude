import type { AgentIntrospection } from "@magnitudedev/agent"
import type { AcnActivityState } from "../activity-tracker"
import type { AcnDisplayViewIntrospection } from "./display-views"

export interface AcnIntrospectionSession {
  readonly sessionId: string
  readonly title: string
  readonly cwd: string
  readonly scratchpadPath: string
  readonly createdAt: number
  readonly updatedAt: number
}

export interface AcnIntrospectionOverview {
  readonly schemaVersion: 1
  readonly timestamp: number
  readonly sessions: readonly AcnIntrospectionSession[]
  readonly activity: AcnActivityState
}

export interface AcnSessionIntrospection {
  readonly schemaVersion: 1
  readonly timestamp: number
  readonly session: AcnIntrospectionSession
  readonly activity: AcnActivityState
  readonly displayViews: readonly AcnDisplayViewIntrospection[]
  readonly introspection: AgentIntrospection | null
}
