import type { AgentIntrospection } from "@magnitudedev/agent"
import type { AcnActivityState } from "../activity-tracker"
import type { SessionRetirementSnapshot } from "../agent-runtime"
import type { ResourceUseGateSnapshot } from "../resource-use-gate"
import type { AcnDisplayViewIntrospection } from "./display-views"

export interface AcnIntrospectionSession {
  readonly sessionId: string
  readonly title: string
  readonly cwd: string
  readonly scratchpadPath: string
  readonly createdAt: number
  readonly updatedAt: number
  readonly generation: number
  readonly residentSince: number
  readonly gate: ResourceUseGateSnapshot
  readonly retirement: SessionRetirementSnapshot | null
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
