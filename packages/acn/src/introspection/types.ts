import type { AgentIntrospection } from "@magnitudedev/agent"
import type { SessionRuntimeSnapshot } from "../agent-runtime"
import type { AcnDisplayViewIntrospection } from "./display-views"

export type AcnIntrospectionSession = SessionRuntimeSnapshot

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
