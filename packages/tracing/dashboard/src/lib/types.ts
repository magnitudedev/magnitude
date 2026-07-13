export type { AgentCallTrace, TokenLogprob, RawInputToken, RawOutputToken, RawLogprobEntry } from '@magnitudedev/tracing'

export interface SessionInfo {
  id: string
  timestamp: string
  traceCount?: number
  meta?: Record<string, any>
}

export interface SessionPage {
  items: SessionInfo[]
  nextCursor: string | null
}

export interface ForkNode {
  forkId: string | null
  name: string
  mode: 'clone' | 'spawn' | 'root'
  parentForkId: string | null
  children: ForkNode[]
  traceCount: number
}
