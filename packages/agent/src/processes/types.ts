import type { ChildProcess } from 'child_process'

export interface BackgroundProcessRecord {
  pid: number
  forkId: string | null
  turnId: string
  command: string
  reason: 'background' | 'timeout_exceeded'
  startedAt: number
  child: ChildProcess
  autoKillTimer?: ReturnType<typeof setTimeout>
  timeoutSeconds?: number
  status: 'running' | 'exited'
  exitCode: number | null
  signal: string | null
  stdoutBuffer: string
  stderrBuffer: string
  demoted: boolean
  stdoutFilePath: string | null
  stderrFilePath: string | null
}

export interface RegisterProcessInput {
  readonly pid: number
  readonly forkId: string | null
  readonly turnId: string
  readonly command: string
  readonly reason: 'background' | 'timeout_exceeded'
  readonly startedAt: number
  readonly child: ChildProcess
  readonly timeoutSeconds?: number
  readonly initialStdout: string
  readonly initialStderr: string
}