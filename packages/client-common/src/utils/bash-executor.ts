/**
 * Bash executor interface — abstracts the RunBash RPC so client-common
 * doesn't depend on the vanilla client. Each app provides its own impl
 * from its agent client.
 */
import { createId } from '@magnitudedev/generate-id'
import type { RunBashResult } from '@magnitudedev/sdk'

export interface BashExecutor {
  runBash(payload: { sessionId: string; command: string }): Promise<RunBashResult>
}

export interface BashResult {
  id: string
  command: string
  stdout: string
  stderr: string
  exitCode: number
  cwd: string
  timestamp: number
}

export interface ExecuteBashCommandParams {
  executor: BashExecutor | null
  sessionId: string | null
  command: string
}

const MAX_OUTPUT_LENGTH = 50_000

const truncate = (text: string): string => {
  if (text.length <= MAX_OUTPUT_LENGTH) return text
  return text.slice(0, MAX_OUTPUT_LENGTH) + '\n... (output truncated)'
}

export async function executeBashCommand({
  executor,
  sessionId,
  command,
}: ExecuteBashCommandParams): Promise<BashResult> {
  if (!executor || !sessionId) {
    return {
      id: createId(),
      command,
      stdout: '',
      stderr: 'Bash execution unavailable: not connected to daemon',
      exitCode: 1,
      cwd: '',
      timestamp: Date.now(),
    }
  }

  try {
    const result = await executor.runBash({
      sessionId,
      command,
    })

    return {
      id: createId(),
      command,
      stdout: truncate(result.stdout.trimEnd()),
      stderr: truncate(result.stderr.trimEnd()),
      exitCode: result.exitCode,
      cwd: result.cwd,
      timestamp: Date.now(),
    }
  } catch (error) {
    return {
      id: createId(),
      command,
      stdout: '',
      stderr: error instanceof Error ? error.message : 'Unknown error executing command',
      exitCode: 1,
      cwd: '',
      timestamp: Date.now(),
    }
  }
}
