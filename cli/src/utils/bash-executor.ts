import { createId } from '@magnitudedev/generate-id'

export interface BashResult {
  id: string
  command: string
  stdout: string
  stderr: string
  exitCode: number
  cwd: string
  timestamp: number
}

const MAX_OUTPUT_LENGTH = 50_000

const textDecoder = new TextDecoder()

const readOutput = (output: unknown): string => {
  if (!output) return ''
  if (typeof output === 'string') return output
  if (output instanceof Uint8Array) return textDecoder.decode(output)
  return ''
}

const truncate = (text: string): string => {
  if (text.length <= MAX_OUTPUT_LENGTH) return text
  return text.slice(0, MAX_OUTPUT_LENGTH) + '\n... (output truncated)'
}

export function executeBashCommand(
  command: string,
  options?: { workspacePath?: string; projectRoot?: string },
): BashResult {
  const cwd = process.cwd()
  const shell = process.env.SHELL || '/bin/sh'
  const projectRoot = options?.projectRoot ?? cwd

  try {
    const result = Bun.spawnSync({
      cmd: [shell, '-c', command],
      cwd,
      stdout: 'pipe',
      stderr: 'pipe',
      env: {
        ...process.env,
        PROJECT_ROOT: projectRoot,
        ...(options?.workspacePath ? { M: options.workspacePath } : {}),
      },
    })

    const stdout = truncate(readOutput(result.stdout).trimEnd())
    const stderr = truncate(readOutput(result.stderr).trimEnd())

    return {
      id: createId(),
      command,
      stdout,
      stderr,
      exitCode: result.exitCode,
      cwd,
      timestamp: Date.now(),
    }
  } catch (error) {
    return {
      id: createId(),
      command,
      stdout: '',
      stderr: error instanceof Error ? error.message : 'Unknown error executing command',
      exitCode: 1,
      cwd,
      timestamp: Date.now(),
    }
  }
}
