import { Effect } from 'effect'
import { createObservable } from '@magnitudedev/agent-definition'
import { ProjectionReaderTag } from './projection-reader'
import type { BackgroundProcessState } from '../projections/background-processes'

const truncateCommand = (command: string): string =>
  command.length > 80 ? `${command.slice(0, 77)}...` : command

const formatProcess = (process: BackgroundProcessState): string => {
  const attrs = [`pid="${process.pid}"`, `status="${process.status}"`, `reason="${process.reason}"`, `command="${truncateCommand(process.command)}"`]
  if (process.status === 'exited' && process.exitCode !== null) attrs.push(`exitCode="${process.exitCode}"`)
  if (process.status === 'killed' && process.signal) attrs.push(`signal="${process.signal}"`)

  const lines = [`<process ${attrs.join(' ')}>`]

  if (process.status !== 'running') {
    lines.push(`<final_stdout>`)
    lines.push(process.unreadStdout || '(no output)')
    lines.push(`</final_stdout>`)
    lines.push(`<final_stderr>`)
    lines.push(process.unreadStderr || '(no output)')
    lines.push(`</final_stderr>`)
    lines.push(`</process>`)
    return lines.join('\n')
  }

  if (!process.demoted) {
    const hasUnread = process.unreadStdout.length > 0 || process.unreadStderr.length > 0
    if (!hasUnread) {
      lines.push(`(no new output since last turn)`)
    } else {
      lines.push(`<new_stdout>`)
      lines.push(process.unreadStdout || '(no new output)')
      lines.push(`</new_stdout>`)
      lines.push(`<new_stderr>`)
      lines.push(process.unreadStderr || '(no new output)')
      lines.push(`</new_stderr>`)
    }
    lines.push(`</process>`)
    return lines.join('\n')
  }

  if (process.stdoutFilePath) {
    if (process.unreadStdout.length > 0) {
      lines.push(`<stdout file="${process.stdoutFilePath}" newLines="${process.newStdoutLines}" totalLines="${process.totalStdoutLines}">`)
      lines.push(process.unreadStdout)
      lines.push(`</stdout>`)
    } else {
      lines.push(`<stdout file="${process.stdoutFilePath}" totalLines="${process.totalStdoutLines}">(no new output since last turn)</stdout>`)
    }
  }

  if (process.stderrFilePath) {
    if (process.unreadStderr.length > 0) {
      lines.push(`<stderr file="${process.stderrFilePath}" newLines="${process.newStderrLines}" totalLines="${process.totalStderrLines}">`)
      lines.push(process.unreadStderr)
      lines.push(`</stderr>`)
    } else {
      lines.push(`<stderr file="${process.stderrFilePath}" totalLines="${process.totalStderrLines}">(no new output since last turn)</stderr>`)
    }
  }

  lines.push(`</process>`)
  return lines.join('\n')
}

export const backgroundProcessesObservable = createObservable({
  name: 'background-processes',
  observe: () => Effect.gen(function* () {
    const reader = yield* ProjectionReaderTag
    const processes = yield* reader.getBackgroundProcesses()
    const formatted = Array.from(processes.values()).map(formatProcess)
    if (formatted.length === 0) return []
    return [{ type: 'text' as const, text: `<background_processes>\n${formatted.join('\n')}\n</background_processes>` }]
  })
})