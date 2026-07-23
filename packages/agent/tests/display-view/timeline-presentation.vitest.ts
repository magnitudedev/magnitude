import { describe, expect, it } from 'vitest'
import { Option } from 'effect'
import type { DisplayMessage, DisplayTimelineWindowInfo, ToolMessage, ToolStepPresentation } from '@magnitudedev/protocol'
import { buildDisplayTimelinePresentation } from '../../src/display-view/timeline-presentation'

const windowFor = (messages: readonly DisplayMessage[]): DisplayTimelineWindowInfo => ({
  start: 0,
  end: messages.length,
  totalCount: messages.length,
  hasMoreBefore: false,
  hasMoreAfter: false,
})

const fileReadPresentation = (path: string): ToolStepPresentation => ({
  toolKey: 'fileRead',
  phase: 'completed',
  tone: 'neutral',
  icon: 'file',
  path,
  lineCount: 1,
  offset: 0,
  limit: 1,
  errorText: null,
  running: false,
  failed: false,
})

const fileRead = (id: string, path: string, timestamp: number): ToolMessage => ({
  id,
  type: 'tool',
  toolKey: 'fileRead',
  cluster: Option.some('read'),
  presentation: Option.some(fileReadPresentation(path)),
  filter: Option.none(),
  resultFilePath: Option.none(),
  timestamp,
})

const shellPresentation = (command: string): ToolStepPresentation => ({
  toolKey: 'shell',
  phase: 'completed',
  tone: 'neutral',
  icon: 'terminal',
  command,
  done: 'completed',
  exitCode: 0,
  pid: null,
  stdout: 'ok',
  stderr: '',
  partialStdout: '',
  partialStderr: '',
  stdoutPath: null,
  stderrPath: null,
  errorText: null,
  running: false,
  failed: false,
})

const shell = (id: string, command: string, timestamp: number): ToolMessage => ({
  id,
  type: 'tool',
  toolKey: 'shell',
  cluster: Option.none(),
  presentation: Option.some(shellPresentation(command)),
  filter: Option.none(),
  resultFilePath: Option.none(),
  timestamp,
})

const spawnWorkerPresentation = (agentId: string, message: string): ToolStepPresentation => ({
  toolKey: 'spawnWorker',
  phase: 'completed',
  tone: 'neutral',
  icon: 'worker',
  agentId,
  role: 'engineer',
  title: 'Build it',
  message,
  running: false,
  failed: false,
})

const spawnWorker = (id: string, agentId: string, message: string, timestamp: number): ToolMessage => ({
  id,
  type: 'tool',
  toolKey: 'spawnWorker',
  cluster: Option.none(),
  presentation: Option.some(spawnWorkerPresentation(agentId, message)),
  filter: Option.none(),
  resultFilePath: Option.none(),
  timestamp,
})

const agentCommunication = (id: string, content: string, timestamp: number): DisplayMessage => ({
  id,
  type: 'agent_communication',
  direction: 'to_agent',
  agentId: 'agent-1',
  streamId: Option.none(),
  agentName: Option.none(),
  agentRole: Option.none(),
  forkId: 'worker-a',
  content,
  preview: content,
  timestamp,
  status: Option.none(),
})

describe('display timeline presentation', () => {
  it('keeps a user bash command as a chronological message entry', () => {
    const messages: DisplayMessage[] = [{
      id: 'bash-1',
      type: 'user_bash_command',
      command: 'pwd',
      cwd: '/tmp',
      exitCode: 0,
      stdout: '/tmp\n',
      stderr: '',
      timestamp: 1,
    }]

    expect(buildDisplayTimelinePresentation({
      scope: 'root',
      mode: 'default',
      timelineMode: 'idle',
      streamingMessageId: null,
      messages,
      window: windowFor(messages),
    }).entries).toEqual([{
      kind: 'message',
      id: 'message:bash-1',
      messageId: 'bash-1',
      timestamp: 1,
      role: 'user',
      streaming: false,
      interrupted: false,
      nextMessageInterrupted: false,
    }])
  })

  it('summarizes only compact summary tools and ignores data-only worker messages between them', () => {
    const messages: DisplayMessage[] = [
      fileRead('tool-1', 'a.ts', 1),
      {
        id: 'worker-resumed-1',
        type: 'worker_resumed',
        workerRole: 'builder',
        workerId: 'agent-1',
        title: 'Build it',
        timestamp: 2,
      },
      fileRead('tool-2', 'b.ts', 3),
    ]

    const presentation = buildDisplayTimelinePresentation({
      scope: 'root',
      mode: 'default',
      timelineMode: 'idle',
      streamingMessageId: null,
      messages,
      window: windowFor(messages),
    })

    expect(presentation.entries).toHaveLength(1)
    expect(presentation.entries[0]).toMatchObject({
      kind: 'tool_summary',
      messageIds: ['tool-1', 'tool-2'],
      summary: { toolKey: 'fileRead', count: 2 },
    })
  })

  it('keeps shell commands as individual tool steps', () => {
    const messages: DisplayMessage[] = [
      shell('shell-1', 'echo one', 1),
      shell('shell-2', 'echo two', 2),
    ]

    const presentation = buildDisplayTimelinePresentation({
      scope: 'root',
      mode: 'default',
      timelineMode: 'idle',
      streamingMessageId: null,
      messages,
      window: windowFor(messages),
    })

    expect(presentation.entries).toMatchObject([
      { kind: 'tool_step', messageId: 'shell-1', step: { toolKey: 'shell', command: 'echo one' } },
      { kind: 'tool_step', messageId: 'shell-2', step: { toolKey: 'shell', command: 'echo two' } },
    ])
  })

  it('moves idle tail interruption into the status slot instead of scrollback', () => {
    const messages: DisplayMessage[] = [
      {
        id: 'assistant-1',
        type: 'assistant_message',
        content: 'Working on it',
        timestamp: 1,
      },
      {
        id: 'interrupt-1',
        type: 'interrupted',
        context: 'root',
        allKilled: Option.some(true),
        timestamp: 2,
      },
    ]

    const presentation = buildDisplayTimelinePresentation({
      scope: 'root',
      mode: 'default',
      timelineMode: 'idle',
      streamingMessageId: null,
      messages,
      window: windowFor(messages),
    })

    expect(presentation.statusSlot).toEqual({
      kind: 'interrupted',
      messageId: 'interrupt-1',
      context: 'root',
      allKilled: true,
    })
    expect(presentation.entries.map((entry) => entry.id)).toEqual(['message:assistant-1'])
  })

  it('keeps default mode free of thinking/status noise and transcript mode includes it', () => {
    const messages: DisplayMessage[] = [
      {
        id: 'thinking-1',
        type: 'thinking',
        content: 'internal notes',
        label: Option.none(),
        timestamp: 1,
      },
      {
        id: 'status-1',
        type: 'status_indicator',
        message: 'Reticulating splines',
        style: 'dim',
        timestamp: 2,
      },
    ]

    const base = {
      scope: 'root' as const,
      timelineMode: 'idle' as const,
      streamingMessageId: null,
      messages,
      window: windowFor(messages),
    }

    expect(buildDisplayTimelinePresentation({ ...base, mode: 'default' }).entries).toEqual([])
    expect(buildDisplayTimelinePresentation({ ...base, mode: 'transcript' }).entries.map((entry) => entry.id)).toEqual([
      'message:thinking-1',
      'message:status-1',
    ])
  })

  it('renders spawnWorker as a worker tool step in transcript mode and hides it in default mode', () => {
    const messages: DisplayMessage[] = [
      spawnWorker('spawn-1', 'agent-worker-a', 'starting work', 1),
    ]

    const base = {
      scope: 'root' as const,
      timelineMode: 'idle' as const,
      streamingMessageId: null,
      messages,
      window: windowFor(messages),
    }

    // Default mode hides spawnWorker entirely (visibility policy).
    expect(buildDisplayTimelinePresentation({ ...base, mode: 'default' }).entries).toEqual([])

    // Transcript mode emits a worker tool step carrying the message body.
    const transcript = buildDisplayTimelinePresentation({ ...base, mode: 'transcript' })
    expect(transcript.entries).toMatchObject([
      {
        kind: 'tool_step',
        messageId: 'spawn-1',
        step: {
          toolKey: 'spawnWorker',
          icon: 'worker',
          agentId: 'agent-worker-a',
          message: 'starting work',
        },
      },
    ])
  })

  it('fork scope keeps agent_communication visible (root scope hides it)', () => {
    const messages: DisplayMessage[] = [
      agentCommunication('comm-1', 'hello from worker', 1),
      fileRead('read-1', 'a.ts', 2),
    ]

    const base = {
      mode: 'default' as const,
      timelineMode: 'idle' as const,
      streamingMessageId: null,
      messages,
      window: windowFor(messages),
    }

    // Root scope hides agent_communication — only the file read summary remains.
    const rootPresentation = buildDisplayTimelinePresentation({ ...base, scope: 'root' })
    expect(rootPresentation.entries.map((entry) => entry.id)).toEqual(['tool-summary:read-1'])

    // Fork scope keeps agent_communication — it must appear as a message entry.
    const forkPresentation = buildDisplayTimelinePresentation({ ...base, scope: 'fork' })
    expect(forkPresentation.entries.map((entry) => entry.id)).toEqual([
      'message:comm-1',
      'tool-summary:read-1',
    ])
    expect(forkPresentation.entries[0]).toMatchObject({
      kind: 'message',
      role: 'agent',
      messageId: 'comm-1',
    })
  })
})
