import { describe, expect, test } from 'vitest'
import { TURN_CONTROL_IDLE, END_TURN_TAG, TURN_CONTROL_IDLE_TAG } from '@magnitudedev/xml-act'
import type { ContentPart } from '../../content'
import type { ResultEntry, TimelineEntry } from '../types'
import { formatInbox } from '../render'
import { formatInterrupted, formatNoop } from '../render-results'
import {
  WORKER_PROGRESS_USER_MESSAGE_REMINDER,
} from '../../prompts/lead-communication-reminders'

const lifecycleReminderFormatters = {
  builder: {
    spawn: (ids: readonly string[]) => `Builder spawned: ${ids.join(', ')}`,
    idle: (ids: readonly string[]) => `Builder idle: ${ids.join(', ')}`,
  },
} as const

const TS0 = 1711641600000 // 2024-03-28 16:00:00 UTC
const TS1 = TS0 + 30_000
const TS2 = TS0 + 60_000
const TS3 = TS0 + 120_000

describe('formatInbox', () => {
  test('returns empty array for empty input', () => {
    expect(formatInbox({ results: [], timeline: [], timezone: 'UTC', lifecycleReminderFormatters })).toEqual([])
  })

  test('renders results-only entries (turn_results, interrupted, error, noop)', () => {
    const results: readonly ResultEntry[] = [
      { kind: 'turn_results', items: [] },
      { kind: 'interrupted' },
      { kind: 'error', message: 'boom' },
      { kind: 'noop' },
    ]

    const out = formatInbox({ results, timeline: [], timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      { type: 'text', text: '<turn_result>' + formatInterrupted() + '<error>boom</error>' + formatNoop() + '\n</turn_result>\n' },
    ])
  })

  test('timeline-only single user message includes marker and user reply reminder', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'user_message', timestamp: TS0, text: 'hello', attachments: [] },
    ]
    expect(formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })).toEqual([
      {
        type: 'text',
        text:
          `--- 2024-03-28 16:00 ---\n<message from="user">hello</message>`,
      },
    ])
  })

  test('timeline-only single user message with attachments renders attachments', () => {
    const timeline: readonly TimelineEntry[] = [
      {
        kind: 'user_message',
        timestamp: TS0,
        text: 'hello',
        attachments: [
          {
            kind: 'mention',
            path: 'src/a.ts',
            contentType: 'text',
            content: 'export const a = 1',
            truncated: true,
            originalBytes: 42,
          },
          { kind: 'image', image: { type: 'image', base64: 'abc', mediaType: 'image/png', width: 1, height: 1 } },
        ],
      },
    ]

    expect(formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })).toEqual([
      {
        type: 'text',
        text: '--- 2024-03-28 16:00 ---\n<message from="user">hello</message>\n<mention path="src/a.ts" type="text" truncated="true" original_bytes="42">export const a = 1</mention>',
      },
      { type: 'image', base64: 'abc', mediaType: 'image/png', width: 1, height: 1 },
    ])
  })

  test('timeline with multiple entries adds time markers', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'user_message', timestamp: TS0, text: 'a', attachments: [] },
      { kind: 'user_message', timestamp: TS2, text: 'b', attachments: [] },
    ]
    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text:
          `--- 2024-03-28 16:00 ---\n<message from="user">a</message>\n\n--- 16:01 ---\n<message from="user">b</message>`,
      },
    ])
  })

  test('preserves timeline input order', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'user_message', timestamp: TS2, text: 'second', attachments: [] },
      { kind: 'user_message', timestamp: TS0, text: 'first', attachments: [] },
    ]
    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out[0]).toEqual({
      type: 'text',
      text:
        `--- 2024-03-28 16:01 ---\n<message from="user">second</message>\n\n--- 16:00 ---\n<message from="user">first</message>`,
    })
  })

  test('renders user message attachments (mentions and images)', () => {
    const timeline: readonly TimelineEntry[] = [
      {
        kind: 'user_message',
        timestamp: TS0,
        text: 'see this',
        attachments: [
          {
            kind: 'mention',
            path: 'b.ts',
            contentType: 'text',
            content: 'const x = 1', truncated: true, originalBytes: 123,
          },
          { kind: 'mention', path: 'c.ts', contentType: 'text', error: 'not found' },
          { kind: 'image', image: { type: 'image', base64: 'abc', mediaType: 'image/png', width: 1, height: 1 } },
        ],
      },
      { kind: 'lifecycle_hook', timestamp: TS1, agentId: 'builder-z', role: 'builder', hookType: 'spawn' },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text:
          '--- 2024-03-28 16:00 ---\n<message from="user">see this</message>\n<mention path="b.ts" type="text" truncated="true" original_bytes="123">const x = 1</mention>\n<mention path="c.ts" type="text" error="not found"/>',
      },
      { type: 'image', base64: 'abc', mediaType: 'image/png', width: 1, height: 1 },
      {
        type: 'text',
        text:
          `\n\n<reminders>\n- Builder spawned: builder-z\n</reminders>`,
      },
    ])
  })

  test('formats task worker spawn reminder with role, task id, and title', () => {
    const timeline: readonly TimelineEntry[] = [
      {
        kind: 'lifecycle_hook',
        timestamp: TS1,
        agentId: 'agent-debug-1',
        role: 'debugger',
        hookType: 'spawn',
        taskId: 'diag-1',
        taskTitle: 'Investigate the crash',
      },
    ]
    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text: '<reminders>\n- Worker `debugger` spawned and working on task diag-1 ("Investigate the crash").\n</reminders>',
      },
    ])
  })

  test('equal timestamp entries preserve input order', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'user_message', timestamp: TS0, text: 'first-input', attachments: [] },
      { kind: 'user_message', timestamp: TS0, text: 'second-input', attachments: [] },
    ]
    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out[0]).toEqual({
      type: 'text',
      text:
        `--- 2024-03-28 16:00 ---\n<message from="user">first-input</message>\n<message from="user">second-input</message>`,
    })
  })

  test('adds attention bullets for user messages and idle agents only when not last', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'user_message', timestamp: TS0, text: 'hi', attachments: [] },
      {
        kind: 'agent_block',
        timestamp: TS1,
        firstAtomTimestamp: TS1,
        lastAtomTimestamp: TS1,
        agentId: 'builder-a',
        role: 'builder',
        atoms: [{ kind: 'idle', timestamp: TS1 }],
      },
      { kind: 'lifecycle_hook', timestamp: TS2, agentId: 'builder-a', role: 'builder', hookType: 'idle' },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out[0]).toEqual({
      type: 'text',
      text:
        `--- 2024-03-28 16:00 ---\n<message from="user">hi</message>\n<agent id="builder-a" role="builder" status="idle">\n${TURN_CONTROL_IDLE}\n</agent>\n\n<reminders>\n- Builder idle: builder-a\n</reminders>\n\n<attention>\n- user message at 16:00\n- builder-a went idle at 16:00\n</attention>`,
    })
  })

  test('passes through observation image ContentParts', () => {
    const img: ContentPart = { type: 'image', base64: 'abc', mediaType: 'image/png', width: 1, height: 1 }
    const timeline: readonly TimelineEntry[] = [
      {
        kind: 'observation',
        timestamp: TS0,
        parts: [{ type: 'text', text: 'seen' }, img],
      },
      { kind: 'lifecycle_hook', timestamp: TS2, agentId: 'builder-a', role: 'builder', hookType: 'spawn' },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text: '--- 2024-03-28 16:00 ---\nseen',
      },
      img,
      {
        type: 'text',
        text: '\n\n<reminders>\n- Builder spawned: builder-a\n</reminders>',
      },
    ])
  })

  test('renders mixed results and timeline', () => {
    const out = formatInbox({
      results: [{ kind: 'error', message: 'failed' }],
      timeline: [{ kind: 'lifecycle_hook', timestamp: TS0, agentId: 'builder-a', role: 'builder', hookType: 'idle' }],
      timezone: 'UTC',
      lifecycleReminderFormatters,
    })

    expect(out).toEqual([
      {
        type: 'text',
        text: '<turn_result><error>failed</error>\n</turn_result>\n\n\n<reminders>\n- Builder idle: builder-a\n</reminders>',
      },
    ])
  })

  test('renders task updates block with expected lines', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'task_update', timestamp: TS0, action: 'created', taskId: 't1', title: 'Title', taskType: 'implement' },
      { kind: 'task_update', timestamp: TS1, action: 'status_changed', taskId: 't1', previousStatus: 'pending', nextStatus: 'working' },
      { kind: 'task_update', timestamp: TS2, action: 'completed', taskId: 't1' },
      { kind: 'task_update', timestamp: TS3 + 1, action: 'cancelled', taskId: 't2', cancelledCount: 3 },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text:
          '<task_updates>\n- Task t1 created: "Title" (implement)\n- Task t1 status changed: pending -> working\n- Task t1 completed\n- Task t2 cancelled (3 tasks removed)\n</task_updates>',
      },
    ])
  })

  test('renders task updates adjacent to task tree', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'task_update', timestamp: TS0, action: 'cancelled', taskId: 't1', cancelledCount: 2 },
      { kind: 'task_tree_view', timestamp: TS1, renderedTree: '- [ ] t3 next' },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text:
          '<task_updates>\n- Task t1 cancelled (2 tasks removed)\n</task_updates>\n\n<task_tree>\n- [ ] t3 next\n</task_tree>',
      },
    ])
  })

  test('does not include task_update entries in chronological stream', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'task_update', timestamp: TS0, action: 'created', taskId: 't1', title: 'Title', taskType: 'implement' },
      { kind: 'user_message', timestamp: TS1, text: 'hello', attachments: [] },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text:
          `--- 2024-03-28 16:00 ---\n<message from="user">hello</message>\n\n<task_updates>\n- Task t1 created: "Title" (implement)\n</task_updates>`,
      },
    ])
  })

  test('renders agent_block atoms (thought, tool_call, message, idle, error)', () => {
    const timeline: readonly TimelineEntry[] = [
      {
        kind: 'agent_block',
        timestamp: TS0,
        firstAtomTimestamp: TS0,
        lastAtomTimestamp: TS3,
        agentId: 'builder-x',
        role: 'builder',
        atoms: [
          { kind: 'thought', timestamp: TS0, text: 'thinking' },
          {
            kind: 'tool_call',
            timestamp: TS1,
            toolCallId: 'tc1',
            tagName: 'read',
            attributes: { path: 'src/a.ts' },
            status: 'success',
          },
          { kind: 'message', timestamp: TS2, direction: 'to_lead', text: 'done?' },
          { kind: 'error', timestamp: TS2, message: 'oops' },
          { kind: 'idle', timestamp: TS3, reason: 'error' },
        ],
      },
      { kind: 'lifecycle_hook', timestamp: TS3 + 1, agentId: 'builder-x', role: 'builder', hookType: 'idle' },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out[0]).toEqual({
      type: 'text',
      text:
        `--- 2024-03-28 16:00 ---\n<agent id="builder-x" role="builder" status="idle">\nthinking\n<read path="src/a.ts"/>\n<message to="lead">done?</message>\n<error>oops</error>\n<${END_TURN_TAG}>\n<${TURN_CONTROL_IDLE_TAG} reason="error"/>\n</${END_TURN_TAG}>\n</agent>\n\n<reminders>\n- ${WORKER_PROGRESS_USER_MESSAGE_REMINDER}\n- Builder idle: builder-x\n</reminders>\n\n<attention>\n- builder-x errored at 16:00\n</attention>`,
    })
  })

  test('renders user_bash_command timeline entry', () => {
    const timeline: readonly TimelineEntry[] = [
      {
        kind: 'user_bash_command',
        timestamp: TS0,
        command: 'ls -la',
        cwd: '/tmp',
        exitCode: 0,
        stdout: 'file-a',
        stderr: '',
      },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text:
          '--- 2024-03-28 16:00 ---\n<user_bash_command cwd="/tmp" exit_code="0">\n<command>ls -la</command>\n<stdout>file-a</stdout>\n<stderr></stderr>\n</user_bash_command>',
      },
    ])
  })

  test('renders all non-observation timeline kinds', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'user_message', timestamp: TS0, text: 'u', attachments: [] },
      { kind: 'user_to_agent', timestamp: TS1, agentId: 'a1', text: 'direct' },
      {
        kind: 'subagent_user_killed',
        timestamp: TS1,
        agentId: 'a2',
        agentType: 'builder',
      },
      { kind: 'user_presence', timestamp: TS1, text: 'back', confirmed: true },

      { kind: 'lifecycle_hook', timestamp: TS2, agentId: 'builder-z', role: 'builder', hookType: 'spawn' },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    const text = out[0]
    expect(text).toEqual({
      type: 'text',
      text:
        `--- 2024-03-28 16:00 ---\n<message from="user">u</message>\n<user-to-agent agent="a1">direct</user-to-agent>\n<subagent-user-killed agent="a2" type="builder"/>\n<user-presence confirmed="true">back</user-presence>\n\n<reminders>\n- Builder spawned: builder-z\n</reminders>`,
    })
  })

  test('does not render user reply reminder when no user_message is present', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'parent_message', timestamp: TS0, text: 'from parent' },
      { kind: 'lifecycle_hook', timestamp: TS1, agentId: 'builder-z', role: 'builder', hookType: 'spawn' },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text:
          '--- 2024-03-28 16:00 ---\n<message from="parent">from parent</message>\n\n<reminders>\n- Builder spawned: builder-z\n</reminders>',
      },
    ])
  })

  test('does not render worker progress reminder for agent_block without to_lead message atom', () => {
    const timeline: readonly TimelineEntry[] = [
      {
        kind: 'agent_block',
        timestamp: TS0,
        firstAtomTimestamp: TS0,
        lastAtomTimestamp: TS2,
        agentId: 'builder-x',
        role: 'builder',
        atoms: [
          { kind: 'thought', timestamp: TS0, text: 'thinking' },
          {
            kind: 'tool_call',
            timestamp: TS1,
            toolCallId: 'tc1',
            tagName: 'read',
            attributes: { path: 'src/a.ts' },
            status: 'success',
          },
          { kind: 'error', timestamp: TS2, message: 'oops' },
        ],
      },
      { kind: 'lifecycle_hook', timestamp: TS3, agentId: 'builder-x', role: 'builder', hookType: 'idle' },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text:
          '--- 2024-03-28 16:00 ---\n<agent id="builder-x" role="builder" status="working">\nthinking\n<read path="src/a.ts"/>\n<error>oops</error>\n</agent>\n\n<reminders>\n- Builder idle: builder-x\n</reminders>\n\n<attention>\n- builder-x errored at 16:00\n</attention>',
      },
    ])
  })

  test('does not render worker progress reminder for lifecycle_hook/task_idle_hook alone', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'lifecycle_hook', timestamp: TS0, agentId: 'builder-z', role: 'builder', hookType: 'spawn' },
      { kind: 'task_idle_hook', timestamp: TS1, taskId: 't1', taskType: 'implement', title: 'Build thing', agentId: 'builder-z' },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text:
          '<reminders>\n- Builder spawned: builder-z\n- Worker builder-z for task t1 ("Build thing") has finished. Review output and either send feedback or mark complete.\n</reminders>',
      },
    ])
  })

  test('renders both communication reminders together when user and worker messages are present', () => {
    const timeline: readonly TimelineEntry[] = [
      { kind: 'user_message', timestamp: TS0, text: 'hello', attachments: [] },
      {
        kind: 'agent_block',
        timestamp: TS1,
        firstAtomTimestamp: TS1,
        lastAtomTimestamp: TS1,
        agentId: 'builder-x',
        role: 'builder',
        atoms: [{ kind: 'message', timestamp: TS1, direction: 'to_lead', text: 'progress update' }],
      },
    ]

    const out = formatInbox({ results: [], timeline, timezone: 'UTC', lifecycleReminderFormatters })
    expect(out).toEqual([
      {
        type: 'text',
        text:
          `--- 2024-03-28 16:00 ---\n<message from="user">hello</message>\n<agent id="builder-x" role="builder" status="working">\n<message to="lead">progress update</message>\n</agent>\n\n<reminders>\n- ${WORKER_PROGRESS_USER_MESSAGE_REMINDER}\n</reminders>`,
      },
    ])
  })
})
