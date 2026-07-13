import { describe, expect, it } from 'vitest'
import { renderTimeline, type RenderTimelineInput } from '../src/window/inbox/render'
import type { AgentAtom, TimelineEntry } from '../src/window/inbox/types'
import { Option } from 'effect'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EPOCH = 1_714_996_800_000  // 2024-05-06T12:00:00Z
const ts = (offsetSeconds: number) => EPOCH + offsetSeconds * 1000

const baseInput: RenderTimelineInput = {
  timeline: [],
  timezone: 'UTC',
}

function textFromParts(parts: ReturnType<typeof renderTimeline>): string {
  return parts
    .filter((p): p is { _tag: 'TextPart'; text: string } => p._tag === 'TextPart')
    .map(p => p.text)
    .join('')
}

function assertMarkers(text: string, expectedMarkers: string[]) {
  const markers = text.match(/---\s+.*?\s+---/g) ?? []
  expect(markers).toEqual(expectedMarkers)
}

function assertNoMarkers(text: string) {
  expect(text).not.toMatch(/---\s+.*?\s+---/)
}

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

function makeUserMessage(text: string, timestamp: number): TimelineEntry {
  return { kind: 'user_message', timestamp, text, attachments: [], synthetic: Option.none() }
}

function makeObservation(timestamp: number, text = 'observing'): TimelineEntry {
  return { kind: 'observation', timestamp, parts: [{ _tag: 'TextPart', text }] }
}

function makeAgentBlock(agentId: string, timestamp: number, atoms: AgentAtom[], status: string = 'working'): TimelineEntry {
  return {
    kind: 'agent_block',
    timestamp,
    firstAtomTimestamp: atoms[0]?.timestamp ?? timestamp,
    lastAtomTimestamp: atoms[atoms.length - 1]?.timestamp ?? timestamp,
    agentId,
    role: 'worker',
    status,
    atoms,
  }
}

function makeThought(text: string, timestamp = 0): AgentAtom {
  return { kind: 'thought', timestamp, text }
}

function makeToolCall(toolName: string, timestamp = 0): AgentAtom {
  return {
    kind: 'tool_call',
    timestamp,
    toolCallId: 'tc-1',
    toolName,
    attributes: {},
    body: Option.none(),
    status: 'success',
    exitCode: Option.none(),
    error: Option.none(),
  }
}

function makeIdle(reason: 'stable' | 'interrupt' | 'error', timestamp = 0): AgentAtom {
  return { kind: 'idle', timestamp, reason: Option.some(reason) }
}

function makeEscalation(timestamp: number): TimelineEntry {
  return {
    kind: 'escalation',
    timestamp,
    observedForkId: null,
    observedTurnId: 't-1',
    justification: 'churn',
    coalesceKey: Option.none(),
  }
}

function makeTurnStart(turnId: string, timestamp: number): TimelineEntry {
  return { kind: 'turn_start', timestamp, turnId }
}

function makeTurnEnd(turnId: string, timestamp: number): TimelineEntry {
  return { kind: 'turn_end', timestamp, turnId }
}

function makeTaskUpdate(action: 'created' | 'cancelled' | 'completed', taskId: string, timestamp: number): TimelineEntry {
  return {
    kind: 'task_update',
    timestamp,
    action,
    taskId,
    title: Option.none(),
    previousStatus: Option.none(),
    nextStatus: Option.none(),
    cancelledCount: Option.none(),
  }
}

function makeTaskTreeView(timestamp: number, renderedTree: string): TimelineEntry {
  return { kind: 'task_tree_view', timestamp, renderedTree }
}

function makeLifecycleHook(timestamp: number, agentId: string, hookType: 'spawn' | 'idle' = 'spawn'): TimelineEntry {
  return { kind: 'lifecycle_hook', timestamp, agentId, role: 'worker', hookType, taskId: Option.none(), taskTitle: Option.none() }
}

function makeTaskIdleHook(timestamp: number, taskId: string, title: string): TimelineEntry {
  return { kind: 'task_idle_hook', timestamp, taskId, title, agentId: 'a' }
}

function makeTaskCompleteHook(timestamp: number, taskId: string, title: string): TimelineEntry {
  return { kind: 'task_complete_hook', timestamp, taskId, title }
}

function makeCoordinatorMessage(text: string, timestamp: number): TimelineEntry {
  return { kind: 'coordinator_message', timestamp, text }
}

function makeUserBashCommand(timestamp: number, command = 'ls', exitCode = 0): TimelineEntry {
  return { kind: 'user_bash_command', timestamp, command, cwd: '/', exitCode, stdout: '', stderr: '' }
}

function makeUserToAgent(timestamp: number, agentId: string, text: string): TimelineEntry {
  return { kind: 'user_to_agent', timestamp, agentId, text }
}

// ---------------------------------------------------------------------------
// 1. Empty / Minimal Scenarios
// ---------------------------------------------------------------------------

describe('empty / minimal', () => {
  it('empty timeline produces no output', () => {
    const text = textFromParts(renderTimeline({ ...baseInput, timeline: [] }))
    expect(text).toBe('')
  })

  it('single user_message emits full-date marker then content', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeUserMessage('hello', ts(0))],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<message from="user">hello</message>')
  })

  it('turn_start and turn_end with no content: turn_start emits marker, turn_end ignored', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeTurnEnd('t1', ts(60)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).not.toContain('12:01:00')
  })

  it('single agent_block emits full-date marker then agent content', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeAgentBlock('worker-1', ts(0), [makeThought('thinking')])],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('thinking')
  })
})

// ---------------------------------------------------------------------------
// 2. Redundancy Scenarios (The Core Bug)
// ---------------------------------------------------------------------------

describe('redundancy', () => {
  it('turn_end and turn_start in same minute: only one marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeUserMessage('first', ts(0)),
        makeTurnEnd('t1', ts(0)),
        makeTurnStart('t2', ts(0)),
        makeUserMessage('second', ts(0)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text.match(/message from="user"/g)).toHaveLength(2)
  })

  it('turn_end and turn_start in different minute: single marker at turn_start', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeUserMessage('first', ts(0)),
        makeTurnEnd('t1', ts(0)),
        makeTurnStart('t2', ts(120)),
        makeUserMessage('second', ts(120)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:02:00 ---',
    ])
  })

  it('multiple turn_start in same minute: only one marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeUserMessage('a', ts(0)),
        makeTurnEnd('t1', ts(0)),
        makeTurnStart('t2', ts(0)),
        makeUserMessage('b', ts(0)),
        makeTurnEnd('t2', ts(0)),
        makeTurnStart('t3', ts(0)),
        makeUserMessage('c', ts(0)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('chronological entries in same minute as turn_start: no extra marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeUserMessage('msg', ts(0)),
        makeAgentBlock('a', ts(30), [makeThought('ok')]),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('chronological entry before turn_start in same minute: chronological entry gets the marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTurnStart('t1', ts(30)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('chronological entry before turn_start in different minute: two markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTurnStart('t1', ts(120)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:02:00 ---',
    ])
  })

  it('back-to-back user messages same minute: no marker between them', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('hi', ts(0)),
        makeUserMessage('hello', ts(5)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text.match(/message from="user"/g)).toHaveLength(2)
  })

  it('back-to-back user messages different minute: one marker per minute', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('hi', ts(0)),
        makeUserMessage('hello', ts(90)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:30 ---',
    ])
  })
})

// ---------------------------------------------------------------------------
// 3. Date Boundary Scenarios
// ---------------------------------------------------------------------------

describe('date boundaries', () => {
  it('entries crossing midnight: full date marker on new date entry', () => {
    const MIDNIGHT_EPOCH = Date.parse('2024-05-06T23:59:00Z')
    const ts2 = (s: number) => MIDNIGHT_EPOCH + s * 1000
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('before midnight', ts2(0)),
        makeUserMessage('after midnight', ts2(120)),  // 00:01:00
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 23:59:00 ---',
      '--- 2024-05-07 00:01:00 ---',
    ])
  })

  it('entries spanning multiple days: full date marker on each new day', () => {
    const DAY = 24 * 60 * 60
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('day 1', ts(0)),
        makeUserMessage('day 2', ts(DAY + 100)),
        makeUserMessage('day 3', ts(2 * DAY + 200)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 2024-05-07 12:01:40 ---',
      '--- 2024-05-08 12:03:20 ---',
    ])
  })

  it('turn boundary crossing midnight: first marker after midnight shows full date', () => {
    const MIDNIGHT_EPOCH = Date.parse('2024-05-06T23:59:00Z')
    const ts2 = (s: number) => MIDNIGHT_EPOCH + s * 1000
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts2(0)),
        makeUserMessage('before midnight', ts2(0)),
        makeTurnEnd('t1', ts2(0)),
        makeTurnStart('t2', ts2(120)),
        makeUserMessage('after midnight', ts2(120)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 23:59:00 ---',
      '--- 2024-05-07 00:01:00 ---',
    ])
  })
})

// ---------------------------------------------------------------------------
// 4. Marker Meaningfulness Scenarios
// ---------------------------------------------------------------------------

describe('marker meaningfulness', () => {
  it('user_message then agent_block different minute: marker before agent_block', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('help', ts(0)),
        makeAgentBlock('a', ts(90), [makeThought('working')]),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:30 ---',
    ])
  })

  it('agent idle atom: marker handled by agent_block timestamp, not by idle event separately', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeAgentBlock('worker', ts(0), [makeThought('done'), makeIdle('stable')]),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<' + 'yield_user/>')
  })

  it('escalation after user_message different minute: marker before escalation, escalation renders as escalation_required block', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeEscalation(ts(90)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:30 ---',
    ])
    expect(text).toContain('<escalation_required>')
  })

  it('worker finishing: same as agent_block rules', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('do work', ts(0)),
        makeAgentBlock('worker', ts(60), [makeToolCall('task_complete'), makeIdle('stable')]),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
  })

  it('multiple agent blocks same minute: single marker before first', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeAgentBlock('a1', ts(0), [makeThought('t1')]),
        makeAgentBlock('a2', ts(10), [makeThought('t2')]),
        makeAgentBlock('a3', ts(20), [makeThought('t3')]),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('entries hours apart: time-only markers with no date repetition', () => {
    const HOUR = 60 * 60
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('morning', ts(0)),
        makeUserMessage('afternoon', ts(2 * HOUR)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 14:00:00 ---',
    ])
  })
})

// ---------------------------------------------------------------------------
// 5. Compaction Scenarios
// ---------------------------------------------------------------------------

describe('compaction', () => {
  it('after compaction, first content shows full date', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeUserMessage('compacted prior', ts(0))],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('new context block after compaction always gets full date marker, independent of previous blocks', () => {
    const block1 = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeUserMessage('old', ts(0))],
    }))
    const block2 = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeUserMessage('new', ts(0))],  // same minute as old block
    }))
    assertMarkers(block1, ['--- 2024-05-06 12:00:00 ---'])
    assertMarkers(block2, ['--- 2024-05-06 12:00:00 ---'])  // full date again, not time-only
  })
})

// ---------------------------------------------------------------------------
// 6. Mixed Structural + Chronological Entries
// ---------------------------------------------------------------------------

describe('structural + chronological', () => {
  it('task_updates and task_tree_view do not trigger markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTaskUpdate('created', 't1', ts(30)),
        makeTaskTreeView(ts(30), '<task/>'),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<task_updates>')
    expect(text).toContain('<task_tree>')
  })

  it('task hooks are filtered out and do not trigger markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTaskIdleHook(ts(30), 't1', 'idle'),
        makeTaskCompleteHook(ts(30), 't1', 'done'),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('lifecycle hooks do not trigger markers but affect attention for last chronological entry', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeUserMessage('msg2', ts(15)),
        makeLifecycleHook(ts(30), 'agent-a'),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<attention>')
  })

  it('structural entries between chronological entries do not reset markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTaskUpdate('created', 't1', ts(30)),  // same minute, structural
        makeTaskUpdate('completed', 't1', ts(90)),  // different minute, structural
        makeUserMessage('next', ts(90)),  // same minute as task_update
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:30 ---',
    ])
  })
})

// ---------------------------------------------------------------------------
// 7. Attention Bullets / Reminder Interaction
// ---------------------------------------------------------------------------

describe('attention and reminders', () => {
  it('idle attention bullet at end of timeline does not add extra markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeUserMessage('msg2', ts(30)),
        makeAgentBlock('a', ts(60), [makeIdle('stable')]),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
    expect(text).toContain('<attention>')
  })
})

// ---------------------------------------------------------------------------
// 8. Timezone Behavior
// ---------------------------------------------------------------------------

describe('timezone', () => {
  it('UTC timezone renders consistent markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timezone: 'UTC',
      timeline: [makeUserMessage('msg', ts(0))],
    }))
    expect(text).toContain('12:00:00')
  })

  it('non-UTC timezone shifts markers', () => {
    const NY_EPOCH = Date.parse('2024-05-06T17:00:00Z')  // 17:00 UTC = 13:00 EDT
    const tsNY = (s: number) => NY_EPOCH + s * 1000
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timezone: 'America/New_York',
      timeline: [makeUserMessage('msg', tsNY(0))],
    }))
    expect(text).toContain('13:00:00')
  })
})

// ---------------------------------------------------------------------------
// 9. Edge Cases with Only Structural Entries
// ---------------------------------------------------------------------------

describe('structural only', () => {
  it('only structural entries produce no markers and no content', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTaskUpdate('created', 't1', ts(0)),
        makeTaskTreeView(ts(0), '<task/>'),
      ],
    }))
    assertNoMarkers(text)
    expect(text).toContain('<task_updates>')
    expect(text).toContain('<task_tree>')
  })

  it('only turn_start produces one marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeTurnStart('t1', ts(0))],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('only turn_end produces no markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeTurnEnd('t1', ts(0))],
    }))
    assertNoMarkers(text)
    expect(text).toBe('')
  })

  it('turn_start with structural entries and turn_end: only turn_start marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeTaskUpdate('created', 't1', ts(0)),
        makeTaskTreeView(ts(0), '<task/>'),
        makeTurnEnd('t1', ts(0)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<task_updates>')
    expect(text).toContain('<task_tree>')
    expect(text).not.toContain('turn_end')
  })
})

// ---------------------------------------------------------------------------
// 10. Complex Mixed Scenarios
// ---------------------------------------------------------------------------

describe('complex scenarios', () => {
  it('full conversation simulation: correct markers at all boundaries', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeUserMessage('hello', ts(0)),
        makeAgentBlock('a', ts(30), [makeThought('thinking')]),
        makeTurnEnd('t1', ts(30)),
        makeTurnStart('t2', ts(120)),
        makeTaskUpdate('created', 't2', ts(120)),
        makeUserMessage('how are you', ts(120)),
        makeAgentBlock('a', ts(180), [makeThought('good'), makeToolCall('ok')]),
        makeTurnEnd('t2', ts(180)),
        makeTurnStart('t3', ts(240)),
        makeUserMessage('thanks', ts(240)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',  // t1 start (first in timeline, full date)
      '--- 12:02:00 ---',              // t2 start (different minute, time-only)
      '--- 12:03:00 ---',              // agent_block at 180s (different minute from t2 start at 120s)
      '--- 12:04:00 ---',              // t3 start (different minute from 180s)
    ])
    const markerLines = text.split('\n').filter(line => line.includes('---'))
    expect(markerLines).toHaveLength(4)
  })

  it('multiple turn_start in same minute separated by content: only one marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeUserMessage('a', ts(0)),
        makeTurnEnd('t1', ts(0)),
        makeTurnStart('t2', ts(0)),  // same minute, no new marker
        makeUserMessage('b', ts(0)),
        makeTurnEnd('t2', ts(0)),
        makeTurnStart('t3', ts(60)), // different minute, new marker
        makeUserMessage('c', ts(60)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
  })

  it('structural entries before chronological content: first chronological gets full date', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTaskUpdate('created', 't1', ts(0)),
        makeTaskTreeView(ts(0), '<task/>'),
        makeUserMessage('hello', ts(30)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:30 ---'])
    expect(text).toContain('<task_updates>')
    expect(text).toContain('<task_tree>')
    expect(text).toContain('<message from="user">hello</message>')
  })
})

// ---------------------------------------------------------------------------
// 11. Additional chronological entry types
// ---------------------------------------------------------------------------

describe('additional chronological entry types', () => {
  it('coordinator_message gets marker and renders correctly', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeCoordinatorMessage('hello from coordinator', ts(0))],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<message from="coordinator">hello from coordinator</message>')
  })

  it('user_bash_command gets marker and renders correctly', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeUserBashCommand(ts(0), 'ls -la', 0)],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<user_bash_command')
  })

  it('user_to_agent gets marker and renders correctly', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeUserToAgent(ts(0), 'agent-1', 'do this')],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<user-to-agent agent="agent-1">do this</user-to-agent>')
  })

  it('observation gets marker and renders correctly', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeObservation(ts(0), 'system event')],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('system event')
  })

  it('multiple chronological entry types in same minute share one marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeUserMessage('msg2', ts(15)),
        makeUserMessage('msg2', ts(30)),
        makeObservation(ts(45), 'obs'),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('chronological entry types across minute boundaries each get markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeCoordinatorMessage('coord', ts(90)),
        makeUserBashCommand(ts(180), 'ls', 0),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:30 ---',
      '--- 12:03:00 ---',
    ])
  })
})

// ---------------------------------------------------------------------------
// 12. Edge case: chronological entries with no turn_start
// ---------------------------------------------------------------------------

describe('no turn_start', () => {
  it('chronological entries without turn_start still get markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('first', ts(0)),
        makeUserMessage('second', ts(60)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
  })

  it('first chronological entry shows full date even without turn_start', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeUserMessage('hello', ts(0))],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })
})

// ---------------------------------------------------------------------------
// 13. Edge case: empty agent block
// ---------------------------------------------------------------------------

describe('empty agent blocks', () => {
  it('empty agent block still gets marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeAgentBlock('a', ts(0), [])],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('</agent>')
  })

  it('empty agent block followed by user message same minute shares marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeAgentBlock('a', ts(0), []),
        makeUserMessage('hi', ts(0)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })
})

// ---------------------------------------------------------------------------
// 14. Edge case: very rapid sequential entries (sub-second)
// ---------------------------------------------------------------------------

describe('rapid entries', () => {
  it('entries within same second share one marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('a', ts(0)),
        makeUserMessage('b', ts(0) + 100),
        makeUserMessage('c', ts(0) + 200),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text.match(/message from="user"/g)).toHaveLength(3)
  })

  it('entries at exactly the same millisecond share one marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('a', ts(0)),
        makeUserMessage('b', ts(0)),
        makeUserMessage('c', ts(0)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text.match(/message from="user"/g)).toHaveLength(3)
  })
})

// ---------------------------------------------------------------------------
// 15. Edge case: turn_end immediately after turn_start (empty turn)
// ---------------------------------------------------------------------------

describe('empty turn', () => {
  it('turn_start immediately followed by turn_end: only one marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeTurnEnd('t1', ts(0)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('empty turn followed by content in same minute: still one marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeTurnEnd('t1', ts(0)),
        makeUserMessage('hello', ts(0)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('hello')
  })

  it('empty turn followed by content in different minute: two markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeTurnEnd('t1', ts(0)),
        makeUserMessage('hello', ts(60)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
  })
})

// ---------------------------------------------------------------------------
// 16. Edge case: task reassigned / task tree dirty / task start hook
// ---------------------------------------------------------------------------

describe('task structural entries', () => {
  it('task_start_hook does not trigger markers or render content', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        { kind: 'task_start_hook', timestamp: ts(30), taskId: 't1', title: 'start' },
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).not.toContain('task_start_hook')
    expect(text).not.toContain('start')
  })

  it('task_tree_dirty does not trigger markers or render content', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        { kind: 'task_tree_dirty', timestamp: ts(30), taskId: 't1' },
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).not.toContain('task_tree_dirty')
  })

  it('task_reassigned does not trigger markers or render content', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        { kind: 'task_reassigned', timestamp: ts(30), text: 'reassigned', oldTaskId: 't1', newTaskId: 't2' },
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).not.toContain('task_reassigned')
  })
})

// ---------------------------------------------------------------------------
// 17. Edge case: worker_user_killed and detached_process_exited
// ---------------------------------------------------------------------------

describe('kill and exit events', () => {
  it('worker_user_killed gets marker and renders', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        { kind: 'worker_user_killed', timestamp: ts(60), agentId: 'w1', agentType: 'worker' },
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
    expect(text).toContain('<subagent-user-killed')
  })

  it('detached_process_exited gets marker and renders', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        {
          kind: 'detached_process_exited',
          timestamp: ts(60),
          pid: 42,
          command: 'sleep 10',
          exitCode: 0,
          stdoutPath: '/tmp/out',
          stderrPath: '/tmp/err',
        },
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
    expect(text).toContain('<detached_process_exited')
  })
})

// ---------------------------------------------------------------------------
// 19. Edge case: content before turn_start in same minute (latent bug)
// ---------------------------------------------------------------------------

describe('content before turn_start', () => {
  it('chronological entry before turn_start in same minute: chronological entry gets marker, not turn_start duplicate', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('before', ts(0)),
        makeTurnStart('t1', ts(30)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('before')
  })

  it('chronological entry before turn_start in different minute: chronological gets full date, turn_start gets time-only', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('before', ts(0)),
        makeTurnStart('t1', ts(120)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:02:00 ---',
    ])
    expect(text).toContain('before')
  })
})

// ---------------------------------------------------------------------------
// 20. Edge case: turn_start after content in same minute
// ---------------------------------------------------------------------------

describe('turn_start after content', () => {
  it('turn_start after content in same minute: no extra marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTurnStart('t1', ts(15)),
        makeUserMessage('next', ts(15)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text.match(/message from="user"/g)).toHaveLength(2)
  })

  it('turn_start after content in different minute: new marker at turn_start', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTurnStart('t1', ts(90)),
        makeUserMessage('next', ts(90)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:30 ---',
    ])
    expect(text.match(/message from="user"/g)).toHaveLength(2)
  })
})

// ---------------------------------------------------------------------------
// 21. Edge case: mixed chronological and structural with attention
// ---------------------------------------------------------------------------

describe('mixed with attention', () => {
  it('lifecycle hook after last chronological entry triggers attention without extra marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeUserMessage('msg2', ts(15)),
        makeLifecycleHook(ts(30), 'agent-1'),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<attention>')
  })

  it('no attention for single user_message with no lifecycle hooks', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeUserMessage('msg', ts(0))],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).not.toContain('<attention>')
  })

  it('attention for multiple chronological entries even without lifecycle hooks', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg1', ts(0)),
        makeUserMessage('msg2', ts(30)),
        makeUserMessage('msg2', ts(60)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
    expect(text).toContain('<attention>')
  })
})

// ---------------------------------------------------------------------------
// 22. Edge case: agent going idle with status map
// ---------------------------------------------------------------------------

describe('agent idle with status', () => {
  it('idle agent with status in map produces attention bullet', () => {
    const agents = new Map()
    agents.set('worker-1', { status: 'idle', role: 'worker', agentId: 'worker-1', forkId: 'f1', taskId: 't1' })
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeUserMessage('msg2', ts(30)),
        makeAgentBlock('worker-1', ts(60), [makeIdle('stable')], 'idle'),
        makeLifecycleHook(ts(90), 'agent-x'),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
    expect(text).toContain('<attention>')
    expect(text).toContain('went idle')
  })

  it('agent with error atoms produces error attention bullet', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeUserMessage('msg2', ts(30)),
        makeAgentBlock('a', ts(60), [{ kind: 'error', timestamp: ts(60), message: 'boom' }]),
        makeLifecycleHook(ts(90), 'agent-x'),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
    expect(text).toContain('<attention>')
    expect(text).toContain('errored')
  })
})

// ---------------------------------------------------------------------------
// 23. Edge case: user message with attachments
// ---------------------------------------------------------------------------

describe('user message with attachments', () => {
  it('user message with file attachment renders attachment text', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        {
          kind: 'user_message',
          timestamp: ts(0),
          text: 'check this',
          synthetic: Option.none(),
          attachments: [
            {
              kind: 'image',
              path: '/attachments/cat.png',
              filename: 'cat.png',
              mediaType: 'image/png',
              width: 0,
              height: 0,
            },
          ],
        },
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('check this')
    expect(text).toContain('<attachment path="/attachments/cat.png" filename="cat.png"')
  })

  it('user message with mention attachment renders mention', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        {
          kind: 'user_message',
          timestamp: ts(0),
          text: 'look at this',
          synthetic: Option.none(),
          attachments: [
            {
              kind: 'mention',
              attachment: { type: 'mention_file', path: '/file.ts' },
              resolution: { status: 'resolved', content: 'const x = 1', truncated: false, originalBytes: 0 },
            },
          ],
        },
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('look at this')
    expect(text).toContain('<mention path="/file.ts"')
  })
})

// ---------------------------------------------------------------------------
// 24. Edge case: turn_end with no following turn_start
// ---------------------------------------------------------------------------

describe('trailing turn_end', () => {
  it('turn_end at end with no following turn_start: no trailing marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTurnEnd('t1', ts(30)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('turn_end at end after content in different minute: no trailing marker, only content marker', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTurnEnd('t1', ts(60)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })
})

// ---------------------------------------------------------------------------
// 25. Edge case: very long gaps between markers
// ---------------------------------------------------------------------------

describe('long gaps', () => {
  it('days apart still show full date on new day', () => {
    const DAY = 24 * 60 * 60
    const WEEK = 7 * DAY
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('old', ts(0)),
        makeUserMessage('new', ts(WEEK)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 2024-05-13 12:00:00 ---',
    ])
  })

  it('months apart still show full date on new day', () => {
    const MONTH = 30 * 24 * 60 * 60
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('old', ts(0)),
        makeUserMessage('new', ts(MONTH)),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 2024-06-05 12:00:00 ---',
    ])
  })
})

// ---------------------------------------------------------------------------
// 26. Edge case: null timezone
// ---------------------------------------------------------------------------

describe('null timezone', () => {
  it('null timezone uses local system time but still produces valid markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timezone: null,
      timeline: [makeUserMessage('msg', ts(0))],
    }))
    expect(text).toMatch(/---\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+---/)
  })
})

// ---------------------------------------------------------------------------
// 27. Edge case: task_updates with different actions
// ---------------------------------------------------------------------------

describe('task update actions', () => {
  it('created action renders with title', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        { kind: 'task_update', timestamp: ts(30), action: 'created', taskId: 't1', title: Option.some('Fix bug'), previousStatus: Option.none(), nextStatus: Option.none(), cancelledCount: Option.none() },
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('Task t1 created: "Fix bug"')
  })

  it('cancelled action renders with count', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        { kind: 'task_update', timestamp: ts(30), action: 'cancelled', taskId: 't1', title: Option.none(), previousStatus: Option.none(), nextStatus: Option.none(), cancelledCount: Option.some(3) },
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('Task t1 cancelled (3 tasks removed)')
  })

  it('status_changed action renders transition', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        { kind: 'task_update', timestamp: ts(30), action: 'status_changed', taskId: 't1', title: Option.none(), previousStatus: Option.some('pending'), nextStatus: Option.some('active'), cancelledCount: Option.none() },
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('Task t1 status changed: pending -> active')
  })
})

// ---------------------------------------------------------------------------
// 28. Edge case: multiple task_updates in same and different minutes
// ---------------------------------------------------------------------------

describe('multiple task updates', () => {
  it('multiple task_updates in same minute do not trigger extra markers', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTaskUpdate('created', 't1', ts(30)),
        makeTaskUpdate('created', 't2', ts(30)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })

  it('task_updates across minutes do not trigger markers themselves', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTaskUpdate('created', 't1', ts(30)),
        makeTaskUpdate('created', 't2', ts(90)),
      ],
    }))
    // task_updates are structural, so no marker between them
    // but the next chronological entry would get a marker if present
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
  })
})

// ---------------------------------------------------------------------------
// 29. Edge case: lifecycle hooks with different hook types
// ---------------------------------------------------------------------------

describe('lifecycle hook types', () => {
  it('spawn lifecycle hook triggers attention', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeUserMessage('msg2', ts(15)),
        makeLifecycleHook(ts(30), 'a1', 'spawn'),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<attention>')
  })

  it('idle lifecycle hook triggers attention', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeUserMessage('msg2', ts(15)),
        makeLifecycleHook(ts(30), 'a1', 'idle'),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<attention>')
  })
})

// ---------------------------------------------------------------------------
// 30. Edge case: background processes
// ---------------------------------------------------------------------------

describe('background processes', () => {
  it('no background processes renders nothing', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeUserMessage('msg', ts(0))],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).not.toContain('<background_processes>')
  })
})

// ---------------------------------------------------------------------------
// 31. Edge case: turn_start immediately before chronological entry at same timestamp
// ---------------------------------------------------------------------------

describe('turn_start and content same timestamp', () => {
  it('turn_start and user_message at exact same timestamp: one marker, both rendered', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeUserMessage('hello', ts(0)),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('hello')
  })

  it('turn_start and agent_block at exact same timestamp: one marker, both rendered', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTurnStart('t1', ts(0)),
        makeAgentBlock('a', ts(0), [makeThought('ok')]),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('ok')
  })
})

// ---------------------------------------------------------------------------
// 32. Edge case: reminders (task idle / complete) without chronological content
// ---------------------------------------------------------------------------

describe('reminders without chronological content', () => {
  it('task idle hook without chronological content renders reminders', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeTaskIdleHook(ts(0), 't1', 'Task 1')],
    }))
    // no chronological content, but task idle hooks generate reminders
    expect(text).toContain('<reminders>')
  })

  it('task complete hook without chronological content renders reminders', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [makeTaskCompleteHook(ts(0), 't1', 'Task 1')],
    }))
    expect(text).toContain('<reminders>')
  })
})

// ---------------------------------------------------------------------------
// 33. Edge case: agent_block with message atoms
// ---------------------------------------------------------------------------

describe('agent block with messages', () => {
  it('agent_block with to_lead message triggers worker progress reminder', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeAgentBlock('a', ts(60), [
          { kind: 'message', timestamp: ts(60), direction: 'to_lead', text: 'progress update' },
        ]),
      ],
    }))
    assertMarkers(text, [
      '--- 2024-05-06 12:00:00 ---',
      '--- 12:01:00 ---',
    ])
    expect(text).toContain('<reminders>')
    expect(text).toContain('progress update')
  })

  it('agent_block with from_user message renders correctly', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeAgentBlock('a', ts(0), [
          { kind: 'message', timestamp: ts(0), direction: 'from_user', text: 'user msg' },
        ]),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('from="user"')
    expect(text).toContain('user msg')
  })

  it('agent_block with from_lead message renders correctly', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeAgentBlock('a', ts(0), [
          { kind: 'message', timestamp: ts(0), direction: 'from_lead', text: 'lead msg' },
        ]),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('from="lead"')
    expect(text).toContain('lead msg')
  })
})

// ---------------------------------------------------------------------------
// 34. Edge case: entry ordering within same minute
// ---------------------------------------------------------------------------

describe('entry ordering within same minute', () => {
  it('structural then chronological in same minute: marker on chronological', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeTaskUpdate('created', 't1', ts(0)),
        makeUserMessage('msg', ts(15)),
      ],
    }))
    // task_update is structural, so marker on user_message
    assertMarkers(text, ['--- 2024-05-06 12:00:15 ---'])
    expect(text).toContain('<message from="user">msg</message>')
    expect(text).toContain('<task_updates>')
  })

  it('chronological then structural then chronological in same minute: marker on first chronological', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg1', ts(0)),
        makeTaskUpdate('created', 't1', ts(15)),
        makeUserMessage('msg2', ts(30)),
      ],
    }))
    // first chronological entry at 12:00:00 gets the full date marker
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text.match(/message from="user"/g)).toHaveLength(2)
  })
})

// ---------------------------------------------------------------------------
// 35. Edge case: task tree view renders only latest
// ---------------------------------------------------------------------------

describe('task tree view', () => {
  it('multiple task_tree_view entries render only the latest', () => {
    const text = textFromParts(renderTimeline({
      ...baseInput,
      timeline: [
        makeUserMessage('msg', ts(0)),
        makeTaskTreeView(ts(30), '<tree1/>'),
        makeTaskTreeView(ts(30), '<tree2/>'),
      ],
    }))
    assertMarkers(text, ['--- 2024-05-06 12:00:00 ---'])
    expect(text).toContain('<task_tree>')
    expect(text).toContain('<tree2/>')
    expect(text).not.toContain('<tree1/>')
  })
})
