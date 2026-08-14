import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type {
  DisplayMessage,
  DisplayRootStatus,
  DisplayTimelineEntry,
  DisplayViewSnapshot,
} from "@magnitudedev/sdk"
import { ForkIdSchema } from "@magnitudedev/sdk"
import { makeDisplayViewSnapshotFixture } from "@magnitudedev/sdk/testing"
import {
  createHeadlessOutputRenderer,
  formatDuration,
  renderUsageSummary,
  sanitizeHeadlessText,
} from "./output"

function snapshot(options: {
  readonly messages?: readonly DisplayMessage[]
  readonly messageOrder?: readonly string[]
  readonly entries?: readonly DisplayTimelineEntry[]
  readonly status?: DisplayRootStatus
  readonly mode?: "idle" | "streaming"
} = {}): DisplayViewSnapshot {
  const messages = options.messages ?? []
  const messageOrder = options.messageOrder ?? messages.map((message) => message.id)
  return makeDisplayViewSnapshotFixture({
    shape: {
      timelines: {
        root: { kind: "tail", limit: 10_000, live: true, presentation: "default" },
      },
    },
    session: { sessionId: "session-1", title: "Headless test", cwd: "/repo" },
    status: options.status,
    messages,
    messageOrder,
    entries: options.entries,
    mode: options.mode,
    streamingMessageId: options.mode === "streaming" ? "assistant-1" : null,
  })
}

function messageEntry(
  message: DisplayMessage,
  options: { readonly role?: "user" | "assistant" | "system" | "agent"; readonly streaming?: boolean } = {},
): DisplayTimelineEntry {
  return {
    kind: "message",
    id: `entry:${message.id}`,
    messageId: message.id,
    timestamp: message.timestamp,
    role: options.role ?? (message.type === "user_message" ? "user" : "assistant"),
    streaming: options.streaming ?? false,
    interrupted: false,
    nextMessageInterrupted: false,
  }
}

const userMessage: DisplayMessage = {
  id: "user-1",
  type: "user_message",
  content: "build the feature",
  timestamp: 1,
  taskMode: false,
  attachments: [],
}

const assistantMessage = (content: string): DisplayMessage => ({
  id: "assistant-1",
  type: "assistant_message",
  content,
  timestamp: 2,
})

const completedShellStep: Extract<DisplayTimelineEntry, { readonly kind: "tool_step" }>["step"] = {
  toolKey: "shell",
  phase: "completed",
  tone: "success",
  icon: "terminal",
  command: "bun test",
  done: "completed",
  exitCode: 0,
  pid: null,
  stdout: "ok",
  stderr: "",
  partialStdout: "",
  partialStderr: "",
  stdoutPath: null,
  stderrPath: null,
  errorText: null,
  running: false,
  failed: false,
}

const toolMessage = (id: string, toolKey = "shell"): DisplayMessage => ({
  id,
  type: "tool",
  toolKey,
  cluster: Option.none(),
  presentation: toolKey === "shell"
    ? Option.some(completedShellStep)
    : Option.none(),
  filter: Option.none(),
  resultFilePath: Option.none(),
  timestamp: 2,
})

describe("headless output renderer", () => {
  it("buffers streaming assistant messages until the authoritative entry is complete", () => {
    const renderer = createHeadlessOutputRenderer()
    const streaming = assistantMessage("partial")

    expect(renderer.handleSnapshot(snapshot({
      messages: [streaming],
      entries: [messageEntry(streaming, { streaming: true })],
      mode: "streaming",
      status: {
        _tag: "Working",
        chainStartedAt: 1,
        detail: { _tag: "Thinking" },
        activeChildCount: 0,
      },
    })).lines).toEqual([])

    const complete = assistantMessage("complete answer")
    expect(renderer.handleSnapshot(snapshot({
      messages: [complete],
      entries: [messageEntry(complete)],
      status: { _tag: "Worked", lastProductiveMs: 25 },
    })).lines).toEqual(["complete answer"])

    expect(renderer.handleSnapshot(snapshot({
      messages: [complete],
      entries: [messageEntry(complete)],
      status: { _tag: "Worked", lastProductiveMs: 25 },
    })).lines).toEqual([])

    expect(renderer.handleSnapshot(snapshot({
      messages: [complete],
      entries: [{ ...messageEntry(complete), id: "replacement-entry:same-message" }],
      status: { _tag: "Worked", lastProductiveMs: 25 },
    })).lines).toEqual([])
  })

  it("renders completed display tools once and counts them", () => {
    const renderer = createHeadlessOutputRenderer()
    const shellMessage = toolMessage("tool-message-1")
    const shellEntry: DisplayTimelineEntry = {
      kind: "tool_step",
      id: "tool-entry-1",
      messageId: shellMessage.id,
      timestamp: 2,
      step: completedShellStep,
    }

    const first = renderer.handleSnapshot(snapshot({
      messages: [userMessage, shellMessage],
      entries: [messageEntry(userMessage), shellEntry],
      status: { _tag: "Worked", lastProductiveMs: 25 },
    }))

    expect(first.lines).toEqual(["> build the feature", "$ bun test · exit 0"])
    expect(first.toolCount).toBe(1)
    expect(renderer.handleSnapshot(snapshot({
      messages: [userMessage, shellMessage],
      entries: [messageEntry(userMessage), shellEntry],
      status: { _tag: "Worked", lastProductiveMs: 25 },
    })).lines).toEqual([])
    expect(renderer.getToolCount()).toBe(1)
  })

  it("rejects tool presentation entries without an authoritative tool message", () => {
    const renderer = createHeadlessOutputRenderer()
    const entries: DisplayTimelineEntry[] = [
      {
        kind: "tool_step",
        id: "forged-step-non-tool",
        messageId: userMessage.id,
        timestamp: 2,
        step: completedShellStep,
      },
      {
        kind: "tool_summary",
        id: "forged-summary-unknown",
        timestamp: 3,
        messageIds: ["missing-tool-message"],
        summary: {
          toolKey: "fileRead",
          phase: "completed",
          tone: "success",
          icon: "file",
          count: 1,
          running: false,
          failed: false,
          matchCount: null,
          fileCount: 1,
          sourceCount: null,
          detail: [],
        },
      },
    ]

    expect(renderer.handleSnapshot(snapshot({
      messages: [userMessage],
      entries: [messageEntry(userMessage), ...entries],
    }))).toEqual({
      lines: ["> build the feature"],
      toolCount: 0,
    })
  })

  it("renders canonical tool presentation instead of a forged entry payload", () => {
    const renderer = createHeadlessOutputRenderer()
    const canonical: Extract<DisplayMessage, { readonly type: "tool" }> = {
      id: "tool-canonical",
      type: "tool",
      toolKey: "shell",
      cluster: Option.none(),
      presentation: Option.some(completedShellStep),
      filter: Option.none(),
      resultFilePath: Option.none(),
      timestamp: 2,
    }
    const forged: DisplayTimelineEntry = {
      kind: "tool_step",
      id: "tool-entry-forged",
      messageId: canonical.id,
      timestamp: canonical.timestamp,
      step: {
        ...completedShellStep,
        command: "FORGED-COMMAND",
        failed: true,
        errorText: "FORGED-ERROR",
      },
    }

    expect(renderer.handleSnapshot(snapshot({ messages: [canonical], entries: [forged] }))).toEqual({
      lines: ["$ bun test · exit 0"],
      toolCount: 1,
    })
  })

  it("waits for canonical tool terminality before accepting a terminal summary", () => {
    const renderer = createHeadlessOutputRenderer()
    const presentation = {
      toolKey: "fileRead" as const,
      phase: "executing" as const,
      tone: "info" as const,
      icon: "file" as const,
      path: "src/index.ts",
      lineCount: 1,
      offset: null,
      limit: null,
      errorText: null,
      running: true,
      failed: false,
    }
    const running: Extract<DisplayMessage, { readonly type: "tool" }> = {
      id: "tool-running-under-terminal-summary",
      type: "tool",
      toolKey: "fileRead",
      cluster: Option.none(),
      presentation: Option.some(presentation),
      filter: Option.none(),
      resultFilePath: Option.none(),
      timestamp: 2,
    }
    const summary: DisplayTimelineEntry = {
      kind: "tool_summary",
      id: "summary-stale-terminal-phase",
      timestamp: 2,
      messageIds: [running.id],
      summary: {
        toolKey: "fileRead",
        phase: "completed",
        tone: "success",
        icon: "file",
        count: 1,
        running: false,
        failed: false,
        matchCount: null,
        fileCount: 1,
        sourceCount: null,
        detail: [],
      },
    }

    expect(renderer.handleSnapshot(snapshot({ messages: [running], entries: [summary] }))).toEqual({
      lines: [],
      toolCount: 0,
    })

    const completed: DisplayMessage = {
      ...running,
      presentation: Option.some({
        ...presentation,
        phase: "completed",
        tone: "success",
        running: false,
      }),
    }
    expect(renderer.handleSnapshot(snapshot({ messages: [completed], entries: [summary] }))).toEqual({
      lines: ["· fileRead × 1"],
      toolCount: 1,
    })
  })

  it("requires a tool summary phase to match its canonical presentations", () => {
    const renderer = createHeadlessOutputRenderer()
    const errored: Extract<DisplayMessage, { readonly type: "tool" }> = {
      id: "tool-error-under-completed-summary",
      type: "tool",
      toolKey: "fileRead",
      cluster: Option.none(),
      presentation: Option.some({
        toolKey: "fileRead",
        phase: "error",
        tone: "error",
        icon: "file",
        path: "src/index.ts",
        lineCount: 0,
        offset: null,
        limit: null,
        errorText: "read failed",
        running: false,
        failed: true,
      }),
      filter: Option.none(),
      resultFilePath: Option.none(),
      timestamp: 2,
    }
    const completedSummary: DisplayTimelineEntry = {
      kind: "tool_summary",
      id: "summary-wrong-terminal-phase",
      timestamp: 2,
      messageIds: [errored.id],
      summary: {
        toolKey: "fileRead",
        phase: "completed",
        tone: "success",
        icon: "file",
        count: 1,
        running: false,
        failed: false,
        matchCount: null,
        fileCount: 1,
        sourceCount: null,
        detail: [],
      },
    }

    expect(renderer.handleSnapshot(snapshot({
      messages: [errored],
      entries: [completedSummary],
    }))).toEqual({ lines: [], toolCount: 0 })

    expect(renderer.handleSnapshot(snapshot({
      messages: [errored],
      entries: [{
        ...completedSummary,
        summary: { ...completedSummary.summary, phase: "error", tone: "error", failed: true },
      }],
    }))).toEqual({ lines: ["✗ fileRead × 1"], toolCount: 1 })
  })

  it("renders mixed terminal tool outcomes using the aggregate error phase", () => {
    const renderer = createHeadlessOutputRenderer()
    const completed: Extract<DisplayMessage, { readonly type: "tool" }> = {
      id: "tool-completed-in-mixed-summary",
      type: "tool",
      toolKey: "fileRead",
      cluster: Option.none(),
      presentation: Option.some({
        toolKey: "fileRead",
        phase: "completed",
        tone: "success",
        icon: "file",
        path: "src/completed.ts",
        lineCount: 1,
        offset: null,
        limit: null,
        errorText: null,
        running: false,
        failed: false,
      }),
      filter: Option.none(),
      resultFilePath: Option.none(),
      timestamp: 2,
    }
    const errored: Extract<DisplayMessage, { readonly type: "tool" }> = {
      ...completed,
      id: "tool-errored-in-mixed-summary",
      presentation: Option.some({
        ...Option.getOrThrow(completed.presentation),
        phase: "error",
        tone: "error",
        path: "src/errored.ts",
        errorText: "read failed",
        failed: true,
      }),
      timestamp: 3,
    }
    const summary: DisplayTimelineEntry = {
      kind: "tool_summary",
      id: "summary-mixed-terminal-phases",
      timestamp: 3,
      messageIds: [completed.id, errored.id],
      summary: {
        toolKey: "fileRead",
        phase: "error",
        tone: "error",
        icon: "file",
        count: 2,
        running: false,
        failed: true,
        matchCount: null,
        fileCount: 2,
        sourceCount: null,
        detail: [],
      },
    }

    expect(renderer.handleSnapshot(snapshot({
      messages: [completed, errored],
      entries: [summary],
    }))).toEqual({ lines: ["✗ fileRead × 2"], toolCount: 2 })

    const rejected: Extract<DisplayMessage, { readonly type: "tool" }> = {
      ...errored,
      id: "tool-rejected-in-mixed-summary",
      presentation: Option.some({
        ...Option.getOrThrow(errored.presentation),
        phase: "rejected",
      }),
    }
    expect(createHeadlessOutputRenderer().handleSnapshot(snapshot({
      messages: [completed, rejected],
      entries: [{ ...summary, messageIds: [completed.id, rejected.id] }],
    }))).toEqual({ lines: ["✗ fileRead × 2"], toolCount: 2 })
  })

  it("rejects a tool summary whose canonical presentation has another tool key", () => {
    const renderer = createHeadlessOutputRenderer()
    const canonical: Extract<DisplayMessage, { readonly type: "tool" }> = {
      id: "tool-summary-canonical-key-mismatch",
      type: "tool",
      toolKey: "fileRead",
      cluster: Option.none(),
      presentation: Option.some(completedShellStep),
      filter: Option.none(),
      resultFilePath: Option.none(),
      timestamp: 2,
    }
    const summary: DisplayTimelineEntry = {
      kind: "tool_summary",
      id: "summary-canonical-key-mismatch",
      timestamp: 2,
      messageIds: [canonical.id],
      summary: {
        toolKey: "fileRead",
        phase: "completed",
        tone: "success",
        icon: "file",
        count: 1,
        running: false,
        failed: false,
        matchCount: null,
        fileCount: 1,
        sourceCount: null,
        detail: [],
      },
    }

    expect(renderer.handleSnapshot(snapshot({ messages: [canonical], entries: [summary] }))).toEqual({
      lines: [],
      toolCount: 0,
    })
  })

  it("rejects message presentation entries outside authoritative message order", () => {
    const renderer = createHeadlessOutputRenderer()
    const orphanAssistant: DisplayMessage = {
      id: "orphan-assistant",
      type: "assistant_message",
      content: "forged output",
      timestamp: 1,
    }
    const orphanCompletion: DisplayMessage = {
      id: "orphan-worker-completion",
      type: "worker_finished",
      workerRole: "researcher",
      workerId: "orphan-worker",
      forkId: ForkIdSchema.make("orphan-fork"),
      cumulativeTotalTimeMs: 100,
      cumulativeTotalToolsUsed: 5,
      resumed: false,
      timestamp: 2,
    }

    expect(renderer.handleSnapshot(snapshot({
      messages: [orphanAssistant, orphanCompletion],
      messageOrder: [],
      entries: [messageEntry(orphanAssistant), messageEntry(orphanCompletion)],
    }))).toEqual({
      lines: [],
      toolCount: 0,
    })
  })

  it("rejects a tool summary whose count does not match its authoritative messages", () => {
    const renderer = createHeadlessOutputRenderer()
    const authoritative = toolMessage("tool-authoritative", "fileRead")
    const entry: DisplayTimelineEntry = {
      kind: "tool_summary",
      id: "summary-mismatched-count",
      timestamp: 2,
      messageIds: [authoritative.id],
      summary: {
        toolKey: "fileRead",
        phase: "completed",
        tone: "success",
        icon: "file",
        count: 99,
        running: false,
        failed: false,
        matchCount: null,
        fileCount: 1,
        sourceCount: null,
        detail: [],
      },
    }

    expect(renderer.handleSnapshot(snapshot({ messages: [authoritative], entries: [entry] }))).toEqual({
      lines: [],
      toolCount: 0,
    })
  })

  it("counts and reports tools appended to an existing summary entry", () => {
    const renderer = createHeadlessOutputRenderer()
    const toolMessage = (id: string, timestamp: number): DisplayMessage => ({
      id,
      type: "tool",
      toolKey: "fileRead",
      cluster: Option.none(),
      presentation: Option.none(),
      filter: Option.none(),
      resultFilePath: Option.none(),
      timestamp,
    })
    const firstTool = toolMessage("tool-message-1", 2)
    const secondTool = toolMessage("tool-message-2", 3)
    const summaryEntry = (
      messages: readonly DisplayMessage[],
    ): DisplayTimelineEntry => ({
      kind: "tool_summary",
      id: "summary:tool-message-1",
      timestamp: 2,
      messageIds: messages.map((message) => message.id),
      summary: {
        toolKey: "fileRead",
        phase: "completed",
        tone: "success",
        icon: "file",
        count: messages.length,
        running: false,
        failed: false,
        matchCount: null,
        fileCount: messages.length,
        sourceCount: null,
        detail: [],
      },
    })

    const first = renderer.handleSnapshot(snapshot({
      messages: [firstTool],
      entries: [summaryEntry([firstTool])],
    }))
    expect(first.lines).toEqual(["· fileRead × 1"])
    expect(first.toolCount).toBe(1)

    const expanded = renderer.handleSnapshot(snapshot({
      messages: [firstTool, secondTool],
      entries: [summaryEntry([firstTool, secondTool])],
    }))
    expect(expanded.lines).toEqual(["· fileRead × +1"])
    expect(expanded.toolCount).toBe(2)
  })

  it("does not move summarized tools across intervening lifecycle records", () => {
    const renderer = createHeadlessOutputRenderer()
    const fileRead = (
      id: string,
      path: string,
    ): Extract<DisplayMessage, { readonly type: "tool" }> => ({
      id,
      type: "tool",
      toolKey: "fileRead",
      cluster: Option.none(),
      presentation: Option.some({
        toolKey: "fileRead",
        phase: "completed",
        tone: "success",
        icon: "file",
        path,
        lineCount: 1,
        offset: null,
        limit: null,
        errorText: null,
        running: false,
        failed: false,
      }),
      filter: Option.none(),
      resultFilePath: Option.none(),
      timestamp: 5,
    })
    const firstTool = fileRead("tool-before-worker", "a.ts")
    const resumed: DisplayMessage = {
      id: "worker-between-tools",
      type: "worker_resumed",
      workerRole: "researcher",
      workerId: "agent-1",
      title: "Research",
      timestamp: 5,
    }
    const secondTool = fileRead("tool-after-worker", "b.ts")
    const summary: DisplayTimelineEntry = {
      kind: "tool_summary",
      id: "summary:interleaved-tools",
      timestamp: 5,
      messageIds: [firstTool.id, secondTool.id],
      summary: {
        toolKey: "fileRead",
        phase: "completed",
        tone: "success",
        icon: "file",
        count: 2,
        running: false,
        failed: false,
        matchCount: null,
        fileCount: 2,
        sourceCount: null,
        detail: [],
      },
    }

    expect(renderer.handleSnapshot(snapshot({
      messages: [firstTool, resumed, secondTool],
      entries: [summary],
    }))).toEqual({
      lines: [
        "→ Read a.ts · 1 line",
        "▶ [agent-1] (researcher) resumed · Research",
        "→ Read b.ts · 1 line",
      ],
      toolCount: 3,
    })

    const fallbackRenderer = createHeadlessOutputRenderer()
    const firstToolWithoutPresentation: DisplayMessage = {
      ...firstTool,
      presentation: Option.none(),
    }
    const secondToolWithoutPresentation: DisplayMessage = {
      ...secondTool,
      presentation: Option.none(),
    }
    expect(fallbackRenderer.handleSnapshot(snapshot({
      messages: [firstToolWithoutPresentation, resumed, secondToolWithoutPresentation],
      entries: [summary],
    })).lines).toEqual([
      "· fileRead × +1",
      "▶ [agent-1] (researcher) resumed · Research",
      "· fileRead × +1",
    ])
  })

  it("renders data-only worker lifecycle and includes delegated tools in the total", () => {
    const renderer = createHeadlessOutputRenderer()
    const started: DisplayMessage = {
      id: "worker-started-1",
      type: "fork_activity",
      forkId: "fork-1",
      name: "Research",
      role: "researcher",
      status: "running",
      createdAt: 2,
      activeSince: 2,
      accumulatedActiveMs: 0,
      completedAt: Option.none(),
      resumeCount: Option.none(),
      toolCounts: {
        commands: 0,
        reads: 0,
        writes: 0,
        edits: 0,
        searches: 0,
        webSearches: 0,
        webFetches: 0,
        artifactWrites: 0,
        artifactUpdates: 0,
        other: 0,
      },
      timestamp: 2,
    }
    const finished: DisplayMessage = {
      id: "worker-finished-1",
      type: "worker_finished",
      workerRole: "researcher",
      workerId: "agent-1",
      forkId: ForkIdSchema.make("fork-1"),
      cumulativeTotalTimeMs: 250,
      cumulativeTotalToolsUsed: 3,
      resumed: false,
      timestamp: 4,
    }
    expect(renderer.handleSnapshot(snapshot({ messages: [started], entries: [] }))).toEqual({
      lines: ["▶ [fork-1] (researcher) started · Research"],
      toolCount: 1,
    })

    const progressed: DisplayMessage = {
      ...started,
      toolCounts: { ...started.toolCounts, commands: 1, reads: 1 },
      timestamp: 3,
    }
    expect(renderer.handleSnapshot(snapshot({ messages: [progressed], entries: [] }))).toEqual({
      lines: ["· [fork-1] +2 worker tools · 2 total"],
      toolCount: 3,
    })

    const completedActivity: DisplayMessage = {
      ...progressed,
      status: "completed",
      toolCounts: { ...progressed.toolCounts, reads: 2 },
      completedAt: Option.some(4),
      timestamp: 4,
    }
    const completed = snapshot({ messages: [completedActivity, finished], entries: [] })
    expect(renderer.handleSnapshot(completed)).toEqual({
      lines: [
        "· [fork-1] +1 worker tool · 3 total",
        "✓ [agent-1] (researcher) done · 3 tools",
      ],
      toolCount: 4,
    })
    expect(renderer.handleSnapshot(completed).lines).toEqual([])
  })

  it("adds tool totals for independent active and completed workers", () => {
    const renderer = createHeadlessOutputRenderer()
    const active: DisplayMessage = {
      id: "worker-active-disjoint",
      type: "fork_activity",
      forkId: "fork-active",
      name: "Active worker",
      role: "researcher",
      status: "running",
      createdAt: 5,
      activeSince: 5,
      accumulatedActiveMs: 0,
      completedAt: Option.none(),
      resumeCount: Option.none(),
      toolCounts: {
        commands: 0,
        reads: 5,
        writes: 0,
        edits: 0,
        searches: 0,
        webSearches: 0,
        webFetches: 0,
        artifactWrites: 0,
        artifactUpdates: 0,
        other: 0,
      },
      timestamp: 5,
    }
    const finished: DisplayMessage = {
      id: "worker-finished-disjoint",
      type: "worker_finished",
      forkId: ForkIdSchema.make("fork-completed"),
      workerRole: "builder",
      workerId: "agent-completed",
      cumulativeTotalTimeMs: 500,
      cumulativeTotalToolsUsed: 3,
      resumed: false,
      timestamp: 6,
    }

    expect(renderer.handleSnapshot(snapshot({
      messages: [active, finished],
      entries: [],
    })).toolCount).toBe(9)
  })

  it("ignores stale lower worker completion records", () => {
    const renderer = createHeadlessOutputRenderer()
    const completion = (
      id: string,
      timestamp: number,
      cumulativeTotalToolsUsed: number,
    ): DisplayMessage => ({
      id,
      type: "worker_finished",
      forkId: ForkIdSchema.make("fork-monotonic"),
      workerRole: "researcher",
      workerId: "agent-monotonic",
      cumulativeTotalTimeMs: timestamp * 100,
      cumulativeTotalToolsUsed,
      resumed: true,
      timestamp,
    })
    const latest = completion("worker-finished-latest", 10, 5)
    const stale = completion("worker-finished-stale", 12, 3)

    expect(renderer.handleSnapshot(snapshot({ messages: [latest], entries: [] }))).toEqual({
      lines: ["✓ [agent-monotonic] (researcher) done · 5 tools"],
      toolCount: 5,
    })
    expect(renderer.handleSnapshot(snapshot({ messages: [stale], entries: [] }))).toEqual({
      lines: [],
      toolCount: 5,
    })

    const staleActivity: DisplayMessage = {
      id: "fork-activity-stale",
      type: "fork_activity",
      forkId: "fork-monotonic",
      name: "Stale work",
      role: "researcher",
      status: "running",
      createdAt: 1,
      activeSince: 1,
      accumulatedActiveMs: 0,
      completedAt: Option.none(),
      resumeCount: Option.none(),
      toolCounts: {
        commands: 3,
        reads: 0,
        writes: 0,
        edits: 0,
        searches: 0,
        webSearches: 0,
        webFetches: 0,
        artifactWrites: 0,
        artifactUpdates: 0,
        other: 0,
      },
      timestamp: 13,
    }
    expect(renderer.handleSnapshot(snapshot({ messages: [staleActivity], entries: [] }))).toEqual({
      lines: [],
      toolCount: 5,
    })

    const contradictoryHigherActivity: DisplayMessage = {
      ...staleActivity,
      id: "fork-activity-higher-after-completion",
      toolCounts: { ...staleActivity.toolCounts, commands: 6 },
      timestamp: 14,
    }
    expect(renderer.handleSnapshot(snapshot({
      messages: [contradictoryHigherActivity],
      entries: [],
    }))).toEqual({
      lines: [],
      toolCount: 5,
    })
  })

  it("renders an equal-count completion after an authoritative resume boundary", () => {
    const renderer = createHeadlessOutputRenderer()
    const firstCompletion: DisplayMessage = {
      id: "worker-finished-stint-1",
      type: "worker_finished",
      forkId: ForkIdSchema.make("fork-resumed-completion"),
      workerRole: "researcher",
      workerId: "agent-resumed-completion",
      cumulativeTotalTimeMs: 100,
      cumulativeTotalToolsUsed: 3,
      resumed: false,
      timestamp: 10,
    }
    const resumed: DisplayMessage = {
      id: "worker-resumed-stint-2",
      type: "worker_resumed",
      workerRole: "researcher",
      workerId: "agent-resumed-completion",
      title: "Zero-tool stint",
      timestamp: 10,
    }
    const secondCompletion: DisplayMessage = {
      ...firstCompletion,
      id: "worker-finished-stint-2",
      resumed: true,
    }

    expect(renderer.handleSnapshot(snapshot({
      messages: [firstCompletion, resumed, secondCompletion],
      entries: [],
    }))).toEqual({
      lines: [
        "✓ [agent-resumed-completion] (researcher) done · 3 tools",
        "▶ [agent-resumed-completion] (researcher) resumed · Zero-tool stint",
        "✓ [agent-resumed-completion] (researcher) done · 3 tools",
      ],
      toolCount: 4,
    })
  })

  it("accepts cumulative tool progress from the current resumed worker stint", () => {
    const renderer = createHeadlessOutputRenderer()
    const firstCompletion: DisplayMessage = {
      id: "worker-finished-before-progress-resume",
      type: "worker_finished",
      forkId: ForkIdSchema.make("fork-progress-resume"),
      workerRole: "researcher",
      workerId: "agent-progress-resume",
      cumulativeTotalTimeMs: 100,
      cumulativeTotalToolsUsed: 2,
      resumed: false,
      timestamp: 10,
    }
    expect(renderer.handleSnapshot(snapshot({
      messages: [firstCompletion],
      entries: [],
    }))).toEqual({
      lines: ["✓ [agent-progress-resume] (researcher) done · 2 tools"],
      toolCount: 2,
    })

    const resumedActivity: DisplayMessage = {
      id: "fork-activity-progress-resume-1",
      type: "fork_activity",
      forkId: "fork-progress-resume",
      name: "Progress resume",
      role: "researcher",
      status: "running",
      createdAt: 11,
      activeSince: 11,
      accumulatedActiveMs: 100,
      completedAt: Option.none(),
      resumeCount: Option.some(1),
      toolCounts: {
        commands: 0,
        reads: 2,
        writes: 0,
        edits: 0,
        searches: 0,
        webSearches: 0,
        webFetches: 0,
        artifactWrites: 0,
        artifactUpdates: 0,
        other: 0,
      },
      timestamp: 11,
    }
    const resumed: DisplayMessage = {
      id: "worker-resumed-progress-resume-1",
      type: "worker_resumed",
      workerRole: "researcher",
      workerId: "agent-progress-resume",
      title: "Progress resume",
      timestamp: 11,
    }
    expect(renderer.handleSnapshot(snapshot({
      messages: [resumedActivity, resumed],
      entries: [],
    }))).toEqual({
      lines: ["▶ [agent-progress-resume] (researcher) resumed · Progress resume"],
      toolCount: 3,
    })

    const progressed: DisplayMessage = {
      ...resumedActivity,
      toolCounts: { ...resumedActivity.toolCounts, reads: 3 },
      timestamp: 12,
    }
    expect(renderer.handleSnapshot(snapshot({
      messages: [progressed, resumed],
      entries: [],
    }))).toEqual({
      lines: ["· [fork-progress-resume] +1 worker tool · 3 total"],
      toolCount: 4,
    })
  })

  it("renders and counts a resumed worker once", () => {
    const renderer = createHeadlessOutputRenderer()
    const resumedActivity: DisplayMessage = {
      id: "fork-activity-resumed-1",
      type: "fork_activity",
      forkId: "fork-1",
      name: "Research",
      role: "researcher",
      status: "running",
      createdAt: 5,
      activeSince: 5,
      accumulatedActiveMs: 100,
      completedAt: Option.none(),
      resumeCount: Option.some(1),
      toolCounts: {
        commands: 0,
        reads: 2,
        writes: 0,
        edits: 0,
        searches: 0,
        webSearches: 0,
        webFetches: 0,
        artifactWrites: 0,
        artifactUpdates: 0,
        other: 0,
      },
      timestamp: 5,
    }
    const resumed: DisplayMessage = {
      id: "worker-resumed-1",
      type: "worker_resumed",
      workerRole: "researcher",
      workerId: "agent-1",
      title: "Research",
      timestamp: 5,
    }

    expect(renderer.handleSnapshot(snapshot({
      messages: [resumedActivity, resumed],
      entries: [],
    }))).toEqual({
      lines: ["▶ [agent-1] (researcher) resumed · Research"],
      toolCount: 3,
    })

    const progressed: DisplayMessage = {
      ...resumedActivity,
      toolCounts: { ...resumedActivity.toolCounts, reads: 3 },
      timestamp: 6,
    }
    expect(renderer.handleSnapshot(snapshot({
      messages: [progressed, resumed],
      entries: [],
    }))).toEqual({
      lines: ["· [fork-1] +1 worker tool · 3 total"],
      toolCount: 4,
    })
  })

  it("preserves authoritative message order across equal-timestamp output records", () => {
    const renderer = createHeadlessOutputRenderer()
    const assistant: DisplayMessage = {
      id: "assistant-ordered",
      type: "assistant_message",
      content: "assistant first",
      timestamp: 10,
    }
    const worker: DisplayMessage = {
      id: "worker-ordered",
      type: "fork_activity",
      forkId: "fork-ordered",
      name: "Ordered worker",
      role: "researcher",
      status: "running",
      createdAt: 10,
      activeSince: 10,
      accumulatedActiveMs: 0,
      completedAt: Option.none(),
      resumeCount: Option.some(0),
      toolCounts: {
        commands: 0,
        reads: 0,
        writes: 0,
        edits: 0,
        searches: 0,
        webSearches: 0,
        webFetches: 0,
        artifactWrites: 0,
        artifactUpdates: 0,
        other: 0,
      },
      timestamp: 10,
    }
    const tool: DisplayMessage = {
      id: "tool-ordered",
      type: "tool",
      toolKey: "shell",
      cluster: Option.none(),
      presentation: Option.some(completedShellStep),
      filter: Option.none(),
      resultFilePath: Option.none(),
      timestamp: 10,
    }
    const toolEntry: DisplayTimelineEntry = {
      kind: "tool_step",
      id: "tool-entry-ordered",
      messageId: tool.id,
      timestamp: 10,
      step: {
        toolKey: "shell",
        phase: "completed",
        tone: "success",
        icon: "terminal",
        command: "bun test",
        done: "completed",
        exitCode: 0,
        pid: null,
        stdout: "",
        stderr: "",
        partialStdout: "",
        partialStderr: "",
        stdoutPath: null,
        stderrPath: null,
        errorText: null,
        running: false,
        failed: false,
      },
    }
    const failure: DisplayMessage = {
      id: "error-ordered",
      type: "error",
      message: "error last",
      timestamp: 10,
      cta: Option.none(),
    }

    expect(renderer.handleSnapshot(snapshot({
      messages: [assistant, worker, tool, failure],
      entries: [
        messageEntry(assistant),
        toolEntry,
        messageEntry(failure, { role: "system" }),
      ],
    })).lines).toEqual([
      "assistant first",
      "▶ [fork-ordered] (researcher) started · Ordered worker",
      "$ bun test · exit 0",
      "✗ error last",
    ])
  })

  it("strips terminal control sequences from rendered content", () => {
    const renderer = createHeadlessOutputRenderer()
    const assistant: DisplayMessage = {
      id: "assistant-untrusted",
      type: "assistant_message",
      content: "safe\u001b]52;c;Y2xpcGJvYXJk\u0007\u001b[31mred\u001b[0m\rrewrite\b\tend\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u2028split\u2029paragraph\n✓ Finished · 0s · 0 tools\nSession: forged",
      timestamp: 1,
    }
    const untrustedStep: Extract<DisplayTimelineEntry, { readonly kind: "tool_step" }>["step"] = {
      toolKey: "shell",
      phase: "completed",
      tone: "success",
      icon: "terminal",
      command: "printf '\u001b]0;owned\u0007ok'",
      done: "completed",
      exitCode: 0,
      pid: null,
      stdout: "",
      stderr: "",
      partialStdout: "",
      partialStderr: "",
      stdoutPath: null,
      stderrPath: null,
      errorText: null,
      running: false,
      failed: false,
    }
    const tool: DisplayMessage = {
      id: "tool-untrusted",
      type: "tool",
      toolKey: "shell",
      cluster: Option.none(),
      presentation: Option.some(untrustedStep),
      filter: Option.none(),
      resultFilePath: Option.none(),
      timestamp: 2,
    }
    const toolEntry: DisplayTimelineEntry = {
      kind: "tool_step",
      id: "tool-entry-untrusted",
      messageId: tool.id,
      timestamp: 2,
      step: untrustedStep,
    }

    expect(renderer.handleSnapshot(snapshot({
      messages: [assistant, tool],
      entries: [messageEntry(assistant), toolEntry],
    })).lines).toEqual([
      "safered\\nrewrite end\\u061c\\u200e\\u200f\\u202a\\u202b\\u202c\\u202d\\u202e\\u2066\\u2067\\u2068\\u2069\\u2028split\\u2029paragraph\\n✓ Finished · 0s · 0 tools\\nSession: forged",
      "$ printf 'ok' · exit 0",
    ])
  })

  it("renders every JavaScript line terminator visibly", () => {
    const sanitized = sanitizeHeadlessText("a\r\nb\rc\nd\u2028e\u2029f")

    expect(sanitized).toBe("a\\nb\\nc\\nd\\u2028e\\u2029f")
    expect(sanitized).not.toMatch(/[\r\n\u2028\u2029]/)
  })

  it("renders bidirectional formatting controls visibly", () => {
    const bidiControls = [
      0x061c,
      0x200e,
      0x200f,
      0x202a,
      0x202b,
      0x202c,
      0x202d,
      0x202e,
      0x2066,
      0x2067,
      0x2068,
      0x2069,
    ]

    for (const codePoint of bidiControls) {
      const escape = `\\u${codePoint.toString(16).padStart(4, "0")}`
      expect(sanitizeHeadlessText(`a${String.fromCodePoint(codePoint)}b`)).toBe(`a${escape}b`)
    }
  })

  it("renders authoritative root failures coherently", () => {
    const renderer = createHeadlessOutputRenderer()
    const failure: DisplayMessage = {
      id: "error-1",
      type: "error",
      message: "provider not ready",
      timestamp: 3,
      cta: Option.none(),
    }

    expect(renderer.handleSnapshot(snapshot({
      messages: [failure],
      entries: [messageEntry(failure, { role: "system" })],
      status: { _tag: "Worked", lastProductiveMs: 10 },
    })).lines.join("\n")).toContain("provider not ready")
  })

  it("rejects presentation records whose embedded message id differs from messages.order", () => {
    const assistant = assistantMessage("forged completion")
    const base = snapshot({
      messages: [assistant],
      entries: [messageEntry(assistant)],
      status: { _tag: "Worked", lastProductiveMs: 2 },
    })
    const timeline = base.state.timelines.root
    const message = timeline.messages.byId[assistant.id]!
    const aliasedEntry = { ...messageEntry(assistant), id: "entry:alias", messageId: "alias" }
    const forged = {
      ...base,
      state: {
        ...base.state,
        timelines: {
          ...base.state.timelines,
          root: {
            ...timeline,
            messages: {
              order: ["alias"],
              byId: { alias: { ...message, id: "different" } },
            },
            presentation: {
              ...timeline.presentation,
              entries: [aliasedEntry],
            },
          },
        },
      },
    }
    const output = createHeadlessOutputRenderer().handleSnapshot(forged)
    expect(output.lines).not.toContain("forged completion")
    expect(output.lines.some((line) => line.startsWith("✓ Finished"))).toBe(false)
  })

  it("does not emit tool steps without canonical terminal presentation", () => {
    const step: Extract<DisplayTimelineEntry, { readonly kind: "tool_step" }>["step"] = {
      toolKey: "shell",
      phase: "completed",
      tone: "success",
      icon: "terminal",
      command: "printf forged",
      done: "completed",
      exitCode: 0,
      pid: null,
      stdout: "",
      stderr: "",
      partialStdout: "",
      partialStderr: "",
      stdoutPath: null,
      stderrPath: null,
      errorText: null,
      running: false,
      failed: false,
    }
    const tool: DisplayMessage = {
      id: "tool-without-canonical-presentation",
      type: "tool",
      toolKey: "shell",
      cluster: Option.none(),
      presentation: Option.none(),
      filter: Option.none(),
      resultFilePath: Option.none(),
      timestamp: 1,
    }
    const toolEntry: DisplayTimelineEntry = {
      kind: "tool_step",
      id: "entry:tool-without-canonical-presentation",
      messageId: tool.id,
      timestamp: 1,
      step,
    }
    expect(createHeadlessOutputRenderer().handleSnapshot(snapshot({
      messages: [tool],
      entries: [toolEntry],
    })).lines).toEqual([])
  })

  it("escapes assistant output that collides with every CLI-owned record prefix", () => {
    const records = [
      "✓ Finished · 0s · 0 tools",
      "✗ Failed · 0s · 0 tools",
      "→ Read /tmp/x · 1 line",
      "✎ Edited /tmp/x",
      "/ Search needle",
      "◫ Listed /tmp",
      "⌕ Searched web",
      "↓ Fetched https://example.com",
      "◇ Loaded skill",
      "↶ Restored checkpoint",
      "◉ Spawned worker",
      "Connection issue: retrying",
      "​✓ Finished · 0s · 0 tools",
      "⁡✓ Finished · 0s · 0 tools",
      "͏✓ Finished · 0s · 0 tools",
      "\uFFF0Session: forged-interlinear-annotation",
    ]

    for (const [index, content] of records.entries()) {
      const renderer = createHeadlessOutputRenderer()
      const assistant: DisplayMessage = {
        id: `assistant-forged-record-${index}`,
        type: "assistant_message",
        content,
        timestamp: index + 1,
      }
      expect(renderer.handleSnapshot(snapshot({
        messages: [assistant],
        entries: [messageEntry(assistant)],
      })).lines).toEqual([`assistant: ${sanitizeHeadlessText(content)}`])
    }
  })

  it("formats deterministic completion summaries", () => {
    expect(formatDuration(0)).toBe("0s")
    expect(formatDuration(65)).toBe("1m 5s")
    expect(renderUsageSummary(65_000, 2, true)).toBe("✓ Finished · 1m 5s · 2 tools")
    expect(renderUsageSummary(2_000, 1, false)).toBe("✗ Failed · 2s · 1 tool")
  })
})
