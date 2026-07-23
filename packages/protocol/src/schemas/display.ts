import { Schema, Option } from "effect"
import { JsonValueSchema } from "@magnitudedev/utils/schema"
import { DisplayAttachment as DisplayAttachmentSchema, ImageAttachment, MentionAttachment } from "./attachments"
import type { DisplayAttachment } from "./attachments"

// ---------------------------------------------------------------------------
// Display Attachments — consolidated with protocol Attachment
// ---------------------------------------------------------------------------

export const DisplayAttachmentImage = ImageAttachment
export type DisplayAttachmentImage = ImageAttachment

export const DisplayAttachmentMention = MentionAttachment
export type DisplayAttachmentMention = MentionAttachment

export { DisplayAttachmentSchema as DisplayAttachment }

// ---------------------------------------------------------------------------
// Message variants
// ---------------------------------------------------------------------------

export const UserMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("user_message"),
  content: Schema.String,
  timestamp: Schema.Number,
  taskMode: Schema.Boolean,
  attachments: Schema.Array(DisplayAttachmentSchema)
})
export type UserMessage = Schema.Schema.Type<typeof UserMessage>

export const QueuedUserMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("queued_user_message"),
  content: Schema.String,
  timestamp: Schema.Number,
  taskMode: Schema.Boolean,
  attachments: Schema.Array(DisplayAttachmentSchema)
})
export type QueuedUserMessage = Schema.Schema.Type<typeof QueuedUserMessage>

export const AssistantMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("assistant_message"),
  content: Schema.String,
  timestamp: Schema.Number
})
export type AssistantMessage = Schema.Schema.Type<typeof AssistantMessage>

export const UserBashCommandMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("user_bash_command"),
  command: Schema.String,
  cwd: Schema.String,
  exitCode: Schema.Number,
  stdout: Schema.String,
  stderr: Schema.String,
  timestamp: Schema.Number
})
export type UserBashCommandMessage = Schema.Schema.Type<typeof UserBashCommandMessage>

export const ThinkingMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("thinking"),
  content: Schema.String,
  label: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  timestamp: Schema.Number
})
export type ThinkingMessage = Schema.Schema.Type<typeof ThinkingMessage>

export const ToolMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("tool"),
  toolKey: Schema.String,
  cluster: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  presentation: Schema.optionalWith(Schema.suspend(() => ToolStepPresentation), { as: "Option", exact: true }),
  filter: Schema.optionalWith(Schema.Union(Schema.String, Schema.Null), { as: "Option", exact: true }),
  resultFilePath: Schema.optionalWith(Schema.Union(Schema.String, Schema.Null), { as: "Option", exact: true }),
  timestamp: Schema.Number
})
export type ToolMessage = Schema.Schema.Type<typeof ToolMessage>

export const StatusIndicatorMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("status_indicator"),
  message: Schema.String,
  style: Schema.Literal("dim"),
  timestamp: Schema.Number
})
export type StatusIndicatorMessage = Schema.Schema.Type<typeof StatusIndicatorMessage>

export const GoalStatusMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("goal_status"),
  status: Schema.Literal("started", "finished"),
  objective: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  evidence: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  timestamp: Schema.Number
})
export type GoalStatusMessage = Schema.Schema.Type<typeof GoalStatusMessage>

export const WorkerResumedMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("worker_resumed"),
  workerRole: Schema.String,
  workerId: Schema.String,
  title: Schema.String,
  timestamp: Schema.Number
})
export type WorkerResumedMessage = Schema.Schema.Type<typeof WorkerResumedMessage>

export const WorkerFinishedMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("worker_finished"),
  workerRole: Schema.String,
  workerId: Schema.String,
  cumulativeTotalTimeMs: Schema.Number,
  cumulativeTotalToolsUsed: Schema.Number,
  resumed: Schema.Boolean,
  timestamp: Schema.Number
})
export type WorkerFinishedMessage = Schema.Schema.Type<typeof WorkerFinishedMessage>

export const WorkerKilledMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("worker_killed"),
  workerRole: Schema.String,
  workerId: Schema.String,
  title: Schema.String,
  timestamp: Schema.Number
})
export type WorkerKilledMessage = Schema.Schema.Type<typeof WorkerKilledMessage>

export const WorkerUserKilledMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("worker_user_killed"),
  workerRole: Schema.String,
  workerId: Schema.String,
  title: Schema.String,
  timestamp: Schema.Number
})
export type WorkerUserKilledMessage = Schema.Schema.Type<typeof WorkerUserKilledMessage>

export const InterruptedMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("interrupted"),
  timestamp: Schema.Number,
  context: Schema.Literal("root", "fork"),
  allKilled: Schema.optionalWith(Schema.Boolean, { as: "Option", exact: true })
})
export type InterruptedMessage = Schema.Schema.Type<typeof InterruptedMessage>

export const ErrorDisplayMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("error"),
  message: Schema.String,
  timestamp: Schema.Number,
  cta: Schema.optionalWith(Schema.Union(
    Schema.Struct({
      kind: Schema.Literal("action"),
      actionId: Schema.String,
      label: Schema.String,
      chord: Schema.String
    }),
    Schema.Struct({
      kind: Schema.Literal("url"),
      url: Schema.String,
      label: Schema.String
    })
  ), { as: "Option", exact: true })
})
export type ErrorDisplayMessage = Schema.Schema.Type<typeof ErrorDisplayMessage>

export const ForkResultMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("fork_result"),
  forkId: Schema.String,
  task: Schema.String,
  result: JsonValueSchema,
  timestamp: Schema.Number
})
export type ForkResultMessage = Schema.Schema.Type<typeof ForkResultMessage>

export const ForkActivityToolCounts = Schema.Struct({
  commands: Schema.Number,
  reads: Schema.Number,
  writes: Schema.Number,
  edits: Schema.Number,
  searches: Schema.Number,
  webSearches: Schema.Number,
  webFetches: Schema.Number,
  artifactWrites: Schema.Number,
  artifactUpdates: Schema.Number,
  other: Schema.Number
})
export type ForkActivityToolCounts = Schema.Schema.Type<typeof ForkActivityToolCounts>

export const ForkActivityMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("fork_activity"),
  forkId: Schema.String,
  name: Schema.String,
  role: Schema.String,
  status: Schema.Literal("running", "completed"),
  createdAt: Schema.Number,
  activeSince: Schema.Number,
  accumulatedActiveMs: Schema.Number,
  completedAt: Schema.optionalWith(Schema.Union(Schema.Number, Schema.Null), { as: "Option", exact: true }),
  resumeCount: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  toolCounts: ForkActivityToolCounts,
  timestamp: Schema.Number
})
export type ForkActivityMessage = Schema.Schema.Type<typeof ForkActivityMessage>

export const AgentCommunicationMessage = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("agent_communication"),
  streamId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  direction: Schema.Literal("to_agent", "from_agent"),
  agentId: Schema.String,
  agentName: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  agentRole: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  forkId: Schema.Union(Schema.String, Schema.Null),
  content: Schema.String,
  preview: Schema.String,
  timestamp: Schema.Number,
  status: Schema.optionalWith(Schema.Literal("streaming", "completed"), { as: "Option", exact: true })
})
export type AgentCommunicationMessage = Schema.Schema.Type<typeof AgentCommunicationMessage>

export const DisplayMessage = Schema.Union(
  UserMessage,
  QueuedUserMessage,
  UserBashCommandMessage,
  AssistantMessage,
  ThinkingMessage,
  ToolMessage,
  StatusIndicatorMessage,
  GoalStatusMessage,
  WorkerResumedMessage,
  WorkerFinishedMessage,
  WorkerKilledMessage,
  WorkerUserKilledMessage,
  InterruptedMessage,
  ErrorDisplayMessage,
  ForkResultMessage,
  ForkActivityMessage,
  AgentCommunicationMessage
)
export type DisplayMessage = Schema.Schema.Type<typeof DisplayMessage>

// ---------------------------------------------------------------------------
// Pending inbound communications
// ---------------------------------------------------------------------------

export const PendingInboundCommunication = Schema.Struct({
  id: Schema.String,
  source: Schema.Literal("agent", "user"),
  direction: Schema.Literal("from_agent", "to_agent"),
  agentId: Schema.String,
  agentName: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  agentRole: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  forkId: Schema.Union(Schema.String, Schema.Null),
  content: Schema.String,
  preview: Schema.String,
  timestamp: Schema.Number,
  arrivedAtTurnId: Schema.Union(Schema.String, Schema.Null),
  readAtTurnId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  requiresAdvisor: Schema.optionalWith(Schema.Literal(true), { as: "Option", exact: true })
})
export type PendingInboundCommunication = Schema.Schema.Type<typeof PendingInboundCommunication>

// ---------------------------------------------------------------------------
// Actor activity
// ---------------------------------------------------------------------------

export const DisplayActivity = Schema.Union(
  Schema.Struct({
    kind: Schema.Literal("thinking"),
    message: Schema.String
  }),
  Schema.Struct({
    kind: Schema.Literal("tool"),
    message: Schema.String,
    decorator: Schema.optionalWith(Schema.Literal("spinner"), { as: "Option", exact: true })
  }),
  Schema.Struct({
    kind: Schema.Literal("advisor"),
    message: Schema.String
  })
)
export type DisplayActivity = Schema.Schema.Type<typeof DisplayActivity>

export const StatusBarActivity = DisplayActivity
export type StatusBarActivity = Schema.Schema.Type<typeof StatusBarActivity>

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

export const TaskAssignee = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("none") }),
  Schema.Struct({
    kind: Schema.Literal("user"),
    label: Schema.Literal("user"),
    tone: Schema.Literal("warning")
  }),
  Schema.Struct({
    kind: Schema.Literal("worker"),
    variant: Schema.Literal("spawning"),
    label: Schema.String,
    icon: Schema.Literal("+"),
    tone: Schema.Literal("active"),
    interactiveForkId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    timer: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
    resumed: Schema.Literal(false),
    continuityKey: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    ghostEligible: Schema.Literal(false)
  }),
  Schema.Struct({
    kind: Schema.Literal("actor"),
    actorKey: Schema.String,
    taskState: Schema.Literal("assigned", "killing"),
    timer: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  })
)
export type TaskAssignee = Schema.Schema.Type<typeof TaskAssignee>

export const TaskWorkerState = Schema.Union(
  Schema.Struct({ status: Schema.Literal("unassigned") }),
  Schema.Struct({
    status: Schema.Literal("spawning"),
    toolCallId: Schema.String,
    role: Schema.optionalWith(Schema.String, { as: "Option", exact: true })
  }),
  Schema.Struct({
    status: Schema.Literal("working"),
    forkId: Schema.String,
    activeSince: Schema.Number,
    accumulatedMs: Schema.Number,
    resumeCount: Schema.Number
  }),
  Schema.Struct({
    status: Schema.Literal("idle"),
    forkId: Schema.String,
    accumulatedMs: Schema.Number,
    completedAt: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
    resumeCount: Schema.Number
  }),
  Schema.Struct({
    status: Schema.Literal("killing"),
    forkId: Schema.String,
    toolCallId: Schema.String
  })
)
export type TaskWorkerState = Schema.Schema.Type<typeof TaskWorkerState>

export const TaskDisplayRow = Schema.Struct({
  rowId: Schema.String,
  kind: Schema.Literal("task"),
  taskId: Schema.String,
  title: Schema.String,
  status: Schema.Literal("pending", "completed"),
  parentId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  depth: Schema.Number,
  updatedAt: Schema.Number,
  assignee: TaskAssignee,
})
export type TaskDisplayRow = Schema.Schema.Type<typeof TaskDisplayRow>

// ---------------------------------------------------------------------------
// Canonical display state
// ---------------------------------------------------------------------------

export const ROOT_FORK_KEY = "root"
export type ForkKey = string

export function forkIdToKey(forkId: string | null): ForkKey {
  return forkId ?? ROOT_FORK_KEY
}

export function forkKeyToForkId(key: ForkKey): string | null {
  return key === ROOT_FORK_KEY ? null : key
}

export const DisplaySession = Schema.Struct({
  sessionId: Schema.String,
  title: Schema.Union(Schema.String, Schema.Null),
  cwd: Schema.String
})
export type DisplaySession = Schema.Schema.Type<typeof DisplaySession>

export const TimelineActivity = DisplayActivity
export type TimelineActivity = Schema.Schema.Type<typeof TimelineActivity>

export const ContextUsageDisplay = Schema.Struct({
  tokenEstimate: Schema.Number,
  isCompacting: Schema.Boolean
})
export type ContextUsageDisplay = Schema.Schema.Type<typeof ContextUsageDisplay>

export const DisplayActorWork = Schema.Struct({
  phase: Schema.Literal("idle", "working", "worked", "interrupted"),
  activeSince: Schema.Union(Schema.Number, Schema.Null),
  lastWorkMs: Schema.Number,
  accumulatedMs: Schema.Number,
  resumeCount: Schema.Number,
  activity: Schema.Union(DisplayActivity, Schema.Null),
  activeChildCount: Schema.Number
})
export type DisplayActorWork = Schema.Schema.Type<typeof DisplayActorWork>

export const DisplayActor = Schema.Struct({
  kind: Schema.Literal("root", "worker"),
  name: Schema.String,
  role: Schema.String,
  parentActorKey: Schema.Union(Schema.String, Schema.Null),
  taskId: Schema.Union(Schema.String, Schema.Null),
  work: DisplayActorWork,
  context: ContextUsageDisplay
})
export type DisplayActor = Schema.Schema.Type<typeof DisplayActor>

/**
 * Normalized timeline messages — heavy content keyed by message id, order as
 * a cheap positional array of ids. Organized so generic JSON patches stay
 * efficient under window movement: growing or sliding the window only churns
 * `order` (short strings); each message body crosses the wire exactly once
 * via `byId`. Mirrors the DisplayTasks byId/order pattern.
 */
export const DisplayTimelineMessages = Schema.Struct({
  byId: Schema.Record({ key: Schema.String, value: DisplayMessage }),
  order: Schema.Array(Schema.String),
})
export type DisplayTimelineMessages = Schema.Schema.Type<typeof DisplayTimelineMessages>

export const DisplayTimelinePresentationMode = Schema.Literal("default", "transcript")
export type DisplayTimelinePresentationMode = Schema.Schema.Type<typeof DisplayTimelinePresentationMode>

export const DisplayTimelineWindowInfo = Schema.Struct({
  start: Schema.Number,
  end: Schema.Number,
  totalCount: Schema.Number,
  hasMoreBefore: Schema.Boolean,
  hasMoreAfter: Schema.Boolean,
})
export type DisplayTimelineWindowInfo = Schema.Schema.Type<typeof DisplayTimelineWindowInfo>

// ---------------------------------------------------------------------------
// Tool presentation — clean per-tool discriminated union
// ---------------------------------------------------------------------------
// The presentation is the display view of the execution state. Each variant
// is discriminated by `toolKey` and carries exactly the fields renderers
// consume. Built by the projection from the typed ToolState. No generic
// segment list. No stringly-typed lookups. No parallel `family` taxonomy.
//
// `ToolState` (in the agent package) is the execution state model — it has
// fields renderers never touch (oldText, newText, body, charCount, …) and
// lacks display-derived fields (tone, icon, running, failed). The
// presentation is a different view, not a duplicate: different purpose,
// different fields, same source.

export const ToolPhase = Schema.Literal(
  "streaming", "executing", "completed", "error", "rejected", "interrupted",
)
export type ToolPhase = Schema.Schema.Type<typeof ToolPhase>

export const ToolTone = Schema.Literal("neutral", "info", "success", "warning", "error", "muted")
export type ToolTone = Schema.Schema.Type<typeof ToolTone>

export const ToolIcon = Schema.Literal(
  "file", "edit", "diff", "search", "tree", "terminal",
  "web", "download", "skill", "worker", "checkpoint", "tool", "image",
)
export type ToolIcon = Schema.Schema.Type<typeof ToolIcon>

export const ToolDiffHunk = Schema.Struct({
  startLine: Schema.Number,
  removedLines: Schema.Array(Schema.String),
  addedLines: Schema.Array(Schema.String),
  contextBefore: Schema.Array(Schema.String),
  contextAfter: Schema.Array(Schema.String),
  streamingCursor: Schema.Boolean,
})
export type ToolDiffHunk = Schema.Schema.Type<typeof ToolDiffHunk>

export const ToolFileRef = Schema.Struct({
  path: Schema.String,
  displayPath: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  section: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})
export type ToolFileRef = Schema.Schema.Type<typeof ToolFileRef>

export const ToolDiffSlot = Schema.Struct({
  hunks: Schema.Array(ToolDiffHunk),
})
export type ToolDiffSlot = Schema.Schema.Type<typeof ToolDiffSlot>

// ── Per-tool presentation variants ─────────────────────────────
// Each variant is a typed struct carrying exactly what renderers consume.
// Discriminated on `toolKey` — no `family` enum, no `detail: Segment[]`.

export const ShellPresentation = Schema.Struct({
  toolKey: Schema.Literal("shell"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("terminal"),
  command: Schema.String,
  done: Schema.Union(Schema.Literal("completed"), Schema.Literal("detached"), Schema.Null),
  exitCode: Schema.Union(Schema.Number, Schema.Null),
  pid: Schema.Union(Schema.Number, Schema.Null),
  stdout: Schema.String,
  stderr: Schema.String,
  partialStdout: Schema.String,
  partialStderr: Schema.String,
  stdoutPath: Schema.Union(Schema.String, Schema.Null),
  stderrPath: Schema.Union(Schema.String, Schema.Null),
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type ShellPresentation = Schema.Schema.Type<typeof ShellPresentation>

export const FileWritePresentation = Schema.Struct({
  toolKey: Schema.Literal("fileWrite"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("edit"),
  path: Schema.Union(Schema.String, Schema.Null),
  displayPath: Schema.Union(Schema.String, Schema.Null),
  lineCount: Schema.Number,
  isScratchpad: Schema.Boolean,
  diff: Schema.Union(ToolDiffSlot, Schema.Null),
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type FileWritePresentation = Schema.Schema.Type<typeof FileWritePresentation>

export const FileEditPresentation = Schema.Struct({
  toolKey: Schema.Literal("fileEdit"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("diff"),
  path: Schema.Union(Schema.String, Schema.Null),
  displayPath: Schema.Union(Schema.String, Schema.Null),
  addedCount: Schema.Number,
  removedCount: Schema.Number,
  isScratchpad: Schema.Boolean,
  streamingTarget: Schema.Union(Schema.Literal("old"), Schema.Literal("new"), Schema.Null),
  diff: Schema.Union(ToolDiffSlot, Schema.Null),
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type FileEditPresentation = Schema.Schema.Type<typeof FileEditPresentation>

export const FileReadPresentation = Schema.Struct({
  toolKey: Schema.Literal("fileRead"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("file"),
  path: Schema.Union(Schema.String, Schema.Null),
  lineCount: Schema.Union(Schema.Number, Schema.Null),
  offset: Schema.Union(Schema.Number, Schema.Null),
  limit: Schema.Union(Schema.Number, Schema.Null),
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type FileReadPresentation = Schema.Schema.Type<typeof FileReadPresentation>

export const FileSearchPresentation = Schema.Struct({
  toolKey: Schema.Literal("fileSearch"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("search"),
  pattern: Schema.Union(Schema.String, Schema.Null),
  matchCount: Schema.Number,
  fileCount: Schema.Number,
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type FileSearchPresentation = Schema.Schema.Type<typeof FileSearchPresentation>

export const FileTreePresentation = Schema.Struct({
  toolKey: Schema.Literal("fileTree"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("tree"),
  path: Schema.String,
  fileCount: Schema.Number,
  dirCount: Schema.Number,
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type FileTreePresentation = Schema.Schema.Type<typeof FileTreePresentation>

export const FileViewPresentation = Schema.Struct({
  toolKey: Schema.Literal("fileView"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("file"),
  path: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type FileViewPresentation = Schema.Schema.Type<typeof FileViewPresentation>

export const WebSearchPresentation = Schema.Struct({
  toolKey: Schema.Literal("webSearch"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("web"),
  query: Schema.Union(Schema.String, Schema.Null),
  sourceCount: Schema.Number,
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type WebSearchPresentation = Schema.Schema.Type<typeof WebSearchPresentation>

export const WebFetchPresentation = Schema.Struct({
  toolKey: Schema.Literal("webFetch"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("download"),
  url: Schema.Union(Schema.String, Schema.Null),
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type WebFetchPresentation = Schema.Schema.Type<typeof WebFetchPresentation>

export const SkillPresentation = Schema.Struct({
  toolKey: Schema.Literal("skill"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("skill"),
  skillName: Schema.Union(Schema.String, Schema.Null),
  skillPath: Schema.Union(Schema.String, Schema.Null),
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type SkillPresentation = Schema.Schema.Type<typeof SkillPresentation>

export const CheckpointPresentation = Schema.Struct({
  toolKey: Schema.Literal("checkpointChanges", "checkpointRollback"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("checkpoint"),
  isRollback: Schema.Boolean,
  since: Schema.Union(Schema.String, Schema.Null),
  fileCount: Schema.Number,
  additions: Schema.Number,
  deletions: Schema.Number,
  files: Schema.Array(ToolFileRef),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type CheckpointPresentation = Schema.Schema.Type<typeof CheckpointPresentation>

export const SpawnWorkerPresentation = Schema.Struct({
  toolKey: Schema.Literal("spawnWorker"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("worker"),
  agentId: Schema.Union(Schema.String, Schema.Null),
  role: Schema.Union(Schema.String, Schema.Null),
  title: Schema.Union(Schema.String, Schema.Null),
  message: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type SpawnWorkerPresentation = Schema.Schema.Type<typeof SpawnWorkerPresentation>

export const QueryImagePresentation = Schema.Struct({
  toolKey: Schema.Literal("queryImage"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("image"),
  path: Schema.Union(Schema.String, Schema.Null),
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type QueryImagePresentation = Schema.Schema.Type<typeof QueryImagePresentation>

export const GenericToolPresentation = Schema.Struct({
  toolKey: Schema.String,
  phase: ToolPhase,
  tone: ToolTone,
  icon: Schema.Literal("tool"),
  label: Schema.String,
  errorText: Schema.Union(Schema.String, Schema.Null),
  running: Schema.Boolean,
  failed: Schema.Boolean,
})
export type GenericToolPresentation = Schema.Schema.Type<typeof GenericToolPresentation>

// ── The union ──────────────────────────────────────────────────

export const ToolStepPresentation = Schema.Union(
  ShellPresentation,
  FileWritePresentation,
  FileEditPresentation,
  FileReadPresentation,
  FileSearchPresentation,
  FileTreePresentation,
  FileViewPresentation,
  WebSearchPresentation,
  WebFetchPresentation,
  SkillPresentation,
  CheckpointPresentation,
  SpawnWorkerPresentation,
  QueryImagePresentation,
  GenericToolPresentation,
)
export type ToolStepPresentation = Schema.Schema.Type<typeof ToolStepPresentation>

export const ToolSummaryDetailItem = Schema.Struct({
  kind: Schema.Literal("path", "pattern", "query"),
  text: Schema.String,
})
export type ToolSummaryDetailItem = Schema.Schema.Type<typeof ToolSummaryDetailItem>

export const ToolSummaryPresentation = Schema.Struct({
  toolKey: Schema.Literal("fileRead", "fileSearch", "webSearch", "webFetch", "fileTree", "fileView"),
  phase: ToolPhase,
  tone: ToolTone,
  icon: ToolIcon,
  count: Schema.Number,
  running: Schema.Boolean,
  failed: Schema.Boolean,
  matchCount: Schema.Union(Schema.Number, Schema.Null),
  fileCount: Schema.Union(Schema.Number, Schema.Null),
  sourceCount: Schema.Union(Schema.Number, Schema.Null),
  detail: Schema.Array(ToolSummaryDetailItem),
})
export type ToolSummaryPresentation = Schema.Schema.Type<typeof ToolSummaryPresentation>

export const DisplayMessageTimelineEntry = Schema.Struct({
  kind: Schema.Literal("message"),
  id: Schema.String,
  messageId: Schema.String,
  timestamp: Schema.Number,
  role: Schema.Literal("user", "assistant", "system", "agent"),
  streaming: Schema.Boolean,
  interrupted: Schema.Boolean,
  nextMessageInterrupted: Schema.Boolean,
})
export type DisplayMessageTimelineEntry = Schema.Schema.Type<typeof DisplayMessageTimelineEntry>

export const DisplayToolSummaryTimelineEntry = Schema.Struct({
  kind: Schema.Literal("tool_summary"),
  id: Schema.String,
  timestamp: Schema.Number,
  messageIds: Schema.Array(Schema.String),
  summary: ToolSummaryPresentation,
})
export type DisplayToolSummaryTimelineEntry = Schema.Schema.Type<typeof DisplayToolSummaryTimelineEntry>

export const DisplayToolStepTimelineEntry = Schema.Struct({
  kind: Schema.Literal("tool_step"),
  id: Schema.String,
  timestamp: Schema.Number,
  messageId: Schema.String,
  step: ToolStepPresentation,
})
export type DisplayToolStepTimelineEntry = Schema.Schema.Type<typeof DisplayToolStepTimelineEntry>

export const DisplayTimelineEntry = Schema.Union(
  DisplayMessageTimelineEntry,
  DisplayToolSummaryTimelineEntry,
  DisplayToolStepTimelineEntry,
)
export type DisplayTimelineEntry = Schema.Schema.Type<typeof DisplayTimelineEntry>

export const DisplayTimelineStatusSlot = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("none") }),
  Schema.Struct({
    kind: Schema.Literal("interrupted"),
    messageId: Schema.String,
    context: Schema.Literal("root", "fork"),
    allKilled: Schema.Boolean,
  }),
)
export type DisplayTimelineStatusSlot = Schema.Schema.Type<typeof DisplayTimelineStatusSlot>

export const DisplayTimelinePresentation = Schema.Struct({
  mode: DisplayTimelinePresentationMode,
  entries: Schema.Array(DisplayTimelineEntry),
  statusSlot: DisplayTimelineStatusSlot,
})
export type DisplayTimelinePresentation = Schema.Schema.Type<typeof DisplayTimelinePresentation>

export const DisplayTimeline = Schema.Struct({
  mode: Schema.Literal("idle", "streaming"),
  messages: DisplayTimelineMessages,
  streamingMessageId: Schema.Union(Schema.String, Schema.Null),
  window: Schema.optionalWith(DisplayTimelineWindowInfo, {
    default: () => ({
      start: 0,
      end: 0,
      totalCount: 0,
      hasMoreBefore: false,
      hasMoreAfter: false,
    }),
  }),
  presentation: Schema.optionalWith(DisplayTimelinePresentation, {
    default: () => ({
      mode: "default" as const,
      entries: [],
      statusSlot: { kind: "none" as const },
    }),
  }),
})
export type DisplayTimeline = Schema.Schema.Type<typeof DisplayTimeline>

export const DisplayAgent = Schema.Struct({
  name: Schema.String,
  role: Schema.String,
  status: Schema.optionalWith(Schema.Literal("working", "idle", "killed"), { as: "Option", exact: true })
})
export type DisplayAgent = Schema.Schema.Type<typeof DisplayAgent>

export const DisplayTasks = Schema.Struct({
  byId: Schema.Record({ key: Schema.String, value: TaskDisplayRow }),
  order: Schema.Array(Schema.String),
  summary: Schema.Struct({
    totalCount: Schema.Number,
    completedCount: Schema.Number,
    incompleteCount: Schema.Number
  })
})
export type DisplayTasks = Schema.Schema.Type<typeof DisplayTasks>

export const DisplayState = Schema.Struct({
  session: DisplaySession,
  timelines: Schema.Record({ key: Schema.String, value: DisplayTimeline }),
  actors: Schema.Record({ key: Schema.String, value: DisplayActor }),
  agents: Schema.Record({ key: Schema.String, value: DisplayAgent }),
  tasks: DisplayTasks
})
export type DisplayState = Schema.Schema.Type<typeof DisplayState>

export const DisplayTimelineWindowShape = Schema.Union(
  Schema.Struct({
    kind: Schema.Literal("tail"),
    limit: Schema.Number,
    live: Schema.Boolean,
    presentation: Schema.optionalWith(DisplayTimelinePresentationMode, { default: () => "default" as const })
  }),
  Schema.Struct({
    kind: Schema.Literal("range"),
    start: Schema.Number,
    limit: Schema.Number,
    live: Schema.Boolean,
    presentation: Schema.optionalWith(DisplayTimelinePresentationMode, { default: () => "default" as const })
  })
)
export type DisplayTimelineWindowShape = Schema.Schema.Type<typeof DisplayTimelineWindowShape>

export const DisplayViewShape = Schema.Struct({
  timelines: Schema.Record({ key: Schema.String, value: DisplayTimelineWindowShape })
})
export type DisplayViewShape = Schema.Schema.Type<typeof DisplayViewShape>

export const sameDisplayTimelineWindowShape = (
  left: DisplayTimelineWindowShape,
  right: DisplayTimelineWindowShape
): boolean => {
  if (left.kind === 'tail') {
    return right.kind === 'tail' &&
      left.limit === right.limit &&
      left.live === right.live &&
      left.presentation === right.presentation
  }
  return right.kind === 'range' &&
    left.start === right.start &&
    left.limit === right.limit &&
    left.live === right.live &&
    left.presentation === right.presentation
}

export const sameDisplayViewShape = (
  left: DisplayViewShape,
  right: DisplayViewShape
): boolean => {
  const leftKeys = Object.keys(left.timelines)
  if (leftKeys.length !== Object.keys(right.timelines).length) return false
  return leftKeys.every((forkKey) => {
    const leftShape = left.timelines[forkKey]
    const rightShape = right.timelines[forkKey]
    return leftShape !== undefined &&
      rightShape !== undefined &&
      sameDisplayTimelineWindowShape(leftShape, rightShape)
  })
}

export const DisplayViewSnapshot = Schema.Struct({
  shape: DisplayViewShape,
  state: DisplayState
})
export type DisplayViewSnapshot = Schema.Schema.Type<typeof DisplayViewSnapshot>
