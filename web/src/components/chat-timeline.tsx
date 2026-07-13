/**
 * ChatTimeline
 *
 * Renders the server-projected timeline presentation. Local state here is
 * limited to scrolling and DOM interaction. Visibility, grouping, and tool
 * semantics come from DisplayView.
 */
import {
  useCallback,
  useMemo,
  useRef,
  type CSSProperties,
  type ReactNode,
} from "react"
import { Atom, useAtomMount } from "@effect-atom/atom-react"
import { Effect } from "effect"
import {
  Download,
  FileDiff,
  FilePen,
  FileText,
  FolderTree,
  GitBranch,
  Globe,
  Image as ImageIcon,
  Search,
  Sparkles,
  Terminal,
  Wrench,
  type LucideIcon,
} from "lucide-react"
import {
  getFork,
  messageForEntry,
  toolSummaryLabel,
  selectedFilePathAtom,
  useDisplayState,
  useTimelineStatus,
  useSelectedSessionId,
  useDisplayViewControllerCore,
  useDisplayReader,
  useRootHistoryLoading,
  TimelineScrollController,
  type TimelineScrollAdapter,
  type ActivityKind,
  TRANSCRIPT_LINE_CAP,
} from "@magnitudedev/client-common"
import { useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import type {
  DisplayTimeline,
  DisplayTimelineEntry,
  ToolIcon,
  ToolTone,
  ToolStepPresentation,
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
  GenericToolPresentation,
  QueryImagePresentation,
} from "@magnitudedev/sdk"
import { MessageDispatch } from "./messages"
import { TimelineLoadingState } from "./timeline-loading-state"
import { ChatEmptyState } from "./chat-empty-state"
import { DiffHunk } from "./diff-hunk"

const DEFAULT_SHELL_LINE_CAP = 8

const TOOL_ICONS: Record<ToolIcon, LucideIcon> = {
  file: FileText,
  edit: FilePen,
  diff: FileDiff,
  search: Search,
  tree: FolderTree,
  terminal: Terminal,
  web: Globe,
  download: Download,
  skill: Sparkles,
  worker: Wrench,
  checkpoint: GitBranch,
  tool: Wrench,
  image: ImageIcon,
}

function toneColor(tone: ToolTone | undefined): string {
  switch (tone) {
    case "info":
      return "var(--accent-info)"
    case "success":
      return "var(--accent-success)"
    case "warning":
      return "var(--accent-warning)"
    case "error":
      return "var(--accent-error)"
    case "muted":
      return "var(--fg-tertiary)"
    case "neutral":
    default:
      return "var(--fg-secondary)"
  }
}

function PathText({ path, displayPath }: { path: string; displayPath?: string | null }): ReactNode {
  const setFilePath = useAtomSet(selectedFilePathAtom)
  return (
    <button
      type="button"
      onClick={() => setFilePath(path)}
      className="hover-text-accent"
      style={{
        border: "none",
        background: "transparent",
        padding: 0,
        margin: 0,
        color: "var(--accent-primary)",
        font: "inherit",
        cursor: "pointer",
        textAlign: "left",
      }}
    >
      {displayPath ?? path}
    </button>
  )
}

function ToolSummaryRow({ entry }: { entry: Extract<DisplayTimelineEntry, { kind: "tool_summary" }> }): ReactNode {
  const summary = entry.summary
  const Icon = TOOL_ICONS[summary.icon] ?? Wrench
  const label = toolSummaryLabel(summary)
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "16px minmax(0, 1fr)",
        gap: 7,
        alignItems: "center",
        fontFamily: "var(--font-sans)",
        fontSize: 13,
        lineHeight: "18px",
        maxWidth: "min(860px, 100%)",
      }}
    >
      <Icon size={14} style={{ color: toneColor(summary.tone), display: "block", justifySelf: "center" }} />
      <div style={{ minWidth: 0, display: "flex", alignItems: "center", gap: 7 }}>
        <span style={{ color: summary.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>
          {label}
        </span>
        {summary.detail.length > 0 && (
          <span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {summary.detail.map((item, index) => (
              <span key={`${item.kind}-${index}`}>
                {index > 0 && <span style={{ color: "var(--fg-tertiary)" }}> · </span>}
                <span style={{ color: "var(--fg-tertiary)" }}>{item.text}</span>
              </span>
            ))}
          </span>
        )}
      </div>
    </div>
  )
}

function shellStatus(step: ShellPresentation): ReactNode {
  const exitCode = step.exitCode
  if (step.phase === "streaming") return <span style={{ color: "var(--fg-tertiary)" }}>▍</span>
  if (step.phase === "executing") return <span style={{ color: "var(--fg-tertiary)" }}>Running...</span>
  if (step.phase === "completed") {
    if (exitCode != null && exitCode !== 0) {
      return <span style={{ color: "var(--accent-error)" }}>✗ Exit {exitCode}</span>
    }
    return <span style={{ color: "var(--accent-success)" }}>✓</span>
  }
  if (step.phase === "rejected") return <span style={{ color: "var(--fg-tertiary)" }}>Rejected (Permission Policy)</span>
  if (step.phase === "interrupted") return <span style={{ color: "var(--fg-tertiary)" }}>Interrupted</span>
  return <span style={{ color: "var(--accent-error)" }}>✗ Error</span>
}

function capLines(text: string, cap: number): string {
  const lines = text.split("\n")
  if (lines.length <= cap) return text
  const hidden = lines.length - cap
  return [...lines.slice(0, cap), `...${hidden} lines hidden`].join("\n")
}

function ShellStep({ step, mode }: { step: ShellPresentation; mode: "default" | "transcript" }): ReactNode {
  const command = step.command
  const stdout = step.phase === "completed" ? step.stdout : step.partialStdout
  const stderr = step.phase === "completed" ? step.stderr : step.partialStderr
  const output = [stderr, stdout].filter(Boolean).join(stderr && stdout ? "\n" : "")
  const failed = step.failed || (step.exitCode != null && step.exitCode !== 0)
  const lineCap = mode === "transcript" ? TRANSCRIPT_LINE_CAP : DEFAULT_SHELL_LINE_CAP
  const capped = output ? capLines(output, lineCap) : ""

  return (
    <div style={{ maxWidth: "min(860px, 100%)" }}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 8,
          fontFamily: "var(--font-mono)",
          fontSize: 13,
          color: "var(--fg-primary)",
        }}
      >
        <span style={{ color: "var(--fg-tertiary)" }}>$</span>
        <span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: mode === "transcript" ? "normal" : "nowrap" }}>
          {command}
        </span>
        <span style={{ flexShrink: 0 }}>{shellStatus(step)}</span>
      </div>
      {capped && (
        <pre
          style={{
            margin: "5px 0 0",
            padding: mode === "transcript" ? "0 0 0 10px" : "0 0 0 18px",
            borderLeft: mode === "transcript" ? "1px solid var(--border-default)" : "none",
            background: "transparent",
            color: failed ? "var(--accent-error)" : "var(--fg-secondary)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            lineHeight: 1.45,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            maxHeight: mode === "transcript" ? 480 : 180,
            overflow: "auto",
          }}
        >
          {capped}
        </pre>
      )}
      {step.phase === "error" && step.errorText && (
        <div style={{ marginTop: 4, paddingLeft: 18, color: "var(--accent-error)", fontFamily: "var(--font-mono)", fontSize: 12 }}>
          {step.errorText}
        </div>
      )}
    </div>
  )
}

function FileWriteStep({ step }: { step: FileWritePresentation }): ReactNode {
  const path = step.path
  const displayPath = step.displayPath ?? step.path ?? "..."
  const lineCount = step.lineCount
  return (
    <div style={{ maxWidth: "min(860px, 100%)", fontFamily: "var(--font-sans)", fontSize: 13 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6, color: step.failed ? "var(--accent-error)" : "var(--fg-primary)" }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--accent-info)" }}>{step.failed ? "✗" : "✎"}</span>
        <span>{step.isScratchpad ? "Write to scratchpad" : "Write"}</span>
        {path ? <PathText path={path} displayPath={displayPath} /> : <span style={{ color: "var(--accent-primary)" }}>{displayPath}</span>}
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : (
          <span style={{ color: "var(--accent-success)" }}>+{lineCount}</span>
        )}
      </div>
      {!step.isScratchpad && step.diff?.hunks.map((hunk, index) => (
        <DiffHunk
          key={`filewrite-${index}`}
          startLine={hunk.startLine}
          contextBefore={hunk.contextBefore}
          removedLines={hunk.removedLines}
          addedLines={hunk.addedLines}
          contextAfter={hunk.contextAfter}
          streamingCursor={hunk.streamingCursor}
        />
      ))}
    </div>
  )
}

function FileEditStep({ step }: { step: FileEditPresentation }): ReactNode {
  const path = step.path
  const displayPath = step.displayPath ?? step.path ?? "..."
  const added = step.addedCount
  const removed = step.removedCount
  return (
    <div style={{ maxWidth: "min(860px, 100%)", fontFamily: "var(--font-sans)", fontSize: 13 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6, color: step.failed ? "var(--accent-error)" : "var(--fg-primary)" }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--accent-info)" }}>{step.failed ? "✗" : "✎"}</span>
        <span>{step.isScratchpad ? "Edit file in scratchpad" : "Edit"}</span>
        {path ? <PathText path={path} displayPath={displayPath} /> : <span style={{ color: "var(--accent-primary)" }}>{displayPath}</span>}
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : (added > 0 || removed > 0) ? (
          <span>
            <span style={{ color: "var(--accent-success)" }}>+{added}</span>
            <span style={{ color: "var(--fg-tertiary)" }}>/</span>
            <span style={{ color: "var(--accent-error)" }}>-{removed}</span>
          </span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : null}
      </div>
      {!step.isScratchpad && step.diff?.hunks.map((hunk, index) => (
        <DiffHunk
          key={`fileedit-${index}`}
          startLine={hunk.startLine}
          contextBefore={hunk.contextBefore}
          removedLines={hunk.removedLines}
          addedLines={hunk.addedLines}
          contextAfter={hunk.contextAfter}
          streamingCursor={hunk.streamingCursor}
        />
      ))}
    </div>
  )
}

function FileReadStep({ step }: { step: FileReadPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  const path = step.path
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>Read</span>
        {path && <PathText path={path} />}
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : step.lineCount != null ? (
          <span style={{ color: "var(--fg-tertiary)" }}>{step.lineCount} lines</span>
        ) : null}
      </div>
    </div>
  )
}

function FileViewStep({ step }: { step: FileViewPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  const path = step.path
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>View</span>
        {path && <PathText path={path} />}
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : null}
      </div>
    </div>
  )
}

function FileSearchStep({ step }: { step: FileSearchPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>Search</span>
        {step.pattern && <span style={{ color: "var(--fg-tertiary)" }}>{step.pattern}</span>}
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : (
          <span style={{ color: "var(--fg-tertiary)" }}>{step.matchCount} matches in {step.fileCount} files</span>
        )}
      </div>
    </div>
  )
}

function FileTreeStep({ step }: { step: FileTreePresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>List files</span>
        {step.path && <span style={{ color: "var(--fg-tertiary)" }}>{step.path}</span>}
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : (
          <span style={{ color: "var(--fg-tertiary)" }}>{step.fileCount} files, {step.dirCount} dirs</span>
        )}
      </div>
    </div>
  )
}

function WebSearchStep({ step }: { step: WebSearchPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>Web search</span>
        {step.query && <span style={{ color: "var(--fg-tertiary)" }}>{step.query}</span>}
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : (
          <span style={{ color: "var(--fg-tertiary)" }}>{step.sourceCount} sources</span>
        )}
      </div>
    </div>
  )
}

function WebFetchStep({ step }: { step: WebFetchPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>Fetch</span>
        {step.url && <span style={{ color: "var(--fg-tertiary)" }}>{step.url}</span>}
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : null}
      </div>
    </div>
  )
}

function SkillStep({ step }: { step: SkillPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>Skill</span>
        {step.skillName && <span style={{ color: "var(--fg-tertiary)" }}>{step.skillName}</span>}
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : null}
        {step.errorText && <span style={{ color: "var(--accent-error)" }}>{step.errorText}</span>}
      </div>
    </div>
  )
}

function CheckpointStep({ step }: { step: CheckpointPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>
          {step.isRollback ? "Roll back" : "Inspect changes"}
        </span>
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : (
          <span style={{ color: "var(--fg-tertiary)" }}>+{step.additions} / -{step.deletions} · {step.fileCount} files</span>
        )}
      </div>
    </div>
  )
}

function SpawnWorkerStep({ step }: { step: SpawnWorkerPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>
          {step.role ?? "Worker"}
          {step.title ? `: ${step.title}` : ""}
        </span>
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : null}
      </div>
    </div>
  )
}

function QueryImageStep({ step }: { step: QueryImagePresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  const path = step.path
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>Inspect image</span>
        {path && <PathText path={path} />}
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : (
          <span style={{ color: "var(--fg-tertiary)" }}>· Done</span>
        )}
      </div>
    </div>
  )
}

function GenericStep({ step }: { step: GenericToolPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div style={{ display: "grid", gridTemplateColumns: "16px minmax(0, 1fr)", gap: 7, alignItems: "center", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: "18px", maxWidth: "min(860px, 100%)" }}>
      <Icon size={14} style={{ color: toneColor(step.tone), display: "block", justifySelf: "center" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <span style={{ color: step.failed ? "var(--accent-error)" : "var(--fg-primary)", flexShrink: 0 }}>{step.label}</span>
        {step.failed ? (
          <span style={{ color: "var(--accent-error)" }}>· Error</span>
        ) : step.running ? (
          <span style={{ color: "var(--fg-tertiary)" }}>...</span>
        ) : null}
        {step.errorText && <span style={{ color: "var(--accent-error)" }}>{step.errorText}</span>}
      </div>
    </div>
  )
}

function ToolStepView({ entry, mode }: {
  entry: Extract<DisplayTimelineEntry, { kind: "tool_step" }>
  mode: "default" | "transcript"
}): ReactNode {
  const step: ToolStepPresentation = entry.step
  // `GenericToolPresentation.toolKey` is `string` (not a literal), so the
  // union does not narrow on `toolKey` comparisons. The projection guarantees
  // the variant matches `toolKey`, so each case casts to its variant type.
  switch (step.toolKey) {
    case "shell":
      return <ShellStep step={step as ShellPresentation} mode={mode} />
    case "fileWrite":
      return <FileWriteStep step={step as FileWritePresentation} />
    case "fileEdit":
      return <FileEditStep step={step as FileEditPresentation} />
    case "fileRead":
      return <FileReadStep step={step as FileReadPresentation} />
    case "fileView":
      return <FileViewStep step={step as FileViewPresentation} />
    case "fileSearch":
      return <FileSearchStep step={step as FileSearchPresentation} />
    case "fileTree":
      return <FileTreeStep step={step as FileTreePresentation} />
    case "webSearch":
      return <WebSearchStep step={step as WebSearchPresentation} />
    case "webFetch":
      return <WebFetchStep step={step as WebFetchPresentation} />
    case "skill":
      return <SkillStep step={step as SkillPresentation} />
    case "checkpointChanges":
    case "checkpointRollback":
      return <CheckpointStep step={step as CheckpointPresentation} />
    case "spawnWorker":
      return <SpawnWorkerStep step={step as SpawnWorkerPresentation} />
    case "queryImage":
      return <QueryImageStep step={step as QueryImagePresentation} />
    default:
      return <GenericStep step={step as GenericToolPresentation} />
  }
}

function isToolEntry(entry: DisplayTimelineEntry): boolean {
  return entry.kind === "tool_summary" || entry.kind === "tool_step"
}

function getEntrySpacing(
  timeline: DisplayTimeline,
  prev: DisplayTimelineEntry | null,
  curr: DisplayTimelineEntry,
): number {
  if (!prev) return 0
  if (isToolEntry(prev) && isToolEntry(curr)) return 8
  if (isToolEntry(prev) || isToolEntry(curr)) return 12
  if (prev.kind !== "message" || curr.kind !== "message") return 12

  const prevMessage = messageForEntry(timeline, prev)
  const currMessage = messageForEntry(timeline, curr)
  const prevSystem = prevMessage?.type === "status_indicator" || prevMessage?.type === "interrupted"
  const currSystem = currMessage?.type === "status_indicator" || currMessage?.type === "interrupted"
  if (prevSystem && currSystem) return 4
  return 16
}

function needsGutter(
  timeline: DisplayTimeline,
  entry: DisplayTimelineEntry,
): boolean {
  if (isToolEntry(entry)) return true
  if (entry.kind !== "message") return true
  const message = messageForEntry(timeline, entry)
  return message?.type !== "user_message" && message?.type !== "queued_user_message" && message?.type !== "interrupted"
}

function TimelineEntryView({
  timeline,
  entry,
}: {
  timeline: DisplayTimeline
  entry: DisplayTimelineEntry
}): ReactNode {
  if (entry.kind === "tool_summary") return <ToolSummaryRow entry={entry} />
  if (entry.kind === "tool_step") return <ToolStepView entry={entry} mode={timeline.presentation.mode} />

  const message = messageForEntry(timeline, entry)
  if (!message) return null
  return (
    <MessageDispatch
      message={message}
      isStreaming={entry.streaming}
      isInterrupted={entry.interrupted}
      mode={timeline.presentation.mode}
    />
  )
}

export interface ChatTimelineProps {
  forkId?: string | null
  loadingTitle?: string
  loadingSubtitle?: string | null
  isVisible?: boolean
}

export function ChatTimeline({
  forkId = null,
  loadingTitle,
  loadingSubtitle,
  isVisible = true,
}: ChatTimelineProps): ReactNode {
  const timeline = useDisplayState((state) => getFork(state, forkId) ?? null)
  const timelineStatus = useTimelineStatus(forkId)
  const selectedSessionId = useSelectedSessionId()
  const displaySession = useDisplayState((state) => state.session)
  const entries = timeline?.presentation.entries ?? []
  const isSessionLoading = selectedSessionId !== null && timelineStatus._tag === "pending"
  const isEmpty = selectedSessionId === null || timelineStatus._tag === "empty"

  const scrollRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)

  // Metrics adapter for TimelineScrollController. The controller is the
  // sole scroll writer besides the user (the container has
  // `overflow-anchor: none`).
  const adapter = useMemo<TimelineScrollAdapter>(
    () => ({
      getScrollMetrics: () => {
        const el = scrollRef.current
        if (!el) return null
        return {
          scrollTop: el.scrollTop,
          viewportHeight: el.clientHeight,
          scrollHeight: el.scrollHeight,
        }
      },
      setScrollTop: (value) => {
        const el = scrollRef.current
        if (el) el.scrollTop = Math.max(0, value)
      },
      // Scroll events (user input) and content size changes (ResizeObserver,
      // post-layout pre-paint) feed the controller with an ActivityKind so it
      // can distinguish user scroll from content growth. The controller is
      // the sole scroll writer — no separate sticky observer.
      subscribeActivity: (handler: (kind: ActivityKind) => void) => {
        const el = scrollRef.current
        if (!el) return () => {}
        const onScroll = (): void => handler("scroll")
        el.addEventListener("scroll", onScroll, { passive: true })
        const content = contentRef.current
        const observer = content ? new ResizeObserver(() => handler("resize")) : null
        if (content && observer) observer.observe(content)
        return () => {
          el.removeEventListener("scroll", onScroll)
          observer?.disconnect()
        }
      },
      stickyThreshold: 8,
      loadThreshold: 200,
    }),
    [],
  )

  const core = useDisplayViewControllerCore()
  const reader = useDisplayReader()
  const isLoadingMore = useRootHistoryLoading()

  const scrollControllerRef = useRef<TimelineScrollController | null>(null)

  // Callback ref: the scroll container's mount/unmount is the lifetime of
  // the scroll controller. The controller is the sole scroll writer — it
  // owns anchoring, bottom-following, and load triggering. No separate
  // sticky observer.
  const attachScrollContainer = useCallback(
    (el: HTMLDivElement | null) => {
      scrollRef.current = el
      if (el) {
        if (scrollControllerRef.current === null) {
          const controller = new TimelineScrollController({ adapter, core, reader, forkId })
          controller.init()
          scrollControllerRef.current = controller
        }
      } else {
        scrollControllerRef.current?.dispose()
        scrollControllerRef.current = null
      }
    },
    [adapter, core, reader, forkId],
  )

  // Suspend/resume the scroll controller when the timeline is hidden behind
  // an overlay. While suspended, the controller preserves all state — window
  // position, scroll distance, followingBottom — so the user returns to
  // exactly what they left.
  const suspendResumeAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          const controller = scrollControllerRef.current
          if (!controller) return
          if (!isVisible) controller.suspend()
          else controller.resume()
          yield* Effect.addFinalizer(() =>
            Effect.sync(() => {
              controller.resume()
            }),
          )
        }),
      ),
    [isVisible],
  )
  useAtomMount(suspendResumeAtom)

  const centerContent = isSessionLoading || (!isSessionLoading && isEmpty)
  const contentStyle: CSSProperties | undefined = centerContent
    ? { flex: 1, minHeight: 0, display: "flex" }
    : undefined

  return (
    <div
      ref={attachScrollContainer}
      className="chat-timeline"
      style={{
        flex: 1,
        overflowY: "auto",
        minHeight: 0,
        // The controller owns anchoring; native scroll anchoring would
        // double-write (and Safari doesn't implement it — this keeps
        // behavior uniform).
        overflowAnchor: "none",
        padding: "12px 12px 24px 12px",
        background: "var(--bg-base)",
        position: "relative",
        display: centerContent ? "flex" : undefined,
        flexDirection: centerContent ? "column" : undefined,
      }}
    >
      <div ref={contentRef} style={contentStyle}>
        {isSessionLoading ? (
          (() => {
            const title =
              loadingTitle ??
              (forkId === null
                ? (displaySession.title?.trim() || undefined)
                : undefined)
            const subtitle =
              loadingSubtitle !== undefined
                ? loadingSubtitle
                : forkId === null
                  ? (displaySession.cwd?.trim() || null)
                  : null
            return (
              <TimelineLoadingState
                title={title ?? ""}
                subtitle={subtitle}
              />
            )
          })()
        ) : isEmpty || !timeline ? (
          forkId === null ? <ChatEmptyState /> : <div style={{ color: "var(--fg-tertiary)", fontSize: 13 }}>No activity yet.</div>
        ) : (
          <>
            {isLoadingMore && forkId === null && (
              <div style={{ display: "flex", justifyContent: "center", marginBottom: "8px" }}>
                <span style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>Loading earlier messages…</span>
              </div>
            )}
            {entries.map((entry, idx) => {
              const prev = idx > 0 ? entries[idx - 1] ?? null : null
              return (
                <div
                  key={entry.id}
                  style={{
                    marginTop: `${getEntrySpacing(timeline, prev, entry)}px`,
                    paddingLeft: needsGutter(timeline, entry) ? "12px" : "0",
                    animation: "fade-in 100ms ease-out",
                  }}
                >
                  <TimelineEntryView timeline={timeline} entry={entry} />
                </div>
              )
            })}
          </>
        )}
      </div>
    </div>
  )
}
