import { Button } from "@/components/ui/button"

/**
 * ChatTimeline
 *
 * Renders the server-projected timeline presentation. Local state here is
 * limited to scrolling and DOM interaction. Visibility, grouping, and tool
 * semantics come from DisplayView.
 */
import { useCallback, useMemo, useRef, type ReactNode } from "react"
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
  selectedCwdAtom,
  selectedProjectIdAtom,
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
  type LocalModelLoadActivity,
} from "@magnitudedev/client-common"
import {
  RelativePathSchema,
  normalizeReferencedPath,
  type DisplayRootStatus,
} from "@magnitudedev/sdk"
import { useAtomValue } from "@effect-atom/atom-react"
import { useOpenWorkspaceFile } from "@/hooks/use-open-workspace-file"
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
import { chatTimelinePlaceholderState } from "./chat-timeline-state"
import { InlineWorkActivity } from "./inline-work-activity"
import { useShowThinking } from "@/stores/conversation-preferences"
import {
  deriveAssistantResponsePresentation,
  type AssistantResponsePresentation,
} from "./assistant-response-presentation"
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
function toneClass(tone: ToolTone | undefined): string {
  switch (tone) {
    case "info":
      return "text-blue-700 dark:text-blue-500"
    case "success":
      return "text-green-700 dark:text-green-500"
    case "warning":
      return "text-orange-700 dark:text-orange-500"
    case "error":
      return "text-red-600 dark:text-red-500"
    case "muted":
      return "text-slate-500"
    case "neutral":
    default:
      return "text-slate-600 dark:text-slate-400"
  }
}
function PathText({
  path,
  displayPath,
}: {
  path: string
  displayPath?: string | null
}): ReactNode {
  const cwd = useAtomValue(selectedCwdAtom)
  const projectId = useAtomValue(selectedProjectIdAtom)
  const openWorkspaceFile = useOpenWorkspaceFile()
  const relativePath = normalizeReferencedPath(cwd && path.startsWith(`${cwd}/`) ? path.slice(cwd.length + 1) : path)
  if (relativePath === null || projectId === null) {
    return <span className="text-slate-600 dark:text-slate-400">{displayPath ?? path}</span>
  }
  return (
    <Button variant="unstyled" size="unstyled"
      type="button"
      onClick={() => {
        openWorkspaceFile(projectId, RelativePathSchema.make(relativePath))
      }}
      className="hover-text-accent border-0 [background:transparent] [padding:0px] [margin:0px] text-blue-700 dark:text-blue-500 [font:inherit] cursor-pointer text-left"
    >
      {displayPath ?? path}
    </Button>
  )
}
function ToolSummaryRow({
  entry,
}: {
  entry: Extract<
    DisplayTimelineEntry,
    {
      kind: "tool_summary"
    }
  >
}): ReactNode {
  const summary = entry.summary
  const Icon = TOOL_ICONS[summary.icon] ?? Wrench
  const label = toolSummaryLabel(summary)
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(summary.tone)} block justify-self-center`}
      />
      <div className="min-w-0 flex items-center [gap:7px]">
        <span
          className={`${
            summary.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          {label}
        </span>
        {summary.detail.length > 0 && (
          <span className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
            {summary.detail.map((item, index) => (
              <span key={`${item.kind}-${index}`}>
                {index > 0 && <span className="text-slate-500"> · </span>}
                <span className="text-slate-500">{item.text}</span>
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
  if (step.phase === "streaming")
    return <span className="text-slate-500">▍</span>
  if (step.phase === "executing")
    return <span className="text-slate-500">Running...</span>
  if (step.phase === "completed") {
    if (exitCode != null && exitCode !== 0) {
      return (
        <span className="text-red-600 dark:text-red-500">
          ✗ Exit {exitCode}
        </span>
      )
    }
    return <span className="text-green-700 dark:text-green-500">✓</span>
  }
  if (step.phase === "rejected")
    return <span className="text-slate-500">Rejected (Permission Policy)</span>
  if (step.phase === "interrupted")
    return <span className="text-slate-500">Interrupted</span>
  return <span className="text-red-600 dark:text-red-500">✗ Error</span>
}
function capLines(text: string, cap: number): string {
  const lines = text.split("\n")
  if (lines.length <= cap) return text
  const hidden = lines.length - cap
  return [...lines.slice(0, cap), `...${hidden} lines hidden`].join("\n")
}
function ShellStep({
  step,
  mode,
}: {
  step: ShellPresentation
  mode: "default" | "transcript"
}): ReactNode {
  const command = step.command
  const stdout = step.phase === "completed" ? step.stdout : step.partialStdout
  const stderr = step.phase === "completed" ? step.stderr : step.partialStderr
  const output = [stderr, stdout]
    .filter(Boolean)
    .join(stderr && stdout ? "\n" : "")
  const failed = step.failed || (step.exitCode != null && step.exitCode !== 0)
  const lineCap =
    mode === "transcript" ? TRANSCRIPT_LINE_CAP : DEFAULT_SHELL_LINE_CAP
  const capped = output ? capLines(output, lineCap) : ""
  return (
    <div className="[max-width:min(860px,_100%)]">
      <div className="flex items-baseline [gap:8px] font-mono text-[13px] text-slate-900 dark:text-slate-200">
        <span className="text-slate-500">$</span>
        <span
          className={`${
            mode === "transcript" ? "whitespace-normal" : "whitespace-nowrap"
          }  min-w-0 overflow-hidden text-ellipsis`}
        >
          {command}
        </span>
        <span className="shrink-0">{shellStatus(step)}</span>
      </div>
      {capped && (
        <pre
          className={`${
            mode === "transcript"
              ? "[padding:0_0_0_10px]"
              : "[padding:0_0_0_18px]"
          } ${
            mode === "transcript"
              ? "border-l border-l-slate-300 dark:border-l-slate-750"
              : "[border-left:none]"
          } ${
            failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-600 dark:text-slate-400"
          } ${
            mode === "transcript" ? "[max-height:480px]" : "[max-height:180px]"
          }  [margin:5px_0_0] [background:transparent] font-mono text-[12px] leading-[1.45] whitespace-pre-wrap [word-break:break-word] overflow-auto`}
        >
          {capped}
        </pre>
      )}
      {step.phase === "error" && step.errorText && (
        <div className="[margin-top:4px] [padding-left:18px] text-red-600 dark:text-red-500 font-mono text-[12px]">
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
    <div className="[max-width:min(860px,_100%)] font-sans text-[13px]">
      <div
        className={`${
          step.failed
            ? "text-red-600 dark:text-red-500"
            : "text-slate-900 dark:text-slate-200"
        }  flex items-baseline [gap:6px]`}
      >
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-blue-700 dark:text-blue-500"
          } `}
        >
          {step.failed ? "✗" : "✎"}
        </span>
        <span>{step.isScratchpad ? "Write to scratchpad" : "Write"}</span>
        {path ? (
          <PathText path={path} displayPath={displayPath} />
        ) : (
          <span className="text-blue-700 dark:text-blue-500">
            {displayPath}
          </span>
        )}
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : (
          <span className="text-green-700 dark:text-green-500">
            +{lineCount}
          </span>
        )}
      </div>
    </div>
  )
}
function FileEditStep({ step }: { step: FileEditPresentation }): ReactNode {
  const path = step.path
  const displayPath = step.displayPath ?? step.path ?? "..."
  const added = step.addedCount
  const removed = step.removedCount
  return (
    <div className="[max-width:min(860px,_100%)] font-sans text-[13px]">
      <div
        className={`${
          step.failed
            ? "text-red-600 dark:text-red-500"
            : "text-slate-900 dark:text-slate-200"
        }  flex items-baseline [gap:6px]`}
      >
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-blue-700 dark:text-blue-500"
          } `}
        >
          {step.failed ? "✗" : "✎"}
        </span>
        <span>{step.isScratchpad ? "Edit file in scratchpad" : "Edit"}</span>
        {path ? (
          <PathText path={path} displayPath={displayPath} />
        ) : (
          <span className="text-blue-700 dark:text-blue-500">
            {displayPath}
          </span>
        )}
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : added > 0 || removed > 0 ? (
          <span>
            <span className="text-green-700 dark:text-green-500">+{added}</span>
            <span className="text-slate-500">/</span>
            <span className="text-red-600 dark:text-red-500">-{removed}</span>
          </span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : null}
      </div>
    </div>
  )
}
function FileReadStep({ step }: { step: FileReadPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  const path = step.path
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          Read
        </span>
        {path && <PathText path={path} />}
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : step.lineCount != null ? (
          <span className="text-slate-500">{step.lineCount} lines</span>
        ) : null}
      </div>
    </div>
  )
}
function FileViewStep({ step }: { step: FileViewPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  const path = step.path
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          View
        </span>
        {path && <PathText path={path} />}
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : null}
      </div>
    </div>
  )
}
function FileSearchStep({ step }: { step: FileSearchPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          Search
        </span>
        {step.pattern && <span className="text-slate-500">{step.pattern}</span>}
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : (
          <span className="text-slate-500">
            {step.matchCount} matches in {step.fileCount} files
          </span>
        )}
      </div>
    </div>
  )
}
function FileTreeStep({ step }: { step: FileTreePresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          List files
        </span>
        {step.path && <span className="text-slate-500">{step.path}</span>}
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : (
          <span className="text-slate-500">
            {step.fileCount} files, {step.dirCount} dirs
          </span>
        )}
      </div>
    </div>
  )
}
function WebSearchStep({ step }: { step: WebSearchPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          Web search
        </span>
        {step.query && <span className="text-slate-500">{step.query}</span>}
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : (
          <span className="text-slate-500">{step.sourceCount} sources</span>
        )}
      </div>
    </div>
  )
}
function WebFetchStep({ step }: { step: WebFetchPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          Fetch
        </span>
        {step.url && <span className="text-slate-500">{step.url}</span>}
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : null}
      </div>
    </div>
  )
}
function SkillStep({ step }: { step: SkillPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          Skill
        </span>
        {step.skillName && (
          <span className="text-slate-500">{step.skillName}</span>
        )}
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : null}
        {step.errorText && (
          <span className="text-red-600 dark:text-red-500">
            {step.errorText}
          </span>
        )}
      </div>
    </div>
  )
}
function CheckpointStep({ step }: { step: CheckpointPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          {step.isRollback ? "Roll back" : "Inspect changes"}
        </span>
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : (
          <span className="text-slate-500">
            +{step.additions} / -{step.deletions} · {step.fileCount} files
          </span>
        )}
      </div>
    </div>
  )
}
function SpawnWorkerStep({
  step,
}: {
  step: SpawnWorkerPresentation
}): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          {step.role ?? "Worker"}
          {step.title ? `: ${step.title}` : ""}
        </span>
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : null}
      </div>
    </div>
  )
}
function QueryImageStep({ step }: { step: QueryImagePresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  const path = step.path
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          Inspect image
        </span>
        {path && <PathText path={path} />}
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : (
          <span className="text-slate-500">· Done</span>
        )}
      </div>
    </div>
  )
}
function GenericStep({ step }: { step: GenericToolPresentation }): ReactNode {
  const Icon = TOOL_ICONS[step.icon]
  return (
    <div className="grid [grid-template-columns:16px_minmax(0,_1fr)] [gap:7px] items-center font-sans text-[13px] leading-[18px] [max-width:min(860px,_100%)]">
      <Icon
        size={14}
        className={`${toneClass(step.tone)} block justify-self-center`}
      />
      <div className="flex items-center [gap:7px] min-w-0">
        <span
          className={`${
            step.failed
              ? "text-red-600 dark:text-red-500"
              : "text-slate-900 dark:text-slate-200"
          }  shrink-0`}
        >
          {step.label}
        </span>
        {step.failed ? (
          <span className="text-red-600 dark:text-red-500">· Error</span>
        ) : step.running ? (
          <span className="text-slate-500">...</span>
        ) : null}
        {step.errorText && (
          <span className="text-red-600 dark:text-red-500">
            {step.errorText}
          </span>
        )}
      </div>
    </div>
  )
}
function ToolStepView({
  entry,
  mode,
}: {
  entry: Extract<
    DisplayTimelineEntry,
    {
      kind: "tool_step"
    }
  >
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
  curr: DisplayTimelineEntry
): number {
  if (!prev) return 0
  if (isToolEntry(prev) && isToolEntry(curr)) return 8
  if (isToolEntry(prev) || isToolEntry(curr)) return 12
  if (prev.kind !== "message" || curr.kind !== "message") return 12
  const prevMessage = messageForEntry(timeline, prev)
  const currMessage = messageForEntry(timeline, curr)
  if (
    prevMessage?.type === "assistant_message" &&
    currMessage?.type === "work_summary"
  ) {
    return 6
  }
  const prevSystem =
    prevMessage?.type === "status_indicator" ||
    prevMessage?.type === "interrupted"
  const currSystem =
    currMessage?.type === "status_indicator" ||
    currMessage?.type === "interrupted"
  if (prevSystem && currSystem) return 4
  return 16
}
function needsGutter(
  timeline: DisplayTimeline,
  entry: DisplayTimelineEntry
): boolean {
  if (isToolEntry(entry)) return true
  if (entry.kind !== "message") return true
  const message = messageForEntry(timeline, entry)
  return (
    message?.type !== "user_message" &&
    message?.type !== "queued_user_message" &&
    message?.type !== "user_bash_command" &&
    message?.type !== "interrupted"
  )
}
function TimelineEntryView({
  timeline,
  entry,
  assistantResponses,
}: {
  timeline: DisplayTimeline
  entry: DisplayTimelineEntry
  assistantResponses: AssistantResponsePresentation
}): ReactNode {
  if (entry.kind === "tool_summary") return <ToolSummaryRow entry={entry} />
  if (entry.kind === "tool_step")
    return <ToolStepView entry={entry} mode={timeline.presentation.mode} />
  const message = messageForEntry(timeline, entry)
  if (!message) return null
  return (
    <MessageDispatch
      message={message}
      isStreaming={entry.streaming}
      isInterrupted={entry.interrupted}
      assistantWorkSummary={message.type === "assistant_message"
        ? assistantResponses.summaryByAssistantId.get(message.id) ?? null
        : null}
      assistantResponseFooter={message.type === "assistant_message"
        ? assistantResponses.footerByAssistantId.get(message.id) ?? null
        : null}
    />
  )
}

function isThinkingEntry(
  timeline: DisplayTimeline,
  entry: DisplayTimelineEntry,
): boolean {
  return entry.kind === "message"
    && messageForEntry(timeline, entry)?.type === "thinking"
}

function entryOwnsActivePresentation(
  timeline: DisplayTimeline,
  entry: DisplayTimelineEntry | undefined,
): boolean {
  if (entry === undefined) return false
  if (entry.kind === "tool_summary") return entry.summary.running
  if (entry.kind === "tool_step") return entry.step.running
  if (entry.kind !== "message") return false
  const message = messageForEntry(timeline, entry)
  return (message?.type === "thinking" && message.phase === "active")
    || (message?.type === "assistant_message" && entry.streaming)
}

export interface ChatTimelineProps {
  forkId?: string | null
  loadingTitle?: string
  loadingSubtitle?: string | null
  isVisible?: boolean
  rootStatus?: DisplayRootStatus | null
  modelLoadActivity?: LocalModelLoadActivity | null
  modelName?: string | null
}
export function ChatTimeline({
  forkId = null,
  loadingTitle,
  loadingSubtitle,
  isVisible = true,
  rootStatus = null,
  modelLoadActivity = null,
  modelName = null,
}: ChatTimelineProps): ReactNode {
  const timeline = useDisplayState((state) => getFork(state, forkId) ?? null)
  const timelineStatus = useTimelineStatus(forkId)
  const selectedSessionId = useSelectedSessionId()
  const displaySession = useDisplayState((state) => state.session)
  const entries = timeline?.presentation.entries ?? []
  const showThinking = useShowThinking()
  const assistantResponses = useMemo(
    () => timeline === null ? null : deriveAssistantResponsePresentation(timeline),
    [timeline],
  )
  const responseEntries = assistantResponses === null
    ? entries
    : assistantResponses.entries
  const presentedEntries = timeline === null || showThinking
    ? responseEntries
    : responseEntries.filter((entry) => !isThinkingEntry(timeline, entry))
  const suppressGenericActivity = timeline !== null
    && entryOwnsActivePresentation(timeline, presentedEntries.at(-1))
  const { isSessionLoading, isEmpty } = chatTimelinePlaceholderState(
    selectedSessionId,
    timelineStatus,
    entries.length,
  )
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
        el.addEventListener("scroll", onScroll, {
          passive: true,
        })
        const content = contentRef.current
        const observer = content
          ? new ResizeObserver(() => handler("resize"))
          : null
        if (content && observer) observer.observe(content)
        return () => {
          el.removeEventListener("scroll", onScroll)
          observer?.disconnect()
        }
      },
      stickyThreshold: 8,
      loadThreshold: 200,
    }),
    []
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
          const controller = new TimelineScrollController({
            adapter,
            core,
            reader,
            forkId,
          })
          controller.init()
          scrollControllerRef.current = controller
        }
      } else {
        scrollControllerRef.current?.dispose()
        scrollControllerRef.current = null
      }
    },
    [adapter, core, reader, forkId]
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
            })
          )
        })
      ),
    [isVisible]
  )
  useAtomMount(suspendResumeAtom)
  const centerContent = isSessionLoading || (!isSessionLoading && isEmpty)
  return (
    <div
      ref={attachScrollContainer}
      className={`chat-timeline flex-1 overflow-y-auto min-h-0 [overflow-anchor:none] [padding:12px_12px_24px_12px] bg-slate-50 dark:bg-slate-900 relative ${
        centerContent ? "flex flex-col" : ""
      }`}
    >
      <div
        ref={contentRef}
        className={`mx-auto w-full max-w-[800px] ${
          centerContent ? "flex min-h-0 flex-1" : ""
        }`}
      >
        {isSessionLoading ? (
          (() => {
            const title =
              loadingTitle ??
              (forkId === null
                ? displaySession.title?.trim() || undefined
                : undefined)
            const subtitle =
              loadingSubtitle !== undefined
                ? loadingSubtitle
                : forkId === null
                ? displaySession.cwd?.trim() || null
                : null
            return (
              <TimelineLoadingState title={title ?? ""} subtitle={subtitle} />
            )
          })()
        ) : isEmpty || !timeline ? (
          forkId === null ? (
            <ChatEmptyState />
          ) : (
            <div className="text-slate-500 text-[13px]">No activity yet.</div>
          )
        ) : (
          <>
            {isLoadingMore && forkId === null && (
              <div className="flex justify-center [margin-bottom:8px]">
                <span className="text-[11px] text-slate-500">
                  Loading earlier messages…
                </span>
              </div>
            )}
            {presentedEntries.map((entry, idx) => {
              const prev = idx > 0 ? presentedEntries[idx - 1] ?? null : null
              return (
                <div
                  key={entry.id}
                  style={{
                    marginTop: `${getEntrySpacing(timeline, prev, entry)}px`,
                  }}
                  className={`${
                    needsGutter(timeline, entry)
                      ? "[padding-left:12px]"
                      : "[padding-left:0]"
                  }  [animation:fade-in_100ms_ease-out]`}
                >
                  <TimelineEntryView
                    timeline={timeline}
                    entry={entry}
                    assistantResponses={assistantResponses!}
                  />
                </div>
              )
            })}
            {forkId === null ? (
              <div className="mt-3 pl-3">
                <InlineWorkActivity
                  rootStatus={rootStatus}
                  modelLoadActivity={modelLoadActivity}
                  modelName={modelName}
                  suppressGeneric={suppressGenericActivity}
                />
              </div>
            ) : null}
          </>
        )}
      </div>
    </div>
  )
}
