import { Option } from 'effect'
import type {
  CheckpointPresentation,
  FileEditPresentation,
  FileReadPresentation,
  FileSearchPresentation,
  FileTreePresentation,
  FileViewPresentation,
  FileWritePresentation,
  GenericToolPresentation,
  QueryImagePresentation,
  ShellPresentation,
  SkillPresentation,
  SpawnWorkerPresentation,
  ToolDiffHunk,
  ToolFileRef,
  ToolIcon,
  ToolStepPresentation,
  ToolSummaryDetailItem,
  ToolSummaryPresentation,
  ToolTone,
  WebFetchPresentation,
  WebSearchPresentation,
} from '@magnitudedev/protocol'
import type { ToolHandleFromSchema } from '../models/tool-handle-schema'
import type {
  CheckpointChangesState,
  CheckpointRollbackState,
  FileEditState,
  FileReadState,
  FileSearchState,
  FileTreeState,
  FileViewState,
  FileWriteState,
  Phase,
  QueryImageState,
  ShellState,
  SkillActivationState,
  SpawnWorkerState,
  ToolState,
  WebFetchState,
  WebSearchState,
} from '../models/tool-state'

const ERROR_PHASES: ReadonlySet<Phase> = new Set(['error', 'rejected', 'interrupted'])

export const HIDDEN_TOOL_KEYS: ReadonlySet<string> = new Set([
  'createTask',
  'updateTask',
  'killWorker',
  'reassignWorker',
  'messageWorker',
  'messageAdvisor',
  'finishGoal',
  'compact',
])

export function isRunningPhase(phase: Phase): boolean {
  return phase === 'streaming' || phase === 'executing'
}

export function isFailedPhase(phase: Phase): boolean {
  return ERROR_PHASES.has(phase)
}

function opt<T>(value: Option.Option<T> | T | undefined | null): T | undefined {
  if (value == null) return undefined
  if (typeof value === 'object' && '_tag' in value) {
    return Option.isSome(value as Option.Option<T>) ? (value as { readonly value: T }).value : undefined
  }
  return value as T
}

/** The set of toolKeys that can be grouped into a summary entry. */
export const SUMMARY_TOOL_KEYS: ReadonlySet<string> = new Set([
  'fileRead',
  'fileSearch',
  'webSearch',
  'webFetch',
  'fileTree',
  'fileView',
])

/**
 * Returns the toolKey if it can be summarized, otherwise null.
 * Replaces the old `getToolSummaryFamily` — `toolKey` is the discriminator now,
 * there is no separate `family` taxonomy.
 */
export function getToolSummaryFamily(toolKey: string): string | null {
  return SUMMARY_TOOL_KEYS.has(toolKey) ? toolKey : null
}

// ── Phase / tone derivation ─────────────────────────────────────

function toneFor(phase: Phase, completedTone: ToolTone = 'neutral'): ToolTone {
  if (isFailedPhase(phase)) return 'error'
  if (isRunningPhase(phase)) return 'info'
  if (phase === 'completed') return completedTone
  return 'neutral'
}

// ── Per-tool presentation builders ──────────────────────────────
// Each takes the typed ToolState and produces the matching presentation
// variant. No generic segments. No `family` tag. No `label` (labels are
// rendering — computed by the client).

function shellPresentation(state: ShellState): ShellPresentation {
  const exitCode = opt(state.exitCode)
  const failedExit = state.phase === 'completed' && exitCode != null && exitCode !== 0
  return {
    toolKey: 'shell',
    phase: state.phase,
    tone: failedExit ? 'error' : toneFor(state.phase),
    icon: 'terminal',
    command: state.command,
    done: opt(state.done) ?? null,
    exitCode: exitCode ?? null,
    pid: opt(state.pid) ?? null,
    stdout: opt(state.stdout) ?? '',
    stderr: opt(state.stderr) ?? '',
    partialStdout: state.partialStdout,
    partialStderr: state.partialStderr,
    stdoutPath: opt(state.stdoutPath) ?? null,
    stderrPath: opt(state.stderrPath) ?? null,
    errorText: opt(state.errorMessage) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function fileWritePresentation(state: FileWriteState): FileWritePresentation {
  const filePath = opt(state.path)
  const displayPath = opt(state.scratchpadDisplayPath) ?? filePath
  const addedLines = state.body
    ? state.body.split('\n').filter((_, index, lines) => index < lines.length - 1 || lines[index] !== '')
    : []
  return {
    toolKey: 'fileWrite',
    phase: state.phase,
    tone: toneFor(state.phase, 'success'),
    icon: 'edit',
    path: filePath ?? null,
    displayPath: displayPath ?? null,
    lineCount: state.lineCount,
    isScratchpad: state.isScratchpad,
    diff: !state.isScratchpad && addedLines.length > 0
      ? {
          hunks: [{
            startLine: 1,
            removedLines: [],
            addedLines,
            contextBefore: [],
            contextAfter: [],
            streamingCursor: isRunningPhase(state.phase),
          }],
        }
      : null,
    errorText: opt(state.errorMessage) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function fileEditPresentation(state: FileEditState): FileEditPresentation {
  const filePath = opt(state.path)
  const displayPath = opt(state.scratchpadDisplayPath) ?? filePath
  const added = state.diffs.reduce((sum, diff) => sum + diff.addedLines.length, 0)
  const removed = state.diffs.reduce((sum, diff) => sum + diff.removedLines.length, 0)
  const streamingTarget = opt(state.streamingTarget)
  return {
    toolKey: 'fileEdit',
    phase: state.phase,
    tone: toneFor(state.phase, 'success'),
    icon: 'diff',
    path: filePath ?? null,
    displayPath: displayPath ?? null,
    addedCount: added,
    removedCount: removed,
    isScratchpad: state.isScratchpad,
    streamingTarget: streamingTarget ?? null,
    diff: !state.isScratchpad && state.diffs.length > 0
      ? {
          hunks: state.diffs.map((diff): ToolDiffHunk => ({
            startLine: diff.startLine,
            removedLines: diff.removedLines,
            addedLines: diff.addedLines,
            contextBefore: diff.contextBefore,
            contextAfter: diff.contextAfter,
            streamingCursor: isRunningPhase(state.phase) && streamingTarget === 'new',
          })),
        }
      : null,
    errorText: opt(state.errorMessage) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function fileReadPresentation(state: FileReadState): FileReadPresentation {
  return {
    toolKey: 'fileRead',
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'file',
    path: opt(state.path) ?? null,
    lineCount: opt(state.lineCount) ?? null,
    offset: opt(state.offset) ?? null,
    limit: opt(state.limit) ?? null,
    errorText: opt(state.errorDetail) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function fileSearchPresentation(state: FileSearchState): FileSearchPresentation {
  return {
    toolKey: 'fileSearch',
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'search',
    pattern: opt(state.pattern) ?? null,
    matchCount: state.matchCount,
    fileCount: state.fileCount,
    errorText: opt(state.errorDetail) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function fileTreePresentation(state: FileTreeState): FileTreePresentation {
  return {
    toolKey: 'fileTree',
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'tree',
    path: opt(state.path) ?? '.',
    fileCount: state.fileCount,
    dirCount: state.dirCount,
    errorText: opt(state.errorDetail) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function fileViewPresentation(state: FileViewState): FileViewPresentation {
  return {
    toolKey: 'fileView',
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'file',
    path: opt(state.path) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function webSearchPresentation(state: WebSearchState): WebSearchPresentation {
  const sources = opt(state.sources) ?? []
  return {
    toolKey: 'webSearch',
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'web',
    query: opt(state.query) ?? null,
    sourceCount: sources.length,
    errorText: opt(state.errorDetail) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function webFetchPresentation(state: WebFetchState): WebFetchPresentation {
  return {
    toolKey: 'webFetch',
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'download',
    url: opt(state.url) ?? null,
    errorText: opt(state.errorDetail) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function skillPresentation(state: SkillActivationState): SkillPresentation {
  return {
    toolKey: 'skill',
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'skill',
    skillName: opt(state.skillName) ?? null,
    skillPath: opt(state.skillPath) ?? null,
    errorText: opt(state.errorDetail) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function checkpointPresentation(
  toolKey: 'checkpointChanges' | 'checkpointRollback',
  state: CheckpointChangesState | CheckpointRollbackState,
): CheckpointPresentation {
  const since = opt(state.since)
  const files = opt(state.files) ?? []
  const fileCount = opt(state.fileCount) ?? files.length
  const additions = opt(state.additions) ?? 0
  const deletions = opt(state.deletions) ?? 0
  const isRollback = toolKey === 'checkpointRollback'
  const fileRefs: ToolFileRef[] = files.map((file) => ({ path: file.path, displayPath: Option.none(), section: Option.none() }))
  return {
    toolKey,
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'checkpoint',
    isRollback,
    since: since ?? null,
    fileCount,
    additions,
    deletions,
    files: fileRefs,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function spawnWorkerPresentation(state: SpawnWorkerState): SpawnWorkerPresentation {
  return {
    toolKey: 'spawnWorker',
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'worker',
    agentId: opt(state.agentId) ?? null,
    role: opt(state.role) ?? null,
    title: opt(state.title) ?? null,
    message: opt(state.message) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function queryImagePresentation(state: QueryImageState): QueryImagePresentation {
  return {
    toolKey: 'queryImage',
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'image',
    path: opt(state.path) ?? null,
    errorText: opt(state.errorMessage) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

function genericPresentation(toolKey: string, state: ToolState): GenericToolPresentation {
  return {
    toolKey,
    phase: state.phase,
    tone: toneFor(state.phase),
    icon: 'tool',
    label: toolKey,
    errorText: opt(state.errorMessage) ?? null,
    running: isRunningPhase(state.phase),
    failed: isFailedPhase(state.phase),
  }
}

// ── Dispatch ────────────────────────────────────────────────────

/**
 * Build a `ToolStepPresentation` from a typed tool handle.
 * Called at event time — the handle is guaranteed to exist.
 */
export function presentToolState(handle: ToolHandleFromSchema): ToolStepPresentation {
  const { toolKey, state } = handle
  switch (toolKey) {
    case 'shell': return shellPresentation(state as ShellState)
    case 'fileWrite': return fileWritePresentation(state as FileWriteState)
    case 'fileEdit': return fileEditPresentation(state as FileEditState)
    case 'fileRead': return fileReadPresentation(state as FileReadState)
    case 'fileSearch': return fileSearchPresentation(state as FileSearchState)
    case 'fileTree': return fileTreePresentation(state as FileTreeState)
    case 'fileView': return fileViewPresentation(state as FileViewState)
    case 'webSearch': return webSearchPresentation(state as WebSearchState)
    case 'webFetch': return webFetchPresentation(state as WebFetchState)
    case 'skill': return skillPresentation(state as SkillActivationState)
    case 'spawnWorker': return spawnWorkerPresentation(state as SpawnWorkerState)
    case 'queryImage': return queryImagePresentation(state as QueryImageState)
    case 'checkpointChanges':
    case 'checkpointRollback':
      return checkpointPresentation(toolKey, state as CheckpointChangesState | CheckpointRollbackState)
    default:
      return genericPresentation(toolKey, state)
  }
}

// ── Summary ─────────────────────────────────────────────────────

function summaryIcon(toolKey: string): ToolIcon {
  switch (toolKey) {
    case 'fileRead': return 'file'
    case 'fileSearch': return 'search'
    case 'webSearch': return 'web'
    case 'webFetch': return 'download'
    case 'fileTree': return 'tree'
    case 'fileView': return 'file'
    default: return 'tool'
  }
}

function isFileSearch(p: ToolStepPresentation): p is FileSearchPresentation {
  return p.toolKey === 'fileSearch'
}
function isWebSearch(p: ToolStepPresentation): p is WebSearchPresentation {
  return p.toolKey === 'webSearch'
}
function isWebFetch(p: ToolStepPresentation): p is WebFetchPresentation {
  return p.toolKey === 'webFetch'
}
function isFileRead(p: ToolStepPresentation): p is FileReadPresentation {
  return p.toolKey === 'fileRead'
}
function isFileTree(p: ToolStepPresentation): p is FileTreePresentation {
  return p.toolKey === 'fileTree'
}
function isFileView(p: ToolStepPresentation): p is FileViewPresentation {
  return p.toolKey === 'fileView'
}

function summaryDetail(
  toolKey: string,
  presentations: readonly ToolStepPresentation[],
): readonly ToolSummaryDetailItem[] {
  switch (toolKey) {
    case 'fileSearch': {
      return presentations
        .filter(isFileSearch)
        .map((p) => p.pattern)
        .filter((p): p is string => p != null && p.length > 0)
        .map((pattern) => ({ kind: 'pattern' as const, text: `"${pattern}"` }))
    }
    case 'webSearch': {
      return presentations
        .filter(isWebSearch)
        .map((p) => p.query)
        .filter((q): q is string => q != null && q.length > 0)
        .map((query) => ({ kind: 'query' as const, text: `"${query}"` }))
    }
    case 'webFetch': {
      return presentations
        .filter(isWebFetch)
        .map((p) => p.url)
        .filter((u): u is string => u != null && u.length > 0)
        .map((url) => ({ kind: 'path' as const, text: url }))
    }
    default: {
      return presentations
        .filter((p): p is FileReadPresentation | FileTreePresentation | FileViewPresentation =>
          isFileRead(p) || isFileTree(p) || isFileView(p))
        .map((p) => p.path)
        .filter((p): p is string => p != null && p.length > 0)
        .map((p) => ({ kind: 'path' as const, text: p }))
    }
  }
}

export function presentToolSummary(
  toolKey: string,
  presentations: readonly ToolStepPresentation[],
): ToolSummaryPresentation {
  const running = presentations.some((p) => isRunningPhase(p.phase))
  const failed = presentations.some((p) => isFailedPhase(p.phase))
  const aggregatePhase: Phase = running ? 'streaming' : failed ? 'error' : 'completed'

  let matchCount: number | null = null
  let fileCount: number | null = null
  let sourceCount: number | null = null

  if (toolKey === 'fileSearch') {
    const searchPresentations = presentations.filter(isFileSearch)
    matchCount = searchPresentations.reduce((sum, p) => sum + p.matchCount, 0)
    fileCount = searchPresentations.reduce((sum, p) => sum + p.fileCount, 0)
  } else if (toolKey === 'webSearch') {
    sourceCount = presentations.filter(isWebSearch).reduce((sum, p) => sum + p.sourceCount, 0)
  }

  return {
    toolKey: toolKey as ToolSummaryPresentation['toolKey'],
    phase: aggregatePhase,
    tone: failed ? 'error' : running ? 'info' : 'neutral',
    icon: summaryIcon(toolKey),
    count: presentations.length,
    running,
    failed,
    matchCount,
    fileCount,
    sourceCount,
    detail: [...summaryDetail(toolKey, presentations).slice(0, 4)],
  }
}
