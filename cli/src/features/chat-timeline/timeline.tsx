import { memo, useMemo, useState, type ReactNode } from 'react'
import { TextAttributes } from '@opentui/core'
import stringWidth from 'string-width'
import type {
  DisplayTimeline,
  DisplayTimelineEntry,
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
  ToolSummaryPresentation,
} from '@magnitudedev/sdk'
import {
  messageForEntry,
  shortenCommandPreview,
  toolSummaryLabel,
  TRANSCRIPT_LINE_CAP,
  truncateToDisplayWidth,
  type SystemMessage,
} from '@magnitudedev/client-common'
import type { ActionId } from '../../types/ui-actions'
import { MessageView } from './message-view'
import { ErrorBoundary } from '../../components/error-boundary'
import { DiffHunk } from '../../components/diff-hunk'
import { ShimmerText } from '../../components/shimmer-text'
import { Button } from '../../components/button'
import { useTheme } from '../../hooks/use-theme'
import { fitItems } from './fit-items'
import { SystemMessageRow } from './messages/system-message-row'

const MAX_COMMAND_DISPLAY_LEN = 80
const PREVIEW_LINE_CAP = 3

interface ChatTimelineProps {
  timeline: DisplayTimeline | null
  chatColumnWidth: number
  themeErrorColor: string
  /** CLI-local system banners from the slash-command surface (not projected). */
  systemMessages?: readonly SystemMessage[]
  onFileClick: (path: string, section?: string) => void
  onForkExpand: (forkId: string) => void
  onErrorAction: (actionId: ActionId) => void
}

function PathButton({
  path,
  displayPath,
  section,
  onFileClick,
}: {
  path: string
  displayPath?: string | null
  section?: string | null
  onFileClick: (path: string, section?: string) => void
}): ReactNode {
  const theme = useTheme()
  const [hovered, setHovered] = useState(false)
  return (
    <Button
      onClick={() => onFileClick(path, section ?? undefined)}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
    >
      <text>
        <span
          style={{ fg: hovered ? theme.link : theme.accent }}
          attributes={TextAttributes.UNDERLINE}
        >
          {displayPath ?? path}
        </span>
      </text>
    </Button>
  )
}

function SummaryRow({
  entry,
  width,
}: {
  entry: Extract<DisplayTimelineEntry, { kind: 'tool_summary' }>
  width: number
}): ReactNode {
  const theme = useTheme()
  const summary = entry.summary
  const icon = (() => {
    if (summary.failed) return '✗ '
    switch (summary.icon) {
      case 'file': return '→ '
      case 'search': return '/ '
      case 'web': return '[⌕] '
      case 'download': return '[↓] '
      case 'tree': return '◫ '
      default: return '• '
    }
  })()
  const iconColor = summary.failed ? theme.status.failure : theme.status.information
  const label = toolSummaryLabel(summary)
  const detailItems = summary.detail.map((item) => item.text).filter(Boolean)
  const prefix = `${icon}${label}`
  const prefixWidth = stringWidth(prefix)
  const detailWidth = width - prefixWidth - 3
  const { shown, remaining } = detailWidth > 0 ? fitItems(detailItems, detailWidth) : { shown: [], remaining: detailItems.length }

  return (
    <text>
      <span style={{ fg: iconColor }}>{icon}</span>
      <span style={{ fg: theme.text.body }}>{label}</span>
      {shown.length > 0 && <span style={{ fg: theme.text.metadata }}>{' ('}</span>}
      {shown.map((item, index) => (
        <span key={`${item}-${index}`}>
          {index > 0 && <span style={{ fg: theme.text.metadata }}>{', '}</span>}
          <span style={{ fg: theme.text.metadata }}>{item}</span>
        </span>
      ))}
      {remaining > 0 && (
        <span style={{ fg: theme.text.metadata }}>{`${shown.length > 0 ? ', ' : ''}+${remaining} more`}</span>
      )}
      {shown.length > 0 && <span style={{ fg: theme.text.metadata }}>{')'}</span>}
    </text>
  )
}

function buildShellPreview(
  lines: string[],
  availableWidth: number,
  maxLines: number,
): { text: string; truncatedCount: number } {
  if (lines.length === 0) return { text: '', truncatedCount: 0 }

  const truncated = lines.map((line) =>
    line.length <= availableWidth ? line : truncateToDisplayWidth(line, availableWidth),
  )

  if (truncated.length <= maxLines) return { text: truncated.join('\n'), truncatedCount: 0 }

  const budget = maxLines - 1
  const prefixBudget = Math.ceil(budget / 2)
  const suffixBudget = budget - prefixBudget
  const prefix = truncated.slice(0, prefixBudget)
  const suffix = truncated.slice(-suffixBudget)
  const collapsedCount = truncated.length - prefixBudget - suffixBudget

  return {
    text: [...prefix, `... ${collapsedCount} lines collapsed`, ...suffix].join('\n'),
    truncatedCount: collapsedCount,
  }
}

function ShellStep({
  step,
  mode,
  width,
}: {
  step: ShellPresentation
  mode: 'default' | 'transcript'
  width: number
}): ReactNode {
  const theme = useTheme()
  const command = step.command
  const exitCode = step.exitCode
  const isStreaming = step.phase === 'streaming'
  const isExecuting = step.phase === 'executing'
  const isCompleted = step.phase === 'completed'
  const isError = step.phase === 'error'
  const isRejected = step.phase === 'rejected'
  const isInterrupted = step.phase === 'interrupted'
  const isFailed = step.failed || (isCompleted && exitCode != null && exitCode !== 0)
  const outputText = isCompleted ? step.stdout : isExecuting ? step.partialStdout : ''
  const errorText = isCompleted ? step.stderr : isExecuting ? step.partialStderr : ''
  const allLines = (errorText ? errorText.split('\n').concat(outputText ? outputText.split('\n') : []) : outputText.split('\n'))
    .filter((line) => line.length > 0)
  const availableWidth = Math.max(10, width - 4)

  const outputDisplayText = (() => {
    if (mode === 'default') {
      return buildShellPreview(allLines, availableWidth, PREVIEW_LINE_CAP * 2 + 1).text
    }
    const horizontal = allLines.map((line) =>
      line.length <= availableWidth ? line : truncateToDisplayWidth(line, availableWidth),
    )
    if (horizontal.length > TRANSCRIPT_LINE_CAP) {
      const hidden = horizontal.length - TRANSCRIPT_LINE_CAP
      return [...horizontal.slice(0, TRANSCRIPT_LINE_CAP), `...${hidden} lines hidden. Output capped at ${TRANSCRIPT_LINE_CAP} lines`].join('\n')
    }
    return horizontal.join('\n')
  })()

  return (
    <box style={{ flexDirection: 'column', marginBottom: 1 }}>
      <text>
        <span style={{ fg: theme.text.metadata }}>{'$ '}</span>
        <span style={{ fg: isStreaming ? theme.text.metadata : theme.text.body }}>
          {mode === 'transcript' ? command : shortenCommandPreview(command, MAX_COMMAND_DISPLAY_LEN)}
        </span>
        {isStreaming && <span style={{ fg: theme.text.metadata }}>{'▍'}</span>}
        {isExecuting && (
          <>
            <span style={{ fg: theme.text.metadata }}>{' · '}</span>
            <ShimmerText text="Running..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
          </>
        )}
        {isCompleted && (
          <span style={{ fg: isFailed ? theme.status.failure : theme.status.success }}>
            {' '}{isFailed ? `✗ Exit ${exitCode ?? 1}` : '✓'}
          </span>
        )}
        {isError && <span style={{ fg: theme.status.failure }}>{' ✗ Error'}</span>}
        {isRejected && <span style={{ fg: theme.status.inactive }}>{' · Rejected (Permission Policy)'}</span>}
        {isInterrupted && <span style={{ fg: theme.status.interrupted }}>{' · Interrupted'}</span>}
      </text>

      {(isExecuting || isCompleted) && outputDisplayText && (
        mode === 'transcript' ? (
          <box style={{ borderStyle: 'single', border: ['left'], borderColor: theme.border.subtle, paddingLeft: 1 }}>
            <text style={{ fg: isFailed ? theme.status.failure : theme.text.metadata, wrapMode: 'none' }}>
              {outputDisplayText}
            </text>
          </box>
        ) : (
          <text style={{ fg: isFailed ? theme.status.failure : theme.text.metadata, paddingLeft: 2, wrapMode: 'none' }}>
            {outputDisplayText}
          </text>
        )
      )}

      {isError && step.errorText && (
        <text style={{ fg: theme.status.failure, marginTop: 1, paddingLeft: 2 }}>
          {step.errorText}
        </text>
      )}
    </box>
  )
}

function FileWriteStep({
  step,
  entryId,
  onFileClick,
}: {
  step: FileWritePresentation
  entryId: string
  onFileClick: (path: string, section?: string) => void
}): ReactNode {
  const theme = useTheme()
  const path = step.path
  const displayPath = step.displayPath ?? step.path ?? '...'
  const lineCount = step.lineCount
  const isRunning = step.running

  if (step.isScratchpad) {
    return (
      <box style={{ flexDirection: 'row', marginBottom: 1 }}>
        <text>
          <span style={{ fg: step.failed ? theme.status.failure : theme.planAccent }}>{step.failed ? '✗ ' : '✎ '}</span>
          <span style={{ fg: theme.text.body }}>{'Write to scratchpad'}</span>
          <span style={{ fg: theme.text.metadata }}>{' · '}</span>
        </text>
        {path ? (
          <PathButton path={path} displayPath={displayPath} onFileClick={onFileClick} />
        ) : (
          <text><span style={{ fg: theme.accent }}>{displayPath}</span></text>
        )}
        <text>
          {step.failed ? (
            <span style={{ fg: theme.status.failure }}>{' · Error'}</span>
          ) : (
            <span style={{ fg: theme.text.metadata }}>{` · ${lineCount} lines`}</span>
          )}
          {isRunning && <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />}
        </text>
      </box>
    )
  }

  return (
    <box style={{ flexDirection: 'column', marginBottom: 1 }}>
      <box style={{ flexDirection: 'row' }}>
        <text>
          <span style={{ fg: step.failed ? theme.status.failure : theme.status.information }}>{step.failed ? '✗ ' : '✎ '}</span>
          <span style={{ fg: theme.text.body }}>{'Write '}</span>
        </text>
        {path ? (
          <PathButton path={path} displayPath={displayPath} onFileClick={onFileClick} />
        ) : (
          <text><span style={{ fg: theme.accent }}>{displayPath}</span></text>
        )}
        <text>
          {step.failed ? (
            <span style={{ fg: theme.status.failure }}>{' · Error'}</span>
          ) : isRunning ? (
          <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
          ) : (
            <span style={{ fg: theme.status.changeAdded }} attributes={TextAttributes.DIM}>{` +${lineCount}`}</span>
          )}
        </text>
      </box>
      {step.diff?.hunks.map((hunk, index) => (
        <DiffHunk
          key={`${entryId}-${index}`}
          startLine={hunk.startLine}
          contextBefore={hunk.contextBefore}
          removedLines={hunk.removedLines}
          addedLines={hunk.addedLines}
          contextAfter={hunk.contextAfter}
          streamingCursor={hunk.streamingCursor}
        />
      ))}
    </box>
  )
}

function FileEditStep({
  step,
  entryId,
  onFileClick,
}: {
  step: FileEditPresentation
  entryId: string
  onFileClick: (path: string, section?: string) => void
}): ReactNode {
  const theme = useTheme()
  const path = step.path
  const displayPath = step.displayPath ?? step.path ?? '...'
  const added = step.addedCount
  const removed = step.removedCount
  const isRunning = step.running

  if (step.isScratchpad) {
    return (
      <box style={{ flexDirection: 'row', marginBottom: 1 }}>
        <text>
          <span style={{ fg: step.failed ? theme.status.failure : theme.planAccent }}>{step.failed ? '✗ ' : '✎ '}</span>
          <span style={{ fg: theme.text.body }}>{'Edit file in scratchpad'}</span>
          <span style={{ fg: theme.text.metadata }}>{' · '}</span>
        </text>
        {path ? (
          <PathButton path={path} displayPath={displayPath} onFileClick={onFileClick} />
        ) : (
          <text><span style={{ fg: theme.accent }}>{displayPath}</span></text>
        )}
        <text>
          {step.failed ? (
            <span style={{ fg: theme.status.failure }}>{' · Error'}</span>
          ) : added > 0 || removed > 0 ? (
            <>
              <span style={{ fg: theme.text.metadata }}>{' ·'}</span>
              <span style={{ fg: theme.status.changeAdded }} attributes={TextAttributes.DIM}>{` +${added}`}</span>
              <span style={{ fg: theme.text.metadata }} attributes={TextAttributes.DIM}>{'/'}</span>
              <span style={{ fg: theme.status.failure }} attributes={TextAttributes.DIM}>{`-${removed}`}</span>
            </>
          ) : isRunning ? (
          <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
          ) : null}
        </text>
      </box>
    )
  }

  return (
    <box style={{ flexDirection: 'column', marginBottom: 1 }}>
      <box style={{ flexDirection: 'row' }}>
        <text>
          <span style={{ fg: step.failed ? theme.status.failure : theme.status.information }}>{step.failed ? '✗ ' : '✎ '}</span>
          <span style={{ fg: theme.text.body }}>{'Edit '}</span>
        </text>
        {path ? (
          <PathButton path={path} displayPath={displayPath} onFileClick={onFileClick} />
        ) : (
          <text><span style={{ fg: theme.accent }}>{displayPath}</span></text>
        )}
        <text>
          {step.failed ? (
            <span style={{ fg: theme.status.failure }}>{' · Error'}</span>
          ) : isRunning && !step.diff ? (
          <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
          ) : added > 0 || removed > 0 ? (
            <>
              <span style={{ fg: theme.status.changeAdded }} attributes={TextAttributes.DIM}>{` +${added}`}</span>
              <span style={{ fg: theme.text.metadata }} attributes={TextAttributes.DIM}>{'/'}</span>
              <span style={{ fg: theme.status.failure }} attributes={TextAttributes.DIM}>{`-${removed}`}</span>
            </>
          ) : null}
        </text>
      </box>
      {step.diff?.hunks.map((hunk, index) => (
        <DiffHunk
          key={`${entryId}-${index}`}
          startLine={hunk.startLine}
          contextBefore={hunk.contextBefore}
          removedLines={hunk.removedLines}
          addedLines={hunk.addedLines}
          contextAfter={hunk.contextAfter}
          streamingCursor={hunk.streamingCursor}
        />
      ))}
    </box>
  )
}

function SkillStep({
  step,
  onFileClick,
}: {
  step: SkillPresentation
  onFileClick: (path: string, section?: string) => void
}): ReactNode {
  const theme = useTheme()
  const skillName = step.skillName
  const skillPath = step.skillPath

  return (
    <box style={{ flexDirection: 'row', marginBottom: 1 }}>
      <text>
        <span style={{ fg: step.failed ? theme.status.failure : theme.status.information }}>{step.failed ? '✗ ' : '✦ '}</span>
        <span style={{ fg: theme.text.body }}>{'Skill'}</span>
        {skillName && (
          <>
            <span style={{ fg: theme.text.metadata }}>{': '}</span>
            <span style={{ fg: theme.text.metadata }}>{skillName}</span>
          </>
        )}
        {skillPath && <span style={{ fg: theme.text.metadata }}>{' · '}</span>}
      </text>
      {skillPath && (
        <PathButton path={skillPath} onFileClick={onFileClick} />
      )}
      <text>
        {step.failed && <span style={{ fg: theme.status.failure }}>{' · Error'}</span>}
        {step.errorText && <span style={{ fg: theme.text.metadata }}>{` (${step.errorText})`}</span>}
      </text>
    </box>
  )
}

function CheckpointStep({ step }: { step: CheckpointPresentation }): ReactNode {
  const theme = useTheme()
  const isRollback = step.isRollback
  const icon = isRollback ? '↺ ' : '/ '
  const additions = step.additions
  const deletions = step.deletions

  if (step.failed) {
    return (
      <text style={{ fg: theme.status.failure }}>
        {isRollback ? '✗ Roll back · Error' : '✗ Inspect changes · Error'}
      </text>
    )
  }

  return (
    <text>
      <span style={{ fg: theme.status.information }}>{icon}</span>
      <span style={{ fg: theme.text.body }}>{isRollback ? 'Roll back' : 'Inspect changes'}</span>
      {step.since && (
        <>
          <span style={{ fg: theme.text.metadata }}>{' · '}</span>
          <span style={{ fg: theme.text.metadata }}>{step.since}</span>
        </>
      )}
      {step.running ? (
        <>
          <span>{' '}</span>
          <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
        </>
      ) : (
        <>
          <span style={{ fg: theme.text.metadata }}>{' · '}</span>
          <span style={{ fg: theme.status.changeAdded }}>{`+${additions}`}</span>
          <span style={{ fg: theme.text.metadata }}>{'/'}</span>
          <span style={{ fg: theme.status.failure }}>{`-${deletions}`}</span>
          <span style={{ fg: theme.text.metadata }}>{` · ${step.fileCount} file${step.fileCount === 1 ? '' : 's'}`}</span>
        </>
      )}
    </text>
  )
}

function SpawnWorkerStep({ step, mode }: { step: SpawnWorkerPresentation; mode: 'default' | 'transcript' }): ReactNode {
  const theme = useTheme()
  if (mode === 'default') return null
  const message = step.message ?? ''
  const lines = message.split('\n')
  const isTruncated = lines.length > 300
  const visibleLines = isTruncated ? lines.slice(0, 300) : lines
  const hidden = lines.length - 300
  return (
    <box style={{ flexDirection: 'column', marginBottom: 1 }}>
      <text>
        <span style={{ fg: theme.planAccent }}>{'▶ '}</span>
        <span style={{ fg: theme.text.metadata }}>
          {step.role ? `${step.role}` : 'Worker'}
          {step.title ? `: ${step.title}` : ''}
        </span>
        {step.failed && <span style={{ fg: theme.status.failure }}>{' · Error'}</span>}
      </text>
      {message && (
        <box style={{ borderStyle: 'single', border: ['left'], borderColor: theme.border.subtle, paddingLeft: 1 }}>
          <text style={{ fg: theme.text.metadata }}>
            {isTruncated
              ? [...visibleLines, `...${hidden} lines hidden. Content capped at 300 lines`].join('\n')
              : visibleLines.join('\n')}
          </text>
        </box>
      )}
    </box>
  )
}

function CompactPathStep({
  step,
  onFileClick,
}: {
  step: FileReadPresentation | FileViewPresentation
  onFileClick: (path: string, section?: string) => void
}): ReactNode {
  const theme = useTheme()
  const path = step.path
  const verb = step.toolKey === 'fileRead' ? 'Read' : 'View'
  return (
    <box style={{ flexDirection: 'row', marginBottom: 1 }}>
      <text>
        <span style={{ fg: step.failed ? theme.status.failure : theme.status.information }}>{step.failed ? '✗ ' : '• '}</span>
        <span style={{ fg: theme.text.body }}>{verb}</span>
        {path && <span style={{ fg: theme.text.metadata }}>{' '}</span>}
      </text>
      {path && <PathButton path={path} onFileClick={onFileClick} />}
      <text>
        {step.failed ? (
          <span style={{ fg: theme.status.failure }}>{' · Error'}</span>
        ) : step.running ? (
          <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
        ) : step.toolKey === 'fileRead' && step.lineCount != null ? (
          <span style={{ fg: theme.text.metadata }}>{` · ${step.lineCount} lines`}</span>
        ) : null}
      </text>
    </box>
  )
}

function FileSearchStep({ step }: { step: FileSearchPresentation }): ReactNode {
  const theme = useTheme()
  return (
    <box style={{ flexDirection: 'row', marginBottom: 1 }}>
      <text>
        <span style={{ fg: step.failed ? theme.status.failure : theme.status.information }}>{step.failed ? '✗ ' : '/ '}</span>
        <span style={{ fg: theme.text.body }}>{'Search'}</span>
        {step.pattern && (
          <>
            <span style={{ fg: theme.text.metadata }}>{': '}</span>
            <span style={{ fg: theme.text.metadata }}>{step.pattern}</span>
          </>
        )}
      </text>
      <text>
        {step.failed ? (
          <span style={{ fg: theme.status.failure }}>{' · Error'}</span>
        ) : step.running ? (
          <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
        ) : (
          <span style={{ fg: theme.text.metadata }}>{` · ${step.matchCount} match${step.matchCount === 1 ? '' : 'es'} in ${step.fileCount} file${step.fileCount === 1 ? '' : 's'}`}</span>
        )}
      </text>
    </box>
  )
}

function FileTreeStep({ step }: { step: FileTreePresentation }): ReactNode {
  const theme = useTheme()
  return (
    <box style={{ flexDirection: 'row', marginBottom: 1 }}>
      <text>
        <span style={{ fg: step.failed ? theme.status.failure : theme.status.information }}>{step.failed ? '✗ ' : '◫ '}</span>
        <span style={{ fg: theme.text.body }}>{'List files'}</span>
        {step.path && (
          <>
            <span style={{ fg: theme.text.metadata }}>{' · '}</span>
            <span style={{ fg: theme.text.metadata }}>{step.path}</span>
          </>
        )}
      </text>
      <text>
        {step.failed ? (
          <span style={{ fg: theme.status.failure }}>{' · Error'}</span>
        ) : step.running ? (
          <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
        ) : (
          <span style={{ fg: theme.text.metadata }}>{` · ${step.fileCount} file${step.fileCount === 1 ? '' : 's'}, ${step.dirCount} dir${step.dirCount === 1 ? '' : 's'}`}</span>
        )}
      </text>
    </box>
  )
}

function WebSearchStep({ step }: { step: WebSearchPresentation }): ReactNode {
  const theme = useTheme()
  return (
    <box style={{ flexDirection: 'row', marginBottom: 1 }}>
      <text>
        <span style={{ fg: step.failed ? theme.status.failure : theme.status.information }}>{step.failed ? '✗ ' : '[⌕] '}</span>
        <span style={{ fg: theme.text.body }}>{'Web search'}</span>
        {step.query && (
          <>
            <span style={{ fg: theme.text.metadata }}>{': '}</span>
            <span style={{ fg: theme.text.metadata }}>{step.query}</span>
          </>
        )}
      </text>
      <text>
        {step.failed ? (
          <span style={{ fg: theme.status.failure }}>{' · Error'}</span>
        ) : step.running ? (
          <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
        ) : (
          <span style={{ fg: theme.text.metadata }}>{` · ${step.sourceCount} source${step.sourceCount === 1 ? '' : 's'}`}</span>
        )}
      </text>
    </box>
  )
}

function WebFetchStep({ step }: { step: WebFetchPresentation }): ReactNode {
  const theme = useTheme()
  return (
    <box style={{ flexDirection: 'row', marginBottom: 1 }}>
      <text>
        <span style={{ fg: step.failed ? theme.status.failure : theme.status.information }}>{step.failed ? '✗ ' : '[↓] '}</span>
        <span style={{ fg: theme.text.body }}>{'Fetch'}</span>
        {step.url && (
          <>
            <span style={{ fg: theme.text.metadata }}>{' '}</span>
            <span style={{ fg: theme.text.metadata }}>{step.url}</span>
          </>
        )}
      </text>
      <text>
        {step.failed ? (
          <span style={{ fg: theme.status.failure }}>{' · Error'}</span>
        ) : step.running ? (
          <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
        ) : (
          <span style={{ fg: theme.status.success }}>{' · Done'}</span>
        )}
      </text>
    </box>
  )
}

function QueryImageStep({
  step,
  onFileClick,
}: {
  step: QueryImagePresentation
  onFileClick: (path: string, section?: string) => void
}): ReactNode {
  const theme = useTheme()
  const path = step.path
  return (
    <box style={{ flexDirection: 'row', marginBottom: 1 }}>
      <text>
        <span style={{ fg: step.failed ? theme.status.failure : theme.status.information }}>{step.failed ? '✗ ' : '▣ '}</span>
        <span style={{ fg: theme.text.body }}>{'Inspect image'}</span>
        {path && <span style={{ fg: theme.text.metadata }}>{': '}</span>}
      </text>
      {path && <PathButton path={path} onFileClick={onFileClick} />}
      <text>
        {step.failed ? (
          <span style={{ fg: theme.status.failure }}>{' · Error'}</span>
        ) : step.running ? (
            <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
        ) : (
          <span style={{ fg: theme.status.success }}>{' · Done'}</span>
        )}
      </text>
    </box>
  )
}

function GenericStep({ step }: { step: GenericToolPresentation }): ReactNode {
  const theme = useTheme()
  const isErrorLike = step.failed
  return (
    <box style={{ flexDirection: 'column', marginBottom: 1 }}>
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: isErrorLike ? theme.status.failure : theme.status.information }}>{isErrorLike ? '✗ ' : '• '}</span>
        <span style={{ fg: theme.text.body }}>{step.label}</span>
        {step.running ? (
          <>
            <span>{' '}</span>
            <ShimmerText text="..." baseColor={theme.text.metadata} highlightColor={theme.text.body} />
          </>
        ) : (
          <span style={{ fg: isErrorLike ? theme.status.failure : theme.status.success }}>
            {isErrorLike ? ' · Error' : ' · Done'}
          </span>
        )}
      </text>
    </box>
  )
}

function ToolStepView({
  entry,
  mode,
  width,
  onFileClick,
}: {
  entry: Extract<DisplayTimelineEntry, { kind: 'tool_step' }>
  mode: 'default' | 'transcript'
  width: number
  onFileClick: (path: string, section?: string) => void
}): ReactNode {
  const step: ToolStepPresentation = entry.step
  // `GenericToolPresentation.toolKey` is `string` (not a literal), so the
  // union does not narrow on `toolKey` comparisons. The projection guarantees
  // the variant matches `toolKey`, so each case casts to its variant type.
  switch (step.toolKey) {
    case 'shell':
      return <ShellStep step={step as ShellPresentation} mode={mode} width={width} />
    case 'fileWrite':
      return <FileWriteStep step={step as FileWritePresentation} entryId={entry.id} onFileClick={onFileClick} />
    case 'fileEdit':
      return <FileEditStep step={step as FileEditPresentation} entryId={entry.id} onFileClick={onFileClick} />
    case 'skill':
      return <SkillStep step={step as SkillPresentation} onFileClick={onFileClick} />
    case 'checkpointChanges':
    case 'checkpointRollback':
      return <CheckpointStep step={step as CheckpointPresentation} />
    case 'spawnWorker':
      return <SpawnWorkerStep step={step as SpawnWorkerPresentation} mode={mode} />
    case 'queryImage':
      return <QueryImageStep step={step as QueryImagePresentation} onFileClick={onFileClick} />
    case 'fileRead':
    case 'fileView':
      return <CompactPathStep step={step as FileReadPresentation | FileViewPresentation} onFileClick={onFileClick} />
    case 'fileSearch':
      return <FileSearchStep step={step as FileSearchPresentation} />
    case 'fileTree':
      return <FileTreeStep step={step as FileTreePresentation} />
    case 'webSearch':
      return <WebSearchStep step={step as WebSearchPresentation} />
    case 'webFetch':
      return <WebFetchStep step={step as WebFetchPresentation} />
    default:
      return <GenericStep step={step as GenericToolPresentation} />
  }
}

function renderEntries(
  timeline: DisplayTimeline,
  width: number,
  themeErrorColor: string,
  onFileClick: (path: string, section?: string) => void,
  onForkExpand: (forkId: string) => void,
  onErrorAction: (actionId: ActionId) => void,
): ReactNode {
  return timeline.presentation.entries.map((entry, index) => {
    if (entry.kind === 'tool_summary') {
      const next = timeline.presentation.entries[index + 1]
      return (
        <box key={entry.id} id={entry.id} style={{ flexDirection: 'column', paddingLeft: 1, marginBottom: next?.kind === 'tool_summary' ? 0 : 1 }}>
          <ErrorBoundary fallback={(err) => (
            <box style={{ paddingLeft: 1 }}>
              <text style={{ fg: themeErrorColor }}>[Render error: {err.message}]</text>
            </box>
          )}>
            <SummaryRow entry={entry} width={width - 4} />
          </ErrorBoundary>
        </box>
      )
    }

    if (entry.kind === 'tool_step') {
      return (
        <box key={entry.id} id={entry.id} style={{ flexDirection: 'column', paddingLeft: 1 }}>
          <ErrorBoundary fallback={(err) => (
            <box style={{ paddingLeft: 1 }}>
              <text style={{ fg: themeErrorColor }}>[Render error: {err.message}]</text>
            </box>
          )}>
            <ToolStepView
              entry={entry}
              mode={timeline.presentation.mode}
              width={width}
              onFileClick={onFileClick}
            />
          </ErrorBoundary>
        </box>
      )
    }

    const message = messageForEntry(timeline, entry)
    if (!message) return null
    return (
      <box key={entry.id} id={entry.id} style={{ flexDirection: 'column' }}>
        <ErrorBoundary fallback={(err) => (
          <box style={{ paddingLeft: 1 }}>
            <text style={{ fg: themeErrorColor }}>[Render error: {err.message}]</text>
          </box>
        )}>
          <MessageView
            message={message}
            isStreaming={entry.streaming}
            isInterrupted={entry.interrupted}
            nextMessageInterrupted={entry.nextMessageInterrupted}
            mode={timeline.presentation.mode}
            onFileClick={onFileClick}
            onForkExpand={onForkExpand}
            onErrorAction={onErrorAction}
          />
        </ErrorBoundary>
      </box>
    )
  })
}

export const ChatTimeline = memo(function ChatTimeline({
  timeline,
  chatColumnWidth,
  themeErrorColor,
  systemMessages,
  onFileClick,
  onForkExpand,
  onErrorAction,
}: ChatTimelineProps) {
  const entries = useMemo(
    () => timeline ? renderEntries(timeline, chatColumnWidth, themeErrorColor, onFileClick, onForkExpand, onErrorAction) : null,
    [timeline, chatColumnWidth, themeErrorColor, onFileClick, onForkExpand, onErrorAction],
  )

  // CLI-local system banners are not part of the projected timeline.
  const localRows = useMemo(() => {
    const rows: ReactNode[] = []
    if (systemMessages) {
      for (const message of systemMessages) {
        rows.push(<SystemMessageRow key={`system:${message.id}`} message={message} />)
      }
    }
    return rows.length > 0 ? rows : null
  }, [systemMessages])

  if (!timeline) return null

  // No wrapper box: OpenTUI viewport culling only culls DIRECT children of
  // the scrollbox content. A single wrapper is always "visible", so every
  // offscreen entry inside it gets walked on each scroll frame (measured
  // ~5.8x scroll-frame cost). Entries must land as direct children.
  return (
    <>
      {entries}
      {localRows}
    </>
  )
})
