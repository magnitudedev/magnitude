/**
 * Filesystem Tool Visuals — Renderers
 *
 * Pure render functions for filesystem tool visual state.
 * State is pre-reduced by DisplayProjection.
 */

import { useState, useCallback } from 'react'
import { TextAttributes } from '@opentui/core'
import type {
  ReadState, WriteState, EditState, TreeState, TreeEntry, SearchState, SearchMatch, EditDiff, ToolResult,
} from '@magnitudedev/agent'
import { render, clusterRender } from './define'
import type { ClusterStepData } from './define'
import { Button } from '../components/button'
import { DiffView, computeDiffStats } from '../components/diff-view'
import { ShimmerText } from '../components/shimmer-text'
import { useTheme } from '../hooks/use-theme'

// =============================================================================
// Constants
// =============================================================================

const SHIMMER_INTERVAL_MS = 160

// =============================================================================
// Helpers
// =============================================================================

function parseMatch(m: string): { line: number; text: string } {
  const pipeIdx = m.indexOf('|')
  if (pipeIdx === -1) return { line: 0, text: m }
  const prefix = m.slice(0, pipeIdx)
  const text = m.slice(pipeIdx + 1)
  const colonIdx = prefix.indexOf(':')
  const line = colonIdx !== -1 ? parseInt(prefix.slice(0, colonIdx), 10) || 0 : 0
  return { line, text }
}

function truncateLine(text: string, max: number): string {
  if (!text) return ''
  const firstLine = text.split('\n').find(l => l.trim() !== '') ?? ''
  if (firstLine.length > max) return firstLine.slice(0, max - 3) + '...'
  return firstLine
}

function formatSearchInputs(state: SearchState): string {
  const parts: string[] = []
  if (state.inputs.pattern !== undefined) parts.push(`pattern="${state.inputs.pattern}"`)
  if (state.inputs.path !== undefined) parts.push(`path="${state.inputs.path}"`)
  if (state.inputs.glob !== undefined) parts.push(`glob="${state.inputs.glob}"`)
  if (state.inputs.limit !== undefined) parts.push(`limit=${state.inputs.limit}`)
  return parts.join(' ')
}

/** Extract system rejection reason from ToolResult, or return false. */
function getSystemRejection(result: ToolResult | undefined): string | false {
  if (result?.status === 'rejected' && result.reason) {
    return result.reason
  }
  return false
}

/** Check if this is a user rejection (rejected without reason). */
function isUserRejection(result: ToolResult | undefined): boolean {
  return result?.status === 'rejected' && !result.reason
}

/** Get edit diffs from ToolResult display data (real before/after diffs from tool execution). */
function getResultDiffs(result: ToolResult | undefined): readonly EditDiff[] | null {
  if (result?.status === 'success' && result.display?.type === 'edit_diff') {
    return result.display.diffs
  }
  return null
}

// =============================================================================
// readRender
// =============================================================================

export const readRender = render<ReadState>(({ state }) => {
  const theme = useTheme()
  const isRunning = state.phase !== 'done'
  const isError = state.result?._tag === 'Error'
  const lineCount = state.result?._tag === 'Success'
    ? state.result.output.replace(/\n$/, '').split('\n').length
    : null

  return (
    <box style={{ flexDirection: 'column' }}>
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '→ '}</span>
        {isRunning ? (
          <>
            <span style={{ fg: theme.foreground }}>{'Reading '}</span>
            <span style={{ fg: theme.muted }}>{state.path || '...'}</span>
            <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
          </>
        ) : isError ? (
          <>
            <span style={{ fg: theme.foreground }}>{'Read '}</span>
            <span style={{ fg: theme.muted }}>{state.path}</span>
            <span style={{ fg: theme.error }}>{' · Error'}</span>
            <span style={{ fg: theme.muted }}>{` (${state.result?._tag === 'Error' ? state.result.error : ''})`}</span>
          </>
        ) : (
          <>
            <span style={{ fg: theme.foreground }}>{'Read '}</span>
            <span style={{ fg: theme.muted }}>{state.path}</span>
            {lineCount !== null && (
              <span style={{ fg: theme.info }}>{` · ${lineCount} ${lineCount === 1 ? 'line' : 'lines'}`}</span>
            )}
          </>
        )}
      </text>
    </box>
  )
})

// =============================================================================
// writeRender — restored system/user rejection distinction
// =============================================================================

export const writeRender = render<WriteState>(({ state, isExpanded, onToggle, stepResult }) => {
  const theme = useTheme()
  const isRunning = state.phase !== 'done'
  const isError = state.result?._tag === 'Error'
  const isRejected = state.result?._tag === 'Rejected'
  const isSuccess = state.result?._tag === 'Success'
  const systemReason = getSystemRejection(stepResult)
  const content = state.contentChunks.join('')
  const lineCount = content ? content.split('\n').length : null
  const writeDiffs: EditDiff[] | null = content
    ? [{ startLine: 1, removedLines: [], addedLines: content.split('\n') }]
    : null

  return (
    <box style={{ flexDirection: 'column' }}>
      <Button onClick={onToggle}>
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: (isRejected || isError) ? theme.error : theme.info }}>
            {(isRejected || isError) ? '✗ ' : '✎ '}
          </span>
          {isRunning ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Writing '}</span>
              <span style={{ fg: theme.muted }}>{state.path || '...'}</span>
              {lineCount !== null && (
                <span style={{ fg: theme.info }}>{` · ${lineCount} ${lineCount === 1 ? 'line' : 'lines'}`}</span>
              )}
              <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
            </>
          ) : isRejected ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Write '}</span>
              <span style={{ fg: theme.muted }}>{state.path}</span>
              <span style={{ fg: theme.error }}>{systemReason ? ' · System Rejected' : ' · User Rejected'}</span>
              {systemReason && <span style={{ fg: theme.muted }}>{` (${systemReason})`}</span>}
            </>
          ) : isError ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Write '}</span>
              <span style={{ fg: theme.muted }}>{state.path}</span>
              <span style={{ fg: theme.error }}>{' · Error'}</span>
              <span style={{ fg: theme.muted }}>{` (${state.result?._tag === 'Error' ? state.result.error : ''})`}</span>
            </>
          ) : (
            <>
              <span style={{ fg: theme.foreground }}>{'Wrote '}</span>
              <span style={{ fg: theme.muted }}>{state.path}</span>
              {lineCount !== null && (
                <>
                  <span style={{ fg: theme.info }}>{` · ${lineCount} ${lineCount === 1 ? 'line' : 'lines'}`}</span>
                  <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                    {isExpanded ? ' (collapse)' : ' (expand)'}
                  </span>
                </>
              )}
            </>
          )}
        </text>
      </Button>
      {isExpanded && isSuccess && writeDiffs && (
        <box style={{ flexDirection: 'column', paddingLeft: 2, marginTop: 1 }}>
          <DiffView diffs={writeDiffs} dimmed={true} />
        </box>
      )}
    </box>
  )
})

// =============================================================================
// editRender — per-step fallback (used when cluster renderer is not available)
// =============================================================================

export const editRender = render<EditState>(({ state, isExpanded, onToggle, stepResult }) => {
  const theme = useTheme()
  const isRunning = state.phase !== 'done'
  const isError = state.result?._tag === 'Error'
  const isRejected = state.result?._tag === 'Rejected'
  const isSuccess = state.result?._tag === 'Success'
  const systemReason = getSystemRejection(stepResult)
  const diffs = getResultDiffs(stepResult) ?? []
  const { totalRemoved, totalAdded } = computeDiffStats(diffs)

  return (
    <box style={{ flexDirection: 'column' }}>
      <Button onClick={onToggle}>
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: (isRejected || isError) ? theme.error : theme.info }}>
            {(isRejected || isError) ? '✗ ' : '✎ '}
          </span>
          {isRunning ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Editing '}</span>
              <span style={{ fg: theme.muted }}>{state.path || '...'}</span>
              <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
            </>
          ) : isRejected ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Edit '}</span>
              <span style={{ fg: theme.muted }}>{state.path}</span>
              <span style={{ fg: theme.error }}>{systemReason ? ' · System Rejected' : ' · User Rejected'}</span>
              {systemReason && <span style={{ fg: theme.muted }}>{` (${systemReason})`}</span>}
            </>
          ) : isError ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Edit '}</span>
              <span style={{ fg: theme.muted }}>{state.path}</span>
              <span style={{ fg: theme.error }}>{' · Error'}</span>
              <span style={{ fg: theme.muted }}>{` (${state.result?._tag === 'Error' ? state.result.error : ''})`}</span>
            </>
          ) : (
            <>
              <span style={{ fg: theme.foreground }}>{'Edited '}</span>
              <span style={{ fg: theme.muted }}>{state.path}</span>
              {diffs.length > 0 && (
                <>
                  <span style={{ fg: theme.info }}>{` · ${diffs.length} ${diffs.length === 1 ? 'change' : 'changes'}`}</span>
                  {(totalRemoved > 0 || totalAdded > 0) && (
                    <span style={{ fg: theme.muted }}>
                      {' ('}
                      {totalRemoved > 0 && <span style={{ fg: theme.error }}>-{totalRemoved}</span>}
                      {totalRemoved > 0 && totalAdded > 0 && ', '}
                      {totalAdded > 0 && <span style={{ fg: theme.success }}>+{totalAdded}</span>}
                      {')'}
                    </span>
                  )}
                  <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                    {isExpanded ? ' (collapse)' : ' (expand)'}
                  </span>
                </>
              )}
            </>
          )}
        </text>
      </Button>

      {isExpanded && isSuccess && diffs.length > 0 && (
        <box style={{ flexDirection: 'column', paddingLeft: 2, marginTop: 1 }}>
          <DiffView diffs={diffs} dimmed={true} />
        </box>
      )}
    </box>
  )
})

// =============================================================================
// editClusterRender — cluster renderer that combines consecutive edit steps
// =============================================================================

interface FileGroup {
  path: string
  steps: readonly ClusterStepData<EditState>[]
}

function groupEditStepsByFile(steps: readonly ClusterStepData<EditState>[]): FileGroup[] {
  const groups: FileGroup[] = []
  for (const step of steps) {
    const path = step.visualState.path
    const last = groups[groups.length - 1]
    if (last && last.path === path) {
      ;(last.steps as ClusterStepData<EditState>[]).push(step)
    } else {
      groups.push({ path, steps: [step] })
    }
  }
  return groups
}

function EditFileRow({ group }: { group: FileGroup }) {
  const theme = useTheme()
  const { path, steps } = group
  const [userExpanded, setUserExpanded] = useState(false)

  const hasCompleted = steps.some(s => s.visualState.phase === 'done')
  const allDone = steps.every(s => s.visualState.phase === 'done')
  const hasError = steps.some(s => s.result?.status === 'error')
  const hasRejection = steps.some(s => s.result?.status === 'rejected')

  // Accumulate diffs from all completed steps
  const allDiffs: EditDiff[] = []
  for (const s of steps) {
    const diffs = getResultDiffs(s.result)
    if (diffs) allDiffs.push(...diffs)
  }

  const { totalRemoved, totalAdded, changeCount } = computeDiffStats(allDiffs)

  // While streaming: always show diffs. Once done: collapsed, user can re-expand.
  const showDiffs = allDone ? userExpanded : true

  const onToggle = useCallback(() => {
    setUserExpanded(prev => !prev)
  }, [])

  return (
    <box style={{ flexDirection: 'column' }}>
      <Button onClick={onToggle}>
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: (hasRejection || hasError) ? theme.error : theme.info }}>
            {(hasRejection || hasError) ? '✗ ' : '✎ '}
          </span>
          {!hasCompleted ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Editing '}</span>
              <span style={{ fg: theme.muted }}>{path || '...'}</span>
              <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
            </>
          ) : (
            <>
              <span style={{ fg: theme.foreground }}>{'Edited '}</span>
              <span style={{ fg: theme.muted }}>{path}</span>
              {changeCount > 0 && (
                <>
                  <span style={{ fg: theme.info }}>{` · ${changeCount} ${changeCount === 1 ? 'change' : 'changes'}`}</span>
                  {(totalRemoved > 0 || totalAdded > 0) && (
                    <span style={{ fg: theme.muted }}>
                      {' ('}
                      {totalRemoved > 0 && <span style={{ fg: theme.error }}>-{totalRemoved}</span>}
                      {totalRemoved > 0 && totalAdded > 0 && ', '}
                      {totalAdded > 0 && <span style={{ fg: theme.success }}>+{totalAdded}</span>}
                      {')'}
                    </span>
                  )}
                  <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                    {showDiffs ? ' (collapse)' : ' (expand)'}
                  </span>
                </>
              )}
            </>
          )}
        </text>
      </Button>

      {showDiffs && allDiffs.length > 0 && (
        <box style={{ flexDirection: 'column', paddingLeft: 2, marginTop: 1 }}>
          <DiffView diffs={allDiffs} dimmed={true} />
        </box>
      )}
    </box>
  )
}

export const editClusterRender = clusterRender<EditState>(({ steps }) => {
  const fileGroups = groupEditStepsByFile(steps)

  return (
    <>
      {fileGroups.map((group) => (
        <EditFileRow
          key={group.path || group.steps[0].id}
          group={group}
        />
      ))}
    </>
  )
})

// =============================================================================
// treeRender
// =============================================================================

export const treeRender = render<TreeState>(({ state, isExpanded, onToggle }) => {
  const theme = useTheme()
  const isRunning = state.phase !== 'done'
  const isError = state.result?._tag === 'Error'
  const entries: readonly TreeEntry[] = state.result?._tag === 'Success' ? state.result.output : []
  const fileCount = entries.filter(e => e.type === 'file').length
  const dirCount = entries.filter(e => e.type === 'dir').length

  return (
    <box style={{ flexDirection: 'column' }}>
      <Button onClick={onToggle}>
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '◫ '}</span>
          {isRunning ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Listing '}</span>
              <span style={{ fg: theme.muted }}>{state.path || '...'}</span>
              <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
            </>
          ) : isError ? (
            <>
              <span style={{ fg: theme.foreground }}>{'List '}</span>
              <span style={{ fg: theme.muted }}>{state.path}</span>
              <span style={{ fg: theme.error }}>{' · Error'}</span>
              <span style={{ fg: theme.muted }}>{` (${state.result?._tag === 'Error' ? state.result.error : ''})`}</span>
            </>
          ) : (
            <>
              <span style={{ fg: theme.foreground }}>{'Listed '}</span>
              <span style={{ fg: theme.muted }}>{state.path}</span>
              {entries.length > 0 ? (
                <>
                  <span style={{ fg: theme.info }}>
                    {` · ${fileCount} ${fileCount === 1 ? 'file' : 'files'}`}
                    {dirCount > 0 ? `, ${dirCount} ${dirCount === 1 ? 'dir' : 'dirs'}` : ''}
                  </span>
                  <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                    {isExpanded ? ' (collapse)' : ' (expand)'}
                  </span>
                </>
              ) : (
                <span style={{ fg: theme.muted }}>{' · empty'}</span>
              )}
            </>
          )}
        </text>
      </Button>

      {isExpanded && entries.length > 0 && (
        <box style={{ flexDirection: 'column', paddingLeft: 2 }}>
          {entries.map((entry, i) => (
            <text key={i}>
              <span style={{ fg: theme.muted }}>{'  '.repeat(entry.depth)}</span>
              {entry.type === 'dir' ? (
                <span style={{ fg: theme.directory }}>{entry.name}/</span>
              ) : (
                <span style={{ fg: theme.muted }}>{entry.name}</span>
              )}
            </text>
          ))}
        </box>
      )}
    </box>
  )
})

// =============================================================================
// searchRender
// =============================================================================

export const searchRender = render<SearchState>(({ state, isExpanded, onToggle }) => {
  const theme = useTheme()
  const isRunning = state.phase !== 'done'
  const isError = state.result?._tag === 'Error'
  const matches: readonly SearchMatch[] = state.result?._tag === 'Success' ? state.result.output : []
  const uniqueFiles = new Set(matches.map(m => m.file)).size
  const inputSummary = formatSearchInputs(state)

  return (
    <box style={{ flexDirection: 'column' }}>
      <Button onClick={onToggle}>
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '/ '}</span>
          {isRunning ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Searching '}</span>
              <span style={{ fg: theme.muted }}>{inputSummary || '...'}</span>
              <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
            </>
          ) : isError ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Searched '}</span>
              <span style={{ fg: theme.muted }}>{inputSummary}</span>
              <span style={{ fg: theme.error }}>{' · Error'}</span>
              <span style={{ fg: theme.muted }}>{` (${state.result?._tag === 'Error' ? state.result.error : ''})`}</span>
            </>
          ) : (
            <>
              <span style={{ fg: theme.foreground }}>{'Searched '}</span>
              <span style={{ fg: theme.muted }}>{inputSummary}</span>
              {matches.length > 0 ? (
                <>
                  <span style={{ fg: theme.info }}>{` · ${matches.length} ${matches.length === 1 ? 'match' : 'matches'} in ${uniqueFiles} ${uniqueFiles === 1 ? 'file' : 'files'}`}</span>
                  <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                    {isExpanded ? ' (collapse)' : ' (expand)'}
                  </span>
                </>
              ) : (
                <span style={{ fg: theme.muted }}>{' · no matches'}</span>
              )}
            </>
          )}
        </text>
      </Button>

      {isExpanded && matches.length > 0 && (
        <box style={{ flexDirection: 'column', paddingLeft: 2 }}>
          {matches.map((match, i) => {
            const parsed = parseMatch(match.match)
            return (
              <text key={i}>
                <span style={{ fg: theme.foreground }}>{'- '}{match.file}</span>
                <span style={{ fg: theme.muted }}>{`:${parsed.line}`}</span>
                <span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{`  ${truncateLine(parsed.text, 60)}`}</span>
              </text>
            )
          })}
        </box>
      )}
    </box>
  )
})
