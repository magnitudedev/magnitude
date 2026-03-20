/**
 * Filesystem Tool Visuals — Renderers
 *
 * Pure render functions for filesystem tool visual state.
 * State is pre-reduced by DisplayProjection.
 */

import { useState, useCallback } from 'react'
import { TextAttributes } from '@opentui/core'
import type {
  ReadState, TreeState, TreeEntry, SearchState, SearchMatch, ToolResult,
} from '@magnitudedev/agent'
import { render } from './define'
import { Button } from '../components/button'

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



// =============================================================================
// readRender
// =============================================================================

export function readLiveText({ state }: { state: ReadState }): string {
  const target = state.path || 'file'
  return state.phase === 'done' ? `Read ${target}` : `Reading ${target}`
}

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


// =============================================================================
// treeRender
// =============================================================================

export function treeLiveText({ state }: { state: TreeState }): string {
  const target = state.path || 'files'
  if (state.phase !== 'done') return `Listing ${target}`
  return state.result?._tag === 'Success' ? `Listed ${target}` : `List ${target}`
}

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

export function searchLiveText({ state }: { state: SearchState }): string {
  const summary = formatSearchInputs(state)
  const target = summary.length > 0 ? summary : 'files'
  return state.phase === 'done' ? `Searched ${target}` : `Searching ${target}`
}

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
