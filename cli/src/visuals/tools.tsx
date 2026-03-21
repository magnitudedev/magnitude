/**
 * Tool Visual Renderers
 *
 * Pure render functions for every tool (except shell and filesystem).
 * State is pre-reduced by DisplayProjection.
 */

import { readFileSync } from 'node:fs'
import { useMemo, useState } from 'react'
import { TextAttributes } from '@opentui/core'

import { useTheme } from '../hooks/use-theme'
import { useStreamingReveal } from '../hooks/use-streaming-reveal'
import { ShimmerText } from '../components/shimmer-text'
import { Button } from '../components/button'
import { DiffHunk } from '../components/diff-hunk'
import { MarkdownContent, StreamingMarkdownContent } from '../markdown/markdown-content'
import { highlightFile } from '../markdown/highlight-file'
import { findUniqueMatchRange } from '../utils/diff-utils'
import { isMarkdownFile, renderCodeLines } from '../utils/file-lang'
import { BOX_CHARS } from '../utils/ui-constants'


type Phase = 'streaming' | 'executing' | 'awaiting_approval' | 'completed' | 'error' | 'rejected' | 'interrupted' | 'done'

function isActive(phase: Phase): boolean {
  return phase === 'streaming' || phase === 'executing' || phase === 'awaiting_approval'
}

type ToolResult<T> =
  | { _tag: 'Success'; output: T }
  | { _tag: 'Error'; error: string }

interface WebSearchSource {
  title: string
  url: string
}

interface WebSearchState {
  phase: Phase
  query?: string
  sources?: WebSearchSource[]
}

interface WebFetchState {
  phase: Phase
  url?: string
}

interface BrowserState {
  phase: Phase
  label?: string
  detail?: string
}

interface WriteState {
  phase: Phase
  path?: string
  contentSoFar?: string
  charCount?: number
  lineCount?: number
  result?: ToolResult<string> | null
}

interface EditState {
  phase: Phase
  path?: string
  oldStringSoFar?: string
  newStringSoFar?: string
  childParsePhase?: 'streaming_old' | 'streaming_new' | 'done'
  result?: ToolResult<string> | null
}

interface AgentCreateState {
  phase: Phase
  id?: string
}

interface AgentIdState {
  phase: Phase
  id?: string
}

interface AgentMessageState {
  phase: Phase
  id?: string
  message?: string
}

interface ParentMessageState {
  phase: Phase
  content?: string
}

interface SkillState {
  phase: Phase
  name?: string
}

import type { ReactNode } from 'react'
import { render } from './define'

// =============================================================================
// Constants
// =============================================================================

const SHIMMER_INTERVAL_MS = 160
const WEB_SEARCH_SHIMMER_MS = 450

// =============================================================================
// Helpers
// =============================================================================

function truncate(s: string, max: number): string {
  if (s.length <= max) return s
  return s.slice(0, max - 1) + '…'
}

function pathSummary(paths: string[], max: number = 3): string {
  if (paths.length === 0) return ''
  const shown = paths.slice(0, max).map(p => {
    const parts = p.split('/')
    return parts.length > 1 ? parts[parts.length - 1] : p
  })
  const rest = paths.length - max
  return rest > 0 ? `${shown.join(', ')} +${rest}` : shown.join(', ')
}



// =============================================================================
// webSearchRender
// =============================================================================

export function webSearchLiveText({ state }: { state: WebSearchState }): string {
  const target = state.query ? `"${state.query}"` : 'the web'
  return isActive(state.phase) ? `Searching web for ${target}` : `Searched web for ${target}`
}



// =============================================================================
// webFetchRender
// =============================================================================

export function webFetchLiveText({ state }: { state: WebFetchState }): string {
  const target = state.url || 'URL'
  if (isActive(state.phase)) return `Fetching ${target}`
  return state.phase === 'error' ? `Fetch ${target}` : `Fetched ${target}`
}



// =============================================================================
// Browser Tools — restored per-tool icons, detail, and colors
// =============================================================================



export function browserLiveText({ state }: { state: BrowserState }): string {
  const label = (state.label ?? '').trim().replace(/\s+/g, ' ')
  const detail = (state.detail ?? '').trim().replace(/\s+/g, ' ')
  if (label.length === 0) return 'Browser action'
  if (detail.length === 0) return label

  const noSpaceBeforeDetail = /^[,.;:!?)]/.test(detail)
  const noSpaceAfterLabel = /[([]$/.test(label)
  const separator = (noSpaceBeforeDetail || noSpaceAfterLabel) ? '' : ' '
  return `${label}${separator}${detail}`
}



export function fsWriteLiveText({ state }: { state: WriteState }): string {
  const target = state.path || 'file'
  return state.phase === 'done' ? `Wrote ${target}` : `Writing ${target}`
}



export function editStreamLiveText({ state }: { state: EditState }): string {
  const target = state.path || 'file'
  return state.phase === 'done' ? `Edited ${target}` : `Editing ${target}`
}



// =============================================================================
// Agent Tools
// =============================================================================

export function agentCreateLiveText({ state }: { state: AgentCreateState }): string {
  const target = state.id ? `agent "${state.id}"` : 'agent'
  if (isActive(state.phase)) return `Starting ${target}`
  return state.phase === 'error' ? `Start ${target}` : `Started ${target}`
}



export function agentDismissLiveText({ state }: { state: AgentIdState }): string {
  const target = state.id ? `agent "${state.id}"` : 'agent'
  if (isActive(state.phase)) return `Dismissing ${target}`
  return `Dismissed ${target}`
}

export const agentDismissRender = render<AgentIdState>(({ state }) => {
  const theme = useTheme()
  const label = state.id ? `Dismissed agent "${state.id}"` : 'Dismissing agent...'
  return (
    <text>
      <span fg={theme.error}>x </span>
      <span fg={theme.foreground}>{label}</span>
    </text>
  )
})

export function agentMessageLiveText({ state }: { state: AgentMessageState }): string {
  const target = state.id ? `agent "${state.id}"` : 'agent'
  if (isActive(state.phase)) return `Messaging ${target}`
  return `Messaged ${target}`
}

export const agentMessageRender = render<AgentMessageState>(({ state, isExpanded, onToggle }) => {
  const theme = useTheme()
  const done = !isActive(state.phase)
  const label = state.id ? `Messaged agent "${state.id}"` : 'Messaging agent...'

  if (state.message) {
    return (
      <box flexDirection="column">
        <Button onClick={onToggle}>
          <text>
            <span fg={done ? theme.foreground : theme.info}>v </span>
            <span fg={theme.foreground}>{label}</span>
            <span fg={theme.secondary} attributes={TextAttributes.DIM}>{isExpanded ? ' (collapse)' : ' (expand)'}</span>
          </text>
        </Button>
        {isExpanded && (
          <box paddingLeft={4}>
            <text fg={theme.muted}>{truncate(state.message, 200)}</text>
          </box>
        )}
      </box>
    )
  }

  return (
    <text>
      <span fg={done ? theme.foreground : theme.info}>v </span>
      <span fg={theme.foreground}>{label}</span>
    </text>
  )
})

// =============================================================================
// parentMessageRender — restored ↑ icon and shimmer
// =============================================================================

export function parentMessageLiveText({ state }: { state: ParentMessageState }): string {
  return isActive(state.phase) ? 'Messaging orchestrator' : 'Messaged orchestrator'
}

export const parentMessageRender = render<ParentMessageState>(({ state, isExpanded, onToggle }) => {
  const theme = useTheme()
  const done = !isActive(state.phase)

  return (
    <box style={{ flexDirection: 'column' }}>
      <Button onClick={onToggle}>
        <text>
          <span style={{ fg: theme.info }}>{'↑ '}</span>
          {!done ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Messaging orchestrator'}</span>
              <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
            </>
          ) : (
            <>
              <span style={{ fg: theme.foreground }}>{'Messaged orchestrator'}</span>
              <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                {isExpanded ? ' (collapse)' : ' (expand)'}
              </span>
            </>
          )}
        </text>
      </Button>
      {isExpanded && state.content && (
        <box style={{ paddingLeft: 2 }}>
          <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
            {state.content}
          </text>
        </box>
      )}
    </box>
  )
})

// =============================================================================
// skillRender
// =============================================================================

export function skillLiveText({ state }: { state: SkillState }): string {
  const target = state.name ? `skill "${state.name}"` : 'skill'
  if (isActive(state.phase)) return `Activating ${target}`
  return `Activated ${target}`
}


