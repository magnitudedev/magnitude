/**
 * Tool Visual Renderers
 *
 * Pure render functions for every tool (except shell and filesystem).
 * State is pre-reduced by DisplayProjection.
 */

import { useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { useStreamingReveal } from '../hooks/use-streaming-reveal'
import { ShimmerText } from '../components/shimmer-text'
import { Button } from '../components/button'
import { MarkdownContent, StreamingMarkdownContent } from '../components/markdown-content'
import { BOX_CHARS } from '../utils/ui-constants'
import { render } from './define'
import type { ToolVisualRenderer } from './define'

import { isActive } from '@magnitudedev/agent'
import { useSelectedArtifact } from '../hooks/use-artifacts'

import type {
  WebSearchState,
  WebFetchState,
  BrowserState,
  ArtifactVisualState,
  ArtifactSyncState,
  AgentCreateState,
  AgentIdState,
  AgentMessageState,
  ParentMessageState,
  SkillState,
} from '@magnitudedev/agent'

import type { ReactNode } from 'react'

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

export const webSearchRender = render<WebSearchState>(({ state, isExpanded, onToggle }) => {
  const theme = useTheme()

  if (isActive(state.phase)) {
    return (
      <Button onClick={onToggle}>
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: theme.info }}>[⌕] </span>
          <span style={{ fg: theme.foreground }}>{'Searching web for '}</span>
          <span style={{ fg: theme.muted }}>{`"${state.query ? truncate(state.query, 50) : '...'}"`}</span>
          <ShimmerText text=" ..." interval={WEB_SEARCH_SHIMMER_MS} primaryColor={theme.info} />
        </text>
      </Button>
    )
  }

  const isError = state.phase === 'error'
  const sources = state.sources

  if (isError) {
    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: theme.error }}>{'✗  '}</span>
        <span style={{ fg: theme.foreground }}>{'Searched web for '}</span>
        <span style={{ fg: theme.muted }}>{`"${state.query ? truncate(state.query, 50) : ''}"`}</span>
        <span style={{ fg: theme.error }}>{' · Error'}</span>
      </text>
    )
  }

  if (sources.length > 0) {
    return (
      <box flexDirection="column">
        <Button onClick={onToggle}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: theme.info }}>[⌕] </span>
            <span style={{ fg: theme.foreground }}>{'Searched web for '}</span>
            <span style={{ fg: theme.muted }}>{`"${truncate(state.query, 50)}"`}</span>
            <span style={{ fg: theme.info }}>{` · ${sources.length} ${sources.length === 1 ? 'source' : 'sources'}`}</span>
            <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>{isExpanded ? ' (collapse)' : ' (expand)'}</span>
          </text>
        </Button>
        {isExpanded && (
          <box style={{ flexDirection: 'column', paddingLeft: 2 }}>
            {sources.map((src, i) => (
              <text key={i}>
                <span style={{ fg: theme.foreground }}>{'- '}{src.title}</span>
                <span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{`: ${truncate(src.url, 60)}`}</span>
              </text>
            ))}
          </box>
        )}
      </box>
    )
  }

  return (
    <text style={{ wrapMode: 'word' }}>
      <span style={{ fg: theme.info }}>[⌕] </span>
      <span style={{ fg: theme.foreground }}>{'Searched web for '}</span>
      <span style={{ fg: theme.muted }}>{`"${truncate(state.query, 50)}"`}</span>
      <span style={{ fg: theme.muted }}>{' · No Sources Found'}</span>
    </text>
  )
})

// =============================================================================
// webFetchRender
// =============================================================================

export function webFetchLiveText({ state }: { state: WebFetchState }): string {
  const target = state.url || 'URL'
  if (isActive(state.phase)) return `Fetching ${target}`
  return state.phase === 'error' ? `Fetch ${target}` : `Fetched ${target}`
}

export const webFetchRender = render<WebFetchState>(({ state, stepResult }) => {
  const theme = useTheme()

  if (isActive(state.phase)) {
    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: theme.info }}>[↓] </span>
        <span style={{ fg: theme.foreground }}>{'Fetching '}</span>
        <span style={{ fg: theme.muted }}>{state.url ? truncate(state.url, 60) : '...'}</span>
        <ShimmerText text=" ..." interval={WEB_SEARCH_SHIMMER_MS} primaryColor={theme.info} />
      </text>
    )
  }

  if (state.phase === 'error') {
    const errorMsg = stepResult?.status === 'error' ? stepResult.message : ''
    return (
      <text style={{ wrapMode: 'word' }}>
        <span style={{ fg: theme.error }}>{'✗  '}</span>
        <span style={{ fg: theme.foreground }}>{'Fetch '}</span>
        <span style={{ fg: theme.muted }}>{truncate(state.url, 60)}</span>
        <span style={{ fg: theme.error }}>{` · Error${errorMsg ? ` (${truncate(errorMsg, 80)})` : ''}`}</span>
      </text>
    )
  }

  return (
    <text style={{ wrapMode: 'word' }}>
      <span style={{ fg: theme.info }}>[↓] </span>
      <span style={{ fg: theme.foreground }}>{'Fetched '}</span>
      <span style={{ fg: theme.muted }}>{truncate(state.url, 60)}</span>
    </text>
  )
})

// =============================================================================
// Browser Tools — restored per-tool icons, detail, and colors
// =============================================================================

function browserRender(config: {
  icon: string
  pendingIcon?: string
}): ToolVisualRenderer {
  return render<BrowserState>(({ state }): ReactNode => {
    const theme = useTheme()
    const icon = isActive(state.phase) ? (config.pendingIcon ?? config.icon) : config.icon
    const isError = state.phase === 'error'

    if (isActive(state.phase)) {
      return (
        <box style={{ flexDirection: 'column' }}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: theme.info }}>{icon} </span>
            <span style={{ fg: theme.foreground }}>{state.label}</span>
            {state.detail ? <span style={{ fg: theme.muted }}>{state.detail}</span> : null}
            <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
          </text>
        </box>
      )
    }

    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : `${icon} `}</span>
          <span style={{ fg: theme.foreground }}>{state.label}</span>
          {state.detail ? <span style={{ fg: theme.muted }}>{state.detail}</span> : null}
          {isError && <span style={{ fg: theme.error }}>{' · Error'}</span>}
        </text>
      </box>
    )
  })
}

export function browserLiveText({ state }: { state: BrowserState }): string {
  const label = state.label.trim().replace(/\s+/g, ' ')
  const detail = (state.detail ?? '').trim().replace(/\s+/g, ' ')
  if (label.length === 0) return 'Browser action'
  if (detail.length === 0) return label

  const noSpaceBeforeDetail = /^[,.;:!?)]/.test(detail)
  const noSpaceAfterLabel = /[([]$/.test(label)
  const separator = (noSpaceBeforeDetail || noSpaceAfterLabel) ? '' : ' '
  return `${label}${separator}${detail}`
}

export const clickRender = browserRender({ icon: '◎' })
export const doubleClickRender = browserRender({ icon: '◎◎' })
export const rightClickRender = browserRender({ icon: '◎' })
export const typeRender = browserRender({ icon: '⌨' })
export const scrollRender = browserRender({ icon: '↕' })
export const dragRender = browserRender({ icon: '⤳' })
export const navigateRender = browserRender({ icon: '→' })
export const goBackRender = browserRender({ icon: '←' })
export const switchTabRender = browserRender({ icon: '⇥' })
export const newTabRender = browserRender({ icon: '+' })
export const screenshotRender = browserRender({ icon: '◻' })
export const evaluateRender = browserRender({ icon: '▶' })

// =============================================================================
// Artifact Tools — restored interactivity, styling, and shimmer
// =============================================================================

export function artifactCreateLiveText({ state }: { state: ArtifactVisualState }): string {
  const target = state.name ? `artifact ${state.name}` : 'artifact'
  if (isActive(state.phase)) return `Creating ${target}`
  return state.phase === 'error' ? `Create ${target}` : `Created ${target}`
}

export const artifactCreateRender = render<ArtifactVisualState>(({ state }) => {
  const theme = useTheme()
  const done = !isActive(state.phase)
  const label = state.name ? `Created artifact "${state.name}"` : 'Creating artifact...'

  if (!done) {
    return (
      <text>
        <span fg={theme.info}>{'≡ + '}</span>
        <ShimmerText text={label} interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
      </text>
    )
  }

  return (
    <text>
      <span fg={state.phase === 'error' ? theme.error : theme.success}>{'≡ + '}</span>
      <span fg={theme.foreground}>{label}</span>
    </text>
  )
})

export function artifactSyncLiveText({ state }: { state: ArtifactSyncState }): string {
  const target = state.name ? `artifact ${state.name}` : 'artifact'
  if (isActive(state.phase)) return `Syncing ${target}`
  return state.phase === 'error' ? `Sync ${target}` : `Synced ${target}`
}

export const artifactSyncRender = render<ArtifactSyncState>(({ state }) => {
  const theme = useTheme()
  const done = !isActive(state.phase)
  const label = state.name
    ? `Synced artifact "${state.name}" → ${state.path || '...'}`
    : 'Syncing artifact...'

  if (!done) {
    return (
      <text>
        <span fg={theme.info}>{'≡ ✎ '}</span>
        <ShimmerText text={label} interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
      </text>
    )
  }

  return (
    <text>
      <span fg={state.phase === 'error' ? theme.error : theme.success}>{'≡ ✎ '}</span>
      <span fg={theme.foreground}>{label}</span>
    </text>
  )
})

export function artifactReadLiveText({ state }: { state: ArtifactVisualState }): string {
  const target = state.name ? `artifact ${state.name}` : 'artifact'
  if (isActive(state.phase)) return `Reading ${target}`
  return `Read ${target}`
}

export const artifactReadRender = render<ArtifactVisualState>(({ state, onArtifactClick }) => {
  const theme = useTheme()
  const done = !isActive(state.phase)
  const isError = state.phase === 'error'
  const [isHovered, setIsHovered] = useState(false)
  const name = state.name

  return (
    <box style={{ flexDirection: 'column' }}>
      <Button
        onClick={() => { if (name) onArtifactClick?.(name) }}
        onMouseOver={() => setIsHovered(true)}
        onMouseOut={() => setIsHovered(false)}
      >
        <text style={{ wrapMode: 'word' }}>
          <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '≡ → '}</span>
          {!done ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Reading artifact '}</span>
              <span style={{ fg: theme.muted }}>{name || '...'}</span>
              <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
            </>
          ) : isError ? (
            <>
              <span style={{ fg: theme.foreground }}>{'Read artifact '}</span>
              <span style={{ fg: theme.muted }}>{name}</span>
              <span style={{ fg: theme.error }}>{' · Error'}</span>
            </>
          ) : (
            <>
              <span style={{ fg: theme.foreground }}>{'Read artifact '}</span>
              <span style={{ fg: isHovered ? theme.link : theme.primary }} attributes={TextAttributes.UNDERLINE}>{name}</span>
            </>
          )}
        </text>
      </Button>
    </box>
  )
})

export function artifactWriteLiveText({ state }: { state: ArtifactVisualState }): string {
  const target = state.name ? `artifact ${state.name}` : 'artifact'
  if (isActive(state.phase)) return `Writing ${target}`
  return state.phase === 'error' ? `Write ${target}` : `Wrote ${target}`
}

export const artifactWriteRender = render<ArtifactVisualState>(({ state, onArtifactClick }) => {
  const theme = useTheme()
  const done = !isActive(state.phase)
  const isError = state.phase === 'error'
  const [isHovered, setIsHovered] = useState(false)
  const name = state.name
  const preview = state.preview?.mode === 'write' ? state.preview : null
  const selectedArtifact = useSelectedArtifact()
  const isOpenInPanel = selectedArtifact === name
  const fullPreviewContent = preview?.contentSoFar ?? ''
  const { displayedContent: revealedFull, showCursor: previewCursor } = useStreamingReveal(fullPreviewContent, !done)


  return (
    <box style={{ flexDirection: 'column' }}>
      <Button
        onClick={() => { if (name) onArtifactClick?.(name) }}
        onMouseOver={() => setIsHovered(true)}
        onMouseOut={() => setIsHovered(false)}
      >
        <box style={{ flexDirection: 'column' }}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '≡ ✎ '}</span>
            {!done ? (
              <>
                <span style={{ fg: theme.foreground }}>{'Writing artifact '}</span>
                <span style={{ fg: theme.muted }}>{name || '...'}</span>
                <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
              </>
            ) : isError ? (
              <>
                <span style={{ fg: theme.foreground }}>{'Write artifact '}</span>
                <span style={{ fg: theme.muted }}>{name}</span>
                <span style={{ fg: theme.error }}>{' · Error'}</span>
              </>
            ) : (
              <>
                <span style={{ fg: theme.foreground }}>{'Wrote artifact '}</span>
                <span style={{ fg: isHovered ? theme.link : theme.primary }} attributes={TextAttributes.UNDERLINE}>{name}</span>
              </>
            )}
          </text>
          {!done && preview && (
            <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
              {`${preview.charCount} chars · ${preview.lineCount} lines`}
            </text>
          )}
          {!done && preview && preview.contentSoFar.length > 0 && !isOpenInPanel && (
            <box style={{
              borderStyle: 'single',
              borderColor: isHovered ? theme.link : theme.border || theme.muted,
              customBorderChars: BOX_CHARS,
              height: 12,
            }}>
              <scrollbox
                stickyScroll
                stickyStart="bottom"
                scrollX={false}
                scrollbarOptions={{ visible: false }}
                verticalScrollbarOptions={{ visible: false }}
                style={{
                  flexGrow: 1,
                  rootOptions: { flexGrow: 1, backgroundColor: 'transparent' },
                  wrapperOptions: { border: false, backgroundColor: 'transparent', paddingLeft: 1, paddingRight: 1 },
                  contentOptions: { justifyContent: 'flex-start' },
                }}
              >
                <StreamingMarkdownContent content={revealedFull} showCursor={previewCursor} />
              </scrollbox>
            </box>
          )}
        </box>
      </Button>
    </box>
  )
})

export function artifactUpdateLiveText({ state }: { state: ArtifactVisualState }): string {
  const target = state.name ? `artifact ${state.name}` : 'artifact'
  if (isActive(state.phase)) return `Updating ${target}`
  return state.phase === 'error' ? `Update ${target}` : `Updated ${target}`
}

export const artifactUpdateRender = render<ArtifactVisualState>(({ state, onArtifactClick }) => {
  const theme = useTheme()
  const done = !isActive(state.phase)
  const isError = state.phase === 'error'
  const [isHovered, setIsHovered] = useState(false)
  const name = state.name
  const preview = state.preview?.mode === 'update' ? state.preview : null
  const selectedArtifact = useSelectedArtifact()
  const isOpenInPanel = selectedArtifact === name
  const fullUpdateContent = preview?.newStringSoFar ?? ''
  const { displayedContent: revealedUpdateFull, showCursor: updatePreviewCursor } = useStreamingReveal(fullUpdateContent, !done)

  const showOldPreview = !done
    && !!preview
    && preview.childPhase === 'streaming_old'
    && preview.oldStringSoFar.length > 0
    && !isOpenInPanel
  const showNewPreview = !done
    && !!preview
    && preview.childPhase === 'streaming_new'
    && preview.newStringSoFar.length > 0
    && !isOpenInPanel

  return (
    <box style={{ flexDirection: 'column' }}>
      <Button
        onClick={() => { if (name) onArtifactClick?.(name) }}
        onMouseOver={() => setIsHovered(true)}
        onMouseOut={() => setIsHovered(false)}
      >
        <box style={{ flexDirection: 'column' }}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '≡ ✎ '}</span>
            {!done ? (
              <>
                <span style={{ fg: theme.foreground }}>{'Updating artifact '}</span>
                <span style={{ fg: theme.muted }}>{name || '...'}</span>
                <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
              </>
            ) : isError ? (
              <>
                <span style={{ fg: theme.foreground }}>{'Update artifact '}</span>
                <span style={{ fg: theme.muted }}>{name}</span>
                <span style={{ fg: theme.error }}>{' · Error'}</span>
              </>
            ) : (
              <>
                <span style={{ fg: theme.foreground }}>{'Updated artifact '}</span>
                <span style={{ fg: isHovered ? theme.link : theme.primary }} attributes={TextAttributes.UNDERLINE}>{name}</span>
              </>
            )}
          </text>
          {!done && preview && (
            <>
              <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
                {`old: ${preview.oldStringSoFar.length} chars → new: ${preview.newStringSoFar.length} chars`}
                {preview.replaceAll ? ' · replace all' : ''}
              </text>
              {showOldPreview && (
                <box style={{
                  borderStyle: 'single',
                  borderColor: theme.warning,
                  customBorderChars: BOX_CHARS,
                  height: 12,
                }}>
                  <scrollbox
                    stickyScroll
                    stickyStart="bottom"
                    scrollX={false}
                    scrollbarOptions={{ visible: false }}
                    verticalScrollbarOptions={{ visible: false }}
                    style={{
                      flexGrow: 1,
                      rootOptions: { flexGrow: 1, backgroundColor: 'transparent' },
                      wrapperOptions: { border: false, backgroundColor: 'transparent', paddingLeft: 1, paddingRight: 1 },
                      contentOptions: { justifyContent: 'flex-start' },
                    }}
                  >
                    <StreamingMarkdownContent content={preview?.oldStringSoFar ?? ''} />
                  </scrollbox>
                </box>
              )}
              {showNewPreview && (
                <box style={{
                  borderStyle: 'single',
                  borderColor: theme.success,
                  customBorderChars: BOX_CHARS,
                  height: 12,
                }}>
                  <scrollbox
                    stickyScroll
                    stickyStart="bottom"
                    scrollX={false}
                    scrollbarOptions={{ visible: false }}
                    verticalScrollbarOptions={{ visible: false }}
                    style={{
                      flexGrow: 1,
                      rootOptions: { flexGrow: 1, backgroundColor: 'transparent' },
                      wrapperOptions: { border: false, backgroundColor: 'transparent', paddingLeft: 1, paddingRight: 1 },
                      contentOptions: { justifyContent: 'flex-start' },
                    }}
                  >
                    <StreamingMarkdownContent content={revealedUpdateFull} showCursor={updatePreviewCursor} />
                  </scrollbox>
                </box>
              )}
            </>
          )}
        </box>
      </Button>
    </box>
  )
})

// =============================================================================
// Agent Tools
// =============================================================================

export function agentCreateLiveText({ state }: { state: AgentCreateState }): string {
  const target = state.id ? `agent "${state.id}"` : 'agent'
  if (isActive(state.phase)) return `Starting ${target}`
  return state.phase === 'error' ? `Start ${target}` : `Started ${target}`
}

export const agentCreateRender = render<AgentCreateState>(({ state }) => {
  const theme = useTheme()
  const label = state.id ? `Started agent "${state.id}"` : 'Starting agent...'

  if (isActive(state.phase)) {
    return (
      <text>
        <span fg={theme.info}>{'> '}</span>
        <ShimmerText text={label} primaryColor={theme.info} />
      </text>
    )
  }

  return (
    <text>
      <span fg={state.phase === 'error' ? theme.error : theme.success}>{'> '}</span>
      <span fg={theme.foreground}>{label}</span>
    </text>
  )
})

export function agentPauseLiveText({ state }: { state: AgentIdState }): string {
  const target = state.id ? `agent "${state.id}"` : 'agent'
  if (isActive(state.phase)) return `Pausing ${target}`
  return `Paused ${target}`
}

export const agentPauseRender = render<AgentIdState>(({ state }) => {
  const theme = useTheme()
  const label = state.id ? `Paused agent "${state.id}"` : 'Pausing agent...'
  return (
    <text>
      <span fg={theme.warning}>|| </span>
      <span fg={theme.foreground}>{label}</span>
    </text>
  )
})

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

export const skillRender = render<SkillState>(({ state }) => {
  const theme = useTheme()
  const done = !isActive(state.phase)
  const label = state.name ? `Activated skill "${state.name}"` : 'Activating skill...'
  return (
    <text>
      <span fg={done ? theme.primary : theme.info}>* </span>
      <span fg={done ? theme.foreground : theme.muted}>{label}</span>
    </text>
  )
})
