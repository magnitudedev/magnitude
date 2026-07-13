import { useCallback, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import type { MentionCandidate } from '@magnitudedev/sdk'
import type { SearchMentionsResult } from '@magnitudedev/sdk'
import type { KeyEvent } from '../types/key-event'

/**
 * Structural type for the subset of the client that `useFileMentions` needs.
 * This decouples the hook from the full `VanillaAcnClient` interface so
 * web/desktop consumers can provide their own adapter.
 */
export interface MentionSearchClient {
  readonly searchMentions: (payload: {
    cwd: string
    query: string
    limit?: number
    visibleLimit?: number
    includeRecent?: boolean
  }) => Promise<SearchMentionsResult>
}

const MAX_RESULTS = 40
const MAX_VISIBLE_RESULTS = 10

export type MentionFileItem = {
  path: string
  kind: 'file' | 'directory'
  contentType: 'text' | 'directory'
  warning?: boolean
  lineRange?: { start: number; end: number }
}

interface FileMentionsState {
  isOpen: boolean
  query: string
  queryLineRange?: { start: number; end: number }
  items: MentionFileItem[]
  recentItems: MentionFileItem[]
  overflowCount: number
  selectedIndex: number
  setSelectedIndex: (index: number) => void
  loading: boolean
  moveUp: () => void
  moveDown: () => void
  select: () => MentionFileItem | null
  confirmSelection: (item: MentionFileItem) => void
  close: () => void
  handleKeyIntercept: (key: KeyEvent) => boolean
}

function parseQueryLineRange(query: string): { filePath: string; lineRange?: { start: number; end: number } } {
  if (query.endsWith(':')) return { filePath: query.slice(0, -1) }

  const rangeMatch = query.match(/:([\d]+)(?:-([\d]+))?$/)
  if (!rangeMatch || rangeMatch.index === 1) return { filePath: query }

  const filePath = query.slice(0, rangeMatch.index)
  const start = parseInt(rangeMatch[1], 10)
  const end = rangeMatch[2] ? parseInt(rangeMatch[2], 10) : start

  if (start < 1 || end < 1 || end < start) return { filePath: query }

  return { filePath, lineRange: { start, end } }
}

function detectQuery(inputText: string, cursorPosition: number): { raw: string; filePath: string; lineRange?: { start: number; end: number } } | null {
  const left = inputText.slice(0, Math.max(0, cursorPosition))
  const match = left.match(/(?:^|\s)@([^\s@]*)$/)
  if (!match) return null
  const raw = match[1] ?? ''
  return { raw, ...parseQueryLineRange(raw) }
}

function toMentionFileItem(candidate: MentionCandidate): MentionFileItem {
  return {
    path: candidate.path,
    kind: candidate.kind,
    contentType: candidate.contentType,
    ...(candidate.warning ? { warning: candidate.warning } : {}),
    ...(candidate.lineRange ? { lineRange: candidate.lineRange } : {}),
  }
}

export interface UseFileMentionsParams {
  inputText: string
  cursorPosition: number
  client: MentionSearchClient | null
  cwd: string | null
  onConfirm?: (item: MentionFileItem) => void
  onExpandDirectory?: (item: MentionFileItem) => void
}

// ---------------------------------------------------------------------------
// Mention search store — powers the async fetch via useSyncExternalStore.
//
// The store is created once per hook instance (useRef) and updated with the
// latest search inputs every render. When the effective search key changes,
// it cancels the in-flight fetch and starts a new one, then notifies
// subscribers. useSyncExternalStore reads the snapshot synchronously.
// ---------------------------------------------------------------------------

interface SearchSnapshot {
  items: MentionFileItem[]
  recentItems: MentionFileItem[]
  overflowCount: number
  resolvedLineRange: { start: number; end: number } | undefined
  loading: boolean
}

const EMPTY_SNAPSHOT: SearchSnapshot = {
  items: [],
  recentItems: [],
  overflowCount: 0,
  resolvedLineRange: undefined,
  loading: false,
}

interface SearchInputs {
  isOpen: boolean
  client: MentionSearchClient | null
  cwd: string | null
  rawQuery: string
}

class MentionSearchStore {
  private snapshot: SearchSnapshot = EMPTY_SNAPSHOT
  private listeners = new Set<() => void>()
  private cancelToken: { cancelled: boolean } | null = null
  private lastKey = ''

  /** Inputs for the current/last search — set every render. */
  updateInputs(inputs: SearchInputs): void {
    const key = `${inputs.isOpen ? 1 : 0}:${inputs.client ? 1 : 0}:${inputs.cwd ?? ''}:${inputs.rawQuery}`

    if (key === this.lastKey) return
    this.lastKey = key

    // Cancel any in-flight fetch.
    if (this.cancelToken) {
      this.cancelToken.cancelled = true
      this.cancelToken = null
    }

    if (!inputs.isOpen || !inputs.client || !inputs.cwd) {
      // Closed or no client — reset to empty.
      this.setSnapshot(EMPTY_SNAPSHOT)
      return
    }

    // Start a new fetch.
    const token: { cancelled: boolean } = { cancelled: false }
    this.cancelToken = token

    this.setSnapshot({ ...this.snapshot, loading: true })

    inputs.client.searchMentions({
      cwd: inputs.cwd,
      query: inputs.rawQuery,
      limit: MAX_RESULTS,
      visibleLimit: MAX_VISIBLE_RESULTS,
      includeRecent: true,
    }).then((result) => {
      if (token.cancelled) return
      this.setSnapshot({
        items: result.candidates.map(toMentionFileItem),
        recentItems: result.recentCandidates.map(toMentionFileItem),
        overflowCount: result.overflowCount,
        resolvedLineRange: result.lineRange,
        loading: false,
      })
    }).catch((error) => {
      if (token.cancelled) return
      console.error('[use-file-mentions] Failed to search mentions:', error)
      this.setSnapshot({
        items: [],
        recentItems: [],
        overflowCount: 0,
        resolvedLineRange: undefined,
        loading: false,
      })
    })
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  getSnapshot = (): SearchSnapshot => this.snapshot

  private setSnapshot(next: SearchSnapshot): void {
    this.snapshot = next
    for (const listener of this.listeners) listener()
  }
}

export function useFileMentions({
  inputText,
  cursorPosition,
  client,
  cwd,
  onConfirm,
  onExpandDirectory,
}: UseFileMentionsParams): FileMentionsState {
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [dismissedQuery, setDismissedQuery] = useState<string | null>(null)
  const justConfirmedRef = useRef(false)

  const queryResult = useMemo(() => detectQuery(inputText, cursorPosition), [inputText, cursorPosition])
  const query = queryResult?.filePath ?? ''
  const rawQuery = queryResult?.raw ?? ''

  // --- dismissedQuery / justConfirmedRef reset (was useEffect #1) ---
  //
  // When the query changes away from the dismissed value, clear dismissedQuery.
  // When there's no query at all, reset justConfirmedRef.
  // Done in the render phase via ref-diff instead of useEffect.
  if (queryResult?.filePath !== dismissedQuery && dismissedQuery !== null) {
    setDismissedQuery(null)
  }
  if (queryResult === null && justConfirmedRef.current) {
    justConfirmedRef.current = false
  }

  const isOpen = queryResult !== null && queryResult.filePath !== dismissedQuery && !justConfirmedRef.current

  // --- Async search (was useEffect #2) ---
  //
  // A MentionSearchStore instance is kept in a ref and fed the latest inputs
  // every render. useSyncExternalStore subscribes to it, giving us the
  // search results reactively without useEffect.
  const storeRef = useRef<MentionSearchStore | null>(null)
  if (storeRef.current === null) {
    storeRef.current = new MentionSearchStore()
  }
  const store = storeRef.current

  // Feed inputs every render — the store diffs internally and only
  // re-fetches when the key actually changes.
  store.updateInputs({ isOpen, client, cwd, rawQuery })

  const { items, recentItems, overflowCount, resolvedLineRange, loading } =
    useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)

  // --- selectedIndex reset/clamp (was useEffect #3) ---
  //
  // Reset to 0 when the items signature changes, clamp when out of bounds.
  // Render-phase ref-diff pattern.
  const prevSignatureRef = useRef<string>('')
  const signature = `${rawQuery}::${items.map(i => i.path).join(',')}`
  if (signature !== prevSignatureRef.current) {
    prevSignatureRef.current = signature
    if (selectedIndex !== 0) {
      setSelectedIndex(0)
    }
  } else if (items.length > 0 && selectedIndex >= items.length) {
    setSelectedIndex(items.length - 1)
  } else if (items.length === 0 && selectedIndex !== 0) {
    setSelectedIndex(0)
  }

  const moveUp = useCallback(() => {
    setSelectedIndex(prev => Math.max(0, prev - 1))
  }, [])

  const moveDown = useCallback(() => {
    setSelectedIndex(prev => Math.min(Math.max(0, items.length - 1), prev + 1))
  }, [items.length])

  const select = useCallback((): MentionFileItem | null => {
    return items[selectedIndex] ?? null
  }, [items, selectedIndex])

  const close = useCallback(() => {
    setSelectedIndex(0)
    setDismissedQuery(query ?? null)
  }, [query])

  const confirmSelection = useCallback((item: MentionFileItem) => {
    justConfirmedRef.current = true
    setDismissedQuery(query ?? null)
    if (onConfirm) onConfirm(item)
  }, [onConfirm, query])

  const handleKeyIntercept = useCallback((key: KeyEvent): boolean => {
    if (!isOpen) return false

    const isUp = key.name === 'up' && !key.ctrl && !key.meta && !key.option
    const isDown = key.name === 'down' && !key.ctrl && !key.meta && !key.option
    const isEnter = (key.name === 'return' || key.name === 'enter') &&
      !key.shift && !key.ctrl && !key.meta && !key.option
    const isEscape = key.name === 'escape'
    const isTab = key.name === 'tab' && !key.shift && !key.ctrl && !key.meta && !key.option

    if (isUp) {
      moveUp()
      return true
    }

    if (isDown) {
      moveDown()
      return true
    }

    if (isEscape) {
      close()
      return true
    }

    if (isEnter || isTab) {
      const item = items[selectedIndex] ?? (isTab && items.length === 1 ? items[0] : null)
      if (!item) return false

      if (isTab && item.kind === 'directory') {
        if (onExpandDirectory) onExpandDirectory(item)
        return true
      }

      confirmSelection(item)
      return true
    }

    return false
  }, [isOpen, items, selectedIndex, moveUp, moveDown, close, confirmSelection, onExpandDirectory])

  return {
    isOpen,
    query,
    queryLineRange: resolvedLineRange ?? queryResult?.lineRange,
    items,
    recentItems,
    overflowCount,
    selectedIndex,
    setSelectedIndex,
    loading,
    moveUp,
    moveDown,
    select,
    confirmSelection,
    close,
    handleKeyIntercept,
  }
}
