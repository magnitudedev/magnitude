import { useCallback, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { Atom, useAtomMount } from '@effect-atom/atom-react'
import { Data, Effect } from 'effect'
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

class MentionSearchError extends Data.TaggedError('MentionSearchError')<{
  readonly cause: unknown
}> {}

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
// The store is created once per hook instance and contains display snapshots
// only. An Effect-owned mounted task performs each search and is interrupted
// when its input atom is replaced or the component unmounts.
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

class MentionSearchStore {
  private snapshot: SearchSnapshot = EMPTY_SNAPSHOT
  private listeners = new Set<() => void>()

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  getSnapshot = (): SearchSnapshot => this.snapshot

  setSnapshot(next: SearchSnapshot): void {
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
  const [selection, setSelection] = useState({ signature: '', index: 0 })
  const [dismissedInput, setDismissedInput] = useState<string | null>(null)

  const queryResult = useMemo(() => detectQuery(inputText, cursorPosition), [inputText, cursorPosition])
  const query = queryResult?.filePath ?? ''
  const rawQuery = queryResult?.raw ?? ''

  const inputIdentity = `${cursorPosition}:${inputText}`
  const isOpen = queryResult !== null && dismissedInput !== inputIdentity

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

  const searchAtom = useMemo(() => Atom.make(
    !isOpen || !client || !cwd
      ? Effect.sync(() => store.setSnapshot(EMPTY_SNAPSHOT))
      : Effect.gen(function* () {
          store.setSnapshot({ ...store.getSnapshot(), loading: true })
          const result = yield* Effect.tryPromise({
            try: () => client.searchMentions({
              cwd,
              query: rawQuery,
              limit: MAX_RESULTS,
              visibleLimit: MAX_VISIBLE_RESULTS,
              includeRecent: true,
            }),
            catch: (cause) => new MentionSearchError({ cause }),
          })
          store.setSnapshot({
            items: result.candidates.map(toMentionFileItem),
            recentItems: result.recentCandidates.map(toMentionFileItem),
            overflowCount: result.overflowCount,
            resolvedLineRange: result.lineRange,
            loading: false,
          })
        }).pipe(
          Effect.catchAll((cause) => Effect.logWarning('Mention search failed').pipe(
            Effect.annotateLogs({ cause: String(cause) }),
            Effect.zipRight(Effect.sync(() => store.setSnapshot(EMPTY_SNAPSHOT))),
          )),
        ),
  ), [client, cwd, isOpen, rawQuery, store])
  useAtomMount(searchAtom)

  const { items, recentItems, overflowCount, resolvedLineRange, loading } =
    useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)

  const signature = `${rawQuery}::${items.map(i => i.path).join(',')}`
  const maximumIndex = Math.max(0, items.length - 1)
  const selectedIndex = selection.signature === signature
    ? Math.min(maximumIndex, Math.max(0, selection.index))
    : 0
  const setSelectedIndex = useCallback((index: number) => {
    setSelection({ signature, index: Math.min(maximumIndex, Math.max(0, index)) })
  }, [maximumIndex, signature])

  const moveUp = useCallback(() => {
    setSelectedIndex(selectedIndex - 1)
  }, [selectedIndex, setSelectedIndex])

  const moveDown = useCallback(() => {
    setSelectedIndex(selectedIndex + 1)
  }, [selectedIndex, setSelectedIndex])

  const select = useCallback((): MentionFileItem | null => {
    return items[selectedIndex] ?? null
  }, [items, selectedIndex])

  const close = useCallback(() => {
    setSelectedIndex(0)
    setDismissedInput(inputIdentity)
  }, [inputIdentity, setSelectedIndex])

  const confirmSelection = useCallback((item: MentionFileItem) => {
    setDismissedInput(inputIdentity)
    if (onConfirm) onConfirm(item)
  }, [inputIdentity, onConfirm])

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
