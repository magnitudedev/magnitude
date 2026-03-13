import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { statSync } from 'node:fs'

import { resolveRgPath } from '@magnitudedev/ripgrep'
import type { KeyEvent } from '@opentui/core'

const CACHE_TTL_MS = 45_000
const MAX_RESULTS = 40
const MAX_VISIBLE_RESULTS = 10
const MAX_RECENT_FILES = 10
const LARGE_FILE_WARNING_BYTES = 500 * 1024

const BINARY_EXTENSIONS = new Set([
  '.exe', '.dll', '.so', '.o', '.pyc', '.class', '.jar', '.zip', '.tar', '.gz', '.bin', '.dat', '.db',
  '.sqlite', '.woff', '.woff2', '.ttf', '.eot', '.ico', '.mp3', '.mp4', '.mov', '.avi', '.pdf',
])

const IMAGE_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.webp',
])

type CachedIndex = {
  files: string[]
  timestamp: number
}

type CachedRecent = {
  files: string[]
  timestamp: number
}

export type MentionFileItem = {
  path: string
  kind: 'file' | 'directory'
  contentType: 'text' | 'image' | 'directory'
  warning?: boolean
}

interface FileMentionsState {
  isOpen: boolean
  query: string
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

let fileIndexCache: CachedIndex | null = null
let inflightLoad: Promise<string[]> | null = null
let recentFileCache: CachedRecent | null = null
let inflightRecentLoad: Promise<string[]> | null = null

function getExt(path: string): string {
  const idx = path.lastIndexOf('.')
  if (idx < 0) return ''
  return path.slice(idx).toLowerCase()
}

function getBase(path: string): string {
  const normalized = path.endsWith('/') ? path.slice(0, -1) : path
  const i = normalized.lastIndexOf('/')
  return i >= 0 ? normalized.slice(i + 1) : normalized
}

function isSubsequence(query: string, text: string): boolean {
  if (!query) return true
  let qi = 0
  let ti = 0
  while (qi < query.length && ti < text.length) {
    if (query[qi] === text[ti]) qi++
    ti++
  }
  return qi === query.length
}

function shouldKeepPath(path: string): boolean {
  const ext = getExt(path)
  if (!ext) return true
  if (IMAGE_EXTENSIONS.has(ext)) return true
  if (BINARY_EXTENSIONS.has(ext)) return false
  return true
}

function rankPath(path: string, queryLower: string): number {
  const base = getBase(path).toLowerCase()
  const full = path.toLowerCase()

  if (base.startsWith(queryLower)) return 0
  if (full.includes(queryLower)) return 1
  if (isSubsequence(queryLower, full)) return 2
  return 999
}

function collectDirectories(paths: string[]): string[] {
  const dirs = new Set<string>()
  for (const filePath of paths) {
    const parts = filePath.split('/').filter(Boolean)
    if (parts.length <= 1) continue
    let current = ''
    for (let i = 0; i < parts.length - 1; i++) {
      current += `${parts[i]}/`
      dirs.add(current)
    }
  }
  return [...dirs].sort((a, b) => a.localeCompare(b))
}

async function loadFileIndex(): Promise<string[]> {
  const now = Date.now()
  if (fileIndexCache && now - fileIndexCache.timestamp < CACHE_TTL_MS) {
    return fileIndexCache.files
  }
  if (inflightLoad) return inflightLoad

  inflightLoad = Promise.resolve().then(async () => {
    const rgPath = await resolveRgPath()
    const proc = Bun.spawn(
      [rgPath, '--files', '--hidden', '-g', '!node_modules/**', '-g', '!dist/**'],
      { cwd: process.cwd(), stdout: 'pipe', stderr: 'pipe' },
    )
    const stdout = await new Response(proc.stdout).text()
    const exitCode = await proc.exited

    if (exitCode !== 0) {
      fileIndexCache = { files: [], timestamp: Date.now() }
      return []
    }

    const files = stdout
      .split('\n')
      .map(line => line.trim())
      .filter(Boolean)
      .filter(shouldKeepPath)

    fileIndexCache = { files, timestamp: Date.now() }
    return files
  }).finally(() => {
    inflightLoad = null
  })

  return inflightLoad
}

async function loadRecentFiles(indexedFiles: string[]): Promise<string[]> {
  const now = Date.now()
  if (recentFileCache && now - recentFileCache.timestamp < CACHE_TTL_MS) {
    return recentFileCache.files
  }
  if (inflightRecentLoad) return inflightRecentLoad

  const indexedSet = new Set(indexedFiles)

  inflightRecentLoad = Promise.resolve().then(async () => {
    const proc = Bun.spawn(
      ['git', 'log', '--diff-filter=M', '--name-only', '--pretty=format:', '-n', '50'],
      { cwd: process.cwd(), stdout: 'pipe', stderr: 'pipe' },
    )
    const stdout = await new Response(proc.stdout).text()
    const exitCode = await proc.exited
    if (exitCode !== 0) {
      recentFileCache = { files: [], timestamp: Date.now() }
      return []
    }

    const seen = new Set<string>()
    const recentFiles: string[] = []

    for (const line of stdout.split('\n')) {
      const path = line.trim()
      if (!path || seen.has(path)) continue
      seen.add(path)
      if (!indexedSet.has(path)) continue
      recentFiles.push(path)
      if (recentFiles.length >= MAX_RECENT_FILES) break
    }

    recentFileCache = { files: recentFiles, timestamp: Date.now() }
    return recentFiles
  }).catch(() => {
    recentFileCache = { files: [], timestamp: Date.now() }
    return []
  }).finally(() => {
    inflightRecentLoad = null
  })

  return inflightRecentLoad
}

function detectQuery(inputText: string, cursorPosition: number): string | null {
  const left = inputText.slice(0, Math.max(0, cursorPosition))
  const match = left.match(/(?:^|\s)@([^\s@]*)$/)
  if (!match) return null
  return match[1] ?? ''
}

function toFileItem(path: string): MentionFileItem {
  const ext = getExt(path)
  const contentType: MentionFileItem['contentType'] = IMAGE_EXTENSIONS.has(ext) ? 'image' : 'text'
  let warning = false
  try {
    const s = statSync(path)
    warning = s.size > LARGE_FILE_WARNING_BYTES
  } catch {
    warning = false
  }
  return { path, kind: 'file', contentType, warning }
}

function toDirectoryItem(path: string): MentionFileItem {
  return { path, kind: 'directory', contentType: 'directory' }
}

export function useFileMentions(
  inputText: string,
  cursorPosition: number,
  onConfirm?: (item: MentionFileItem) => void,
): FileMentionsState {
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  const [indexedFiles, setIndexedFiles] = useState<string[]>([])
  const [recentFiles, setRecentFiles] = useState<string[]>([])
  const [dismissedQuery, setDismissedQuery] = useState<string | null>(null)
  const justConfirmedRef = useRef(false)

  const query = useMemo(() => detectQuery(inputText, cursorPosition), [inputText, cursorPosition])
  const queryLower = (query ?? '').toLowerCase()
  const isOpen = query !== null && query !== dismissedQuery && !justConfirmedRef.current

  useEffect(() => {
    if (query !== dismissedQuery) setDismissedQuery(null)
    if (query === null) justConfirmedRef.current = false
  }, [query, dismissedQuery])

  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    setLoading(true)
    loadFileIndex()
      .then(files => {
        if (!cancelled) setIndexedFiles(files)
        return files
      })
      .then(files => loadRecentFiles(files))
      .then(files => {
        if (!cancelled) setRecentFiles(files)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [isOpen])

  const { items, recentItems, overflowCount } = useMemo(() => {
    if (!isOpen) return { items: [], recentItems: [], overflowCount: 0 }

    const directories = collectDirectories(indexedFiles)
    const allCandidates: MentionFileItem[] = [
      ...indexedFiles.map(toFileItem),
      ...directories.map(toDirectoryItem),
    ]

    if (!queryLower) {
      const recent = recentFiles.map(toFileItem)
      const recentSet = new Set(recent.map((item) => item.path))
      const rest = allCandidates
        .filter((item) => !(item.kind === 'file' && recentSet.has(item.path)))
        .sort((a, b) => a.path.localeCompare(b.path))
        .slice(0, MAX_RESULTS)

      const ranked = [...recent, ...rest]
      const visible = ranked.slice(0, MAX_VISIBLE_RESULTS)
      const visibleRecentCount = Math.min(recent.length, visible.length)
      return {
        items: visible,
        recentItems: visible.slice(0, visibleRecentCount),
        overflowCount: Math.max(0, ranked.length - MAX_VISIBLE_RESULTS),
      }
    }

    const ranked = allCandidates
      .map(item => ({ item, rank: rankPath(item.path, queryLower) }))
      .filter(x => x.rank < 999)
      .sort((a, b) => (a.rank - b.rank) || a.item.path.localeCompare(b.item.path))
      .slice(0, MAX_RESULTS)
      .map(x => x.item)

    return {
      items: ranked.slice(0, MAX_VISIBLE_RESULTS),
      recentItems: [],
      overflowCount: Math.max(0, ranked.length - MAX_VISIBLE_RESULTS),
    }
  }, [isOpen, indexedFiles, recentFiles, queryLower])

  const prevSignatureRef = useRef<string>('')
  useEffect(() => {
    const signature = `${queryLower}::${items.map(i => i.path).join(',')}`
    if (signature !== prevSignatureRef.current) {
      prevSignatureRef.current = signature
      setSelectedIndex(0)
    } else if (selectedIndex >= items.length && items.length > 0) {
      setSelectedIndex(items.length - 1)
    } else if (items.length === 0 && selectedIndex !== 0) {
      setSelectedIndex(0)
    }
  }, [queryLower, items, selectedIndex])

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
      confirmSelection(item)
      return true
    }

    return false
  }, [isOpen, items, selectedIndex, moveUp, moveDown, close, confirmSelection])

  return {
    isOpen,
    query: query ?? '',
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