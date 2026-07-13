/**
 * ChatEmptyState
 *
 * No session has been selected yet. The first message creates a session using
 * the agent-host working directory selected here.
 */
import { useCallback, useEffect, useMemo, useState, type ReactNode, type UIEvent } from "react"
import { Folder, Loader2, Search } from "lucide-react"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { formatCwdForDisplay, formatRelativeTime, selectedCwdAtom, useAgentClient } from "@magnitudedev/client-common"
import type { DirectoryCandidate, SearchDirectoriesResult } from "@magnitudedev/sdk"

const DIRECTORY_PAGE_SIZE = 14

function directoryFallbackLabel(path: string): string {
  if (path === ".") return "Current workspace"
  const parts = path.split("/").filter(Boolean)
  return parts.at(-1) ?? path
}

function DirectoryRow({
  candidate,
  selected,
  onSelect,
}: {
  candidate: DirectoryCandidate
  selected: boolean
  onSelect: () => void
}): ReactNode {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="hover-surface"
      style={{
        width: "100%",
        minHeight: 46,
        padding: "6px 10px",
        border: `1px solid ${selected ? "var(--accent-primary)" : "transparent"}`,
        borderRadius: 4,
        background: selected ? "var(--bg-surface-elevated)" : undefined,
        color: "var(--fg-primary)",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        gap: 10,
        textAlign: "left",
        fontFamily: "var(--font-sans)",
        transition: "background 100ms",
      }}
    >
      <Folder size={16} style={{ color: selected ? "var(--accent-primary)" : "var(--fg-tertiary)", flexShrink: 0 }} />
      <span style={{ minWidth: 0, flex: 1 }}>
        <span
          style={{
            display: "block",
            fontSize: 13,
            fontWeight: selected ? 650 : 500,
            color: selected ? "var(--accent-primary)" : "var(--fg-primary)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {candidate.label}
        </span>
        <span
          style={{
            display: "block",
            marginTop: 2,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--fg-secondary)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {formatCwdForDisplay(candidate.path, { maxLen: 70, abbreviateHome: true })}
        </span>
      </span>
      {candidate.lastActivity !== undefined && (
        <span style={{ flexShrink: 0, fontSize: 11, color: "var(--fg-tertiary)" }}>
          {formatRelativeTime(candidate.lastActivity)}
        </span>
      )}
    </button>
  )
}

function DirectoryPicker(): ReactNode {
  const client = useAgentClient()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const setSelectedCwd = useAtomSet(selectedCwdAtom)
  const [query, setQuery] = useState("")
  const [limitState, setLimitState] = useState({
    query: "",
    limit: DIRECTORY_PAGE_SIZE,
  })
  const [lastLoaded, setLastLoaded] = useState<{
    query: string
    limit: number
    candidates: readonly DirectoryCandidate[]
  } | null>(null)
  const trimmedQuery = query.trim()
  const visibleLimit = limitState.query === trimmedQuery ? limitState.limit : DIRECTORY_PAGE_SIZE

  const directoriesAtom = useMemo(
    () => client.query("SearchDirectories", {
      query: trimmedQuery,
      limit: visibleLimit,
      includeRecent: true,
    }, { reactivityKeys: ["sessions"] }),
    [client, trimmedQuery, visibleLimit],
  )
  const result = useAtomValue(directoriesAtom)
  const isLoading = Result.isInitial(result)
  const successCandidates = Result.isSuccess(result)
    ? (result.value as SearchDirectoriesResult).candidates
    : null

  useEffect(() => {
    if (!successCandidates) return
    setLastLoaded({
      query: trimmedQuery,
      limit: visibleLimit,
      candidates: successCandidates,
    })
  }, [successCandidates, trimmedQuery, visibleLimit])

  const cachedCandidates = lastLoaded?.query === trimmedQuery ? lastLoaded.candidates : []
  const candidates = successCandidates ?? cachedCandidates
  const loadedLimit = successCandidates ? visibleLimit : lastLoaded?.query === trimmedQuery ? lastLoaded.limit : 0
  const loadingMore = isLoading && candidates.length > 0 && visibleLimit > loadedLimit
  const hasMore = Result.isSuccess(result)
    ? candidates.length >= visibleLimit
    : lastLoaded?.query === trimmedQuery
      ? lastLoaded.candidates.length >= lastLoaded.limit
      : false

  useEffect(() => {
    if (selectedCwd) return
    if (candidates.length > 0) {
      setSelectedCwd(candidates[0]!.path)
      return
    }
    if (!Result.isInitial(result)) {
      setSelectedCwd(".")
    }
  }, [candidates, selectedCwd, result, setSelectedCwd])

  const selectedPath = selectedCwd ?? candidates[0]?.path ?? "."
  const selectedCandidate = candidates.find((candidate) => candidate.path === selectedPath)
  const selectedLabel = selectedCandidate?.label ?? directoryFallbackLabel(selectedPath)

  const handleSelect = (path: string) => {
    setSelectedCwd(path)
    setQuery("")
  }

  const handleDirectoryScroll = useCallback(
    (event: UIEvent<HTMLDivElement>) => {
      if (!hasMore || isLoading) return
      const element = event.currentTarget
      const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight
      if (distanceFromBottom < 96) {
        setLimitState((current) => {
          const currentLimit = current.query === trimmedQuery ? current.limit : DIRECTORY_PAGE_SIZE
          return {
            query: trimmedQuery,
            limit: currentLimit + DIRECTORY_PAGE_SIZE,
          }
        })
      }
    },
    [hasMore, isLoading, trimmedQuery],
  )

  return (
    <div
      style={{
        width: "min(640px, 100%)",
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div
        style={{
          marginBottom: 6,
          fontFamily: "var(--font-sans)",
        }}
      >
        <div style={{ color: "var(--fg-primary)", fontSize: 18, fontWeight: 650 }}>
          Start a new chat in <span style={{ color: "var(--accent-primary)" }}>{selectedLabel}</span>
        </div>
        <div style={{ color: "var(--fg-secondary)", fontSize: 13, marginTop: 4, fontFamily: "var(--font-mono)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {formatCwdForDisplay(selectedPath, { maxLen: 86, abbreviateHome: true })}
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          height: 34,
          padding: "0 2px",
          background: "transparent",
          borderBottom: `1px solid ${trimmedQuery ? "var(--accent-primary)" : "var(--border-default)"}`,
        }}
      >
        <Search size={16} style={{ color: "var(--fg-tertiary)", flexShrink: 0 }} />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search recent directories or paste a path"
          style={{
            flex: 1,
            minWidth: 0,
            background: "transparent",
            border: "none",
            outline: "none",
            color: "var(--fg-primary)",
            fontFamily: "var(--font-mono)",
            fontSize: 14,
          }}
        />
      </div>

      <div
        onScroll={handleDirectoryScroll}
        style={{
          border: "1px solid var(--border-subtle)",
          borderRadius: 6,
          background: "var(--bg-surface)",
          padding: 6,
          height: 320,
          overflowY: "auto",
        }}
      >
        <div
          style={{
            padding: "4px 4px 8px",
            color: "var(--fg-tertiary)",
            fontFamily: "var(--font-sans)",
            fontSize: 11,
            fontWeight: 650,
            textTransform: "uppercase",
          }}
        >
          {trimmedQuery ? "Matching directories" : "Recent directories"}
        </div>
        {candidates.length > 0 ? (
          <>
            {candidates.map((candidate) => (
              <DirectoryRow
                key={`${candidate.source}:${candidate.path}`}
                candidate={candidate}
                selected={candidate.path === selectedPath}
                onSelect={() => handleSelect(candidate.path)}
              />
            ))}
            {loadingMore && (
              <div
                style={{
                  height: 32,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "var(--fg-tertiary)",
                }}
              >
                <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
              </div>
            )}
          </>
        ) : (
          <div
            style={{
              padding: "18px 10px",
              color: "var(--fg-tertiary)",
              fontFamily: "var(--font-sans)",
              fontSize: 13,
              textAlign: "center",
            }}
          >
            {isLoading ? "Loading directories..." : "No matching directories"}
          </div>
        )}
      </div>
    </div>
  )
}

export function ChatEmptyState(): ReactNode {
  return (
    <div
      style={{
        flex: 1,
        width: "100%",
        minHeight: 0,
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "32px 24px",
        animation: "fade-in 200ms ease-out",
      }}
    >
      <DirectoryPicker />
    </div>
  )
}
