/**
 * ChatEmptyState
 *
 * No session has been selected yet. The first message creates a session using
 * the agent-host working directory selected here.
 */
import {
  useCallback,
  useMemo,
  useState,
  type ReactNode,
  type UIEvent,
} from "react"
import { Folder, Loader2, Search } from "lucide-react"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import {
  formatCwdForDisplay,
  formatRelativeTime,
  selectedCwdAtom,
  useAgentClient,
} from "@magnitudedev/client-common"
import type {
  DirectoryCandidate,
  SearchDirectoriesResult,
} from "@magnitudedev/sdk"
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
      className={`${
        selected
          ? "border-blue-700 bg-slate-100 dark:border-blue-500 dark:bg-slate-800"
          : "border-transparent bg-transparent hover:bg-slate-100 dark:hover:bg-slate-800"
      } w-full min-h-[46px] border px-2.5 py-1.5 rounded text-slate-900 dark:text-slate-200 cursor-pointer flex items-center gap-2.5 text-left font-sans transition-colors duration-100`}
    >
      <Folder
        size={16}
        className={`${
          selected ? "text-blue-700 dark:text-blue-500" : "text-slate-500"
        }  shrink-0`}
      />
      <span className="min-w-0 [flex:1]">
        <span
          className={`${selected ? "font-[650]" : "font-medium"} ${
            selected
              ? "text-blue-700 dark:text-blue-500"
              : "text-slate-900 dark:text-slate-200"
          }  block text-[13px] overflow-hidden text-ellipsis whitespace-nowrap`}
        >
          {candidate.label}
        </span>
        <span className="block [margin-top:2px] font-mono text-[11px] text-slate-600 dark:text-slate-400 overflow-hidden text-ellipsis whitespace-nowrap">
          {formatCwdForDisplay(candidate.path, {
            maxLen: 70,
            abbreviateHome: true,
          })}
        </span>
      </span>
      {candidate.lastActivity !== undefined && (
        <span className="shrink-0 text-[11px] text-slate-500">
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
  const trimmedQuery = query.trim()
  const visibleLimit =
    limitState.query === trimmedQuery ? limitState.limit : DIRECTORY_PAGE_SIZE
  const directoriesAtom = useMemo(
    () =>
      client.rpc.query(
        "SearchDirectories",
        {
          query: trimmedQuery,
          limit: visibleLimit,
          includeRecent: true,
        },
        {
          reactivityKeys: ["sessions"],
        }
      ),
    [client, trimmedQuery, visibleLimit]
  )
  const result = useAtomValue(directoriesAtom)
  const isLoading = Result.isInitial(result)
  const candidates = Result.match(result, {
    onInitial: () => [] as DirectoryCandidate[],
    onFailure: (f) =>
      f.previousSuccess._tag === "Some"
        ? (f.previousSuccess.value.value as SearchDirectoriesResult).candidates
        : [],
    onSuccess: (s) => (s.value as SearchDirectoriesResult).candidates,
  })
  const loadedLimit = Result.isSuccess(result) ? visibleLimit : 0
  const loadingMore =
    isLoading && candidates.length > 0 && visibleLimit > loadedLimit
  const hasMore = Result.isSuccess(result) && candidates.length >= visibleLimit
  const selectedPath = selectedCwd ?? candidates[0]?.path ?? "."
  const selectedCandidate = candidates.find(
    (candidate) => candidate.path === selectedPath
  )
  const selectedLabel =
    selectedCandidate?.label ?? directoryFallbackLabel(selectedPath)
  const handleSelect = (path: string) => {
    setSelectedCwd(path)
    setQuery("")
  }
  const handleDirectoryScroll = useCallback(
    (event: UIEvent<HTMLDivElement>) => {
      if (!hasMore || isLoading) return
      const element = event.currentTarget
      const distanceFromBottom =
        element.scrollHeight - element.scrollTop - element.clientHeight
      if (distanceFromBottom < 96) {
        setLimitState((current) => {
          const currentLimit =
            current.query === trimmedQuery ? current.limit : DIRECTORY_PAGE_SIZE
          return {
            query: trimmedQuery,
            limit: currentLimit + DIRECTORY_PAGE_SIZE,
          }
        })
      }
    },
    [hasMore, isLoading, trimmedQuery]
  )
  return (
    <div className="[width:min(640px,_100%)] flex flex-col [gap:10px]">
      <div className="[margin-bottom:6px] font-sans">
        <div className="text-slate-900 dark:text-slate-200 text-[18px] font-[650]">
          Start a new chat in{" "}
          <span className="text-blue-700 dark:text-blue-500">
            {selectedLabel}
          </span>
        </div>
        {selectedPath !== "." && (
          <div className="text-slate-600 dark:text-slate-400 text-[13px] [margin-top:4px] font-mono overflow-hidden text-ellipsis whitespace-nowrap">
            {formatCwdForDisplay(selectedPath, {
              maxLen: 86,
              abbreviateHome: true,
            })}
          </div>
        )}
      </div>

      <div
        className={`${
          trimmedQuery
            ? "border-blue-700 dark:border-blue-500"
            : "border-slate-300 dark:border-slate-750"
        } flex h-[34px] items-center gap-2 border-b bg-transparent px-0.5`}
      >
        <Search size={16} className="text-slate-500 shrink-0" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search recent directories or paste a path"
          className="[flex:1] min-w-0 [background:transparent] border-0 [outline:none] text-slate-900 dark:text-slate-200 font-mono text-[14px]"
        />
      </div>

      <div
        onScroll={handleDirectoryScroll}
        className="border border-slate-200 dark:border-slate-800 rounded-[6px] bg-white dark:bg-slate-875 [padding:6px] [height:320px] overflow-y-auto"
      >
        <div className="[padding:4px_4px_8px] text-slate-500 font-sans text-[11px] font-[650] [text-transform:uppercase]">
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
              <div className="[height:32px] flex items-center justify-center text-slate-500">
                <Loader2
                  size={14}
                  className="[animation:spin_1s_linear_infinite]"
                />
              </div>
            )}
          </>
        ) : (
          <div className="[padding:18px_10px] text-slate-500 font-sans text-[13px] text-center">
            {isLoading ? "Loading directories..." : "No matching directories"}
          </div>
        )}
      </div>
    </div>
  )
}
export function ChatEmptyState(): ReactNode {
  return (
    <div className="[flex:1] w-full min-h-0 box-border flex flex-col items-center justify-center [padding:32px_24px] [animation:fade-in_200ms_ease-out]">
      <DirectoryPicker />
    </div>
  )
}
