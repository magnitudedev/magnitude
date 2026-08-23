import { useMemo, useState, type ReactNode } from "react"
import { ArchiveRestore, Search, Trash2 } from "lucide-react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  formatCwdForDisplay,
  formatRelativeTime,
  useAgentClient,
  useProjectPages,
  useSessionPages,
} from "@magnitudedev/client-common"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"
import { ActionTooltip } from "@/components/ui/tooltip"
import { notify } from "@/lib/notifications"

const pluralizeChats = (count: number): string =>
  `${count} archived ${count === 1 ? "chat" : "chats"}`

export function ArchivedChatsView(): ReactNode {
  const client = useAgentClient()
  const projects = useProjectPages()
  const [search, setSearch] = useState("")
  const [selectedIds, setSelectedIds] = useState<ReadonlySet<string>>(new Set())
  const [selectingAll, setSelectingAll] = useState(false)
  const [deleteIds, setDeleteIds] = useState<ReadonlyArray<string> | null>(null)
  const trimmedSearch = search.trim()
  const page = useSessionPages({
    archive: "archived",
    query: trimmedSearch || undefined,
    pageSize: 100,
  })
  const restoreAtom = client.Sessions.RestoreSession
  const deleteAtom = client.Sessions.DeleteArchivedSession
  const restoreResult = useAtomValue(restoreAtom)
  const deleteResult = useAtomValue(deleteAtom)
  const restoreSession = useAtomSet(restoreAtom, { mode: "promise" })
  const deleteSession = useAtomSet(deleteAtom, { mode: "promise" })
  const actionPending = Result.isWaiting(restoreResult) || Result.isWaiting(deleteResult)
  const projectNamesByCwd = useMemo(
    () => new Map(projects.projects.map((project) => [project.cwd as string, project.name])),
    [projects.projects],
  )
  const allLoadedSelected = page.sessions.length > 0
    && page.sessions.every((session) => selectedIds.has(session.id))
    && !page.hasMore
  const someSelected = selectedIds.size > 0

  const toggleSession = (sessionId: string, selected: boolean) => {
    setSelectedIds((current) => {
      const next = new Set(current)
      if (selected) next.add(sessionId)
      else next.delete(sessionId)
      return next
    })
  }

  const restore = async (sessionIds: ReadonlyArray<string>) => {
    const failed: string[] = []
    for (const sessionId of sessionIds) {
      try {
        await restoreSession({ sessionId })
      } catch {
        failed.push(sessionId)
      }
    }
    setSelectedIds(new Set(failed))
    if (failed.length === 0) {
      notify("success", sessionIds.length === 1 ? "Chat restored." : `${sessionIds.length} chats restored.`)
    } else {
      notify("error", `Could not restore ${failed.length} ${failed.length === 1 ? "chat" : "chats"}.`)
    }
  }

  const permanentlyDelete = async (sessionIds: ReadonlyArray<string>) => {
    const failed: string[] = []
    for (const sessionId of sessionIds) {
      try {
        await deleteSession({ sessionId })
      } catch {
        failed.push(sessionId)
      }
    }
    setDeleteIds(null)
    setSelectedIds(new Set(failed))
    if (failed.length === 0) {
      notify("success", sessionIds.length === 1 ? "Chat permanently deleted." : `${sessionIds.length} chats permanently deleted.`)
    } else {
      notify("error", `Could not delete ${failed.length} ${failed.length === 1 ? "chat" : "chats"}.`)
    }
  }

  return (
    <div className="mx-auto flex min-h-full w-full max-w-[1040px] flex-col px-7 py-8 max-[700px]:px-4">
      <header className="mb-7">
        <h1 className="font-heading text-[28px] font-semibold text-slate-900 dark:text-slate-100">
          Archived chats
        </h1>
        <p className="mt-1 font-sans text-[13px] text-slate-500">
          Restore chats to their projects or permanently delete their history.
        </p>
      </header>

      <div className="mb-3 flex items-center gap-3">
        <div className="flex h-9 min-w-0 flex-1 items-center gap-2 rounded-md border border-slate-300 bg-white px-2.5 dark:border-slate-750 dark:bg-slate-850">
          <Search size={16} className="shrink-0 text-slate-500" aria-hidden="true" />
          <Input
            value={search}
            disabled={selectingAll}
            onChange={(event) => {
              setSearch(event.target.value)
              setSelectedIds(new Set())
              setSelectingAll(false)
            }}
            placeholder="Search archived chats"
            aria-label="Search archived chats"
            className="h-full min-w-0 flex-1 border-0 bg-transparent p-0 shadow-none focus-visible:ring-0 dark:bg-transparent"
          />
        </div>
        <span className="shrink-0 whitespace-nowrap font-sans text-[12px] font-medium text-slate-500">
          {`${pluralizeChats(page.sessions.length)}${page.hasMore ? "+" : ""}`}
        </span>
      </div>

      <div className="flex h-11 shrink-0 items-center gap-3 border-y border-slate-200 px-3 dark:border-slate-800">
        <Checkbox
          checked={allLoadedSelected}
          indeterminate={someSelected && !allLoadedSelected}
          disabled={page.loading || page.sessions.length === 0 || actionPending || selectingAll}
          onCheckedChange={(checked) => {
            if (!checked) {
              setSelectingAll(false)
              setSelectedIds(new Set())
              return
            }
            setSelectingAll(true)
            void page.loadAll()
              .then((sessions) => {
                setSelectedIds(new Set(sessions.map((session) => session.id)))
              })
              .catch(() => {
                notify("error", "Could not select all archived chats.")
              })
              .finally(() => {
                setSelectingAll(false)
              })
          }}
          aria-label="Select all matching archived chats"
        />
        <span className="min-w-0 flex-1 font-sans text-[12px] text-slate-500">
          {selectingAll
            ? "Selecting all matching chats…"
            : someSelected
            ? `${selectedIds.size} selected`
            : "Select all"}
        </span>
        {someSelected ? (
          <>
            <Button
              variant="outline"
              size="sm"
              disabled={actionPending}
              onClick={() => void restore([...selectedIds])}
              aria-label="Restore selected chats"
            >
              <ArchiveRestore data-icon="inline-start" />
              Restore
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={actionPending}
              className="text-red-600 hover:bg-red-200/40 hover:text-red-700 dark:text-red-400 dark:hover:bg-red-800/30 dark:hover:text-red-300"
              onClick={() => setDeleteIds([...selectedIds])}
            >
              <Trash2 data-icon="inline-start" />
              Delete permanently
            </Button>
          </>
        ) : null}
      </div>

      <div className="min-h-[240px]">
        {page.loading || projects.loading ? (
          <div className="flex min-h-[240px] items-center justify-center gap-2 font-sans text-[13px] text-slate-500">
            <Spinner className="size-5 text-blue-600 dark:text-blue-400" />
            Loading archived chats…
          </div>
        ) : page.error ? (
          <div className="flex min-h-[240px] items-center justify-center font-sans text-[13px] text-red-600 dark:text-red-400">
            Could not load archived chats.
          </div>
        ) : page.sessions.length === 0 ? (
          <div className="flex min-h-[240px] flex-col items-center justify-center text-center">
            <ArchiveRestore size={25} className="mb-3 text-slate-500" aria-hidden="true" />
            <strong className="font-sans text-[13px] font-semibold text-slate-800 dark:text-slate-200">
              {trimmedSearch ? "No archived chats match your search" : "No archived chats"}
            </strong>
            <span className="mt-1 font-sans text-[12px] text-slate-500">
              {trimmedSearch ? "Try a different title or project name." : "Chats you archive will appear here."}
            </span>
          </div>
        ) : (
          <div aria-label="Archived chats" role="list">
            {page.sessions.map((session) => {
              const selected = selectedIds.has(session.id)
              return (
                <div
                  key={session.id}
                  role="listitem"
                  data-selected={selected || undefined}
                  className="grid min-h-[58px] grid-cols-[auto_minmax(0,1fr)_minmax(110px,180px)_auto] items-center gap-3 border-b border-slate-200 px-3 py-2.5 data-[selected]:bg-slate-100 dark:border-slate-800 dark:data-[selected]:bg-slate-850 max-[760px]:grid-cols-[auto_minmax(0,1fr)_auto]"
                >
                  <Checkbox
                    checked={selected}
                    disabled={actionPending || selectingAll}
                    onCheckedChange={(checked) => toggleSession(session.id, checked)}
                    aria-label={`Select ${session.title}`}
                  />
                  <div className="min-w-0">
                    <div className="truncate font-sans text-[13px] font-semibold text-slate-800 dark:text-slate-200">
                      {session.title}
                    </div>
                    <div className="mt-0.5 truncate font-sans text-[11px] text-slate-500">
                      {projectNamesByCwd.get(session.workingDirectory)
                        ?? formatCwdForDisplay(session.workingDirectory)}
                    </div>
                  </div>
                  <span className="font-sans text-[11px] text-slate-500 max-[760px]:hidden">
                    Last active {formatRelativeTime(session.timestamp)}
                  </span>
                  <div className="flex items-center gap-1">
                    <ActionTooltip
                      label="Restore chat"
                      trigger={
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          disabled={actionPending || selectingAll}
                          onClick={() => void restore([session.id])}
                          aria-label={`Restore ${session.title}`}
                        >
                          <ArchiveRestore />
                        </Button>
                      }
                    />
                    <ActionTooltip
                      label="Delete permanently"
                      trigger={
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          disabled={actionPending || selectingAll}
                          onClick={() => setDeleteIds([session.id])}
                          aria-label={`Permanently delete ${session.title}`}
                          className="text-slate-500 hover:text-red-600 dark:hover:text-red-400"
                        >
                          <Trash2 />
                        </Button>
                      }
                    />
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {page.hasMore && !selectingAll ? (
        <div className="flex justify-center py-4">
          <Button variant="outline" disabled={page.loadingMore} onClick={() => page.loadMore()}>
            {page.loadingMore ? <Spinner className="size-4" /> : null}
            {page.loadingMore ? "Loading…" : "Load more"}
          </Button>
        </div>
      ) : null}

      <AlertDialog open={deleteIds !== null} onOpenChange={(open) => {
        if (!open && !actionPending) setDeleteIds(null)
      }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Permanently delete {deleteIds?.length === 1 ? "this chat" : `${deleteIds?.length ?? 0} chats`}?
            </AlertDialogTitle>
            <AlertDialogDescription>
              This removes the complete chat history and cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={actionPending}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={actionPending || !deleteIds?.length}
              onClick={() => {
                if (deleteIds) void permanentlyDelete(deleteIds)
              }}
            >
              {actionPending ? "Deleting…" : "Delete permanently"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
