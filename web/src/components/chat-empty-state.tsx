import { useMemo, useState, type ReactNode } from "react"
import { Effect } from "effect"
import {
  Atom,
  useAtomMount,
  useAtomSet,
  useAtomValue,
} from "@effect-atom/atom-react"
import {
  Check,
  FolderPlus,
  MagnifyingGlass,
} from "@phosphor-icons/react"
import {
  selectedCwdAtom,
  selectedProjectIdAtom,
  useProjects,
} from "@magnitudedev/client-common"
import type { ProjectSummary } from "@magnitudedev/sdk"
import { MagnitudeMark } from "./magnitude-mark"
import { ProjectFormDialog } from "./project-dialogs"
import { collapsedProjectIdsAtom } from "../state/web-atoms"

const isAvailable = (summary: ProjectSummary): boolean =>
  summary.directoryState._tag === "available"

export function ChatEmptyState(): ReactNode {
  const { projects, loading, error } = useProjects()
  const selectedProjectId = useAtomValue(selectedProjectIdAtom)
  const setSelectedCwd = useAtomSet(selectedCwdAtom)
  const setSelectedProjectId = useAtomSet(selectedProjectIdAtom)
  const setCollapsedProjects = useAtomSet(collapsedProjectIdsAtom)
  const [chooserOpen, setChooserOpen] = useState(false)
  const [query, setQuery] = useState("")
  const [creatingProject, setCreatingProject] = useState(false)

  const selectedById = projects.find(
    ({ project }) => project.projectId === selectedProjectId,
  )
  const selected = selectedById && isAvailable(selectedById)
    ? selectedById
    : projects.find(isAvailable) ?? null

  const initializeDraftProjectAtom = useMemo(
    () => Atom.make(Effect.sync(() => {
      if (selected === null || selectedProjectId === selected.project.projectId) return
      setSelectedProjectId(selected.project.projectId)
      setSelectedCwd(selected.project.sourceDirectory)
    })),
    [selected, selectedProjectId, setSelectedCwd, setSelectedProjectId],
  )
  useAtomMount(initializeDraftProjectAtom)

  const selectProject = (project: ProjectSummary["project"]) => {
    setSelectedProjectId(project.projectId)
    setSelectedCwd(project.sourceDirectory)
    setChooserOpen(false)
    setQuery("")
  }

  const visibleProjects = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return projects.filter((summary) =>
      isAvailable(summary) && (
        !normalized ||
        summary.project.name.toLowerCase().includes(normalized) ||
        summary.project.sourceDirectory.toLowerCase().includes(normalized)
      ),
    )
  }, [projects, query])

  return (
    <div className="flex min-h-0 w-full flex-1 flex-col items-center justify-center px-6 py-8 [animation:fade-in_200ms_ease-out]">
      <div className="flex w-full max-w-[520px] flex-col items-center text-center">
        <MagnitudeMark className="mb-6 h-auto w-[68px]" />
        <div className="relative">
          {selected ? (
            <>
              {chooserOpen ? (
                <button
                  type="button"
                  aria-label="Close project chooser"
                  onClick={() => setChooserOpen(false)}
                  className="fixed inset-0 z-20 cursor-default border-0 bg-transparent"
                />
              ) : null}
              <h1 className="font-mono text-[24px] font-semibold leading-[1.4] text-slate-900 dark:text-slate-100">
                What would you like to do in{" "}
                <button
                  type="button"
                  aria-haspopup="listbox"
                  aria-expanded={chooserOpen}
                  onClick={() => setChooserOpen((open) => !open)}
                  className="relative z-30 -mx-1 rounded border-0 bg-transparent px-1 py-0.5 [font:inherit] text-inherit underline decoration-current decoration-1 underline-offset-4 hover:bg-slate-150 dark:hover:bg-slate-800"
                >
                  {selected.project.name}
                </button>
                ?
              </h1>
              {chooserOpen ? (
                <div className="absolute bottom-[calc(100%+12px)] left-1/2 z-30 w-[min(360px,calc(100vw-32px))] -translate-x-1/2 rounded-lg border border-slate-300 bg-white p-1.5 text-left shadow-[0_8px_24px_rgba(0,0,0,.16)] dark:border-slate-600 dark:bg-slate-750 dark:shadow-[0_8px_24px_rgba(0,0,0,.36)]">
                  <div className="mb-1 flex h-9 items-center gap-2 border-b border-slate-200 px-2 dark:border-slate-600">
                    <MagnifyingGlass size={16} className="shrink-0 text-slate-500" />
                    <input
                      autoFocus
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Escape") setChooserOpen(false)
                      }}
                      placeholder="Search projects"
                      className="min-w-0 flex-1 border-0 bg-transparent font-sans text-[13px] text-slate-900 outline-none placeholder:text-slate-500 dark:text-slate-100"
                    />
                  </div>
                  <div role="listbox" aria-label="Projects" className="max-h-[240px] overflow-y-auto">
                    {visibleProjects.length === 0 ? (
                      <div className="px-3 py-6 text-center font-sans text-[12px] text-slate-500">
                        No matching projects.
                      </div>
                    ) : visibleProjects.map((summary) => {
                      const active = summary.project.projectId === selected.project.projectId
                      return (
                        <button
                          key={summary.project.projectId}
                          type="button"
                          role="option"
                          aria-selected={active}
                          onClick={() => selectProject(summary.project)}
                          className="flex h-9 w-full items-center gap-2 rounded-md border-0 bg-transparent px-2.5 text-left font-sans text-[13px] text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-700"
                        >
                          <span className="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
                            {summary.project.name}
                          </span>
                          {active ? <Check size={15} className="shrink-0 text-blue-600 dark:text-blue-400" /> : null}
                        </button>
                      )
                    })}
                  </div>
                </div>
              ) : null}
            </>
          ) : loading ? (
            <>
              <h1 className="font-mono text-[24px] font-semibold leading-[1.4] text-slate-900 dark:text-slate-100">
                What would you like to do?
              </h1>
              <span className="mt-4 block font-sans text-[14px] text-slate-500">Loading projects…</span>
            </>
          ) : error ? (
            <>
              <h1 className="font-mono text-[24px] font-semibold leading-[1.4] text-slate-900 dark:text-slate-100">
                What would you like to do?
              </h1>
              <span className="mt-4 block font-sans text-[13px] text-red-600 dark:text-red-400">{error}</span>
            </>
          ) : (
            <button
              type="button"
              onClick={() => setCreatingProject(true)}
              className="flex h-9 items-center justify-center gap-2 rounded-md border border-blue-700 bg-blue-700 px-3 font-sans text-[13px] font-semibold text-white hover:bg-blue-800 dark:border-blue-500 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400"
            >
              <FolderPlus size={16} /> New project
            </button>
          )}
        </div>
      </div>

      {creatingProject ? (
        <ProjectFormDialog
          onDismiss={() => setCreatingProject(false)}
          onSaved={(project) => {
            setCreatingProject(false)
            setCollapsedProjects((current) => {
              if (!current.has(project.projectId)) return current
              const next = new Set(current)
              next.delete(project.projectId)
              return next
            })
            selectProject(project)
          }}
        />
      ) : null}
    </div>
  )
}
