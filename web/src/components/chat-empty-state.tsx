import { useMemo, useState, type ReactNode } from "react"
import { Effect } from "effect"
import { Atom, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { FolderPlus } from "@phosphor-icons/react"
import { selectedCwdAtom, selectedProjectIdAtom, useProjectPages } from "@magnitudedev/client-common"
import type { Project } from "@magnitudedev/sdk"
import { MagnitudeMark } from "./magnitude-mark"
import { ProjectFormDialog } from "./project-dialogs"
import { collapsedProjectIdsAtom } from "../state/web-atoms"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"

export function ChatEmptyState(): ReactNode {
  const { projects, loading, error } = useProjectPages()
  const selectedProjectId = useAtomValue(selectedProjectIdAtom)
  const setSelectedCwd = useAtomSet(selectedCwdAtom)
  const setSelectedProjectId = useAtomSet(selectedProjectIdAtom)
  const setCollapsedProjects = useAtomSet(collapsedProjectIdsAtom)
  const [chooserOpen, setChooserOpen] = useState(false)
  const [query, setQuery] = useState("")
  const [creatingProject, setCreatingProject] = useState(false)

  const selected =
    projects.find((project) => project.projectId === selectedProjectId) ?? projects[0] ?? null

  const initializeDraftProjectAtom = useMemo(
    () =>
      Atom.make(
        Effect.sync(() => {
          if (selected === null || selectedProjectId === selected.projectId) return
          setSelectedProjectId(selected.projectId)
          setSelectedCwd(selected.cwd)
        })
      ),
    [selected, selectedProjectId, setSelectedCwd, setSelectedProjectId]
  )
  useAtomMount(initializeDraftProjectAtom)

  const selectProject = (project: Project) => {
    setSelectedProjectId(project.projectId)
    setSelectedCwd(project.cwd)
    setChooserOpen(false)
    setQuery("")
  }

  const visibleProjects = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return projects.filter(
      (project) =>
        !normalized ||
        project.name.toLowerCase().includes(normalized) ||
        project.cwd.toLowerCase().includes(normalized)
    )
  }, [projects, query])

  return (
    <div className="flex min-h-0 w-full flex-1 flex-col items-center justify-center px-6 py-8 [animation:fade-in_200ms_ease-out]">
      <div className="flex w-full max-w-[800px] flex-col items-center text-center">
        <MagnitudeMark className="mb-6 h-auto w-[68px]" />
        <div className="relative">
          {selected ? (
            <Popover open={chooserOpen} onOpenChange={setChooserOpen}>
              <h1 className="font-mono text-[24px] font-semibold leading-[1.4] text-slate-900 dark:text-slate-100">
                What would you like to do in{" "}
                <PopoverTrigger
                  render={
                    <Button
                      type="button"
                      variant="unstyled"
                      size="unstyled"
                      className="-mx-1 rounded border-0 px-1 py-0.5 [font:inherit] text-inherit underline decoration-current decoration-1 underline-offset-4 hover:bg-slate-150 dark:hover:bg-slate-800"
                    />
                  }
                >
                  {selected.name}
                </PopoverTrigger>
                ?
              </h1>
              <PopoverContent
                side="top"
                sideOffset={12}
                align="center"
                className="w-[min(360px,calc(100vw-32px))] border border-slate-300 p-1.5 text-left dark:border-slate-600 dark:bg-slate-750"
              >
                <Command shouldFilter={false} className="bg-transparent dark:bg-transparent">
                  <CommandInput
                    autoFocus
                    value={query}
                    onValueChange={setQuery}
                    placeholder="Search projects"
                    className="text-[13px]"
                  />
                  <CommandList className="max-h-[240px]">
                    <CommandEmpty className="text-[12px] text-slate-500">
                      No matching projects.
                    </CommandEmpty>
                    {visibleProjects.map((project) => {
                      const active = project.projectId === selected.projectId
                      return (
                        <CommandItem
                          key={project.projectId}
                          value={`${project.name} ${project.cwd}`}
                          data-checked={active || undefined}
                          onSelect={() => selectProject(project)}
                          className="h-9 px-2.5 text-[13px] text-slate-700 dark:text-slate-200"
                        >
                          <span className="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
                            {project.name}
                          </span>
                        </CommandItem>
                      )
                    })}
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>
          ) : loading ? (
            <h1 className="font-mono text-[24px] font-semibold leading-[1.4] text-slate-900 dark:text-slate-100">
              What would you like to do in{" "}
              <span
                role="status"
                aria-label="Loading project"
                className="inline-flex min-w-8 items-center justify-center align-middle"
              >
                <Spinner className="size-5 text-blue-600 motion-reduce:animate-none dark:text-blue-400" />
              </span>
              ?
            </h1>
          ) : error ? (
            <>
              <h1 className="font-mono text-[24px] font-semibold leading-[1.4] text-slate-900 dark:text-slate-100">
                What would you like to do?
              </h1>
              <span className="mt-4 block font-sans text-[13px] text-red-600 dark:text-red-400">
                Failed to load projects.
              </span>
            </>
          ) : (
            <Button
              type="button"
              onClick={() => setCreatingProject(true)}
              className="h-9 bg-blue-700 text-[13px] font-semibold text-white hover:bg-blue-800 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400"
            >
              <FolderPlus size={16} /> New project
            </Button>
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
