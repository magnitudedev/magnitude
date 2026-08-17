import { useMemo, useState, type ReactNode } from "react"
import { useAtomSet, useAtomValue, Result } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import { FolderOpen } from "@phosphor-icons/react"
import {
  useAgentClient,
  usePlatform,
} from "@magnitudedev/client-common"
import type { ProjectRecord } from "@magnitudedev/sdk"
import {
  Dialog,
  DialogActions,
  dialogPrimaryButton,
  dialogSecondaryButton,
} from "./dialog"

const basename = (path: string): string =>
  path.replace(/[\\/]+$/, "").split(/[\\/]/).at(-1) ?? path

export function ProjectFormDialog({
  project,
  onDismiss,
  onSaved,
}: {
  readonly project?: ProjectRecord
  readonly onDismiss: () => void
  readonly onSaved: (project: ProjectRecord) => void
}): ReactNode {
  const client = useAgentClient()
  const platform = usePlatform()
  const [sourceDirectory, setSourceDirectory] = useState(project?.sourceDirectory ?? "")
  const [name, setName] = useState(project?.name ?? "")
  const [nameWasEdited, setNameWasEdited] = useState(project !== undefined)
  const [sourcePickerFailed, setSourcePickerFailed] = useState(false)
  const createMutation = useMemo(() => client.rpc.mutation("CreateProject"), [client])
  const editMutation = useMemo(() => client.rpc.mutation("EditProject"), [client])
  const createState = useAtomValue(createMutation)
  const editState = useAtomValue(editMutation)
  const create = useAtomSet(createMutation, { mode: "promise" })
  const edit = useAtomSet(editMutation, { mode: "promise" })
  const mutationState = project ? editState : createState
  const pending = Result.isWaiting(mutationState)
  const failed = Result.isFailure(mutationState)
  const failure = failed
    ? Option.getOrNull(Cause.failureOption(mutationState.cause))
    : null
  const failureMessage = failure === null
    ? (project ? "Could not update this project." : "Could not create this project.")
    : failure._tag === "InvalidProjectSource"
      ? `That source cannot be used: ${failure.reason}`
      : failure._tag === "ProjectSourceAlreadyRegistered"
        ? "That source already belongs to another project."
        : failure._tag === "ProjectBusy"
          ? "Wait for this project's active work to finish before changing its source."
          : failure._tag === "InvalidProjectName"
            ? "Enter a project name."
            : project
              ? "Could not update this project."
              : "Could not create this project."

  const updateSource = (path: string) => {
    setSourceDirectory(path)
    if (!nameWasEdited || !name.trim()) setName(basename(path))
  }

  const selectSource = async () => {
    setSourcePickerFailed(false)
    try {
      const selected = await platform.dialogs.openDirectory()
      if (selected) updateSource(selected)
    } catch {
      setSourcePickerFailed(true)
    }
  }

  const save = async () => {
    const cleanName = name.trim()
    const cleanSource = sourceDirectory.trim()
    if (!cleanName || !cleanSource || pending) return
    try {
      const saved = project
        ? await edit({
            payload: {
              projectId: project.projectId,
              name: cleanName,
              sourceDirectory: cleanSource,
            },
            reactivityKeys: ["projects", "sessions"],
          })
        : await create({
            payload: { name: cleanName, sourceDirectory: cleanSource },
            reactivityKeys: ["projects", "sessions"],
          })
      onSaved(saved)
    } catch {
      // Mutation Result owns the rendered failure state.
    }
  }

  return (
    <Dialog
      title={project ? "Edit project" : "New project"}
      description={project
        ? "Change the name or source. Every existing chat in this project will use the new source directory."
        : "Choose a folder for Magnitude to work in."
      }
      onDismiss={onDismiss}
    >
      <div className="space-y-4 px-5 pb-5 pt-2">
        <label className="block">
          <span className="mb-1.5 block font-sans text-[12px] font-semibold text-slate-700 dark:text-slate-300">
            Source
          </span>
          <div className="flex gap-2">
            <input
              value={sourceDirectory}
              onChange={(event) => updateSource(event.target.value)}
              placeholder="/path/to/project"
              readOnly={platform.id === "desktop"}
              className="h-9 min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-3 font-sans text-[13px] text-slate-900 outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-500/15 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-blue-500"
            />
            {platform.id === "desktop" ? (
              <button
                autoFocus
                type="button"
                onClick={selectSource}
                className={dialogSecondaryButton}
              >
                <span className="flex items-center gap-2"><FolderOpen size={16} /> {project ? "Change source" : "Select source"}</span>
              </button>
            ) : null}
          </div>
          <span className="mt-1.5 block font-sans text-[11px] text-slate-500">
            {platform.id === "desktop"
              ? "Select a source folder on this computer."
              : "Enter an absolute directory on the agent host."
            }
          </span>
          {sourcePickerFailed ? (
            <span role="alert" className="mt-1.5 block font-sans text-[11px] text-red-600 dark:text-red-400">
              Could not open the folder picker.
            </span>
          ) : null}
        </label>

        <label className="block">
          <span className="mb-1.5 block font-sans text-[12px] font-semibold text-slate-700 dark:text-slate-300">
            Project name
          </span>
          <input
            autoFocus={platform.id !== "desktop"}
            value={name}
            onChange={(event) => {
              setNameWasEdited(true)
              setName(event.target.value)
            }}
            placeholder="Project name"
            className="h-9 w-full rounded-md border border-slate-300 bg-white px-3 font-sans text-[13px] text-slate-900 outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-500/15 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-blue-500"
            onKeyDown={(event) => {
              if (event.key === "Enter") void save()
            }}
          />
        </label>
        {failed ? (
          <p role="alert" className="font-sans text-[12px] text-red-600 dark:text-red-400">
            {failureMessage}
          </p>
        ) : null}
      </div>
      <DialogActions>
        <button type="button" onClick={onDismiss} disabled={pending} className={dialogSecondaryButton}>
          Cancel
        </button>
        <button
          type="button"
          onClick={() => void save()}
          disabled={pending || !sourceDirectory.trim() || !name.trim()}
          className={dialogPrimaryButton}
        >
          {pending ? "Saving…" : project ? "Save changes" : "Create new"}
        </button>
      </DialogActions>
    </Dialog>
  )
}

export function RemoveProjectDialog({
  project,
  onDismiss,
  onRemoved,
}: {
  readonly project: ProjectRecord
  readonly onDismiss: () => void
  readonly onRemoved: () => void
}): ReactNode {
  const client = useAgentClient()
  const mutation = useMemo(() => client.rpc.mutation("RemoveProject"), [client])
  const state = useAtomValue(mutation)
  const remove = useAtomSet(mutation, { mode: "promise" })
  const pending = Result.isWaiting(state)
  const failed = Result.isFailure(state)
  const confirm = async () => {
    try {
      await remove({
        payload: { projectId: project.projectId },
        reactivityKeys: ["projects", "sessions"],
      })
      onRemoved()
    } catch {
      // Mutation Result owns the rendered failure state.
    }
  }
  return (
    <Dialog
      title="Remove project?"
      description={`Remove “${project.name}” from the sidebar? Its sessions and files will not be deleted.`}
      onDismiss={onDismiss}
      size="small"
    >
      {failed ? (
        <p role="alert" className="px-5 pb-4 font-sans text-[12px] text-red-600 dark:text-red-400">
          Could not remove this project.
        </p>
      ) : null}
      <DialogActions>
        <button type="button" onClick={onDismiss} disabled={pending} className={dialogSecondaryButton}>
          Cancel
        </button>
        <button
          type="button"
          onClick={() => void confirm()}
          disabled={pending}
          className="h-9 rounded-md border border-red-600 bg-red-600 px-3.5 font-sans text-[13px] font-semibold text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50 dark:border-red-500 dark:bg-red-500 dark:text-slate-925 dark:hover:bg-red-400"
        >
          {pending ? "Removing…" : "Remove"}
        </button>
      </DialogActions>
    </Dialog>
  )
}
