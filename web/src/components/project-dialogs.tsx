import { useMemo, useState, type ReactNode } from "react"
import { useAtomSet, useAtomValue, Result } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import { FolderOpen } from "@phosphor-icons/react"
import { useAgentClient, usePlatform } from "@magnitudedev/client-common"
import { Projects, type Project } from "@magnitudedev/sdk"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
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
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

const basename = (path: string): string =>
  path
    .replace(/[\\/]+$/, "")
    .split(/[\\/]/)
    .at(-1) ?? path

export function ProjectFormDialog({
  project,
  onDismiss,
  onSaved,
}: {
  readonly project?: Project
  readonly onDismiss: () => void
  readonly onSaved: (project: Project) => void
}): ReactNode {
  const client = useAgentClient()
  const platform = usePlatform()
  const [cwd, setCwd] = useState(project?.cwd ?? "")
  const [name, setName] = useState(project?.name ?? "")
  const [nameWasEdited, setNameWasEdited] = useState(project !== undefined)
  const [sourcePickerFailed, setSourcePickerFailed] = useState(false)
  const createMutation = useMemo(() => client.mutation(Projects.CreateProject), [client])
  const editMutation = useMemo(() => client.mutation(Projects.EditProject), [client])
  const createState = useAtomValue(createMutation)
  const editState = useAtomValue(editMutation)
  const create = useAtomSet(createMutation, { mode: "promise" })
  const edit = useAtomSet(editMutation, { mode: "promise" })
  const mutationState = project ? editState : createState
  const pending = Result.isWaiting(mutationState)
  const failed = Result.isFailure(mutationState)
  const failure = failed ? Option.getOrNull(Cause.failureOption(mutationState.cause)) : null
  const failureMessage =
    failure === null
      ? project
        ? "Could not update this project."
        : "Could not create this project."
      : failure._tag === "InvalidProjectName"
      ? "Enter a project name."
      : failure._tag === "InvalidDirectoryPath" || failure._tag === "PathNotDirectory"
      ? "Enter an absolute folder path."
      : failure._tag === "DirectoryNotFound"
      ? "That folder does not exist on the agent host."
      : failure._tag === "DirectoryAccessDenied"
      ? "That folder cannot be accessed."
      : failure._tag === "ProjectCwdAlreadyRegistered"
      ? "That folder already belongs to another project."
      : project
      ? "Could not update this project."
      : "Could not create this project."

  const updateSource = (path: string) => {
    setCwd(path)
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
    const cleanCwd = cwd.trim()
    if (!cleanName || !cleanCwd || pending) return
    try {
      const saved = project
        ? await edit({
            projectId: project.projectId,
            name: cleanName,
            cwd: cleanCwd,
          })
        : await create({ name: cleanName, cwd: cleanCwd })
      onSaved(saved)
    } catch {
      // Mutation Result owns the rendered failure state.
    }
  }

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onDismiss()
      }}
    >
      <DialogContent className="max-w-[520px] gap-0 overflow-hidden p-0 dark:bg-slate-800">
        <DialogHeader className="px-5 pb-3 pt-5">
          <DialogTitle className="text-[15px] font-semibold">
            {project ? "Edit project" : "New project"}
          </DialogTitle>
          <DialogDescription className="mt-1.5 text-[13px] leading-5 text-slate-600 dark:text-slate-400">
            {project
              ? "Change the name or folder. Chats stay grouped by the folder they ran in."
              : "Choose a folder for Magnitude to work in."}
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 px-5 pb-5 pt-2">
          <div>
            <Label className="mb-1.5 block text-[12px] font-semibold text-slate-700 dark:text-slate-300">
              Source
            </Label>
            <div className="flex gap-2">
              <Input
                value={cwd}
                onChange={(event) => updateSource(event.target.value)}
                placeholder="/path/to/project"
                readOnly={platform.id === "desktop"}
                className="h-9 min-w-0 flex-1 border-slate-300 bg-white px-3 text-[13px] dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
              {platform.id === "desktop" ? (
                <Button
                  autoFocus
                  type="button"
                  onClick={selectSource}
                  variant="outline"
                  className="h-9"
                >
                  <span className="flex items-center gap-2">
                    <FolderOpen size={16} /> {project ? "Change source" : "Select source"}
                  </span>
                </Button>
              ) : null}
            </div>
            <span className="mt-1.5 block text-[11px] text-slate-500">
              {platform.id === "desktop"
                ? "Select a source folder on this computer."
                : "Enter an absolute directory on the agent host."}
            </span>
            {sourcePickerFailed ? (
              <span
                role="alert"
                className="mt-1.5 block text-[11px] text-red-600 dark:text-red-400"
              >
                Could not open the folder picker.
              </span>
            ) : null}
          </div>

          <div>
            <Label className="mb-1.5 block text-[12px] font-semibold text-slate-700 dark:text-slate-300">
              Project name
            </Label>
            <Input
              autoFocus={platform.id !== "desktop"}
              value={name}
              onChange={(event) => {
                setNameWasEdited(true)
                setName(event.target.value)
              }}
              placeholder="Project name"
              className="h-9 w-full border-slate-300 bg-white px-3 text-[13px] dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              onKeyDown={(event) => {
                if (event.key === "Enter") void save()
              }}
            />
          </div>
          {failed ? (
            <p role="alert" className="text-[12px] text-red-600 dark:text-red-400">
              {failureMessage}
            </p>
          ) : null}
        </div>
        <DialogFooter className="border-t border-slate-200 px-5 py-4 dark:border-slate-750">
          <Button
            type="button"
            onClick={onDismiss}
            disabled={pending}
            variant="outline"
            className="h-9"
          >
            Cancel
          </Button>
          <Button
            type="button"
            onClick={() => void save()}
            disabled={pending || !cwd.trim() || !name.trim()}
            className="h-9 bg-blue-700 text-white hover:bg-blue-800 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400"
          >
            {pending ? "Saving…" : project ? "Save changes" : "Create new"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function RemoveProjectDialog({
  project,
  onDismiss,
  onRemoved,
}: {
  readonly project: Project
  readonly onDismiss: () => void
  readonly onRemoved: () => void
}): ReactNode {
  const client = useAgentClient()
  const mutation = useMemo(() => client.mutation(Projects.RemoveProject), [client])
  const state = useAtomValue(mutation)
  const remove = useAtomSet(mutation, { mode: "promise" })
  const pending = Result.isWaiting(state)
  const failed = Result.isFailure(state)
  const confirm = async () => {
    try {
      await remove({ projectId: project.projectId })
      onRemoved()
    } catch {
      // Mutation Result owns the rendered failure state.
    }
  }
  return (
    <AlertDialog
      open
      onOpenChange={(open) => {
        if (!open) onDismiss()
      }}
    >
      <AlertDialogContent className="max-w-[400px] dark:bg-slate-800">
        <AlertDialogHeader>
          <AlertDialogTitle className="text-[15px] font-semibold">Remove project?</AlertDialogTitle>
          <AlertDialogDescription className="text-[13px] leading-5">
            Remove “{project.name}” from the sidebar? Its sessions and files will not be deleted.
          </AlertDialogDescription>
        </AlertDialogHeader>
        {failed ? (
          <p role="alert" className="text-[12px] text-red-600 dark:text-red-400">
            Could not remove this project.
          </p>
        ) : null}
        <AlertDialogFooter>
          <AlertDialogCancel onClick={onDismiss} disabled={pending} className="h-9">
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            type="button"
            onClick={() => void confirm()}
            disabled={pending}
            variant="destructive"
            className="h-9 bg-red-600 text-white hover:bg-red-700 dark:bg-red-500 dark:text-slate-925 dark:hover:bg-red-400"
          >
            {pending ? "Removing…" : "Remove"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
