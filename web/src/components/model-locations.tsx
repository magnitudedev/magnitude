import { useState } from "react"
import { Cause, Option } from "effect"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { FolderPlusIcon } from "@phosphor-icons/react"
import { unreadableModelLocationMessage, useAgentClient } from "@magnitudedev/client-common"
import type { AutomaticModelLocationKind, ModelLocation, ModelLocationKind } from "@magnitudedev/sdk"
import { Button } from "./ui/button"
import { Switch } from "./ui/switch"
import { ErrorNotice, type NoticeContent } from "./error-notice"
import { SettingsRow } from "./settings-row"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "./ui/dialog"
import { SkeletonLine } from "./page-skeletons"
import { FolderPicker } from "./folder-picker"
import { modelLocationErrorNotice } from "./candidate-errors"

const locationNames: Record<ModelLocationKind, string> = { HuggingFaceCache: "Hugging Face cache", LmStudio: "LM Studio", LlamaCpp: "llama.cpp", Ollama: "Ollama", Folder: "Folder" }

const folderName = (path: string) => path.split(/[\\/]/).filter(part => part.length > 0).at(-1) ?? path

const folderCount = (count: number) => `${count} ${count === 1 ? "folder" : "folders"}`

/** "Hugging Face cache, LM Studio · 1 folder": the automatic locations found and on, then how many folders are added. */
export const modelLocationsSummary = (locations: readonly ModelLocation[]): string => {
  const automatic = locations.flatMap(location => location.kind !== "Folder" && location.state === "Found" && location.enabled ? [locationNames[location.kind]] : [])
  const folders = locations.filter(location => location.kind === "Folder").length
  const parts = [...(automatic.length > 0 ? [automatic.join(", ")] : []), ...(folders > 0 ? [folderCount(folders)] : [])]
  return parts.length > 0 ? parts.join(" · ") : "Magnitude isn’t looking anywhere else for models"
}

const pathText = "block truncate text-xs text-slate-500"
const sectionHeading = "font-heading text-sm font-semibold"
const listClass = "divide-y divide-slate-200 rounded-lg border border-slate-200 dark:divide-slate-750 dark:border-slate-750"
const rowClass = "flex items-center gap-3 px-3 py-2"

const locationPath = (location: ModelLocation) =>
  <span className={pathText} title={location.path}>{location.state === "Found" ? location.path : `${location.state === "Unreadable" ? "Can’t read" : "Not found"} · ${location.path}`}</span>

const unreadableHint = (location: ModelLocation) =>
  location.state === "Unreadable" && <span className="mt-0.5 block text-xs text-orange-700 dark:text-orange-300">{unreadableModelLocationMessage(location.kind)}</span>

/**
 * Manage…: the automatic locations here with an on/off switch, including ones Magnitude can't read
 * with what to do about it, the ones not found tucked away, then your folders.
 */
export function ModelLocationsManager({ locations, busy, notice, onToggle, onRemove, onAdd }: {
  locations: readonly ModelLocation[]
  busy: boolean
  notice: NoticeContent | undefined
  onToggle: (kind: AutomaticModelLocationKind, enabled: boolean) => void
  onRemove: (path: string) => void
  onAdd: () => void
}) {
  const automatic = locations.flatMap(location => location.kind === "Folder" ? [] : [{ ...location, kind: location.kind }])
  const found = automatic.filter(location => location.state !== "NotFound")
  const missing = automatic.filter(location => location.state === "NotFound")
  const folders = locations.filter(location => location.kind === "Folder")
  return <div className="space-y-4">
    {notice && <ErrorNotice {...notice} />}
    <section aria-label="Found on this computer" className="space-y-2">
      <h3 className={sectionHeading}>Found on this computer</h3>
      {found.length === 0
        ? <p className="text-xs text-slate-500">No other apps’ model folders were found.</p>
        : <ul className={listClass}>
          {found.map(location => {
            const name = locationNames[location.kind]
            return <li key={location.kind} className={rowClass}>
              <span className="min-w-0 flex-1"><span className="block text-sm font-medium">{name}</span>{locationPath(location)}{unreadableHint(location)}</span>
              <Switch aria-label={`Look for models in ${name}`} checked={location.enabled} disabled={busy} onCheckedChange={checked => onToggle(location.kind, checked)} />
            </li>
          })}
        </ul>}
      {missing.length > 0 && <details className="text-xs text-slate-500">
        <summary className="cursor-pointer select-none">Not found on this computer ({missing.length})</summary>
        <ul className="mt-1.5 space-y-1 pl-4">
          {missing.map(location => <li key={location.kind} className="min-w-0"><span>{locationNames[location.kind]}</span><span className="block truncate" title={location.path}>{location.path}</span></li>)}
        </ul>
      </details>}
    </section>
    <section aria-label="Your folders" className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <h3 className={sectionHeading}>Your folders</h3>
        <Button size="sm" variant="outline" disabled={busy} onClick={onAdd}><FolderPlusIcon />Add folder…</Button>
      </div>
      <p className="text-xs text-slate-500">{folders.length === 0
        ? "Add a folder of GGUF files no other app manages, such as an external drive. Files added to it later appear automatically."
        : "Magnitude watches these folders, so files added or deleted later appear or disappear."}</p>
      {folders.length > 0 && <ul className={listClass}>
        {folders.map(folder => <li key={folder.path} className={rowClass}>
          <span className="min-w-0 flex-1"><span className="block truncate text-sm font-medium">{folderName(folder.path)}</span>{locationPath(folder)}{unreadableHint(folder)}</span>
          <Button size="sm" variant="ghost" disabled={busy} aria-label={`Remove ${folder.path}`} onClick={() => onRemove(folder.path)}>Remove</Button>
        </li>)}
      </ul>}
    </section>
  </div>
}

/** Settings → Model locations: one row summarizing where Magnitude looks, with Manage… for the details. */
export function ModelLocationsRow() {
  const client = useAgentClient()
  const locations = useAtomValue(client.Models.ListModelLocations({})).result
  const setEnabled = useAtomSet(client.Models.SetModelLocationEnabled)
  const addFolder = useAtomSet(client.Models.AddModelFolder)
  const removeFolder = useAtomSet(client.Models.RemoveModelFolder)
  const toggling = useAtomValue(client.Models.SetModelLocationEnabled)
  const adding = useAtomValue(client.Models.AddModelFolder)
  const removing = useAtomValue(client.Models.RemoveModelFolder)
  const [managing, setManaging] = useState(false)
  const [choosing, setChoosing] = useState(false)
  const busy = toggling.waiting || adding.waiting || removing.waiting
  const notice: NoticeContent | undefined = busy ? undefined
    : Result.isFailure(adding) ? Option.match(Cause.failureOption(adding.cause), { onNone: () => ({ title: "The folder wasn’t added" }), onSome: modelLocationErrorNotice })
    : Result.isFailure(removing) ? Option.match(Cause.failureOption(removing.cause), { onNone: () => ({ title: "The folder wasn’t removed" }), onSome: modelLocationErrorNotice })
    : Result.isFailure(toggling) ? { title: "The location wasn’t changed", description: "Try again." }
    : undefined
  return <SettingsRow label="Model locations"
    hint={Result.isSuccess(locations) ? modelLocationsSummary(locations.value.locations)
      : Result.isFailure(locations) ? undefined
      : <SkeletonLine className="h-4 text-xs" width="220px" />}
    alert={Result.isFailure(locations) ? <ErrorNotice title="Couldn’t read model locations" description="Magnitude keeps looking in the locations it already uses." /> : undefined}
    control={<Button size="sm" variant="outline" disabled={!Result.isSuccess(locations)} onClick={() => setManaging(true)}>Manage…</Button>}>
    <Dialog open={managing} onOpenChange={setManaging}>
      <DialogContent className="max-h-[85vh] grid-cols-[minmax(0,1fr)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Model locations</DialogTitle>
          <DialogDescription>Where Magnitude looks for GGUF files. It reads them and never changes them.</DialogDescription>
        </DialogHeader>
        {Result.isSuccess(locations) && <ModelLocationsManager locations={locations.value.locations} busy={busy} notice={notice}
          onToggle={(kind, enabled) => setEnabled({ kind, enabled })} onRemove={path => removeFolder({ path })} onAdd={() => setChoosing(true)} />}
      </DialogContent>
    </Dialog>
    <FolderPicker open={choosing} onOpenChange={setChoosing} initialPath={Option.none()} title="Add a folder of models"
      description="Folders on the computer running Magnitude. Magnitude reads GGUF files there and never changes them." confirmLabel="Add this folder"
      onChoose={path => addFolder({ path })} />
  </SettingsRow>
}
