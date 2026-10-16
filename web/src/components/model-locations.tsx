import { useState } from "react"
import { Cause, Option } from "effect"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { FolderPlusIcon } from "@phosphor-icons/react"
import { useAgentClient } from "@magnitudedev/client-common"
import type { AutomaticModelLocationKind, ModelLocation, ModelLocationKind } from "@magnitudedev/sdk"
import { Button } from "./ui/button"
import { Switch } from "./ui/switch"
import { ErrorNotice, type NoticeContent } from "./error-notice"
import { SettingsGroup, SettingsRow } from "./settings-row"
import { SkeletonLine } from "./page-skeletons"
import { FolderPicker } from "./folder-picker"
import { modelLocationErrorNotice } from "./candidate-errors"

const locationNames: Record<ModelLocationKind, string> = { HuggingFaceCache: "Hugging Face cache", LmStudio: "LM Studio", LlamaCpp: "llama.cpp", Ollama: "Ollama", Folder: "Folder" }

const folderName = (path: string) => path.split(/[\\/]/).filter(part => part.length > 0).at(-1) ?? path

/** The automatic locations Magnitude found, each of which can be turned off, then your folders with Add and Remove. */
export function ModelLocationRows({ locations, busy, onToggle, onRemove, onAdd }: {
  locations: readonly ModelLocation[]
  busy: boolean
  onToggle: (kind: AutomaticModelLocationKind, enabled: boolean) => void
  onRemove: (path: string) => void
  onAdd: () => void
}) {
  const folders = locations.filter(location => location.kind === "Folder")
  return <>
    {locations.map(location => {
      const kind = location.kind
      if (kind === "Folder") return null
      const name = locationNames[kind]
      return <SettingsRow key={location.kind} label={name}
        hint={<span className="block truncate" title={location.path}>{location.found ? location.path : `Not found · ${location.path}`}</span>}
        control={location.found ? <Switch aria-label={`Look for models in ${name}`} checked={location.enabled} disabled={busy} onCheckedChange={checked => onToggle(kind, checked)} /> : undefined} />
    })}
    {folders.map(folder => <SettingsRow key={folder.path} label={folderName(folder.path)}
      hint={<span className="block truncate" title={folder.path}>{folder.found ? folder.path : `Not found · ${folder.path}`}</span>}
      control={<Button size="sm" variant="ghost" disabled={busy} aria-label={`Remove ${folder.path}`} onClick={() => onRemove(folder.path)}>Remove</Button>} />)}
    <SettingsRow label="Your folders" hint={folders.length === 0 ? "Add a folder of GGUF files no other app manages, such as an external drive. Files added to it later appear automatically." : "Magnitude watches these folders, so files added or deleted later appear or disappear."}
      control={<Button size="sm" variant="outline" disabled={busy} onClick={onAdd}><FolderPlusIcon />Add folder…</Button>} />
  </>
}

/** Settings → Model locations. */
export function ModelLocationsGroup() {
  const client = useAgentClient()
  const locations = useAtomValue(client.Models.ListModelLocations({})).result
  const setEnabled = useAtomSet(client.Models.SetModelLocationEnabled)
  const addFolder = useAtomSet(client.Models.AddModelFolder)
  const removeFolder = useAtomSet(client.Models.RemoveModelFolder)
  const toggling = useAtomValue(client.Models.SetModelLocationEnabled)
  const adding = useAtomValue(client.Models.AddModelFolder)
  const removing = useAtomValue(client.Models.RemoveModelFolder)
  const [choosing, setChoosing] = useState(false)
  const busy = toggling.waiting || adding.waiting || removing.waiting
  const notice: NoticeContent | undefined = busy ? undefined
    : Result.isFailure(adding) ? Option.match(Cause.failureOption(adding.cause), { onNone: () => ({ title: "The folder wasn’t added" }), onSome: modelLocationErrorNotice })
    : Result.isFailure(removing) ? Option.match(Cause.failureOption(removing.cause), { onNone: () => ({ title: "The folder wasn’t removed" }), onSome: modelLocationErrorNotice })
    : Result.isFailure(toggling) ? { title: "The location wasn’t changed", description: "Try again." }
    : undefined
  return <SettingsGroup label="Model locations">
    <FolderPicker open={choosing} onOpenChange={setChoosing} initialPath={Option.none()} title="Add a folder of models"
      description="Folders on the computer running Magnitude. Magnitude reads GGUF files there and never changes them." confirmLabel="Add this folder"
      onChoose={path => addFolder({ path })} />
    {notice && <div className="px-4 py-3"><ErrorNotice {...notice} /></div>}
    {Result.isSuccess(locations)
      ? <ModelLocationRows locations={locations.value.locations} busy={busy} onToggle={(kind, enabled) => setEnabled({ kind, enabled })} onRemove={path => removeFolder({ path })} onAdd={() => setChoosing(true)} />
      : Result.isFailure(locations)
        ? <div className="px-4 py-3"><ErrorNotice title="Couldn’t read model locations" description="Magnitude keeps looking in the locations it already uses." /></div>
        : <div className="px-4 py-3"><SkeletonLine className="h-5 text-sm" width="240px" /><SkeletonLine className="mt-1 h-4 text-xs" width="320px" /></div>}
  </SettingsGroup>
}
