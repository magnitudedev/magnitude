import { useState } from "react"
import { EyeSlashIcon, TrashIcon } from "@phosphor-icons/react"
import { catalogComponentName, formatLocalModelDisplayName, removalKind } from "@magnitudedev/client-common"
import { type LocalModel } from "@magnitudedev/sdk"
import { Button } from "./ui/button"
import { ConfirmDialog } from "./confirm-dialog"
import { addedModelAwaitingFiles } from "./model-card-state"

/** Whether the files Remove deletes are only Magnitude's copy, with the same bytes also in another location that keeps the model listed. */
export const keepsExternalCopy = (model: LocalModel): boolean => {
  const acquisition = model.acquisitionState
  return removalKind(model) === "delete" && "installation" in acquisition && acquisition.installation.ownership === "Mixed"
}

/** The confirmation for Remove: forgetting a file, deleting only Magnitude's copy, forgetting an added model not yet downloaded, or deleting the download. */
export const removalConfirmation = (model: LocalModel, running: boolean): { readonly title: string; readonly description: string; readonly confirmLabel: string } => {
  const name = formatLocalModelDisplayName(model)
  const confirmLabel = running ? "Stop and remove" : "Remove"
  if (removalKind(model) === "forget") return {
    title: running ? `Stop ${name} and remove it from Magnitude?` : `Remove ${name} from Magnitude?`,
    description: "The file stays where it is. You can add it again with Add GGUF file…",
    confirmLabel: running ? confirmLabel : "Remove from Magnitude",
  }
  if (model._tag === "Catalog" && removalKind(model) === "parts") {
    const parts = model.storedComponents.map(component => catalogComponentName(component).toLowerCase()).join(" and ")
    if (parts.length === 0) return {
      title: `Cancel the download Magnitude has in progress for ${name}?`,
      description: "The model file is in another app's location, which Magnitude never deletes, so the model stays in My Models and runs from there. Hide it to stop listing it.",
      confirmLabel: running ? "Stop and cancel" : "Cancel download",
    }
    return {
      title: running ? `Stop ${name} and remove the ${parts} Magnitude downloaded?` : `Remove the ${parts} Magnitude downloaded for ${name}?`,
      description: "The model file is in another app's location, which Magnitude never deletes, so the model stays in My Models and runs from there. Hide it to stop listing it.",
      confirmLabel,
    }
  }
  if (addedModelAwaitingFiles(model)) return {
    title: `Remove ${name} from Magnitude?`,
    description: "Anything Magnitude downloaded for it so far is deleted. You can add it again from Hugging Face.",
    confirmLabel,
  }
  if (keepsExternalCopy(model)) return {
    title: running ? `Stop ${name} and remove Magnitude’s copy?` : `Remove Magnitude’s copy of ${name}?`,
    description: "Only the files Magnitude downloaded are deleted. The same model is also in another location on this computer, so it stays in My Models from there.",
    confirmLabel,
  }
  return {
    title: running ? `Stop ${name} and remove its downloaded files?` : `Remove the downloaded files for ${name}?`,
    description: "You can download it again later.",
    confirmLabel,
  }
}

/** Remove for any model: deleting or forgetting asks first; hiding, offered with `onHide`, is undone from the Hidden filter. */
export function ModelRemoveAction({ model, running, disabled, onRemove, onHide }: {
  model: LocalModel
  running: boolean
  disabled: boolean
  onRemove: () => void
  onHide?: () => void
}) {
  const [confirming, setConfirming] = useState(false)
  const name = formatLocalModelDisplayName(model)
  const kind = removalKind(model)
  const hide = onHide && <Button variant="ghost" size="icon" aria-label={`Hide ${name} from Magnitude`} title="Hide from Magnitude" disabled={disabled} onClick={onHide}><EyeSlashIcon /></Button>
  if (kind === "hide") return hide
  const forget = kind === "forget"
  return <>
    {kind === "parts" && hide}
    <Button variant="ghost" size="icon" aria-label={forget ? `Remove ${name} from Magnitude` : kind === "parts" ? `Remove the files Magnitude downloaded for ${name}` : `Remove ${name}`} title={forget ? "Remove from Magnitude" : "Remove download"} disabled={disabled} onClick={() => setConfirming(true)}><TrashIcon /></Button>
    <ConfirmDialog open={confirming} onOpenChange={setConfirming} destructive={!forget} {...removalConfirmation(model, running)} onConfirm={onRemove} />
  </>
}
