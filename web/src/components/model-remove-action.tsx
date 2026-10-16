import { useState } from "react"
import { EyeSlashIcon, TrashIcon } from "@phosphor-icons/react"
import { formatLocalModelDisplayName } from "@magnitudedev/client-common"
import { installedAcquisition, type LocalModel } from "@magnitudedev/sdk"
import { Button } from "./ui/button"
import { ConfirmDialog } from "./confirm-dialog"

/** What Remove does follows who owns the files: delete Magnitude's own, forget an imported file, hide a found one. */
export type RemovalKind = "delete" | "forget" | "hide"

export const removalKind = (model: LocalModel): RemovalKind => {
  if (model._tag === "Catalog") return installedAcquisition(model.acquisitionState)?.installation.ownership === "External" ? "hide" : "delete"
  return model.origin._tag === "Added" ? "delete" : model.origin._tag === "Imported" ? "forget" : "hide"
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
  if (kind === "hide") return onHide && <Button variant="ghost" size="icon" aria-label={`Hide ${name} from Magnitude`} title="Hide from Magnitude" disabled={disabled} onClick={onHide}><EyeSlashIcon /></Button>
  const forget = kind === "forget"
  return <>
    <Button variant="ghost" size="icon" aria-label={forget ? `Remove ${name} from Magnitude` : `Remove ${name}`} title={forget ? "Remove from Magnitude" : "Remove download"} disabled={disabled} onClick={() => setConfirming(true)}><TrashIcon /></Button>
    <ConfirmDialog open={confirming} onOpenChange={setConfirming} destructive={!forget}
      title={forget
        ? running ? `Stop ${name} and remove it from Magnitude?` : `Remove ${name} from Magnitude?`
        : running ? `Stop ${name} and remove its downloaded files?` : `Remove the downloaded files for ${name}?`}
      description={forget ? "The file stays where it is. You can add it again with Add GGUF file…." : "You can download it again later."}
      confirmLabel={running ? "Stop and remove" : forget ? "Remove from Magnitude" : "Remove"} onConfirm={onRemove} />
  </>
}
