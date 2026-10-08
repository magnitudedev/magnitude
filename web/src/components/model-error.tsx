import { Option } from "effect"
import type { ModelAcquisitionFailure, ModelInstanceFailure, ModelFailure } from "@magnitudedev/sdk"
import { MEMORY_SHORTAGE_TITLE, describeMemoryShortage, formatStorageSize, type LocalModelCommandFailure } from "@magnitudedev/client-common"
import { ErrorNotice, type NoticeContent } from "./error-notice"
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip"
import { WarningCircleIcon } from "@phosphor-icons/react"
import type { ReactNode } from "react"

export const modelLoadNotice = (failure: ModelInstanceFailure): NoticeContent => {
  if ("_tag" in failure) return { title: MEMORY_SHORTAGE_TITLE, description: describeMemoryShortage(failure.shortage) }
  return { title: "This model couldn’t start", description: "Try loading it again. If it still won’t start, choose another model." }
}

export function ModelLoadNotice({ failure, actions }: { failure: ModelInstanceFailure; actions?: ReactNode }) {
  return <ErrorNotice {...modelLoadNotice(failure)} actions={actions} />
}

/** A compact amber marker for a row whose last load failed; hovering explains the failure without displacing the row's neighbours. */
export function ModelLoadFailureIndicator({ failure }: { failure: ModelInstanceFailure }) {
  const notice = modelLoadNotice(failure)
  return <Tooltip>
    <TooltipTrigger render={<span role="img" aria-label={notice.title} className="inline-flex shrink-0 text-amber-500 dark:text-amber-400" />}>
      <WarningCircleIcon aria-hidden="true" weight="fill" className="size-4" />
    </TooltipTrigger>
    <TooltipContent side="top" sideOffset={6} className="max-w-sm flex-col items-start gap-1 border border-slate-300 bg-white px-3 py-2.5 text-left text-slate-900 shadow-md dark:border-slate-600 dark:bg-slate-750 dark:text-slate-100">
      <span className="text-[12px] font-semibold leading-4">{notice.title}</span>
      {notice.description && <span className="text-xs text-slate-600 dark:text-slate-400">{notice.description}</span>}
    </TooltipContent>
  </Tooltip>
}

export const downloadNotice = (failure: ModelAcquisitionFailure): NoticeContent => {
  switch (failure._tag) {
    case "Interrupted": return { title: "Download interrupted", description: "Try the download again. Saved progress will be reused where possible." }
    case "InsufficientDiskSpace": return { title: "Not enough disk space", description: `Free up ${formatStorageSize(Math.max(0, failure.requiredBytes - failure.availableBytes))} on the drive where your models are stored, then try again.` }
    case "SourceUnavailable": return { title: "This download is unavailable", description: "The model source no longer provides the required files. Choose another model." }
    case "NetworkUnavailable": return { title: "Can’t reach the model source", description: "Check your internet connection, then try the download again." }
    case "LocalStorageFailure": return { title: "Couldn’t save the model", description: "Check that your model folder is writable and its drive is connected, then try again." }
    case "CorruptDownload": return { title: "The download couldn’t be verified", description: "Try downloading the model again." }
    case "Internal": return { title: "The download couldn’t finish", description: "Try the download again." }
  }
}

export const modelRemovalNotice = (failure: ModelFailure): NoticeContent => {
  switch (failure.code) {
    case "model_removal_retained_external": return { title: "This model is managed outside Magnitude", description: "Its files are in an external model cache and have not been removed." }
    case "model_removal_retained_shared": return { title: "These model files are shared", description: "Another model uses these files, so they have not been removed." }
    default: return { title: "Couldn’t remove the model", description: "The files may still be on disk. Check that the model folder is accessible before trying again." }
  }
}

export const modelCommandNotice = (failure: LocalModelCommandFailure): NoticeContent => {
  if (failure.operation === "load" && Option.isSome(failure.rejection) && failure.rejection.value.code === "model_instance_stopped") return {
    severity: "info", title: "Loading was stopped", description: "Load the model again when you’re ready.",
  }
  if (Option.isSome(failure.rejection) && failure.rejection.value.code === "memory_shortage") return {
    title: MEMORY_SHORTAGE_TITLE, description: "Quit apps you aren’t using, or choose a smaller model.",
  }
  switch (failure.operation) {
    case "load": return { title: "This model couldn’t load", description: "Try loading it again, or choose another model." }
    case "install": return { title: "The download couldn’t start", description: "Check your connection and available disk space, then try again." }
    case "cancel": return { title: "Couldn’t cancel the download", description: "The download may still be running. Check its progress before trying again." }
    case "dismiss": return { title: "Couldn’t dismiss the download error", description: "The previous download error is still available." }
    case "remove": return modelRemovalNotice(Option.getOrElse(failure.rejection, () => ({ code: "unknown", message: "", retryable: true })))
  }
}
