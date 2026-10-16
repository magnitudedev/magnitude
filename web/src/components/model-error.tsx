import { Option } from "effect"
import type { ModelAcquisitionFailure, ModelInstanceFailure, ModelFailure } from "@magnitudedev/sdk"
import { MEMORY_SHORTAGE_TITLE, describeMemoryShortage, formatStorageSize, memoryShortageTitle, type LocalModelCommandFailure } from "@magnitudedev/client-common"
import { ErrorNotice, type NoticeContent } from "./error-notice"
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip"
import { WarningCircleIcon } from "@phosphor-icons/react"
import type { ReactNode } from "react"
import { readLanguage, translate } from "../i18n"

const tr = (key: Parameters<typeof translate>[1]) => translate(readLanguage(), key)

export const modelLoadNotice = (failure: ModelInstanceFailure): NoticeContent => {
  if ("_tag" in failure) return { title: memoryShortageTitle(failure.shortage), description: describeMemoryShortage(failure.shortage) }
  return { title: tr("modelCouldNotStart"), description: tr("tryLoadAnother") }
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
    case "Interrupted": return { title: tr("downloadInterrupted"), description: tr("retryDownloadSaved") }
    case "InsufficientDiskSpace": return { title: tr("notEnoughDiskSpace"),  description: tr("freeDiskSpace").replace("{size}", formatStorageSize(Math.max(0, failure.requiredBytes - failure.availableBytes))) }
    case "SourceUnavailable": return { title: tr("downloadUnavailable"), description: tr("sourceNoLongerProvides") }
    case "NetworkUnavailable": return { title: tr("cannotReachSource"), description: tr("checkInternet") }
    case "LocalStorageFailure": return { title: tr("couldNotSaveModel"), description: tr("checkModelFolder") }
    case "CorruptDownload": return { title: tr("downloadNotVerified"), description: tr("retryDownloadAgain") }
    case "Internal": return { title: tr("downloadCouldNotFinish"), description: "Try the download again." }
  }
}

export const modelRemovalNotice = (failure: ModelFailure): NoticeContent => {
  switch (failure.code) {
    case "model_removal_retained_external": return { title: tr("modelExternalManaged"), description: tr("externalCacheNotRemoved") }
    case "model_removal_retained_shared": return { title: tr("modelFilesShared"), description: tr("sharedFilesRetained") }
    default: return { title: tr("couldNotRemoveModel"), description: tr("filesMayRemain") }
  }
}

export const modelCommandNotice = (failure: LocalModelCommandFailure): NoticeContent => {
  if (failure.operation === "load" && Option.isSome(failure.rejection) && failure.rejection.value.code === "model_instance_stopped") return {
    severity: "info", title: tr("loadingStopped"), description: tr("loadWhenReady"),
  }
  if (failure.operation === "load" && Option.isSome(failure.rejection) && failure.rejection.value.code === "model_does_not_fit") return {
    title: tr("modelDoesNotFit"), description: tr("needsMoreMemory"),
  }
  if (Option.isSome(failure.rejection) && failure.rejection.value.code === "memory_shortage") return {
    title: MEMORY_SHORTAGE_TITLE, description: tr("quitUnusedApps"),
  }
  switch (failure.operation) {
    case "load": return { title: tr("modelCouldNotLoad"), description: tr("tryLoadOrChoose") }
    case "install": return { title: tr("downloadCouldNotStart"), description: tr("checkConnectionDisk") }
    case "cancel": return { title: tr("couldNotCancelDownload"), description: tr("downloadMayStillRun") }
    case "dismiss": return { title: tr("couldNotDismissError"), description: tr("previousDownloadError") }
    case "remove": return modelRemovalNotice(Option.getOrElse(failure.rejection, () => ({ code: "unknown", message: "", retryable: true })))
  }
}
