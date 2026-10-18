import { Option } from "effect"
import { ArrowUpRightIcon, LightningIcon } from "@phosphor-icons/react"
import {
  customModelDrafterOffer, formatLocalModelDisplayName,
} from "@magnitudedev/client-common"
import type { CustomLocalModel, DrafterCandidate, LocalModel, ModelTransferProgress } from "@magnitudedev/sdk"
import { Button } from "./ui/button"
import { ErrorNotice, NoticeAction } from "./error-notice"
import { downloadNotice } from "./model-error"

/** Where a custom model's files are: an added repository pinned to its commit, or a path Magnitude references. */
export const customModelSource = (model: CustomLocalModel): { readonly label: string; readonly href: Option.Option<string> } => {
  const origin = model.origin
  if (origin._tag === "Added") return { label: `${origin.repository} · pinned to ${origin.commit.slice(0, 7)}`, href: Option.some(`https://huggingface.co/${origin.repository}`) }
  if (origin._tag === "Imported") return { label: origin.path, href: Option.none() }
  const state = model.acquisitionState
  const path = (state._tag === "External" || state._tag === "Unavailable") && state.installation._tag === "Resolved" ? ` · ${state.installation.primaryPath}` : ""
  return { label: `${model.presentation.sourceLabel}${path}`, href: Option.none() }
}

const textButton = "h-auto px-1.5 py-0.5 text-xs"

/** A newer revision of the custom model's attached catalog drafter exists, and nothing is downloading it right now. */
export const drafterUpdateAvailable = (model: LocalModel): boolean =>
  model._tag === "Custom" && Option.exists(model.drafter, drafter => drafter.updateAvailable && drafter.availability._tag !== "Downloading")

const transferPercent = (progress: ModelTransferProgress) =>
  progress.totalBytes > 0 ? `${Math.floor(progress.completedBytes * 100 / progress.totalBytes)}%` : "Preparing"

/** The card line offering the recommended drafter: one click downloads only the drafter. */
export function DrafterOffer({ model, disabled, onAdd }: { model: CustomLocalModel; disabled: boolean; onAdd: (candidate: DrafterCandidate) => void }) {
  return Option.match(customModelDrafterOffer(model), {
    onNone: () => null,
    onSome: offer => <Button variant="link" size="unstyled" disabled={disabled} onClick={() => onAdd(offer.candidate)} className="inline-flex items-center gap-1 text-sm text-blue-700 dark:text-blue-400">
      <LightningIcon aria-hidden="true" className="size-3.5" />{offer.label}
    </Button>,
  })
}

/** The Drafter row of a custom model's details: the attached drafter with Change and Turn off, or the offer and Change. */
export function CustomModelDrafterRow({ model, disabled, onChangeDrafter, onTurnOffDrafter, onAddDrafter }: {
  model: CustomLocalModel
  disabled: boolean
  onChangeDrafter: () => void
  onTurnOffDrafter: () => void
  onAddDrafter: (candidate: DrafterCandidate) => void
}) {
  return <dl><div className="min-w-0"><dt className="text-xs text-slate-500">Drafter</dt><dd className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
    {Option.match(model.drafter, {
      onSome: drafter => <>
        <span className="min-w-0 truncate" title={drafter.name}>{drafter.name}{drafter.updateAvailable ? " · Update available" : ""}</span>
        <DrafterOffer model={model} disabled={disabled} onAdd={onAddDrafter} />
        <Button variant="outline" size="unstyled" className={textButton} disabled={disabled} onClick={onChangeDrafter}>Change</Button>
        <Button variant="ghost" size="unstyled" className={textButton} disabled={disabled} onClick={onTurnOffDrafter}>Turn off</Button>
      </>,
      onNone: () => <>
        {Option.isSome(customModelDrafterOffer(model)) ? <DrafterOffer model={model} disabled={disabled} onAdd={onAddDrafter} /> : <span className="text-slate-500">None</span>}
        <Button variant="outline" size="unstyled" className={textButton} disabled={disabled} onClick={onChangeDrafter}>Change</Button>
      </>,
    })}
  </dd></div></dl>
}

/** The Source of a custom model's details ("LM Studio · path", a pinned repository), and the catalog model it is a variant of. */
export function CustomModelSource({ model, variantOfName }: { model: CustomLocalModel; variantOfName: Option.Option<string> }) {
  const source = customModelSource(model)
  return <dl className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
    <div className="min-w-0"><dt className="text-xs text-slate-500">Source</dt><dd className="mt-1 truncate" title={source.label}>
      {Option.match(source.href, {
        onNone: () => source.label,
        onSome: href => <a className="inline-flex max-w-full items-center gap-1 text-slate-600 hover:underline dark:text-slate-300" href={href} target="_blank" rel="noreferrer"><span className="truncate">{source.label}</span><ArrowUpRightIcon aria-hidden="true" className="size-3.5 shrink-0" /></a>,
      })}
    </dd></div>
    {Option.isSome(variantOfName) && <div className="min-w-0"><dt className="text-xs text-slate-500">Variant of</dt><dd className="mt-1 truncate">{variantOfName.value}</dd></div>}
  </dl>
}

/** A drafter-only download's progress and failure under a custom model's card. */
export function DrafterNotices({ model, disabled, onRetry, onCancel, onDismiss }: {
  model: CustomLocalModel
  disabled: boolean
  onRetry: () => void
  onCancel: () => void
  onDismiss: () => void
}) {
  const availability = Option.map(model.drafter, drafter => drafter.availability)
  return <>
    {Option.isSome(availability) && availability.value._tag === "Downloading" && <div className="col-span-full mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-slate-500">
      <p role="status">Downloading the drafter for {formatLocalModelDisplayName(model)} · {transferPercent(availability.value.progress)}</p>
      <Button variant="ghost" size="unstyled" className={textButton} disabled={disabled} onClick={onCancel}>Cancel</Button>
    </div>}
    {Option.isSome(availability) && availability.value._tag === "DownloadFailed" && <ErrorNotice {...downloadNotice(availability.value.failure)} className="col-span-full mt-3"
      actions={<><NoticeAction disabled={disabled} onClick={onRetry}>Retry drafter download</NoticeAction><NoticeAction disabled={disabled} onClick={onDismiss}>Dismiss</NoticeAction></>} />}
  </>
}
