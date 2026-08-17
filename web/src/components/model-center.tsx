import { useMemo, useState, type ReactNode } from "react"
import { Exit, Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import {
  Activity,
  AlertTriangle,
  Check,
  Cpu,
  Download,
  Gauge,
  Heart,
  Layers3,
  Loader2,
  MemoryStick,
  PackageOpen,
  Play,
  RefreshCw,
  Square,
  Trash2,
  X,
} from "lucide-react"
import {
  deriveHardwareMemoryView,
  formatLocalModelDisplayName,
  installedLocalModels,
  localModelConfigurationId,
  localModelProviderModelId,
  localModelRadarAxes,
  modelDownloadFailureMessage,
  modelSlotResidentAllocation,
  useCatalogModels,
  useLocalInferenceHardware,
  useLocalModelActions,
  useLocalModels,
  useModelSlotActions,
  useModelSlots,
  usePreviewModelLoad,
  type CatalogModelReconciliationState,
  type CatalogModelView,
} from "@magnitudedev/client-common"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  servableModelBundlePackages,
  type LocalModel,
  type ModelSlot,
  type ModelSlotsState,
  type SlotId,
} from "@magnitudedev/sdk"
import type { SettingsTab } from "../state/web-atoms"
import {
  formatBytes,
  formatContext,
  intentLabel,
  modelContextLength,
  slotStatus,
  transferLabel,
  transferProgress,
} from "./local-inference-format"
import { ModelRadarChart } from "./model-radar-chart"
const localProviderId = ProviderIdSchema.make("local")
const recommendationOrder = {
  balanced: 0,
  smartest: 1,
  fastest: 2,
  lightweight: 3,
} as const
const valueOf = <A, E>(result: Result.Result<A, E>): A | null =>
  Option.getOrNull(Result.value(result))
const modelKey = (model: LocalModel): string =>
  Option.getOrElse(localModelConfigurationId(model), () =>
    formatLocalModelDisplayName(model)
  )
const fitsAssessment = (model: LocalModel) =>
  model.servingState._tag === "Assessed" &&
  model.servingState.assessment._tag === "Fits"
    ? model.servingState.assessment
    : null
function QueryNotice({
  result,
  label,
}: {
  readonly result: Result.Result<unknown, unknown>
  readonly label: string
}): ReactNode {
  if (Result.isFailure(result)) {
    return (
      <div className="flex items-center gap-2 rounded-[7px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 px-3 py-2.5 text-slate-600 dark:text-slate-400 text-xs [&.danger]:border-red-300 [&.danger]:text-red-600 dark:[&.danger]:border-red-700 dark:[&.danger]:text-red-400 danger">
        <AlertTriangle size={15} />
        Unable to load {label}.
      </div>
    )
  }
  if (Result.isWaiting(result) && Option.isNone(Result.value(result))) {
    return (
      <div className="flex items-center gap-2 rounded-[7px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 px-3 py-2.5 text-slate-600 dark:text-slate-400 text-xs [&.danger]:border-red-300 [&.danger]:text-red-600 dark:[&.danger]:border-red-700 dark:[&.danger]:text-red-400">
        <Loader2 className="animate-spin" size={15} />
        Loading {label}…
      </div>
    )
  }
  return null
}
const modelFailure = (model: LocalModel): string | null => {
  if (model.acquisitionState._tag === "Failed")
    return modelDownloadFailureMessage(model.acquisitionState.failure)
  if (model.upgradeState._tag === "Failed")
    return modelDownloadFailureMessage(model.upgradeState.failure)
  if (model.servingState._tag === "Failed")
    return model.servingState.failure.message
  if (model.servingState._tag === "Assessed") {
    if (model.servingState.assessment._tag === "DoesNotFit") {
      return `Needs ${formatBytes(
        model.servingState.assessment.deficitBytes
      )} more ${model.servingState.assessment.limitingResource}.`
    }
    if (model.servingState.assessment._tag === "Incompatible")
      return model.servingState.assessment.failure.message
    if (model.servingState.availabilityState._tag === "Unavailable")
      return model.servingState.availabilityState.failure.message
  }
  return null
}
const modelTransfer = (model: LocalModel) =>
  model.acquisitionState._tag === "Downloading"
    ? model.acquisitionState
    : model.upgradeState._tag === "Upgrading"
    ? model.upgradeState
    : null
const modelDownloadId = (model: LocalModel) => {
  const acquisition = model.acquisitionState
  if (
    acquisition._tag === "Downloading" ||
    acquisition._tag === "Failed" ||
    acquisition._tag === "Cancelled"
  ) {
    return acquisition.downloadId
  }
  return model.upgradeState._tag === "Upgrading"
    ? model.upgradeState.downloadId
    : null
}
type StatusTone = "neutral" | "success" | "progress" | "warning" | "danger"
const modelStatus = (
  model: LocalModel,
  reconciliation: CatalogModelReconciliationState | null = null
): { readonly label: string; readonly tone: StatusTone } => {
  if (reconciliation?._tag === "Starting")
    return {
      label: "Starting",
      tone: "progress",
    }
  if (reconciliation?._tag === "Removing")
    return {
      label: "Removing",
      tone: "progress",
    }
  if (reconciliation?._tag === "RemoveFailed")
    return {
      label: "Remove failed",
      tone: "danger",
    }
  if (reconciliation?._tag === "Transferring")
    return {
      label: `${
        reconciliation.operation === "Update" ? "Updating" : "Downloading"
      } ${transferProgress(reconciliation)}%`,
      tone: "progress",
    }
  if (reconciliation?._tag === "Failed")
    return {
      label: `${reconciliation.operation} failed`,
      tone: "danger",
    }
  if (model.servingState._tag === "Resolving")
    return {
      label: "Resolving",
      tone: "progress",
    }
  if (model.servingState._tag === "Assessing")
    return {
      label: "Assessing",
      tone: "progress",
    }
  if (model.servingState._tag === "Failed")
    return {
      label: "Assessment failed",
      tone: "danger",
    }
  if (model.servingState.assessment._tag === "DoesNotFit")
    return {
      label: "Doesn’t fit",
      tone: "danger",
    }
  if (model.servingState.assessment._tag === "Incompatible")
    return {
      label: "Incompatible",
      tone: "danger",
    }
  if (model.servingState.availabilityState._tag === "Unavailable")
    return {
      label: "Unavailable",
      tone: "danger",
    }
  if (model.acquisitionState._tag === "Installed") {
    if (model.upgradeState._tag === "Available")
      return {
        label: "Update available",
        tone: "warning",
      }
    return {
      label: "Installed",
      tone: "success",
    }
  }
  if (model.acquisitionState._tag === "Downloading")
    return {
      label: `Downloading ${transferProgress(model.acquisitionState)}%`,
      tone: "progress",
    }
  if (model.acquisitionState._tag === "Failed")
    return {
      label: "Download failed",
      tone: "danger",
    }
  return {
    label: "Available",
    tone: "neutral",
  }
}
const statusToneClass = (tone: string): string => {
  switch (tone) {
    case "success":
      return "text-green-700 dark:text-green-500"
    case "progress":
      return "text-blue-700 dark:text-blue-500"
    case "warning":
      return "text-orange-700 dark:text-orange-500"
    case "danger":
      return "text-red-600 dark:text-red-500"
    case "neutral":
    default:
      return "text-slate-600 dark:text-slate-400"
  }
}
function RuntimeSlot({
  slot,
  label,
  slotId,
  models,
  slots,
}: {
  readonly slot: ModelSlot
  readonly label: string
  readonly slotId: SlotId
  readonly models: readonly LocalModel[]
  readonly slots: ModelSlotsState
}): ReactNode {
  const actions = useModelSlotActions()
  const [controlling, setControlling] = useState(false)
  const [controlFailed, setControlFailed] = useState(false)
  const status = slotStatus(slot)
  const busy =
    controlling ||
    Result.isWaiting(actions.assignResult) ||
    Result.isWaiting(actions.clearResult) ||
    Result.isWaiting(actions.favoriteResult)
  const failed =
    controlFailed ||
    Result.isFailure(actions.assignResult) ||
    Result.isFailure(actions.clearResult) ||
    Result.isFailure(actions.favoriteResult)
  const isFavorite =
    slot._tag !== "Unassigned" &&
    slots.favoriteModels.some(
      (favorite) =>
        favorite.providerId === slot.selection.providerId &&
        favorite.providerModelId === slot.selection.providerModelId
    )
  const options = models.flatMap((model) => {
    if (
      model.servingState._tag !== "Assessed" ||
      model.servingState.assessment._tag !== "Fits" ||
      model.servingState.availabilityState._tag !== "Selectable"
    )
      return []
    return [
      {
        model,
        providerModelId: model.servingState.availabilityState.providerModelId,
      },
    ]
  })
  const selectedKey =
    slot._tag === "Unassigned" || slot.selection.providerId !== localProviderId
      ? ""
      : slot.selection.providerModelId
  const selectedModel = options.find(
    ({ providerModelId }) => providerModelId === selectedKey
  )?.model
  const allocation = Option.getOrNull(modelSlotResidentAllocation(slot))
  const residentBytes = allocation?.memoryDomains.reduce(
    (total, domain) =>
      total +
      domain.modelBytes +
      domain.contextBytes +
      domain.computeBytes +
      domain.auxiliaryBytes,
    0
  )
  const control = async (kind: "load" | "stop") => {
    setControlling(true)
    setControlFailed(false)
    const exit = await actions[kind](slotId)
    setControlFailed(Exit.isFailure(exit))
    setControlling(false)
  }
  return (
    <article
      className="relative grid min-h-[116px] grid-cols-[minmax(220px,1.1fr)_minmax(240px,.9fr)_auto] items-center gap-[22px] px-[18px] py-5 bg-white dark:bg-slate-850 border-b border-slate-300 dark:border-slate-750 first:rounded-t-[9px] last:rounded-b-[9px] max-[1050px]:grid-cols-[minmax(200px,1fr)_minmax(230px,1fr)] max-[620px]:grid-cols-1 max-[620px]:gap-3.5"
      data-slot={slotId}
    >
      <div className="min-w-0 [&_h3]:text-slate-900 dark:[&_h3]:text-slate-200 [&_h3]:text-base [&_h3]:leading-[1.3]">
        <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
          {label}
        </span>
        <h3>
          {slot._tag === "Unassigned"
            ? "No model configured"
            : selectedModel
            ? formatLocalModelDisplayName(selectedModel)
            : `${slot.descriptor.displayName}${Option.match(
                slot.descriptor.variantLabel,
                {
                  onNone: () => "",
                  onSome: (variant) => ` (${variant})`,
                }
              )}`}
        </h3>
        <div className="mt-[7px] flex flex-wrap gap-x-3 gap-y-[5px] font-sans text-[11px] leading-[normal] text-slate-500">
          <span
            className={`${statusToneClass(
              status.tone
            )} inline-flex items-center gap-1 text-[10px] font-bold whitespace-nowrap`}
          >
            {status.label}
          </span>
          {selectedModel && modelContextLength(selectedModel) !== null && (
            <span>
              {formatContext(modelContextLength(selectedModel)!)} context
            </span>
          )}
          {residentBytes !== undefined && (
            <span>{formatBytes(residentBytes)} resident</span>
          )}
        </div>
        {status.detail && (
          <p className="!text-red-600 dark:!text-red-500">{status.detail}</p>
        )}
      </div>
      <div className="min-w-0 [&_select]:w-full [&_select]:min-h-9 [&_select]:rounded-md [&_select]:border [&_select]:border-slate-300 dark:[&_select]:border-slate-750 [&_select]:bg-slate-50 dark:[&_select]:bg-slate-925 [&_select]:text-slate-900 dark:[&_select]:text-slate-200 [&_select]:px-2.5 [&_select]:font-sans [&_select]:text-xs">
        <label
          className="mt-0 mb-1.5 block text-[11px] font-semibold text-slate-600 dark:text-slate-400"
          htmlFor={`slot-${slotId}`}
        >
          Model
        </label>
        <select
          id={`slot-${slotId}`}
          value={selectedKey}
          disabled={busy}
          onChange={(event) => {
            const selected = options.find(
              ({ providerModelId }) => providerModelId === event.target.value
            )
            if (!selected) return actions.clear(slotId)
            const capabilities =
              selected.model.servingState._tag === "Assessed"
                ? selected.model.servingState.capabilities
                : null
            actions.assign(slotId, {
              providerId: localProviderId,
              providerModelId: selected.providerModelId,
              reasoningEffort: capabilities
                ? Option.getOrElse(capabilities.reasoning.defaultEffort, () =>
                    ReasoningEffortSchema.make("none")
                  )
                : ReasoningEffortSchema.make("none"),
            })
          }}
        >
          <option value="">Choose from installed models</option>
          {options.map(({ model, providerModelId }) => (
            <option key={providerModelId} value={providerModelId}>
              {formatLocalModelDisplayName(model)}
              {modelContextLength(model)
                ? ` · ${formatContext(modelContextLength(model)!)} context`
                : ""}
            </option>
          ))}
        </select>
      </div>
      <div className="flex flex-wrap items-center justify-end gap-[7px] max-[1050px]:col-span-full max-[1050px]:justify-start max-[620px]:col-auto">
        {slot._tag === "ConfiguredLocal" &&
          slot.actions.some(
            (action) => action === "Load" || action === "RetryLoad"
          ) && (
            <button
              type="button"
              className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-blue-700 text-slate-50 hover:bg-blue-800 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400"
              disabled={busy}
              onClick={() => void control("load")}
            >
              <Play size={14} />
              {slot.actions.includes("RetryLoad") ? "Retry load" : "Load"}
            </button>
          )}
        {slot._tag === "ConfiguredLocal" && slot.actions.includes("Stop") && (
          <button
            type="button"
            className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-200 border border-slate-300 dark:border-slate-750 hover:bg-slate-150 dark:hover:bg-slate-750"
            disabled={busy}
            onClick={() => void control("stop")}
          >
            <Square size={13} />
            {slot.residency._tag === "Loading" ||
            slot.residency._tag === "Requested"
              ? "Cancel load"
              : "Stop"}
          </button>
        )}
        {slot._tag === "ConfiguredLocal" && (
          <button
            type="button"
            className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 w-8 !px-0 bg-transparent text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-750 hover:bg-slate-150 hover:text-slate-900 dark:hover:bg-slate-750 dark:hover:text-slate-200"
            title={isFavorite ? "Remove favorite" : "Favorite model"}
            aria-label={isFavorite ? "Remove favorite" : "Favorite model"}
            disabled={busy}
            onClick={() => actions.setFavorite(slot.selection, !isFavorite)}
          >
            <Heart size={15} fill={isFavorite ? "currentColor" : "none"} />
          </button>
        )}
        {slot._tag !== "Unassigned" && (
          <button
            type="button"
            className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-transparent text-slate-600 dark:text-slate-400 !px-1 hover:text-slate-900 dark:hover:text-slate-200"
            disabled={busy}
            onClick={() => actions.clear(slotId)}
          >
            Clear
          </button>
        )}
      </div>
      {failed && (
        <div className="flex items-center gap-2 rounded-[7px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 px-3 py-2.5 text-slate-600 dark:text-slate-400 text-xs [&.danger]:border-red-300 [&.danger]:text-red-600 dark:[&.danger]:border-red-700 dark:[&.danger]:text-red-400 danger col-span-full">
          <AlertTriangle size={14} />
          The model action failed. The configured slot was not changed.
        </div>
      )}
    </article>
  )
}
function TransferQueue({
  models,
}: {
  readonly models: readonly LocalModel[]
}): ReactNode {
  const actions = useLocalModelActions()
  if (models.length === 0) return null
  return (
    <section
      className="flex flex-col gap-3.5"
      aria-labelledby="model-activity-title"
    >
      <div className="flex items-end justify-between gap-5 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
        <div>
          <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
            Activity
          </span>
          <h2 id="model-activity-title">Transfers and failures</h2>
        </div>
        <span className="min-w-6 text-right font-mono text-xs leading-[normal] font-semibold text-slate-500">
          {models.length}
        </span>
      </div>
      <div className="border-t border-slate-300 dark:border-slate-750">
        {models.map((model) => {
          const configurationId = Option.getOrNull(
            localModelConfigurationId(model)
          )
          const transfer = modelTransfer(model)
          const downloadId = modelDownloadId(model)
          return (
            <article
              className="grid min-h-[66px] grid-cols-[auto_minmax(0,1fr)_auto_auto] items-center gap-3 border-b border-slate-200 px-2.5 py-3 text-slate-500 dark:border-slate-800 max-[620px]:grid-cols-[auto_minmax(0,1fr)_auto] max-[620px]:[&>span]:hidden"
              key={modelKey(model)}
            >
              <Activity size={16} aria-hidden="true" />
              <div className="min-w-0 [&>div]:mb-2 [&>div]:flex [&>div]:justify-between [&>div]:gap-3 [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_strong]:text-xs [&_span]:text-slate-500 [&_span]:text-[11px]">
                <div>
                  <strong>{formatLocalModelDisplayName(model)}</strong>
                  <span>
                    {transfer ? transferLabel(transfer) : modelFailure(model)}
                  </span>
                </div>
                {transfer && (
                  <progress
                    max={100}
                    value={transferProgress(transfer)}
                    aria-label={`${formatLocalModelDisplayName(
                      model
                    )} transfer progress`}
                  />
                )}
              </div>
              <span
                className={`${statusToneClass(
                  transfer ? "progress" : "danger"
                )} inline-flex items-center gap-1 text-[10px] font-bold whitespace-nowrap`}
              >
                {transfer ? `${transferProgress(transfer)}%` : "Failed"}
              </span>
              <div className="flex items-center gap-1.5 max-[620px]:col-[2/-1]">
                {transfer && downloadId && (
                  <button
                    type="button"
                    className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-transparent text-slate-600 dark:text-slate-400 !px-1 hover:text-slate-900 dark:hover:text-slate-200"
                    onClick={() => actions.cancel(downloadId)}
                  >
                    Cancel
                  </button>
                )}
                {!transfer && configurationId && (
                  <button
                    type="button"
                    className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-transparent text-slate-600 dark:text-slate-400 !px-1 hover:text-slate-900 dark:hover:text-slate-200"
                    onClick={() => actions.install(configurationId)}
                  >
                    Retry
                  </button>
                )}
                {!transfer && downloadId && (
                  <button
                    type="button"
                    className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-transparent text-slate-600 dark:text-slate-400 !px-1 hover:text-slate-900 dark:hover:text-slate-200"
                    onClick={() => actions.dismissFailure(downloadId)}
                  >
                    Dismiss
                  </button>
                )}
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}
function InstalledLibrary({
  models,
}: {
  readonly models: readonly LocalModel[]
}): ReactNode {
  const actions = useLocalModelActions()
  return (
    <section
      className="flex flex-col gap-3.5"
      aria-labelledby="installed-models-title"
    >
      <div className="flex items-end justify-between gap-5 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
        <div>
          <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
            Local library
          </span>
          <h2 id="installed-models-title">Installed models</h2>
          <p>Model files available on this machine.</p>
        </div>
        <span className="min-w-6 text-right font-mono text-xs leading-[normal] font-semibold text-slate-500">
          {models.length}
        </span>
      </div>
      {models.length === 0 ? (
        <div className="rounded-[10px] border border-dashed border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-[26px] text-center text-[13px] text-slate-500">
          No models are installed yet. Open Catalog to choose one for this
          machine.
        </div>
      ) : (
        <div
          className="border-t border-slate-300 dark:border-slate-750"
          role="list"
        >
          {models.map((model) => {
            const configurationId = Option.getOrNull(
              localModelConfigurationId(model)
            )
            const context = modelContextLength(model)
            const installedBytes =
              model.acquisitionState._tag === "Installed"
                ? model.acquisitionState.installedBytes
                : model.downloadBytes
            return (
              <article
                className="grid min-h-[78px] grid-cols-[auto_minmax(210px,1fr)_minmax(300px,.8fr)_auto] items-center gap-3.5 border-b border-slate-200 dark:border-slate-800 px-2.5 py-[13px] text-slate-500 max-[1050px]:grid-cols-[auto_minmax(180px,1fr)_minmax(250px,.9fr)_auto] max-[840px]:grid-cols-[auto_minmax(0,1fr)_auto]"
                role="listitem"
                key={modelKey(model)}
              >
                <PackageOpen size={17} aria-hidden="true" />
                <div className="flex min-w-0 flex-col gap-[3px] [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_strong]:text-[13px] [&_span]:overflow-hidden [&_span]:text-ellipsis [&_span]:whitespace-nowrap [&_span]:text-slate-500 [&_span]:text-[11px]">
                  <strong>{formatLocalModelDisplayName(model)}</strong>
                  <span>{model.presentation.description}</span>
                </div>
                <dl className="grid grid-cols-3 gap-4 [&_div]:min-w-0 [&_dt]:text-[9px] [&_dt]:tracking-[.06em] [&_dt]:uppercase [&_dt]:text-slate-500 [&_dd]:overflow-hidden [&_dd]:text-ellipsis [&_dd]:whitespace-nowrap [&_dd]:text-[11px] [&_dd]:text-slate-600 dark:[&_dd]:text-slate-400 max-[840px]:col-start-2 max-[620px]:grid-cols-2 max-[620px]:[&>div:last-child]:hidden">
                  <div>
                    <dt>Context</dt>
                    <dd>
                      {context === null ? "Pending" : formatContext(context)}
                    </dd>
                  </div>
                  <div>
                    <dt>Storage</dt>
                    <dd>{formatBytes(installedBytes)}</dd>
                  </div>
                  <div>
                    <dt>License</dt>
                    <dd>
                      {Option.getOrElse(
                        model.presentation.license,
                        () => "Unknown"
                      )}
                    </dd>
                  </div>
                </dl>
                <button
                  type="button"
                  className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 w-8 !px-0 bg-transparent text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-750 hover:bg-slate-150 hover:text-slate-900 dark:hover:bg-slate-750 dark:hover:text-slate-200 !border-transparent !text-slate-500 enabled:hover:!border-red-400 enabled:hover:!text-red-600 dark:enabled:hover:!border-red-700 dark:enabled:hover:!text-red-400"
                  title="Delete model files"
                  aria-label={`Delete ${formatLocalModelDisplayName(
                    model
                  )} files`}
                  disabled={configurationId === null}
                  onClick={() =>
                    configurationId &&
                    window.confirm(
                      `Delete ${formatLocalModelDisplayName(
                        model
                      )} from this computer?`
                    ) &&
                    actions.delete(configurationId)
                  }
                >
                  <Trash2 size={14} />
                </button>
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}
function ModelsView(): ReactNode {
  const modelsResult = useLocalModels()
  const slotsResult = useModelSlots()
  const models = valueOf(modelsResult)
  const slots = valueOf(slotsResult)
  const installed =
    models?.models.filter(
      (model) => model.acquisitionState._tag === "Installed"
    ) ?? []
  const selectableModels = models ? installedLocalModels(models) : []
  const active =
    models?.models.filter(
      (model) =>
        modelTransfer(model) !== null ||
        model.acquisitionState._tag === "Failed" ||
        model.upgradeState._tag === "Failed"
    ) ?? []
  return (
    <div className="box-border mx-auto flex w-full max-w-[1240px] flex-col gap-[34px] px-[clamp(18px,4vw,48px)] pt-[34px] pb-[72px] max-[640px]:pt-16 max-[620px]:px-3.5">
      <QueryNotice result={slotsResult} label="model slots" />
      <QueryNotice result={modelsResult} label="local models" />
      {slots && models && (
        <section
          className="flex flex-col gap-3.5"
          aria-labelledby="runtime-title"
        >
          <div className="flex items-end justify-between gap-5 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
            <div>
              <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                Runtime
              </span>
              <h2 id="runtime-title">Configured models</h2>
              <p>
                Select what each agent role uses and control local residency.
              </p>
            </div>
            <div className="inline-flex items-center gap-1.5 text-slate-600 dark:text-slate-400 text-[11px]">
              <Gauge size={15} />
              {
                [slots.slots.primary, slots.slots.secondary].filter(
                  (slot) =>
                    slot._tag === "ConfiguredLocal" &&
                    slot.residency._tag === "Ready"
                ).length
              }{" "}
              ready
            </div>
          </div>
          <div className="border-t border-slate-300 dark:border-slate-750">
            <RuntimeSlot
              slot={slots.slots.primary}
              label="Primary"
              slotId={PRIMARY_SLOT_ID}
              models={selectableModels}
              slots={slots}
            />
            <RuntimeSlot
              slot={slots.slots.secondary}
              label="Secondary"
              slotId={SECONDARY_SLOT_ID}
              models={selectableModels}
              slots={slots}
            />
          </div>
        </section>
      )}
      <TransferQueue models={active} />
      <InstalledLibrary models={installed} />
    </div>
  )
}
const repositoryUrl = (model: LocalModel): string | null => {
  const repository = servableModelBundlePackages(model.bundle).find(
    ({ source }) => source._tag === "HuggingFace"
  )?.source
  return repository?._tag === "HuggingFace"
    ? `https://huggingface.co/${repository.repository}`
    : null
}
const recommendationRank = (view: CatalogModelView): number => {
  const serving = view.model.servingState
  const intent =
    serving._tag === "Assessed" ? serving.recommendations[0]?.intent : undefined
  return intent === undefined ? 4 : recommendationOrder[intent]
}
function CatalogCandidate({
  view,
  selected,
  onSelect,
}: {
  readonly view: CatalogModelView
  readonly selected: boolean
  readonly onSelect: () => void
}): ReactNode {
  const { model, reconciliationState } = view
  const status = modelStatus(model, reconciliationState)
  const recommendation =
    model.servingState._tag === "Assessed"
      ? model.servingState.recommendations[0] ?? null
      : null
  return (
    <button
      type="button"
      className="appearance-none block min-h-[66px] w-full border-b border-slate-200 bg-transparent px-3.5 py-3 text-left text-slate-600 cursor-pointer hover:bg-white data-[selected=true]:bg-slate-100 dark:border-slate-800 dark:text-slate-400 dark:hover:bg-slate-850 dark:data-[selected=true]:bg-slate-800"
      data-selected={selected}
      aria-pressed={selected}
      onClick={onSelect}
    >
      <span className="flex min-w-0 flex-col gap-1.5 [&>strong]:text-[12.5px] [&>strong]:leading-[1.35] [&>strong]:text-slate-900 dark:[&>strong]:text-slate-200 [&>strong]:[overflow-wrap:anywhere]">
        <strong>{formatLocalModelDisplayName(model)}</strong>
        <span className="flex flex-wrap items-center gap-[5px] text-[10px] text-slate-500">
          {recommendation && <span>{intentLabel(recommendation.intent)}</span>}
          {recommendation && <span aria-hidden="true">·</span>}
          <span
            className={`${statusToneClass(
              status.tone
            )} inline-flex items-center gap-1 text-[10px] font-bold whitespace-nowrap`}
          >
            {status.label}
          </span>
        </span>
      </span>
    </button>
  )
}
function CatalogInspector({
  view,
}: {
  readonly view: CatalogModelView
}): ReactNode {
  const { model, reconciliationState } = view
  const modelActions = useLocalModelActions()
  const slotActions = useModelSlotActions()
  const configurationId = Option.getOrNull(localModelConfigurationId(model))
  const providerModelId = Option.getOrNull(localModelProviderModelId(model))
  const status = modelStatus(model, reconciliationState)
  const assessment = fitsAssessment(model)
  const serving =
    model.servingState._tag === "Assessed" ? model.servingState : null
  const recommendation = serving?.recommendations[0] ?? null
  const axes = Option.getOrNull(localModelRadarAxes(model))
  const transfer =
    reconciliationState._tag === "Transferring"
      ? reconciliationState
      : modelTransfer(model)
  const downloadId =
    reconciliationState._tag === "Transferring"
      ? reconciliationState.downloadId
      : modelDownloadId(model)
  const installed = model.acquisitionState._tag === "Installed"
  const selectable =
    serving?.availabilityState._tag === "Selectable" &&
    assessment !== null &&
    providerModelId !== null
  const starting =
    reconciliationState._tag === "Starting" ||
    reconciliationState._tag === "Removing"
  const source = repositoryUrl(model)
  const actions = (
    <div className="flex flex-wrap items-center justify-end gap-[7px] max-[620px]:justify-start">
      {!installed && !transfer && configurationId && (
        <button
          className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-blue-700 text-slate-50 hover:bg-blue-800 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400"
          type="button"
          disabled={starting}
          onClick={() => modelActions.install(configurationId)}
        >
          {model.acquisitionState._tag === "Failed" ? (
            <RefreshCw size={14} />
          ) : (
            <Download size={14} />
          )}
          {model.acquisitionState._tag === "Failed"
            ? "Retry download"
            : "Download"}
        </button>
      )}
      {installed &&
        (model.upgradeState._tag === "Available" ||
          model.upgradeState._tag === "Failed") &&
        configurationId && (
          <button
            className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-blue-700 text-slate-50 hover:bg-blue-800 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400"
            type="button"
            disabled={starting}
            onClick={() => modelActions.install(configurationId)}
          >
            <RefreshCw size={14} />
            {model.upgradeState._tag === "Failed" ? "Retry update" : "Update"}
          </button>
        )}
      {transfer && downloadId && (
        <button
          className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-200 border border-slate-300 dark:border-slate-750 hover:bg-slate-150 dark:hover:bg-slate-750"
          type="button"
          onClick={() => modelActions.cancel(downloadId)}
        >
          <X size={14} /> Cancel
        </button>
      )}
      {selectable && serving && (
        <button
          className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-blue-700 text-slate-50 hover:bg-blue-800 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400"
          type="button"
          onClick={() =>
            slotActions.assign(PRIMARY_SLOT_ID, {
              providerId: localProviderId,
              providerModelId,
              reasoningEffort: Option.getOrElse(
                serving.capabilities.reasoning.defaultEffort,
                () => ReasoningEffortSchema.make("none")
              ),
            })
          }
        >
          <Check size={14} /> Select as primary
        </button>
      )}
      {installed && configurationId && (
        <button
          className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-transparent text-slate-600 dark:text-slate-400 !px-1 hover:text-slate-900 dark:hover:text-slate-200 text-red-600 dark:text-red-400"
          type="button"
          disabled={starting}
          onClick={() =>
            window.confirm(
              `Delete ${formatLocalModelDisplayName(model)} from this computer?`
            ) && modelActions.delete(configurationId)
          }
        >
          <Trash2 size={14} /> Uninstall
        </button>
      )}
    </div>
  )
  return (
    <article className="grid min-w-0 max-h-full grid-rows-[auto_minmax(0,1fr)] overflow-hidden rounded-[10px] border border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 max-[840px]:max-h-none max-[840px]:overflow-visible">
      <header className="flex items-start justify-between gap-6 border-b border-slate-300 dark:border-slate-750 px-[22px] py-5 max-[620px]:flex-col [&_h2]:mb-[5px] [&_h2]:text-xl [&_h2]:leading-tight [&_h2]:tracking-[-.02em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_h2]:[overflow-wrap:anywhere] [&_p]:text-[11.5px] [&_p]:leading-[1.45] [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
        <div className="min-w-0">
          <h2>{formatLocalModelDisplayName(model)}</h2>
          <p>{model.presentation.description}</p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-[9px] max-[620px]:w-full max-[620px]:items-start">
          <span
            className={`${statusToneClass(
              status.tone
            )} inline-flex items-center gap-1 text-[10px] font-bold whitespace-nowrap`}
          >
            {status.label}
          </span>
          {actions}
        </div>
      </header>

      <div className="min-h-0 overflow-y-auto p-[22px] max-[840px]:overflow-visible">
        {transfer && (
          <div className="mb-[18px] [&>div]:mb-1.5 [&>div]:flex [&>div]:justify-between [&>div]:text-[10px] [&>div]:text-slate-600 dark:[&>div]:text-slate-400">
            <div>
              <span>{transferLabel(transfer)}</span>
              <strong>{transferProgress(transfer)}%</strong>
            </div>
            <progress
              max={100}
              value={transferProgress(transfer)}
              aria-label={`${formatLocalModelDisplayName(
                model
              )} transfer progress`}
            />
          </div>
        )}
        {modelFailure(model) && (
          <p className="!text-red-600 dark:!text-red-500">
            {modelFailure(model)}
          </p>
        )}

        {recommendation && (
          <section
            className="border-b border-slate-200 dark:border-slate-800 pb-5 [&_h3]:mb-[5px] [&_h3]:text-[14px] [&_h3]:text-slate-900 dark:[&_h3]:text-slate-200 [&_p]:text-[11.5px] [&_p]:leading-[1.55] [&_p]:text-slate-600 dark:[&_p]:text-slate-400"
            aria-labelledby="why-model-title"
          >
            <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
              Why this model
            </span>
            <h3 id="why-model-title">{intentLabel(recommendation.intent)}</h3>
            <p>{recommendation.explanation}</p>
          </section>
        )}

        {axes ? (
          <section className="py-[22px]" aria-labelledby="model-profile-title">
            <div className="flex items-end justify-between gap-3 [&_h3]:text-[15px] [&_h3]:text-slate-900 dark:[&_h3]:text-slate-200 [&>span]:text-[9px] [&>span]:text-slate-500">
              <h3 id="model-profile-title">Model profile</h3>
            </div>
            <ModelRadarChart axes={axes} />
          </section>
        ) : (
          <div className="my-[18px] flex items-center gap-2 text-[11px] text-slate-500">
            <AlertTriangle size={15} />A complete comparison profile is not
            available for this configuration.
          </div>
        )}

        <dl className="flex flex-wrap gap-x-10 gap-y-[18px] border-t border-slate-300 dark:border-slate-750 pt-[18px] [&_div]:min-w-[140px] [&_dt]:text-[9px] [&_dt]:tracking-[.06em] [&_dt]:uppercase [&_dt]:text-slate-500 [&_dd]:mt-1 [&_dd]:text-[11px] [&_dd]:text-slate-600 dark:[&_dd]:text-slate-400 [&_a]:text-blue-700 dark:[&_a]:text-blue-400 [&_a]:no-underline hover:[&_a]:underline">
          <div>
            <dt>License</dt>
            <dd>
              {Option.getOrElse(model.presentation.license, () => "Unknown")}
            </dd>
          </div>
          {source && (
            <div>
              <dt>Source</dt>
              <dd>
                <a href={source} target="_blank" rel="noreferrer">
                  View on Hugging Face ↗
                </a>
              </dd>
            </div>
          )}
        </dl>
      </div>
    </article>
  )
}
function CatalogView(): ReactNode {
  const catalogResult = useCatalogModels()
  const modelActions = useLocalModelActions()
  const catalog = valueOf(catalogResult)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const candidates = useMemo(
    () =>
      [...(catalog?.models ?? [])].sort(
        (left, right) =>
          recommendationRank(left) - recommendationRank(right) ||
          formatLocalModelDisplayName(left.model).localeCompare(
            formatLocalModelDisplayName(right.model)
          )
      ),
    [catalog]
  )
  const selected =
    candidates.find(({ model }) => modelKey(model) === selectedKey) ??
    candidates[0] ??
    null
  const discovery = catalog?.discoveryState ?? null
  return (
    <div className="box-border mx-auto flex w-full max-w-[1240px] flex-col gap-[34px] px-[clamp(18px,4vw,48px)] pt-[34px] pb-[72px] max-[640px]:pt-16 max-[620px]:px-3.5 !max-w-[1360px]">
      <QueryNotice result={catalogResult} label="local catalog" />
      {modelActions.latestInstallationFailed && (
        <div className="flex items-center gap-2 rounded-[7px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 px-3 py-2.5 text-slate-600 dark:text-slate-400 text-xs [&.danger]:border-red-300 [&.danger]:text-red-600 dark:[&.danger]:border-red-700 dark:[&.danger]:text-red-400 danger">
          <AlertTriangle size={14} />
          The latest model installation or update request failed.
        </div>
      )}
      {discovery?._tag === "Loading" && (
        <section className="flex items-center gap-3.5 rounded-lg border border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 px-[18px] py-4 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
          <Loader2 className="animate-spin" size={20} />
          <div>
            <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
              Local assessment
            </span>
            <h2>Preparing models for this machine</h2>
            <p>Hardware detection and native assessment are running locally.</p>
          </div>
        </section>
      )}
      {discovery?._tag === "Failed" && (
        <div className="flex items-center gap-2 rounded-[7px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 px-3 py-2.5 text-slate-600 dark:text-slate-400 text-xs [&.danger]:border-red-300 [&.danger]:text-red-600 dark:[&.danger]:border-red-700 dark:[&.danger]:text-red-400 danger">
          <AlertTriangle size={15} /> {discovery.failure.message}
        </div>
      )}
      <div className="flex items-end justify-between gap-5 -mb-3.5 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
        <div>
          <h2>Catalog</h2>
          <p>Models assessed for this computer.</p>
        </div>
        <span className="min-w-6 text-right font-mono text-xs leading-[normal] font-semibold text-slate-500">
          {candidates.length}
        </span>
      </div>
      {candidates.length === 0 && discovery?._tag === "Ready" ? (
        <div className="rounded-[10px] border border-dashed border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-[26px] text-center text-[13px] text-slate-500">
          No local catalog models are currently available.
        </div>
      ) : (
        <div className="grid h-[calc(100vh-250px)] min-h-[500px] grid-cols-[minmax(330px,.72fr)_minmax(480px,1.28fr)] items-stretch gap-[22px] max-[1050px]:grid-cols-[minmax(280px,.8fr)_minmax(400px,1.2fr)] max-[1050px]:gap-3.5 max-[840px]:h-auto max-[840px]:min-h-0 max-[840px]:grid-cols-1">
          <div
            className="min-w-0 max-h-full overflow-y-auto border-t border-slate-300 dark:border-slate-750 max-[840px]:max-h-[360px] max-[840px]:overflow-auto"
            role="group"
            aria-label="Catalog models"
          >
            {candidates.map((candidate) => {
              const key = modelKey(candidate.model)
              return (
                <CatalogCandidate
                  key={key}
                  view={candidate}
                  selected={
                    selected !== null && modelKey(selected.model) === key
                  }
                  onSelect={() => setSelectedKey(key)}
                />
              )
            })}
          </div>
          {selected && <CatalogInspector view={selected} />}
        </div>
      )}
    </div>
  )
}
const percentage = (part: number, total: number): number =>
  total <= 0 ? 0 : Math.min(100, Math.max(0, (part / total) * 100))
function HardwareView(): ReactNode {
  const hardwareResult = useLocalInferenceHardware()
  const slotsResult = useModelSlots()
  const preview = usePreviewModelLoad(PRIMARY_SLOT_ID)
  const hardware = valueOf(hardwareResult)
  const slots = valueOf(slotsResult)
  const primary = slots?.slots.primary ?? null
  const allocation = primary
    ? modelSlotResidentAllocation(primary)
    : Option.none()
  const memory = hardware
    ? deriveHardwareMemoryView(hardware, allocation)
    : null
  return (
    <div className="box-border mx-auto flex w-full max-w-[1240px] flex-col gap-[34px] px-[clamp(18px,4vw,48px)] pt-[34px] pb-[72px] max-[640px]:pt-16 max-[620px]:px-3.5">
      <QueryNotice result={hardwareResult} label="hardware" />
      <QueryNotice result={slotsResult} label="model runtime" />
      {hardware && (
        <>
          <section className="flex items-center gap-4 border-b border-slate-300 dark:border-slate-750 pb-[26px] max-[620px]:flex-wrap max-[620px]:items-start [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-[12px] [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
            <Cpu
              className="shrink-0 text-blue-700 dark:text-blue-500"
              size={24}
              aria-hidden="true"
            />
            <div className="min-w-0">
              <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                This machine
              </span>
              <h2>
                {Option.getOrElse(hardware.productName, () =>
                  Option.getOrElse(hardware.processor, () => "Local system")
                )}
              </h2>
              <p>
                {hardware.platform} · {hardware.architecture} ·{" "}
                {hardware.logicalCores} logical cores
              </p>
            </div>
            <div className="ml-auto text-right max-[620px]:ml-[62px] max-[620px]:w-full max-[620px]:text-left [&_strong]:block [&_strong]:font-mono [&_strong]:text-[22px] [&_strong]:leading-[normal] [&_strong]:font-semibold [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_span]:text-[10px] [&_span]:text-slate-500">
              <strong>{formatBytes(hardware.totalSystemMemoryBytes)}</strong>
              <span>system memory</span>
            </div>
          </section>

          <section
            className="flex flex-col gap-3.5"
            aria-labelledby="footprint-title"
          >
            <div className="flex items-end justify-between gap-5 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
              <div>
                <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                  Runtime footprint
                </span>
                <h2 id="footprint-title">Primary model</h2>
              </div>
            </div>
            <article className="grid min-h-[68px] grid-cols-[auto_minmax(180px,1fr)_minmax(300px,auto)] items-center gap-3.5 border-t border-slate-300 border-b border-b-slate-200 px-2.5 py-3.5 dark:border-t-slate-750 dark:border-b-slate-800 [&>div]:flex [&>div]:flex-col [&_strong]:text-[13px] [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_span]:text-[10px] [&_span]:text-slate-500 max-[620px]:grid-cols-[auto_minmax(0,1fr)]">
              <Layers3 size={18} aria-hidden="true" />
              <div>
                <strong>
                  {primary?._tag === "Unassigned" || primary === null
                    ? "No local model selected"
                    : primary.descriptor.displayName}
                </strong>
                <span>
                  {primary
                    ? slotStatus(primary).detail ?? slotStatus(primary).label
                    : "Not configured"}
                </span>
              </div>
              {Result.isSuccess(preview) &&
                primary?._tag === "ConfiguredLocal" &&
                primary.residency._tag === "Unloaded" && (
                  <dl className="grid grid-cols-3 gap-4 [&_div]:min-w-0 [&_dt]:text-[9px] [&_dt]:tracking-[.06em] [&_dt]:uppercase [&_dt]:text-slate-500 [&_dd]:overflow-hidden [&_dd]:text-ellipsis [&_dd]:whitespace-nowrap [&_dd]:text-[11px] [&_dd]:text-slate-600 dark:[&_dd]:text-slate-400 max-[620px]:col-start-2">
                    <div>
                      <dt>Required memory</dt>
                      <dd>
                        {formatBytes(preview.value.requiredSystemMemoryBytes)}
                      </dd>
                    </div>
                    <div>
                      <dt>Context</dt>
                      <dd>
                        {formatContext(preview.value.contextWindowTokens)}
                      </dd>
                    </div>
                    <div>
                      <dt>Parallel</dt>
                      <dd>{preview.value.parallelSequences}</dd>
                    </div>
                  </dl>
                )}
            </article>
          </section>

          <section
            className="flex flex-col gap-3.5"
            aria-labelledby="domains-title"
          >
            <div className="flex items-end justify-between gap-5 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
              <div>
                <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                  Physical topology
                </span>
                <h2 id="domains-title">Memory domains</h2>
                <p>Each allocation is charged once to its physical owner.</p>
              </div>
              <span className="min-w-6 text-right font-mono text-xs leading-[normal] font-semibold text-slate-500">
                {memory?.domains.length ?? 0}
              </span>
            </div>
            <div className="border-t border-slate-300 dark:border-slate-750">
              {memory?.domains.map((domain) => (
                <article
                  className="grid grid-cols-[auto_minmax(0,1fr)] gap-3 border-b border-slate-200 dark:border-slate-800 px-2.5 py-4 text-slate-500"
                  key={domain.id}
                >
                  <MemoryStick size={17} aria-hidden="true" />
                  <div className="min-w-0 [&_p]:mt-2 [&_p]:text-[10px] [&_p]:text-slate-500">
                    <div className="flex justify-between gap-3.5 [&_strong]:text-[12px] [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_span]:font-mono [&_span]:text-[10px] [&_span]:text-slate-600 dark:[&_span]:text-slate-400">
                      <strong>{domain.label}</strong>
                      <span>
                        {domain.usedBytes === null
                          ? `${formatBytes(domain.totalBytes)} total`
                          : `${formatBytes(domain.usedBytes)} of ${formatBytes(
                              domain.totalBytes
                            )} used`}
                      </span>
                    </div>
                    {domain.freeBytes === null || domain.usedBytes === null ? (
                      <p>{domain.notice}</p>
                    ) : (
                      <>
                        <div
                          className="mt-2.5 flex h-2 overflow-hidden rounded-sm bg-slate-100 dark:bg-slate-800"
                          role="progressbar"
                          aria-label={`${
                            domain.label
                          } memory use: ${formatBytes(
                            domain.fixedBytes ?? 0
                          )} model weights, ${formatBytes(
                            domain.kvCacheBytes ?? 0
                          )} context cache, ${formatBytes(
                            domain.systemAndAppsBytes ?? 0
                          )} system and apps, ${formatBytes(
                            domain.freeBytes
                          )} free`}
                          aria-valuemin={0}
                          aria-valuemax={domain.totalBytes}
                          aria-valuenow={domain.usedBytes}
                        >
                          <span
                            className="bg-blue-700 dark:bg-blue-500"
                            style={{
                              width: `${percentage(
                                domain.fixedBytes ?? 0,
                                domain.totalBytes
                              )}%`,
                            }}
                          />
                          <span
                            className="bg-violet-700 dark:bg-violet-500"
                            style={{
                              width: `${percentage(
                                domain.kvCacheBytes ?? 0,
                                domain.totalBytes
                              )}%`,
                            }}
                          />
                          <span
                            className="bg-slate-500 opacity-55"
                            style={{
                              width: `${percentage(
                                domain.systemAndAppsBytes ?? 0,
                                domain.totalBytes
                              )}%`,
                            }}
                          />
                        </div>
                        <dl className="mt-2.5 grid grid-cols-4 gap-x-[18px] gap-y-2.5 max-[620px]:grid-cols-2 [&_div]:relative [&_div]:min-w-0 [&_div]:pl-[11px] [&_div]:before:absolute [&_div]:before:left-0 [&_div]:before:top-1 [&_div]:before:h-3 [&_div]:before:w-[5px] [&_div]:before:rounded-px [&_dt]:text-[9px] [&_dt]:leading-tight [&_dt]:text-slate-500 [&_dd]:mt-0.5 [&_dd]:font-mono [&_dd]:text-[10px] [&_dd]:leading-[normal] [&_dd]:text-slate-600 dark:[&_dd]:text-slate-400">
                          <div className="before:bg-blue-700 dark:before:bg-blue-500">
                            <dt>Model weights</dt>
                            <dd>{formatBytes(domain.fixedBytes ?? 0)}</dd>
                          </div>
                          <div className="before:bg-violet-700 dark:before:bg-violet-500">
                            <dt>Context cache</dt>
                            <dd>{formatBytes(domain.kvCacheBytes ?? 0)}</dd>
                          </div>
                          <div className="before:bg-slate-500 before:opacity-55">
                            <dt>System &amp; apps</dt>
                            <dd>
                              {formatBytes(domain.systemAndAppsBytes ?? 0)}
                            </dd>
                          </div>
                          <div className="before:border before:border-slate-300 before:bg-slate-100 dark:before:border-slate-750 dark:before:bg-slate-800">
                            <dt>Free</dt>
                            <dd>{formatBytes(domain.freeBytes)}</dd>
                          </div>
                        </dl>
                      </>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section
            className="flex flex-col gap-3.5"
            aria-labelledby="accelerators-title"
          >
            <div className="flex items-end justify-between gap-5 [&_h2]:mb-1 [&_h2]:text-[19px] [&_h2]:leading-tight [&_h2]:tracking-[-.015em] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
              <div>
                <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                  Compute
                </span>
                <h2 id="accelerators-title">Accelerators</h2>
              </div>
              <span className="min-w-6 text-right font-mono text-xs leading-[normal] font-semibold text-slate-500">
                {hardware.accelerators.length}
              </span>
            </div>
            {hardware.accelerators.length === 0 ? (
              <div className="rounded-[10px] border border-dashed border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-[26px] text-center text-[13px] text-slate-500">
                CPU inference · no accelerator reported
              </div>
            ) : (
              <div className="border-t border-slate-300 dark:border-slate-750">
                {hardware.accelerators.map((accelerator) => (
                  <article
                    className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 border-b border-slate-200 dark:border-slate-800 px-2.5 py-[13px] text-slate-500 [&>div]:flex [&>div]:flex-col [&_strong]:text-[12px] [&_strong]:text-slate-900 dark:[&_strong]:text-slate-200 [&_span]:text-[10px] [&_span]:text-slate-500"
                    key={accelerator.acceleratorId}
                  >
                    <Cpu size={17} aria-hidden="true" />
                    <div>
                      <strong>{accelerator.name}</strong>
                      <span>{accelerator.backend}</span>
                    </div>
                    <span>Local inference</span>
                  </article>
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  )
}
export function SettingsCenter({
  tab,
}: {
  readonly tab: SettingsTab
}): ReactNode {
  return (
    <div className="min-w-0 min-h-0 flex-1 overflow-auto">
      {tab === "models" ? (
        <ModelsView />
      ) : tab === "catalog" ? (
        <CatalogView />
      ) : (
        <HardwareView />
      )}
    </div>
  )
}
