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
      <div className="model-notice danger">
        <AlertTriangle size={15} />
        Unable to load {label}.
      </div>
    )
  }
  if (Result.isWaiting(result) && Option.isNone(Result.value(result))) {
    return (
      <div className="model-notice">
        <Loader2 className="spin" size={15} />
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

const modelStatus = (
  model: LocalModel,
  reconciliation: CatalogModelReconciliationState | null = null
) => {
  if (reconciliation?._tag === "Starting")
    return { label: "Starting", tone: "progress" }
  if (reconciliation?._tag === "Removing")
    return { label: "Removing", tone: "progress" }
  if (reconciliation?._tag === "RemoveFailed")
    return { label: "Remove failed", tone: "danger" }
  if (reconciliation?._tag === "Transferring")
    return {
      label: `${
        reconciliation.operation === "Update" ? "Updating" : "Downloading"
      } ${transferProgress(reconciliation)}%`,
      tone: "progress",
    }
  if (reconciliation?._tag === "Failed")
    return { label: `${reconciliation.operation} failed`, tone: "danger" }
  if (model.servingState._tag === "Resolving")
    return { label: "Resolving", tone: "progress" }
  if (model.servingState._tag === "Assessing")
    return { label: "Assessing", tone: "progress" }
  if (model.servingState._tag === "Failed")
    return { label: "Assessment failed", tone: "danger" }
  if (model.servingState.assessment._tag === "DoesNotFit")
    return { label: "Doesn’t fit", tone: "danger" }
  if (model.servingState.assessment._tag === "Incompatible")
    return { label: "Incompatible", tone: "danger" }
  if (model.servingState.availabilityState._tag === "Unavailable")
    return { label: "Unavailable", tone: "danger" }
  if (model.acquisitionState._tag === "Installed") {
    if (model.upgradeState._tag === "Available")
      return { label: "Update available", tone: "warning" }
    return { label: "Installed", tone: "success" }
  }
  if (model.acquisitionState._tag === "Downloading")
    return {
      label: `Downloading ${transferProgress(model.acquisitionState)}%`,
      tone: "progress",
    }
  if (model.acquisitionState._tag === "Failed")
    return { label: "Download failed", tone: "danger" }
  return { label: "Available", tone: "neutral" }
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
    <article className="mc-runtime-slot" data-slot={slotId}>
      <div className="mc-runtime-identity">
        <span className="eyebrow">{label}</span>
        <h3>
          {slot._tag === "Unassigned"
            ? "No model configured"
            : selectedModel
              ? formatLocalModelDisplayName(selectedModel)
              : `${slot.descriptor.displayName}${Option.match(
                  slot.descriptor.variantLabel,
                  { onNone: () => "", onSome: (variant) => ` (${variant})` }
                )}`}
        </h3>
        <div className="mc-runtime-facts">
          <span className={`mc-state-label ${status.tone}`}>{status.label}</span>
          {selectedModel && modelContextLength(selectedModel) !== null && (
            <span>{formatContext(modelContextLength(selectedModel)!)} context</span>
          )}
          {residentBytes !== undefined && (
            <span>{formatBytes(residentBytes)} resident</span>
          )}
        </div>
        {status.detail && <p className="model-error-text">{status.detail}</p>}
      </div>
      <div className="mc-runtime-control">
        <label className="field-label" htmlFor={`slot-${slotId}`}>
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
      <div className="mc-runtime-actions">
        {slot._tag === "ConfiguredLocal" &&
          slot.actions.some(
            (action) => action === "Load" || action === "RetryLoad"
          ) && (
            <button
              type="button"
              className="primary-button"
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
            className="secondary-button"
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
            className="icon-button"
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
            className="text-button"
            disabled={busy}
            onClick={() => actions.clear(slotId)}
          >
            Clear
          </button>
        )}
      </div>
      {failed && (
        <div className="model-notice danger mc-runtime-error">
          <AlertTriangle size={14} />
          The model action failed. The configured slot was not changed.
        </div>
      )}
    </article>
  )
}

function TransferQueue({ models }: { readonly models: readonly LocalModel[] }): ReactNode {
  const actions = useLocalModelActions()
  if (models.length === 0) return null
  return (
    <section className="mc-section" aria-labelledby="model-activity-title">
      <div className="mc-section-heading">
        <div>
          <span className="eyebrow">Activity</span>
          <h2 id="model-activity-title">Transfers and failures</h2>
        </div>
        <span className="mc-section-count">{models.length}</span>
      </div>
      <div className="mc-activity-list">
        {models.map((model) => {
          const configurationId = Option.getOrNull(localModelConfigurationId(model))
          const transfer = modelTransfer(model)
          const downloadId = modelDownloadId(model)
          return (
            <article className="mc-activity-row" key={modelKey(model)}>
              <Activity size={16} aria-hidden="true" />
              <div className="mc-activity-main">
                <div>
                  <strong>{formatLocalModelDisplayName(model)}</strong>
                  <span>{transfer ? transferLabel(transfer) : modelFailure(model)}</span>
                </div>
                {transfer && (
                  <progress
                    max={100}
                    value={transferProgress(transfer)}
                    aria-label={`${formatLocalModelDisplayName(model)} transfer progress`}
                  />
                )}
              </div>
              <span className={`mc-state-label ${transfer ? "progress" : "danger"}`}>
                {transfer ? `${transferProgress(transfer)}%` : "Failed"}
              </span>
              <div className="mc-row-actions">
                {transfer && downloadId && (
                  <button type="button" className="text-button" onClick={() => actions.cancel(downloadId)}>
                    Cancel
                  </button>
                )}
                {!transfer && configurationId && (
                  <button type="button" className="text-button" onClick={() => actions.install(configurationId)}>
                    Retry
                  </button>
                )}
                {!transfer && downloadId && (
                  <button type="button" className="text-button" onClick={() => actions.dismissFailure(downloadId)}>
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

function InstalledLibrary({ models }: { readonly models: readonly LocalModel[] }): ReactNode {
  const actions = useLocalModelActions()
  return (
    <section className="mc-section" aria-labelledby="installed-models-title">
      <div className="mc-section-heading">
        <div>
          <span className="eyebrow">Local library</span>
          <h2 id="installed-models-title">Installed models</h2>
          <p>Model files available on this machine.</p>
        </div>
        <span className="mc-section-count">{models.length}</span>
      </div>
      {models.length === 0 ? (
        <div className="empty-panel">
          No models are installed yet. Open Catalog to choose one for this machine.
        </div>
      ) : (
        <div className="mc-library" role="list">
          {models.map((model) => {
            const configurationId = Option.getOrNull(localModelConfigurationId(model))
            const context = modelContextLength(model)
            const installedBytes =
              model.acquisitionState._tag === "Installed"
                ? model.acquisitionState.installedBytes
                : model.downloadBytes
            return (
              <article className="mc-library-row" role="listitem" key={modelKey(model)}>
                <PackageOpen size={17} aria-hidden="true" />
                <div className="mc-library-identity">
                  <strong>{formatLocalModelDisplayName(model)}</strong>
                  <span>{model.presentation.description}</span>
                </div>
                <dl className="mc-library-metrics">
                  <div>
                    <dt>Context</dt>
                    <dd>{context === null ? "Pending" : formatContext(context)}</dd>
                  </div>
                  <div>
                    <dt>Storage</dt>
                    <dd>{formatBytes(installedBytes)}</dd>
                  </div>
                  <div>
                    <dt>License</dt>
                    <dd>{Option.getOrElse(model.presentation.license, () => "Unknown")}</dd>
                  </div>
                </dl>
                <button
                  type="button"
                  className="icon-button mc-delete-button"
                  title="Delete model files"
                  aria-label={`Delete ${formatLocalModelDisplayName(model)} files`}
                  disabled={configurationId === null}
                  onClick={() =>
                    configurationId &&
                    window.confirm(
                      `Delete ${formatLocalModelDisplayName(model)} from this computer?`
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
    models?.models.filter((model) => model.acquisitionState._tag === "Installed") ?? []
  const selectableModels = models ? installedLocalModels(models) : []
  const active =
    models?.models.filter(
      (model) =>
        modelTransfer(model) !== null ||
        model.acquisitionState._tag === "Failed" ||
        model.upgradeState._tag === "Failed"
    ) ?? []
  return (
    <div className="model-center-view">
      <QueryNotice result={slotsResult} label="model slots" />
      <QueryNotice result={modelsResult} label="local models" />
      {slots && models && (
        <section className="mc-section" aria-labelledby="runtime-title">
          <div className="mc-section-heading mc-runtime-heading">
            <div>
              <span className="eyebrow">Runtime</span>
              <h2 id="runtime-title">Configured models</h2>
              <p>Select what each agent role uses and control local residency.</p>
            </div>
            <div className="mc-runtime-summary">
              <Gauge size={15} />
              {
                [slots.slots.primary, slots.slots.secondary].filter(
                  (slot) =>
                    slot._tag === "ConfiguredLocal" && slot.residency._tag === "Ready"
                ).length
              }{" "}
              ready
            </div>
          </div>
          <div className="mc-runtime-stack">
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
      className="mc-candidate"
      data-selected={selected}
      aria-pressed={selected}
      onClick={onSelect}
    >
      <span className="mc-candidate-copy">
        <strong>{formatLocalModelDisplayName(model)}</strong>
        <span className="mc-candidate-status">
          {recommendation && <span>{intentLabel(recommendation.intent)}</span>}
          {recommendation && <span aria-hidden="true">·</span>}
          <span className={`mc-state-label ${status.tone}`}>{status.label}</span>
        </span>
      </span>
    </button>
  )
}

function CatalogInspector({ view }: { readonly view: CatalogModelView }): ReactNode {
  const { model, reconciliationState } = view
  const modelActions = useLocalModelActions()
  const slotActions = useModelSlotActions()
  const configurationId = Option.getOrNull(localModelConfigurationId(model))
  const providerModelId = Option.getOrNull(localModelProviderModelId(model))
  const status = modelStatus(model, reconciliationState)
  const assessment = fitsAssessment(model)
  const serving = model.servingState._tag === "Assessed" ? model.servingState : null
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
    <div className="mc-inspector-actions">
      {!installed && !transfer && configurationId && (
        <button
          className="primary-button"
          type="button"
          disabled={starting}
          onClick={() => modelActions.install(configurationId)}
        >
          {model.acquisitionState._tag === "Failed" ? (
            <RefreshCw size={14} />
          ) : (
            <Download size={14} />
          )}
          {model.acquisitionState._tag === "Failed" ? "Retry download" : "Download"}
        </button>
      )}
      {installed &&
        (model.upgradeState._tag === "Available" ||
          model.upgradeState._tag === "Failed") &&
        configurationId && (
          <button
            className="primary-button"
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
          className="secondary-button"
          type="button"
          onClick={() => modelActions.cancel(downloadId)}
        >
          <X size={14} /> Cancel
        </button>
      )}
      {selectable && serving && (
        <button
          className="primary-button"
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
          className="text-button mc-destructive-text"
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
    <article className="mc-inspector">
      <header className="mc-inspector-header">
        <div className="mc-inspector-title">
          <h2>{formatLocalModelDisplayName(model)}</h2>
          <p>{model.presentation.description}</p>
        </div>
        <div className="mc-inspector-command">
          <span className={`mc-state-label ${status.tone}`}>{status.label}</span>
          {actions}
        </div>
      </header>

      <div className="mc-inspector-body">
        {transfer && (
          <div className="mc-inspector-transfer">
            <div>
              <span>{transferLabel(transfer)}</span>
              <strong>{transferProgress(transfer)}%</strong>
            </div>
            <progress
              max={100}
              value={transferProgress(transfer)}
              aria-label={`${formatLocalModelDisplayName(model)} transfer progress`}
            />
          </div>
        )}
        {modelFailure(model) && <p className="model-error-text">{modelFailure(model)}</p>}

        {recommendation && (
          <section className="mc-why" aria-labelledby="why-model-title">
            <span className="eyebrow">Why this model</span>
            <h3 id="why-model-title">{intentLabel(recommendation.intent)}</h3>
            <p>{recommendation.explanation}</p>
          </section>
        )}

        {axes ? (
          <section className="mc-radar-section" aria-labelledby="model-profile-title">
            <div className="mc-subsection-heading">
              <h3 id="model-profile-title">Model profile</h3>
            </div>
            <ModelRadarChart axes={axes} />
          </section>
        ) : (
          <div className="mc-evidence-unavailable">
            <AlertTriangle size={15} />
            A complete comparison profile is not available for this configuration.
          </div>
        )}

        <dl className="mc-source-details">
          <div>
            <dt>License</dt>
            <dd>{Option.getOrElse(model.presentation.license, () => "Unknown")}</dd>
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
    candidates.find(({ model }) => modelKey(model) === selectedKey) ?? candidates[0] ?? null
  const discovery = catalog?.discoveryState ?? null
  return (
    <div className="model-center-view mc-catalog-view">
      <QueryNotice result={catalogResult} label="local catalog" />
      {modelActions.latestInstallationFailed && (
        <div className="model-notice danger">
          <AlertTriangle size={14} />
          The latest model installation or update request failed.
        </div>
      )}
      {discovery?._tag === "Loading" && (
        <section className="mc-preparation-panel">
          <Loader2 className="spin" size={20} />
          <div>
            <span className="eyebrow">Local assessment</span>
            <h2>Preparing models for this machine</h2>
            <p>Hardware detection and native assessment are running locally.</p>
          </div>
        </section>
      )}
      {discovery?._tag === "Failed" && (
        <div className="model-notice danger">
          <AlertTriangle size={15} /> {discovery.failure.message}
        </div>
      )}
      <div className="mc-catalog-heading">
        <div>
          <h2>Catalog</h2>
          <p>Models assessed for this computer.</p>
        </div>
        <span className="mc-section-count">{candidates.length}</span>
      </div>
      {candidates.length === 0 && discovery?._tag === "Ready" ? (
        <div className="empty-panel">No local catalog models are currently available.</div>
      ) : (
        <div className="mc-catalog-workspace">
          <div className="mc-candidate-list" role="group" aria-label="Catalog models">
            {candidates.map((candidate) => {
              const key = modelKey(candidate.model)
              return (
                <CatalogCandidate
                  key={key}
                  view={candidate}
                  selected={selected !== null && modelKey(selected.model) === key}
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
  const allocation = primary ? modelSlotResidentAllocation(primary) : Option.none()
  const memory = hardware ? deriveHardwareMemoryView(hardware, allocation) : null
  return (
    <div className="model-center-view">
      <QueryNotice result={hardwareResult} label="hardware" />
      <QueryNotice result={slotsResult} label="model runtime" />
      {hardware && (
        <>
          <section className="mc-hardware-hero">
            <Cpu className="mc-hardware-mark" size={24} aria-hidden="true" />
            <div className="mc-hardware-identity">
              <span className="eyebrow">This machine</span>
              <h2>
                {Option.getOrElse(hardware.productName, () =>
                  Option.getOrElse(hardware.processor, () => "Local system")
                )}
              </h2>
              <p>
                {hardware.platform} · {hardware.architecture} · {hardware.logicalCores} logical cores
              </p>
            </div>
            <div className="mc-hardware-total">
              <strong>{formatBytes(hardware.totalSystemMemoryBytes)}</strong>
              <span>system memory</span>
            </div>
          </section>

          <section className="mc-section" aria-labelledby="footprint-title">
            <div className="mc-section-heading">
              <div>
                <span className="eyebrow">Runtime footprint</span>
                <h2 id="footprint-title">Primary model</h2>
              </div>
            </div>
            <article className="mc-footprint-row">
              <Layers3 size={18} aria-hidden="true" />
              <div>
                <strong>
                  {primary?._tag === "Unassigned" || primary === null
                    ? "No local model selected"
                    : primary.descriptor.displayName}
                </strong>
                <span>
                  {primary ? slotStatus(primary).detail ?? slotStatus(primary).label : "Not configured"}
                </span>
              </div>
              {Result.isSuccess(preview) &&
                primary?._tag === "ConfiguredLocal" &&
                primary.residency._tag === "Unloaded" && (
                  <dl className="mc-footprint-preview">
                    <div><dt>Required memory</dt><dd>{formatBytes(preview.value.requiredSystemMemoryBytes)}</dd></div>
                    <div><dt>Context</dt><dd>{formatContext(preview.value.contextWindowTokens)}</dd></div>
                    <div><dt>Parallel</dt><dd>{preview.value.parallelSequences}</dd></div>
                  </dl>
                )}
            </article>
          </section>

          <section className="mc-section" aria-labelledby="domains-title">
            <div className="mc-section-heading">
              <div>
                <span className="eyebrow">Physical topology</span>
                <h2 id="domains-title">Memory domains</h2>
                <p>Each allocation is charged once to its physical owner.</p>
              </div>
              <span className="mc-section-count">{memory?.domains.length ?? 0}</span>
            </div>
            <div className="mc-domain-list">
              {memory?.domains.map((domain) => (
                <article className="mc-domain-row" key={domain.id}>
                  <MemoryStick size={17} aria-hidden="true" />
                  <div className="mc-domain-main">
                    <div className="mc-domain-heading">
                      <strong>{domain.label}</strong>
                      <span>
                        {domain.usedBytes === null
                          ? `${formatBytes(domain.totalBytes)} total`
                          : `${formatBytes(domain.usedBytes)} of ${formatBytes(domain.totalBytes)} used`}
                      </span>
                    </div>
                    {domain.freeBytes === null || domain.usedBytes === null ? (
                      <p>{domain.notice}</p>
                    ) : (
                      <>
                        <div
                          className="mc-domain-track"
                          role="progressbar"
                          aria-label={`${domain.label} memory use: ${formatBytes(domain.fixedBytes ?? 0)} model weights, ${formatBytes(domain.kvCacheBytes ?? 0)} context cache, ${formatBytes(domain.systemAndAppsBytes ?? 0)} system and apps, ${formatBytes(domain.freeBytes)} free`}
                          aria-valuemin={0}
                          aria-valuemax={domain.totalBytes}
                          aria-valuenow={domain.usedBytes}
                        >
                          <span
                            className="mc-domain-model"
                            style={{ width: `${percentage(domain.fixedBytes ?? 0, domain.totalBytes)}%` }}
                          />
                          <span
                            className="mc-domain-context"
                            style={{ width: `${percentage(domain.kvCacheBytes ?? 0, domain.totalBytes)}%` }}
                          />
                          <span
                            className="mc-domain-system"
                            style={{ width: `${percentage(domain.systemAndAppsBytes ?? 0, domain.totalBytes)}%` }}
                          />
                        </div>
                        <dl className="mc-domain-legend">
                          <div className="model">
                            <dt>Model weights</dt>
                            <dd>{formatBytes(domain.fixedBytes ?? 0)}</dd>
                          </div>
                          <div className="context">
                            <dt>Context cache</dt>
                            <dd>{formatBytes(domain.kvCacheBytes ?? 0)}</dd>
                          </div>
                          <div className="system">
                            <dt>System &amp; apps</dt>
                            <dd>{formatBytes(domain.systemAndAppsBytes ?? 0)}</dd>
                          </div>
                          <div className="free">
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

          <section className="mc-section" aria-labelledby="accelerators-title">
            <div className="mc-section-heading">
              <div>
                <span className="eyebrow">Compute</span>
                <h2 id="accelerators-title">Accelerators</h2>
              </div>
              <span className="mc-section-count">{hardware.accelerators.length}</span>
            </div>
            {hardware.accelerators.length === 0 ? (
              <div className="empty-panel">CPU inference · no accelerator reported</div>
            ) : (
              <div className="mc-accelerator-list">
                {hardware.accelerators.map((accelerator) => (
                  <article className="mc-accelerator-row" key={accelerator.acceleratorId}>
                    <Cpu size={17} aria-hidden="true" />
                    <div><strong>{accelerator.name}</strong><span>{accelerator.backend}</span></div>
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
    <div className="model-center">
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
