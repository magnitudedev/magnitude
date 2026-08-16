import { useMemo, useState, type ReactNode } from "react"
import { Exit, Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import {
  AlertTriangle,
  Check,
  Cpu,
  Download,
  HardDrive,
  Heart,
  Loader2,
  MemoryStick,
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
  type LocalModel,
  type ModelSlot,
  type ModelSlotsState,
  type SlotId,
} from "@magnitudedev/sdk"
import type { ModelCenterTab } from "../state/web-atoms"
import {
  formatBytes,
  formatContext,
  intentLabel,
  modelContextLength,
  slotStatus,
  transferLabel,
  transferProgress,
} from "./local-inference-format"

const localProviderId = ProviderIdSchema.make("local")

const valueOf = <A, E>(result: Result.Result<A, E>): A | null =>
  Option.getOrNull(Result.value(result))

function QueryNotice({
  result,
  label,
}: {
  result: Result.Result<unknown, unknown>
  label: string
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

function SlotCard({
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
  const status = slotStatus(slot)
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
  const control = async (kind: "load" | "stop") => {
    setControlling(true)
    setControlFailed(false)
    const exit = await actions[kind](slotId)
    setControlFailed(Exit.isFailure(exit))
    setControlling(false)
  }

  return (
    <article className="model-card slot-card">
      <div className="model-card-header">
        <div>
          <span className="eyebrow">{label}</span>
          <h3>
            {slot._tag === "Unassigned"
              ? "Choose a local model"
              : slot.descriptor.displayName}
          </h3>
        </div>
        <span className={`status-pill ${status.tone}`}>{status.label}</span>
      </div>
      {status.detail && <p className="model-error-text">{status.detail}</p>}
      {failed && (
        <div className="model-notice danger">
          <AlertTriangle size={14} />
          The model action failed. The authoritative slot state was not changed.
        </div>
      )}
      <label className="field-label" htmlFor={`slot-${slotId}`}>
        Selected model
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
      <div className="model-actions">
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
            className="secondary-button"
            disabled={busy}
            onClick={() => actions.clear(slotId)}
          >
            <X size={14} />
            Clear
          </button>
        )}
      </div>
    </article>
  )
}

function InstalledModelCard({
  model,
}: {
  readonly model: LocalModel
}): ReactNode {
  const actions = useLocalModelActions()
  const configurationId = Option.getOrNull(localModelConfigurationId(model))
  const context = modelContextLength(model)
  const installedBytes =
    model.acquisitionState._tag === "Installed"
      ? model.acquisitionState.installedBytes
      : model.downloadBytes
  return (
    <article className="model-card compact">
      <div className="model-card-header">
        <div>
          <h3>{formatLocalModelDisplayName(model)}</h3>
          <p>
            {context === null
              ? "Configuration pending"
              : `${formatContext(context)} context`}
          </p>
        </div>
        <span className="status-pill success">
          <Check size={12} />
          Installed
        </span>
      </div>
      <p>{model.presentation.description}</p>
      <div className="model-meta-row">
        <span>
          <HardDrive size={13} />
          {formatBytes(installedBytes)}
        </span>
        {model.presentation.license._tag === "Some" && (
          <span>{model.presentation.license.value}</span>
        )}
      </div>
      <div className="model-actions">
        <button
          type="button"
          className="danger-button"
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
          Delete files
        </button>
      </div>
    </article>
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
  const actions = useLocalModelActions()
  return (
    <div className="model-center-view">
      <QueryNotice result={slotsResult} label="model slots" />
      <QueryNotice result={modelsResult} label="local models" />
      {slots && models && (
        <section className="model-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Runtime</span>
              <h2>Selected models</h2>
            </div>
          </div>
          <div className="model-grid two-column">
            <SlotCard
              slot={slots.slots.primary}
              label="Primary"
              slotId={PRIMARY_SLOT_ID}
              models={selectableModels}
              slots={slots}
            />
            <SlotCard
              slot={slots.slots.secondary}
              label="Secondary"
              slotId={SECONDARY_SLOT_ID}
              models={selectableModels}
              slots={slots}
            />
          </div>
        </section>
      )}
      {active.length > 0 && (
        <section className="model-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Transfers</span>
              <h2>Downloads and failures</h2>
            </div>
          </div>
          <div className="model-grid">
            {active.map((model) => {
              const configurationId = Option.getOrNull(
                localModelConfigurationId(model)
              )
              const transfer = modelTransfer(model)
              const downloadId = modelDownloadId(model)
              return (
                <article
                  className="model-card compact"
                  key={configurationId ?? formatLocalModelDisplayName(model)}
                >
                  <div className="model-card-header">
                    <h3>{formatLocalModelDisplayName(model)}</h3>
                    <span
                      className={`status-pill ${
                        transfer ? "progress" : "danger"
                      }`}
                    >
                      {transfer ? `${transferProgress(transfer)}%` : "Failed"}
                    </span>
                  </div>
                  <p>
                    {transfer ? transferLabel(transfer) : modelFailure(model)}
                  </p>
                  {transfer && (
                    <progress
                      max={100}
                      value={transferProgress(transfer)}
                      aria-label={`${formatLocalModelDisplayName(
                        model
                      )} download progress`}
                    />
                  )}
                  <div className="model-actions">
                    {transfer && downloadId && (
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => actions.cancel(downloadId)}
                      >
                        <X size={14} />
                        Cancel
                      </button>
                    )}
                    {!transfer && configurationId && (
                      <button
                        className="primary-button"
                        type="button"
                        onClick={() => actions.install(configurationId)}
                      >
                        <RefreshCw size={14} />
                        Retry
                      </button>
                    )}
                    {!transfer && downloadId && (
                      <button
                        className="secondary-button"
                        type="button"
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
      )}
      <section className="model-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Storage</span>
            <h2>Installed models</h2>
          </div>
          <span className="section-count">{installed.length}</span>
        </div>
        {installed.length === 0 ? (
          <div className="empty-panel">
            No models are installed yet. Open Catalog to choose one for this
            machine.
          </div>
        ) : (
          <div className="model-grid">
            {installed.map((model) => (
              <InstalledModelCard
                key={Option.getOrElse(localModelConfigurationId(model), () =>
                  formatLocalModelDisplayName(model)
                )}
                model={model}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

function CatalogCard({ view }: { readonly view: CatalogModelView }): ReactNode {
  const { model, reconciliationState } = view
  const modelActions = useLocalModelActions()
  const slotActions = useModelSlotActions()
  const configurationId = Option.getOrNull(localModelConfigurationId(model))
  const providerModelId = Option.getOrNull(localModelProviderModelId(model))
  const status = modelStatus(model, reconciliationState)
  const transfer =
    reconciliationState._tag === "Transferring"
      ? reconciliationState
      : modelTransfer(model)
  const downloadId =
    reconciliationState._tag === "Transferring"
      ? reconciliationState.downloadId
      : modelDownloadId(model)
  const serving =
    model.servingState._tag === "Assessed" ? model.servingState : null
  const assessment =
    serving?.assessment._tag === "Fits" ? serving.assessment : null
  const recommendation = serving?.recommendations[0] ?? null
  const selectable =
    serving?.availabilityState._tag === "Selectable" &&
    assessment !== null &&
    providerModelId !== null
  const installed = model.acquisitionState._tag === "Installed"
  const starting =
    reconciliationState._tag === "Starting" ||
    reconciliationState._tag === "Removing"
  return (
    <article className="model-card catalog-card">
      <div className="model-card-header">
        <div>
          <div className="card-badges">
            {recommendation && (
              <span className="recommendation-badge">
                {intentLabel(recommendation.intent)}
              </span>
            )}
            <span className="subtle-badge">
              {model.presentation.variantLabel}
            </span>
          </div>
          <h3>{model.presentation.displayName}</h3>
        </div>
        <span className={`status-pill ${status.tone}`}>{status.label}</span>
      </div>
      <p>{model.presentation.description}</p>
      {recommendation && (
        <p className="recommendation-copy">{recommendation.explanation}</p>
      )}
      {serving && assessment && (
        <div className="evidence-grid">
          <span>
            <MemoryStick size={14} />
            <strong>{formatBytes(assessment.memory.totalRequiredBytes)}</strong>
            <small>memory</small>
          </span>
          <span>
            <Cpu size={14} />
            <strong>
              ~{Math.round(assessment.performance[0]!.estimatedTokensPerSecond)}{" "}
              tok/s
            </strong>
            <small>expected speed</small>
          </span>
          <span>
            <HardDrive size={14} />
            <strong>{formatBytes(model.downloadBytes)}</strong>
            <small>download</small>
          </span>
          <span>
            <strong>
              {formatContext(serving.configuration.profile.contextLength)}
            </strong>
            <small>context</small>
          </span>
        </div>
      )}
      {serving && assessment && (
        <div className="capability-row">
          {serving.capabilities.tools && <span>Tools</span>}
          {serving.capabilities.vision && <span>Vision</span>}
          {serving.capabilities.reasoning.supported && <span>Reasoning</span>}
          {model.presentation.license._tag === "Some" && (
            <span>{model.presentation.license.value}</span>
          )}
        </div>
      )}
      {transfer && (
        <>
          <progress
            max={100}
            value={transferProgress(transfer)}
            aria-label={`${formatLocalModelDisplayName(
              model
            )} transfer progress`}
          />
          <p>{transferLabel(transfer)}</p>
        </>
      )}
      {modelFailure(model) && (
        <p className="model-error-text">{modelFailure(model)}</p>
      )}
      <div className="model-actions">
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
            <X size={14} />
            Cancel
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
            <Check size={14} />
            Select as primary
          </button>
        )}
        {installed && configurationId && (
          <button
            className="danger-button"
            type="button"
            disabled={starting}
            onClick={() =>
              window.confirm(
                `Delete ${formatLocalModelDisplayName(
                  model
                )} from this computer?`
              ) && modelActions.delete(configurationId)
            }
          >
            <Trash2 size={14} />
            Uninstall
          </button>
        )}
      </div>
    </article>
  )
}

function CatalogView(): ReactNode {
  const catalogResult = useCatalogModels()
  const modelActions = useLocalModelActions()
  const catalog = valueOf(catalogResult)
  const candidates = useMemo(
    () =>
      [...(catalog?.models ?? [])].sort(
        (left, right) =>
          Number(
            right.model.servingState._tag === "Assessed" &&
              right.model.servingState.recommendations.length > 0
          ) -
            Number(
              left.model.servingState._tag === "Assessed" &&
                left.model.servingState.recommendations.length > 0
            ) ||
          formatLocalModelDisplayName(left.model).localeCompare(
            formatLocalModelDisplayName(right.model)
          )
      ),
    [catalog]
  )
  const discovery = catalog?.discoveryState ?? null
  return (
    <div className="model-center-view">
      <QueryNotice result={catalogResult} label="local catalog" />
      {modelActions.latestInstallationFailed && (
        <div className="model-notice danger">
          <AlertTriangle size={14} />
          The latest model installation or update request failed.
        </div>
      )}
      {discovery?._tag === "Loading" && (
        <section className="preparation-panel">
          <Loader2 className="spin" size={20} />
          <div>
            <h2>Preparing models for this machine</h2>
            <p>Hardware detection and native assessment run locally.</p>
          </div>
        </section>
      )}
      {discovery && discovery.progress.length > 0 && (
        <ol className="progress-steps">
          {discovery.progress.map((step, index) => (
            <li
              key={`${step.id}:${index}`}
              data-state={step.status._tag.toLowerCase()}
            >
              <span>
                {step.status._tag === "Completed" ? (
                  <Check size={13} />
                ) : step.status._tag === "Running" ? (
                  <Loader2 className="spin" size={13} />
                ) : step.status._tag === "Failed" ? (
                  <AlertTriangle size={13} />
                ) : (
                  index + 1
                )}
              </span>
              <div>
                <strong>
                  {step.id.charAt(0).toUpperCase() + step.id.slice(1)}
                </strong>
                {step.status._tag === "Failed" && (
                  <small>{step.status.failure.message}</small>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
      {discovery?._tag === "Failed" && (
        <div className="model-notice danger">
          <AlertTriangle size={15} />
          {discovery.failure.message}
        </div>
      )}
      <section className="model-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Assessed locally</span>
            <h2>Catalog</h2>
            <p>
              Configurations and compatibility come directly from the local
              daemon.
            </p>
          </div>
          <span className="section-count">{candidates.length}</span>
        </div>
        {candidates.length === 0 && discovery?._tag === "Ready" ? (
          <div className="empty-panel">
            No local catalog models are currently available.
          </div>
        ) : (
          <div className="model-grid">
            {candidates.map((candidate) => (
              <CatalogCard
                key={Option.getOrElse(
                  localModelConfigurationId(candidate.model),
                  () => formatLocalModelDisplayName(candidate.model)
                )}
                view={candidate}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

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
    <div className="model-center-view">
      <QueryNotice result={hardwareResult} label="hardware" />
      <QueryNotice result={slotsResult} label="model runtime" />
      {hardware && (
        <>
          <section className="hardware-hero">
            <div className="hardware-icon">
              <Cpu size={24} />
            </div>
            <div>
              <span className="eyebrow">This machine</span>
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
            <div className="hardware-capacity">
              <strong>{formatBytes(hardware.totalSystemMemoryBytes)}</strong>
              <span>system memory</span>
            </div>
          </section>
          <section className="model-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Runtime</span>
                <h2>Current model</h2>
              </div>
            </div>
            {primary && (
              <article className="model-card compact">
                <div className="model-card-header">
                  <div>
                    <h3>
                      {primary._tag === "Unassigned"
                        ? "No local model selected"
                        : primary.descriptor.displayName}
                    </h3>
                    {slotStatus(primary).detail && (
                      <p>{slotStatus(primary).detail}</p>
                    )}
                  </div>
                  <span className={`status-pill ${slotStatus(primary).tone}`}>
                    {slotStatus(primary).label}
                  </span>
                </div>
                {Result.isSuccess(preview) &&
                  primary._tag === "ConfiguredLocal" &&
                  primary.residency._tag === "Unloaded" && (
                    <p>
                      Load preview: {preview.value.parallelSequences} parallel
                      sequences ·{" "}
                      {formatContext(preview.value.contextWindowTokens)} context
                      · {formatBytes(preview.value.requiredSystemMemoryBytes)}{" "}
                      system memory
                    </p>
                  )}
              </article>
            )}
          </section>
          <section className="model-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Physical domains</span>
                <h2>Memory</h2>
              </div>
            </div>
            <div className="memory-grid">
              {memory?.domains.map((domain) => (
                <article className="memory-card" key={domain.id}>
                  <div>
                    <MemoryStick size={17} />
                    <strong>{domain.label}</strong>
                  </div>
                  <span>{formatBytes(domain.totalBytes)} total</span>
                  {domain.freeBytes === null ? (
                    <p>{domain.notice}</p>
                  ) : (
                    <>
                      <progress
                        max={domain.totalBytes}
                        value={domain.usedBytes ?? 0}
                        aria-label={`${domain.label} used memory`}
                      />
                      <p>
                        {formatBytes(domain.usedBytes ?? 0)} used ·{" "}
                        {formatBytes(domain.freeBytes)} free
                      </p>
                      {domain.fixedBytes !== null && domain.fixedBytes > 0 && (
                        <small>
                          {formatBytes(domain.fixedBytes)} model ·{" "}
                          {formatBytes(domain.kvCacheBytes ?? 0)} context cache
                        </small>
                      )}
                    </>
                  )}
                </article>
              ))}
            </div>
          </section>
          <section className="model-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Compute</span>
                <h2>Accelerators</h2>
              </div>
            </div>
            {hardware.accelerators.length === 0 ? (
              <div className="empty-panel">
                CPU inference · no accelerator reported
              </div>
            ) : (
              <div className="model-grid">
                {hardware.accelerators.map((accelerator) => (
                  <article
                    className="model-card compact"
                    key={accelerator.acceleratorId}
                  >
                    <div className="model-card-header">
                      <div>
                        <h3>{accelerator.name}</h3>
                        <p>{accelerator.backend}</p>
                      </div>
                      <Cpu size={18} />
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
          <section className="reserve-row">
            <span>
              Available system memory{" "}
              <strong>
                {formatBytes(hardware.availableSystemMemoryBytes)}
              </strong>
            </span>
            <span>
              Allocation headroom{" "}
              <strong>
                {formatBytes(hardware.systemAllocationHeadroomBytes)}
              </strong>
            </span>
            <span>
              Abort reserve{" "}
              <strong>{formatBytes(hardware.abortReserveBytes)}</strong>
            </span>
          </section>
        </>
      )}
    </div>
  )
}

export function ModelCenter({
  tab,
  onTabChange,
}: {
  readonly tab: ModelCenterTab
  readonly onTabChange: (tab: ModelCenterTab) => void
}): ReactNode {
  return (
    <div className="model-center">
      <nav className="model-center-tabs" aria-label="Model Center sections">
        {(["models", "catalog", "hardware"] as const).map((item) => (
          <button
            key={item}
            type="button"
            aria-current={tab === item ? "page" : undefined}
            onClick={() => onTabChange(item)}
          >
            {item === "models"
              ? "Models"
              : item === "catalog"
              ? "Catalog"
              : "Hardware"}
          </button>
        ))}
      </nav>
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
