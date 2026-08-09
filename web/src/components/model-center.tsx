import { useMemo, type ReactNode } from "react"
import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
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
  modelSlotResidentAllocation,
  useLocalConfigurationSelection,
  useLocalInferenceHardware,
  useLocalModelActions,
  useLocalModels,
  useModelSlotActions,
  useModelSlots,
  usePreviewModelLoad,
  useProviderModelCatalog,
} from "@magnitudedev/client-common"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogLifecycle,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type LocalModel,
  type LocalModelCatalogCandidate,
  type ModelSlot,
  type ModelSlotsState,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type SlotId,
} from "@magnitudedev/sdk"
import type { ModelCenterTab } from "../state/web-atoms"
import {
  downloadLabel,
  downloadProgress,
  formatBytes,
  formatContext,
  intentLabel,
  modelSpeedLabel,
  slotStatus,
} from "./local-inference-format"

const localProviderId = ProviderIdSchema.make("local")

const valueOf = <A, E>(result: Result.Result<A, E>): A | null =>
  Option.getOrNull(Result.value(result))

const catalogModels = (catalog: ProviderModelCatalogState | null): readonly ProviderModelCatalogEntry[] =>
  catalog === null ? [] : ProviderModelCatalogLifecycle.match(catalog, {
    Loading: () => [],
    Ready: ({ models }) => models.filter(({ providerId }) => providerId === localProviderId),
    Refreshing: ({ models }) => models.filter(({ providerId }) => providerId === localProviderId),
    Degraded: ({ models }) => models.filter(({ providerId }) => providerId === localProviderId),
    Unavailable: () => [],
  })

function QueryNotice({ result, label }: { result: Result.Result<unknown, unknown>; label: string }): ReactNode {
  if (Result.isFailure(result)) {
    return <div className="model-notice danger"><AlertTriangle size={15} />Unable to load {label}.</div>
  }
  if (Result.isWaiting(result) && Option.isNone(Result.value(result))) {
    return <div className="model-notice"><Loader2 className="spin" size={15} />Loading {label}…</div>
  }
  return null
}

function MutationFailure({ result, message }: { result: Result.Result<unknown, unknown>; message: string }): ReactNode {
  return Result.isFailure(result)
    ? <div className="model-notice danger"><AlertTriangle size={15} />{message}</div>
    : null
}

function CatalogLifecycleNotice({ catalog }: { catalog: ProviderModelCatalogState | null }): ReactNode {
  if (catalog === null || catalog._tag === "Ready") return null
  if (catalog._tag === "Loading") {
    return <div className="model-notice"><Loader2 className="spin" size={15} />Loading local model offerings…</div>
  }
  const localFailures = catalog.failures.filter((failure) =>
    failure._tag === "CatalogFailure" || failure.providerId === localProviderId)
  if (catalog._tag === "Refreshing" && localFailures.length === 0) {
    return <div className="model-notice"><Loader2 className="spin" size={15} />Refreshing local model offerings…</div>
  }
  const fallback = catalog._tag === "Unavailable"
    ? "The local model catalog is unavailable."
    : "The local model catalog is degraded; showing the last available offerings."
  return (
    <div className={`model-notice ${catalog._tag === "Unavailable" ? "danger" : ""}`}>
      <AlertTriangle size={15} />
      {localFailures.map(({ message }) => message).join(" · ") || fallback}
    </div>
  )
}

function SlotCard({
  slot,
  label,
  slotId,
  catalog,
  slots,
}: {
  slot: ModelSlot
  label: string
  slotId: SlotId
  catalog: readonly ProviderModelCatalogEntry[]
  slots: ModelSlotsState
}): ReactNode {
  const actions = useModelSlotActions()
  const busy = Result.isWaiting(actions.assignResult)
    || Result.isWaiting(actions.clearResult)
    || Result.isWaiting(actions.loadResult)
    || Result.isWaiting(actions.stopResult)
    || Result.isWaiting(actions.favoriteResult)
  const failed = Result.isFailure(actions.assignResult)
    || Result.isFailure(actions.clearResult)
    || Result.isFailure(actions.loadResult)
    || Result.isFailure(actions.stopResult)
    || Result.isFailure(actions.favoriteResult)
  const status = slotStatus(slot)
  const isFavorite = slot._tag !== "Unassigned" && slots.favoriteModels.some((favorite) =>
    favorite.providerId === slot.selection.providerId
    && favorite.providerModelId === slot.selection.providerModelId)
  const localOptions = catalog.filter(({ supportedSlots, availability }) =>
    supportedSlots.includes(slotId) && availability._tag === "Available")
  const selectedKey = slot._tag === "Unassigned" || slot.selection.providerId !== localProviderId
    ? ""
    : slot.selection.providerModelId
  const instance = slot._tag === "ConfiguredLocal" ? Option.getOrNull(slot.instance) : null

  return (
    <article className="model-card slot-card">
      <div className="model-card-header">
        <div>
          <span className="eyebrow">{label}</span>
          <h3>{slot._tag === "Unassigned" ? "Choose a local model" : slot.descriptor.displayName}</h3>
        </div>
        <span className={`status-pill ${status.tone}`}>{status.label}</span>
      </div>
      {status.detail && <p className="model-error-text">{status.detail}</p>}
      {failed && <div className="model-notice danger"><AlertTriangle size={14} />The model action failed. The authoritative slot state was not changed.</div>}
      <label className="field-label" htmlFor={`slot-${slotId}`}>Selected model</label>
      <select
        id={`slot-${slotId}`}
        value={selectedKey}
        disabled={busy}
        onChange={(event) => {
          const model = localOptions.find(({ providerModelId }) => providerModelId === event.target.value)
          if (!model) {
            actions.clear(slotId)
            return
          }
          actions.assign(slotId, {
            providerId: localProviderId,
            providerModelId: model.providerModelId,
            reasoningEffort: Option.getOrElse(
              model.capabilities.reasoning.defaultEffort,
              () => ReasoningEffortSchema.make("none"),
            ),
          })
        }}
      >
        <option value="">Choose from installed models</option>
        {localOptions.map((model) => (
          <option key={model.providerModelId} value={model.providerModelId}>
            {model.displayName} · {formatContext(model.contextWindow)} context
          </option>
        ))}
      </select>
      <div className="model-actions">
        {slot._tag === "ConfiguredLocal" && slot.actions.some((action) => action === "Load" || action === "RetryLoad") && (
          <button type="button" className="primary-button" disabled={busy} onClick={() => actions.load(slotId)}>
            <Play size={14} />{slot.actions.includes("RetryLoad") ? "Retry load" : "Load"}
          </button>
        )}
        {slot._tag === "ConfiguredLocal" && slot.actions.includes("Stop") && instance && (
          <button type="button" className="secondary-button" disabled={busy} onClick={() => actions.stop(instance.id)}>
            <Square size={13} />Stop
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
          <button type="button" className="secondary-button" disabled={busy} onClick={() => actions.clear(slotId)}>
            <X size={14} />Clear
          </button>
        )}
      </div>
    </article>
  )
}

function InstalledModelCard({ model }: { model: LocalModel }): ReactNode {
  const actions = useLocalModelActions()
  const deleting = Result.isWaiting(actions.deleteResult)
  return (
    <article className="model-card compact">
      <div className="model-card-header">
        <div>
          <h3>{model.displayName}</h3>
          <p>{model.quantization} · {formatContext(model.maximumContextLength)} max context</p>
        </div>
        <span className="status-pill success"><Check size={12} />Installed</span>
      </div>
      <p>{model.description}</p>
      <div className="model-meta-row">
        <span><HardDrive size={13} />{model.download._tag === "Downloaded" ? formatBytes(model.download.installedBytes) : formatBytes(model.downloadBytes)}</span>
        <span>{model.offerings.length} configured {model.offerings.length === 1 ? "offering" : "offerings"}</span>
      </div>
      <div className="model-actions">
        <button
          type="button"
          className="danger-button"
          disabled={deleting}
          onClick={() => {
            if (window.confirm(`Delete ${model.displayName} from this computer?`)) actions.delete(model.targetId)
          }}
        >
          <Trash2 size={14} />Delete files
        </button>
      </div>
      <MutationFailure result={actions.deleteResult} message="Failed to delete this model." />
    </article>
  )
}

function ModelsView(): ReactNode {
  const modelsResult = useLocalModels()
  const slotsResult = useModelSlots()
  const catalogResult = useProviderModelCatalog()
  const models = valueOf(modelsResult)
  const slots = valueOf(slotsResult)
  const catalogState = valueOf(catalogResult)
  const catalog = catalogModels(catalogState)
  const installed = models?.models.filter(({ download }) => download._tag === "Downloaded") ?? []
  const active = models?.models.filter(({ download }) => download._tag === "Downloading" || download._tag === "Failed") ?? []
  const actions = useLocalModelActions()
  const transferBusy = Result.isWaiting(actions.downloadResult)
    || Result.isWaiting(actions.cancelResult)
    || Result.isWaiting(actions.dismissFailureResult)

  return (
    <div className="model-center-view">
      <QueryNotice result={slotsResult} label="model slots" />
      <QueryNotice result={modelsResult} label="local models" />
      <QueryNotice result={catalogResult} label="model catalog" />
      <CatalogLifecycleNotice catalog={catalogState} />
      {slots && (
        <section className="model-section">
          <div className="section-heading"><div><span className="eyebrow">Runtime</span><h2>Selected models</h2></div></div>
          <div className="model-grid two-column">
            <SlotCard slot={slots.slots.primary} label="Primary" slotId={PRIMARY_SLOT_ID} catalog={catalog} slots={slots} />
            <SlotCard slot={slots.slots.secondary} label="Secondary" slotId={SECONDARY_SLOT_ID} catalog={catalog} slots={slots} />
          </div>
        </section>
      )}
      {active.length > 0 && (
        <section className="model-section">
          <div className="section-heading"><div><span className="eyebrow">Transfers</span><h2>Downloads and failures</h2></div></div>
          <MutationFailure result={actions.downloadResult} message="Failed to start the download." />
          <MutationFailure result={actions.cancelResult} message="Failed to cancel the download." />
          <MutationFailure result={actions.dismissFailureResult} message="Failed to dismiss the failure." />
          <div className="model-grid">
            {active.map((model) => {
              const download = model.download
              const progress = downloadProgress(download)
              return (
                <article className="model-card compact" key={model.targetId}>
                  <div className="model-card-header"><h3>{model.displayName}</h3><span className={`status-pill ${model.download._tag === "Failed" ? "danger" : "progress"}`}>{model.download._tag}</span></div>
                  <p>{downloadLabel(model.download)}</p>
                  {progress !== null && <progress max={100} value={progress} aria-label={`${model.displayName} download progress`} />}
                  <div className="model-actions">
                    {download._tag === "Downloading" && <button className="secondary-button" type="button" disabled={transferBusy} onClick={() => actions.cancel(download.attemptIds)}><X size={14} />Cancel</button>}
                    {model.download._tag === "Failed" && <>
                      <button className="primary-button" type="button" disabled={transferBusy} onClick={() => actions.download(model.targetId)}><RefreshCw size={14} />Retry</button>
                      <button className="secondary-button" type="button" disabled={transferBusy} onClick={() => actions.dismissFailure(model.targetId)}>Dismiss</button>
                    </>}
                  </div>
                </article>
              )
            })}
          </div>
        </section>
      )}
      <section className="model-section">
        <div className="section-heading"><div><span className="eyebrow">Storage</span><h2>Installed models</h2></div><span className="section-count">{installed.length}</span></div>
        {installed.length === 0
          ? <div className="empty-panel">No models are installed yet. Open Catalog to choose one for this machine.</div>
          : <div className="model-grid">{installed.map((model) => <InstalledModelCard key={model.targetId} model={model} />)}</div>}
      </section>
    </div>
  )
}

function CandidateCard({
  candidate,
  recommendation,
}: {
  candidate: LocalModelCatalogCandidate
  recommendation: { intent: "balanced" | "best_quality" | "fastest" | "lightweight"; explanation: string } | null
}): ReactNode {
  const modelActions = useLocalModelActions()
  const selection = useLocalConfigurationSelection()
  const download = candidate.download
  const progress = downloadProgress(download)
  const totalMemory = candidate.memory.reduce((sum, domain) => sum + domain.requiredBytes, 0)
  const busy = Result.isWaiting(selection.result)
    || Result.isWaiting(modelActions.downloadResult)
    || Result.isWaiting(modelActions.cancelResult)
    || Result.isWaiting(modelActions.dismissFailureResult)
  return (
    <article className="model-card catalog-card">
      <div className="model-card-header">
        <div>
          <div className="card-badges">
            {recommendation && <span className="recommendation-badge">{intentLabel(recommendation.intent)}</span>}
            <span className="subtle-badge">{candidate.quantizationName}</span>
          </div>
          <h3>{candidate.displayName}</h3>
        </div>
        <span className={`status-pill ${candidate.download._tag === "Downloaded" ? "success" : candidate.download._tag === "Failed" ? "danger" : candidate.download._tag === "Downloading" ? "progress" : "neutral"}`}>
          {candidate.download._tag === "Downloaded" ? "Installed" : candidate.download._tag === "Downloading" ? `${progress}%` : candidate.download._tag === "Failed" ? "Failed" : "Available"}
        </span>
      </div>
      <p>{candidate.description}</p>
      {recommendation && <p className="recommendation-copy">{recommendation.explanation}</p>}
      <div className="evidence-grid">
        <span><MemoryStick size={14} /><strong>{formatBytes(totalMemory)}</strong><small>memory</small></span>
        <span><Cpu size={14} /><strong>{modelSpeedLabel(candidate)}</strong><small>expected speed</small></span>
        <span><HardDrive size={14} /><strong>{formatBytes(candidate.downloadBytes)}</strong><small>download</small></span>
        <span><strong>{formatContext(candidate.profile.contextLength)}</strong><small>context</small></span>
      </div>
      <div className="capability-row">
        {candidate.capabilities.tools && <span>Tools</span>}
        {candidate.capabilities.vision && <span>Vision</span>}
        {candidate.capabilities.reasoning.supported && <span>Reasoning</span>}
        <span>{candidate.license}</span>
      </div>
      {progress !== null && candidate.download._tag === "Downloading" && <progress max={100} value={progress} aria-label={`${candidate.displayName} download progress`} />}
      {candidate.download._tag === "Failed" && <p className="model-error-text">{candidate.download.failure.message}</p>}
      {candidate.availability._tag === "Unavailable" && <p className="model-error-text">{candidate.availability.failure.message}</p>}
      <MutationFailure result={selection.result} message="Failed to configure and select this model." />
      <MutationFailure result={modelActions.downloadResult} message="Failed to start the download." />
      <MutationFailure result={modelActions.cancelResult} message="Failed to cancel the download." />
      <MutationFailure result={modelActions.dismissFailureResult} message="Failed to dismiss the download failure." />
      <div className="model-actions">
        {(candidate.download._tag === "NotDownloaded" || candidate.download._tag === "Cancelled" || candidate.download._tag === "Failed") && (
          <button className="primary-button" type="button" disabled={busy} onClick={() => modelActions.download(candidate.targetId)}>
            {candidate.download._tag === "Failed" ? <RefreshCw size={14} /> : <Download size={14} />}
            {candidate.download._tag === "Failed" ? "Retry download" : "Download"}
          </button>
        )}
        {download._tag === "Downloading" && (
          <button className="secondary-button" type="button" disabled={busy} onClick={() => modelActions.cancel(download.attemptIds)}><X size={14} />Cancel</button>
        )}
        {candidate.download._tag === "Failed" && (
          <button className="secondary-button" type="button" disabled={busy} onClick={() => modelActions.dismissFailure(candidate.targetId)}>Dismiss</button>
        )}
        {candidate.download._tag === "Downloaded" && candidate.availability._tag === "Available" && (
          <button
            className="primary-button"
            type="button"
            disabled={busy}
            onClick={() => selection.select({
              slotId: PRIMARY_SLOT_ID,
              targetId: candidate.targetId,
              configurationId: candidate.configurationId,
              reasoningEffort: Option.getOrElse(
                candidate.capabilities.reasoning.defaultEffort,
                () => ReasoningEffortSchema.make("none"),
              ),
            })}
          >
            {busy ? <Loader2 className="spin" size={14} /> : <Check size={14} />}Select as primary
          </button>
        )}
      </div>
    </article>
  )
}

function CatalogView(): ReactNode {
  const modelsResult = useLocalModels()
  const models = valueOf(modelsResult)
  const lifecycle = models?.recommendations ?? null
  const candidates = lifecycle?._tag === "Ready" ? lifecycle.catalog : []
  const recommendations = lifecycle?._tag === "Ready" ? lifecycle.entries : []
  const progress = lifecycle?.progress ?? []
  const candidatesWithRecommendation = useMemo(() => candidates.map((candidate) => ({
    candidate,
    recommendation: recommendations.find((entry) => entry.candidate.configurationId === candidate.configurationId) ?? null,
  })).sort((left, right) => Number(right.recommendation !== null) - Number(left.recommendation !== null)), [candidates, recommendations])

  return (
    <div className="model-center-view">
      <QueryNotice result={modelsResult} label="local catalog" />
      {lifecycle?._tag === "Loading" && (
        <section className="preparation-panel"><Loader2 className="spin" size={20} /><div><h2>Preparing models for this machine</h2><p>Hardware detection and native assessment run locally.</p></div></section>
      )}
      {progress.length > 0 && (
        <ol className="progress-steps">
          {progress.map((step, index) => <li key={`${step.id}:${index}`} data-state={step.status._tag.toLowerCase()}><span>{step.status._tag === "Completed" ? <Check size={13} /> : step.status._tag === "Running" ? <Loader2 className="spin" size={13} /> : step.status._tag === "Failed" ? <AlertTriangle size={13} /> : index + 1}</span><div><strong>{step.id.charAt(0).toUpperCase() + step.id.slice(1)}</strong>{step.status._tag === "Failed" && <small>{step.status.failure.message}</small>}</div></li>)}
        </ol>
      )}
      {lifecycle?._tag === "Failed" && <div className="model-notice danger"><AlertTriangle size={15} />{lifecycle.failure.message}</div>}
      <section className="model-section">
        <div className="section-heading"><div><span className="eyebrow">Assessed locally</span><h2>Catalog</h2><p>Compatible configurations derived by the daemon for this hardware.</p></div><span className="section-count">{candidates.length}</span></div>
        {candidatesWithRecommendation.length === 0 && lifecycle?._tag === "Ready"
          ? <div className="empty-panel">No compatible local configurations are currently available.</div>
          : <div className="model-grid">{candidatesWithRecommendation.map(({ candidate, recommendation }) => <CandidateCard key={candidate.configurationId} candidate={candidate} recommendation={recommendation} />)}</div>}
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
  const allocation = primary ? modelSlotResidentAllocation(primary) : Option.none()
  const memory = hardware ? deriveHardwareMemoryView(hardware, allocation) : null

  return (
    <div className="model-center-view">
      <QueryNotice result={hardwareResult} label="hardware" />
      <QueryNotice result={slotsResult} label="model runtime" />
      {hardware && <>
        <section className="hardware-hero">
          <div className="hardware-icon"><Cpu size={24} /></div>
          <div><span className="eyebrow">This machine</span><h2>{Option.getOrElse(hardware.productName, () => Option.getOrElse(hardware.processor, () => "Local system"))}</h2><p>{hardware.platform} · {hardware.architecture} · {hardware.logicalCores} logical cores</p></div>
          <div className="hardware-capacity"><strong>{formatBytes(hardware.totalSystemMemoryBytes)}</strong><span>system memory</span></div>
        </section>
        <section className="model-section">
          <div className="section-heading"><div><span className="eyebrow">Runtime</span><h2>Current model</h2></div></div>
          {primary && <article className="model-card compact">
            <div className="model-card-header"><div><h3>{primary._tag === "Unassigned" ? "No local model selected" : primary.descriptor.displayName}</h3>{primary._tag !== "Unassigned" && <p>{slotStatus(primary).detail}</p>}</div><span className={`status-pill ${slotStatus(primary).tone}`}>{slotStatus(primary).label}</span></div>
            {Result.isSuccess(preview) && primary._tag === "ConfiguredLocal" && Option.isNone(primary.instance) && <p>Load preview: {preview.value.parallelSequences} parallel sequences · {formatContext(preview.value.contextWindowTokens)} context · {formatBytes(preview.value.requiredSystemMemoryBytes)} system memory</p>}
          </article>}
        </section>
        <section className="model-section">
          <div className="section-heading"><div><span className="eyebrow">Physical domains</span><h2>Memory</h2></div></div>
          <div className="memory-grid">
            {memory?.domains.map((domain) => <article className="memory-card" key={domain.id}>
              <div><MemoryStick size={17} /><strong>{domain.label}</strong></div>
              <span>{formatBytes(domain.totalBytes)} total</span>
              {domain.freeBytes === null ? <p>{domain.notice}</p> : <>
                <progress max={domain.totalBytes} value={domain.usedBytes ?? 0} aria-label={`${domain.label} used memory`} />
                <p>{formatBytes(domain.usedBytes ?? 0)} used · {formatBytes(domain.freeBytes)} free</p>
                {domain.fixedBytes !== null && domain.fixedBytes > 0 && <small>{formatBytes(domain.fixedBytes)} model · {formatBytes(domain.kvCacheBytes ?? 0)} context cache</small>}
              </>}
            </article>)}
          </div>
        </section>
        <section className="model-section">
          <div className="section-heading"><div><span className="eyebrow">Compute</span><h2>Accelerators</h2></div></div>
          {hardware.accelerators.length === 0 ? <div className="empty-panel">CPU inference · no accelerator reported</div> : <div className="model-grid">{hardware.accelerators.map((accelerator) => <article className="model-card compact" key={accelerator.acceleratorId}><div className="model-card-header"><div><h3>{accelerator.name}</h3><p>{accelerator.backend}</p></div><Cpu size={18} /></div></article>)}</div>}
        </section>
        <section className="reserve-row"><span>Assessment reserve <strong>{formatBytes(hardware.assessReserveBytes)}</strong></span><span>Warning reserve <strong>{formatBytes(hardware.warningReserveBytes)}</strong></span><span>Abort reserve <strong>{formatBytes(hardware.abortReserveBytes)}</strong></span></section>
      </>}
    </div>
  )
}

export function ModelCenter({ tab, onTabChange }: { tab: ModelCenterTab; onTabChange: (tab: ModelCenterTab) => void }): ReactNode {
  return (
    <div className="model-center">
      <nav className="model-center-tabs" aria-label="Model Center sections">
        {(["models", "catalog", "hardware"] as const).map((item) => (
          <button key={item} type="button" aria-current={tab === item ? "page" : undefined} onClick={() => onTabChange(item)}>
            {item === "models" ? "Models" : item === "catalog" ? "Catalog" : "Hardware"}
          </button>
        ))}
      </nav>
      {tab === "models" ? <ModelsView /> : tab === "catalog" ? <CatalogView /> : <HardwareView />}
    </div>
  )
}
