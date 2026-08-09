import { useMemo, type ReactNode } from "react"
import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { AlertTriangle, Check, Cpu, Download, HardDrive, Loader2, X } from "lucide-react"
import {
  buildLocalInferenceSelections,
  deriveOnboardingModelSetupView,
  useOnboardingModelSetup,
  type LocalInferenceSelection,
} from "@magnitudedev/client-common"
import { ReasoningEffortSchema } from "@magnitudedev/sdk"
import { downloadLabel, downloadProgress, formatBytes, formatContext } from "./local-inference-format"

const selectionIntent = (selection: LocalInferenceSelection): string | null =>
  selection.kind === "recommendation" && selection.recommendation._tag === "Recommended"
    ? ({ balanced: "Balanced", best_quality: "Best quality", fastest: "Fastest", lightweight: "Lightweight" } as const)[selection.recommendation.value.intent]
    : null

export function LocalModelOnboarding({
  onSkip,
  completing,
  completionFailed,
}: {
  onSkip: () => void
  completing: boolean
  completionFailed: boolean
}): ReactNode {
  const setup = useOnboardingModelSetup()
  const hardware = Option.getOrNull(Result.value(setup.hardware))
  const models = Option.getOrNull(Result.value(setup.models))
  const catalog = Option.getOrNull(Result.value(setup.catalog))
  const slots = Option.getOrNull(Result.value(setup.slots))
  const selections = useMemo(() => models && catalog && slots
    ? buildLocalInferenceSelections(models, catalog, slots).filter((selection) =>
        selection.kind !== "recommendation" || selection.recommendation._tag === "Recommended")
    : [], [catalog, models, slots])
  const submitting = Result.isWaiting(setup.workflowResult)
  const view = models && slots ? deriveOnboardingModelSetupView({
    active: true,
    submission: setup.submission,
    providerModelId: setup.providerModelId,
    submitting,
    models,
    slots,
  }) : null
  const progress = models?.recommendations.progress ?? []
  const failed = Result.isFailure(setup.hardware) || Result.isFailure(setup.models)
    || Result.isFailure(setup.catalog) || Result.isFailure(setup.slots)
  const choose = (selection: LocalInferenceSelection) => {
    if (selection.kind === "running") {
      onSkip()
      return
    }
    const reasoningEffort = Option.getOrElse(
      selection.reasoningEffort,
      () => ReasoningEffortSchema.make("none"),
    )
    if (selection.kind === "stored" && Option.isSome(selection.providerModelId)) {
      setup.load({
        providerModelId: selection.providerModelId.value,
        displayName: selection.model.displayName,
        reasoningEffort,
      })
      return
    }
    if (selection.kind === "stored") {
      setup.configureThenLoad({
        targetId: selection.model.targetId,
        configurationId: selection.configurationId,
        displayName: selection.model.displayName,
        reasoningEffort,
      })
      return
    }
    if (selection.recommendation._tag === "Recommended") {
      const candidate = selection.recommendation.value.candidate
      setup.configureThenLoad({
        targetId: candidate.targetId,
        configurationId: candidate.configurationId,
        displayName: candidate.displayName,
        reasoningEffort,
      })
    }
  }

  const operationCandidate = view?._tag === "Downloading" || view?._tag === "DownloadFailed"
    || view?._tag === "Configuring" ? view.candidate : null
  const operationProgress = operationCandidate ? downloadProgress(operationCandidate.download) : null

  return (
    <main className="onboarding-page">
      <div className="onboarding-shell">
        <header className="onboarding-header">
          <div className="onboarding-mark"><Cpu size={26} /></div>
          <div>
            <span className="eyebrow">Local inference setup</span>
            <h1>Choose the model that powers Magnitude</h1>
            <p>Everything runs on this machine. Magnitude assessed these configurations against your actual hardware.</p>
          </div>
        </header>

        {hardware && (
          <div className="onboarding-hardware">
            <Cpu size={16} />
            <span>{Option.getOrElse(hardware.productName, () => Option.getOrElse(hardware.processor, () => "This computer"))}</span>
            <span>{hardware.logicalCores} cores</span>
            <span>{formatBytes(hardware.totalSystemMemoryBytes)} memory</span>
            {hardware.accelerators.map((accelerator) => <span key={accelerator.acceleratorId}>{accelerator.name} · {accelerator.backend}</span>)}
          </div>
        )}

        {progress.length > 0 && models?.recommendations._tag !== "Ready" && (
          <ol className="progress-steps onboarding-progress">
            {progress.map((step, index) => (
              <li key={`${step.id}:${index}`} data-state={step.status._tag.toLowerCase()}>
                <span>{step.status._tag === "Completed" ? <Check size={13} /> : step.status._tag === "Running" ? <Loader2 className="spin" size={13} /> : step.status._tag === "Failed" ? <AlertTriangle size={13} /> : index + 1}</span>
                <div><strong>{step.id.charAt(0).toUpperCase() + step.id.slice(1)}</strong>{step.status._tag === "Failed" && <small>{step.status.failure.message}</small>}</div>
              </li>
            ))}
          </ol>
        )}

        {failed && <div className="model-notice danger"><AlertTriangle size={16} />The local model state could not be loaded. Reconnect to the daemon and try again.</div>}
        {models?.recommendations._tag === "Failed" && <div className="model-notice danger"><AlertTriangle size={16} />{models.recommendations.failure.message}</div>}
        {catalog?._tag === "Unavailable" && <div className="model-notice danger"><AlertTriangle size={16} />The local model catalog is unavailable.</div>}
        {Result.isFailure(setup.workflowResult) && <div className="model-notice danger"><AlertTriangle size={16} />The setup command failed. You can retry or choose another model.</div>}
        {completionFailed && <div className="model-notice danger"><AlertTriangle size={16} />Setup could not be completed. Check the daemon connection and try again.</div>}

        {view?._tag === "Activating" && (
          <section className="onboarding-operation">
            {view.phase === "Failed" ? <AlertTriangle size={24} /> : <Loader2 className={view.phase === "Ready" ? "" : "spin"} size={24} />}
            <div>
              <span className="eyebrow">{view.phase}</span>
              <h2>{view.displayName}</h2>
              <p>{view.failure ?? (view.phase === "Ready" ? "Finishing setup…" : "The daemon is preparing the selected model.")}</p>
            </div>
            {view.phase === "Failed" && <button type="button" className="secondary-button" onClick={setup.cancel}>Choose another</button>}
            {(view.phase === "Loading" || view.phase === "Stopping") && <button type="button" className="secondary-button" onClick={setup.cancel} disabled={setup.cancelling}><X size={14} />{setup.cancelling ? "Cancelling…" : "Cancel"}</button>}
          </section>
        )}

        {operationCandidate && view?._tag !== "Activating" && (
          <section className="onboarding-operation">
            {view?._tag === "DownloadFailed" ? <AlertTriangle size={24} /> : <Loader2 className="spin" size={24} />}
            <div>
              <span className="eyebrow">{view?._tag === "Configuring" ? "Configuring" : view?._tag === "DownloadFailed" ? "Download failed" : "Downloading"}</span>
              <h2>{operationCandidate.displayName}</h2>
              <p>{downloadLabel(operationCandidate.download)}</p>
              {operationProgress !== null && <progress max={100} value={operationProgress} aria-label="Setup download progress" />}
            </div>
            <div className="model-actions">
              {view?._tag === "DownloadFailed" && <button type="button" className="primary-button" onClick={() => setup.configureThenLoad({
                targetId: operationCandidate.targetId,
                configurationId: operationCandidate.configurationId,
                displayName: operationCandidate.displayName,
                reasoningEffort: Option.getOrElse(operationCandidate.capabilities.reasoning.defaultEffort, () => ReasoningEffortSchema.make("none")),
              })}>Retry</button>}
              <button type="button" className="secondary-button" onClick={setup.cancel} disabled={setup.cancelling}><X size={14} />{setup.cancelling ? "Cancelling…" : "Choose another"}</button>
            </div>
          </section>
        )}

        {(!view || view._tag === "Choosing") && models?.recommendations._tag === "Ready" && catalog && slots && (
          <section className="onboarding-choices">
            {selections.map((selection) => {
              const recommendation = selection.kind === "recommendation" && selection.recommendation._tag === "Recommended"
                ? selection.recommendation.value.candidate
                : null
              return (
                <button type="button" className="onboarding-choice" key={selection.id} onClick={() => choose(selection)} disabled={submitting || completing}>
                  <div className="model-card-header">
                    <div>
                      {selectionIntent(selection) && <span className="recommendation-badge">{selectionIntent(selection)}</span>}
                      <h3>{selection.model.displayName}</h3>
                      <p>{selection.model.description}</p>
                    </div>
                    <span className={`status-pill ${selection.kind === "running" ? "success" : selection.kind === "stored" ? "neutral" : "progress"}`}>
                      {selection.kind === "running" ? "Ready" : selection.kind === "stored" ? "Installed" : "Download"}
                    </span>
                  </div>
                  <div className="model-meta-row">
                    <span><HardDrive size={13} />{formatBytes(selection.model.downloadBytes)}</span>
                    <span>{selection.model.quantization}</span>
                    <span>{formatContext(selection.contextLength)} context</span>
                    {recommendation && <span><Download size={13} />{formatBytes(recommendation.downloadBytes)}</span>}
                  </div>
                </button>
              )
            })}
            {selections.length === 0 && <div className="empty-panel">No compatible model choices are available yet.</div>}
          </section>
        )}

        <footer className="onboarding-footer">
          <span>Model files and inference stay on this machine.</span>
          <button type="button" className="text-button" onClick={onSkip} disabled={submitting || completing}>Continue without loading a model</button>
        </footer>
      </div>
    </main>
  )
}
