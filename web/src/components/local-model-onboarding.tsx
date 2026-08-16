import type { ReactNode } from "react"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { AlertTriangle, Check, Cpu, HardDrive, Loader2, X } from "lucide-react"
import {
  formatLocalModelDisplayName,
  localModelConfigurationId,
  modelDownloadFailureMessage,
  onboardingModelSetupFailureMessage,
  type useOnboardingModelSetup,
} from "@magnitudedev/client-common"
import type {
  LocalModel,
  LocalModelRecommendationProgressStep,
} from "@magnitudedev/sdk"
import {
  formatBytes,
  formatContext,
  intentLabel,
  modelContextLength,
  transferLabel,
  transferProgress,
} from "./local-inference-format"

type Setup = ReturnType<typeof useOnboardingModelSetup>

const progressIcon = (
  step: LocalModelRecommendationProgressStep,
  index: number
): ReactNode => {
  switch (step.status._tag) {
    case "Completed":
      return <Check size={13} />
    case "Running":
      return <Loader2 className="spin" size={13} />
    case "Failed":
      return <AlertTriangle size={13} />
    case "Pending":
      return index + 1
  }
}

const ModelSummary = ({ model }: { model: LocalModel }): ReactNode => {
  const context = modelContextLength(model)
  return (
    <div className="model-meta-row">
      <span>
        <HardDrive size={13} />
        {formatBytes(model.downloadBytes)}
      </span>
      {context !== null && <span>{formatContext(context)} context</span>}
      {model.bundle._tag === "SpeculativeDecoding" && (
        <span>{model.bundle.method._tag} speculative decoding</span>
      )}
    </div>
  )
}

export function LocalModelOnboarding({
  setup,
}: {
  readonly setup: Setup
}): ReactNode {
  const hardware = Option.getOrNull(Result.value(setup.hardware))
  const state = Option.getOrNull(Result.value(setup.view))
  if (state === null || state._tag === "Closed") return null

  const notice = Option.match(state.notice, {
    onNone: () => null,
    onSome: onboardingModelSetupFailureMessage,
  })
  const content = state.content
  const operation =
    content._tag === "Chooser" ? Option.getOrNull(content.operation) : null
  const operationModel = operation?.model ?? null
  const transfer =
    operationModel?.acquisitionState._tag === "Downloading"
      ? operationModel.acquisitionState
      : operationModel?.upgradeState._tag === "Upgrading"
      ? operationModel.upgradeState
      : null
  const operationFailure =
    (operation?._tag === "Loading" ? operation.failure?.message : null) ??
    (operationModel?.acquisitionState._tag === "Failed"
      ? modelDownloadFailureMessage(operationModel.acquisitionState.failure)
      : null)

  return (
    <main className="onboarding-page">
      <div className="onboarding-shell">
        <header className="onboarding-header">
          <div className="onboarding-mark">
            <Cpu size={26} />
          </div>
          <div>
            <span className="eyebrow">Local inference setup</span>
            <h1>Choose the model that powers Magnitude</h1>
            <p>
              Everything runs on this machine. Magnitude assessed these
              configurations against your actual hardware.
            </p>
          </div>
        </header>

        {hardware && (
          <div className="onboarding-hardware">
            <Cpu size={16} />
            <span>
              {Option.getOrElse(hardware.productName, () =>
                Option.getOrElse(hardware.processor, () => "This computer")
              )}
            </span>
            <span>{hardware.logicalCores} cores</span>
            <span>{formatBytes(hardware.totalSystemMemoryBytes)} memory</span>
            {hardware.accelerators.map((accelerator) => (
              <span key={accelerator.acceleratorId}>
                {accelerator.name} · {accelerator.backend}
              </span>
            ))}
          </div>
        )}

        {notice && (
          <div className="model-notice danger">
            <AlertTriangle size={16} />
            {notice}
          </div>
        )}
        {Result.isFailure(setup.hardware) && (
          <div className="model-notice danger">
            <AlertTriangle size={16} />
            Hardware details are unavailable.
          </div>
        )}

        {content._tag === "Preparation" && (
          <>
            <section className="preparation-panel">
              {content.discoveryFailure ? (
                <AlertTriangle size={20} />
              ) : (
                <Loader2 className="spin" size={20} />
              )}
              <div>
                <h2>Preparing models for this machine</h2>
                <p>
                  {content.discoveryFailure?.message ??
                    "Hardware detection and native assessment run locally."}
                </p>
              </div>
            </section>
            {content.progress.length > 0 && (
              <ol className="progress-steps onboarding-progress">
                {content.progress.map((step, index) => (
                  <li
                    key={`${step.id}:${index}`}
                    data-state={step.status._tag.toLowerCase()}
                  >
                    <span>{progressIcon(step, index)}</span>
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
          </>
        )}

        {content._tag === "Closing" && (
          <section className="onboarding-operation">
            <Loader2 className="spin" size={24} />
            <div>
              <span className="eyebrow">Finishing</span>
              <h2>Saving setup</h2>
              <p>Preparing the workspace…</p>
            </div>
          </section>
        )}

        {operationModel && operation && (
          <section className="onboarding-operation">
            {operation._tag === "Loading" && operation.phase === "Failed" ? (
              <AlertTriangle size={24} />
            ) : (
              <Loader2
                className={
                  operation._tag === "Loading" && operation.phase === "Ready"
                    ? ""
                    : "spin"
                }
                size={24}
              />
            )}
            <div>
              <span className="eyebrow">
                {operation._tag === "Loading"
                  ? operation.phase
                  : operation._tag}
              </span>
              <h2>{formatLocalModelDisplayName(operationModel)}</h2>
              <p>
                {operationFailure ??
                  (transfer
                    ? transferLabel(transfer)
                    : operation._tag === "Completing"
                    ? "Finishing setup…"
                    : "The daemon is preparing the selected model.")}
              </p>
              {transfer && (
                <progress
                  max={100}
                  value={transferProgress(transfer)}
                  aria-label="Setup download progress"
                />
              )}
            </div>
            <div className="model-actions">
              {operation._tag === "Loading" && operation.phase === "Failed" && (
                <button
                  type="button"
                  className="primary-button"
                  onClick={() => setup.select(operation.configurationId)}
                >
                  Retry
                </button>
              )}
              {operation._tag !== "Completing" &&
                !(
                  operation._tag === "Loading" && operation.phase === "Ready"
                ) && (
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={setup.cancel}
                    disabled={"cancelling" in operation && operation.cancelling}
                  >
                    <X size={14} />
                    {"cancelling" in operation && operation.cancelling
                      ? "Cancelling…"
                      : "Choose another"}
                  </button>
                )}
            </div>
          </section>
        )}

        {content._tag === "Chooser" && operation === null && (
          <section className="onboarding-choices">
            {content.options.map((option) => {
              const configurationId = Option.getOrNull(
                localModelConfigurationId(option.model)
              )
              return (
                <button
                  type="button"
                  className="onboarding-choice"
                  key={option.id}
                  onClick={() =>
                    configurationId && setup.select(configurationId)
                  }
                  disabled={configurationId === null}
                >
                  <div className="model-card-header">
                    <div>
                      {option.recommendations[0] && (
                        <span className="recommendation-badge">
                          {intentLabel(option.recommendations[0].intent)}
                        </span>
                      )}
                      <h3>{formatLocalModelDisplayName(option.model)}</h3>
                      <p>{option.model.presentation.description}</p>
                    </div>
                    <span
                      className={`status-pill ${
                        option.kind === "running"
                          ? "success"
                          : option.kind === "stored"
                          ? "neutral"
                          : "progress"
                      }`}
                    >
                      {option.kind === "running"
                        ? "Ready"
                        : option.kind === "stored"
                        ? "Installed"
                        : "Download"}
                    </span>
                  </div>
                  {option.recommendations[0] && (
                    <p className="recommendation-copy">
                      {option.recommendations[0].explanation}
                    </p>
                  )}
                  <ModelSummary model={option.model} />
                </button>
              )
            })}
            {content.options.length === 0 && (
              <div className="empty-panel">
                No compatible model choices are available yet.
              </div>
            )}
          </section>
        )}

        <footer className="onboarding-footer">
          <span>Model files and inference stay on this machine.</span>
          {content._tag !== "Closing" && operation === null && (
            <button type="button" className="text-button" onClick={setup.exit}>
              {state.exitKind === "Skip"
                ? "Continue without loading a model"
                : "Close"}
            </button>
          )}
        </footer>
      </div>
    </main>
  )
}
