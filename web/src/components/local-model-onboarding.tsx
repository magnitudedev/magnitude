import type { ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { AlertTriangle, Cpu, HardDrive, Loader2, X } from "lucide-react"
import {
  formatLocalModelDisplayName,
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

const formatDurationMs = (durationMs: number): string =>
  durationMs < 1_000
    ? `${(durationMs / 1_000).toFixed(1)}s`
    : durationMs < 60_000
      ? `${Math.round(durationMs / 1_000)}s`
      : `${Math.floor(durationMs / 60_000)}m ${Math.round(
          (durationMs % 60_000) / 1_000
        )}s`

const preparationLabel = (
  step: LocalModelRecommendationProgressStep,
  completed: boolean
): string => {
  if (step.id === "hardware") {
    return completed ? "Detected hardware" : "Detecting hardware"
  }
  if (step.id === "inventory") {
    if (!completed) return "Checking downloaded models"
    const count = Option.getOrElse(step.completedItems, () => 0)
    return `Found ${count} downloaded ${count === 1 ? "model" : "models"}`
  }
  if (step.id === "assessment") {
    if (!completed) return "Assessing models for this machine"
    const count = Option.getOrElse(step.completedItems, () => 0)
    return `Assessed ${count} models for this machine`
  }
  if (!completed) return "Preparing recommendations"
  const count = Option.getOrElse(step.completedItems, () => 0)
  return `Prepared ${count} recommendations`
}

const preparationMetadata = (
  step: LocalModelRecommendationProgressStep
): string | null => {
  if (step.status._tag === "Failed") return step.status.failure.message
  if (step.status._tag === "Completed") {
    return step.id === "assessment" && !step.status.cached
      ? `Completed in ${formatDurationMs(step.status.durationMs)}`
      : null
  }
  if (step.status._tag !== "Running") return null
  const count =
    step.id === "assessment"
      ? Option.match(step.totalItems, {
          onNone: () => null,
          onSome: (total) =>
            Option.match(step.completedItems, {
              onNone: () => `${total} models`,
              onSome: (completed) => `${completed} of ${total} models`,
            }),
        })
      : null
  const estimate = Option.match(step.estimatedRemainingMs, {
    onNone: () => null,
    onSome: (remainingMs) =>
      `About ${formatDurationMs(remainingMs)} remaining`,
  })
  return [count, estimate].filter((value) => value !== null).join(" · ") || null
}

const preparationFraction = (
  step: LocalModelRecommendationProgressStep
): number | null => {
  if (step.status._tag !== "Running") return null
  return Option.match(
    Option.all({
      completed: step.completedItems,
      total: step.totalItems,
    }),
    {
      onNone: () => null,
      onSome: ({ completed, total }) =>
        total <= 0 ? null : Math.max(0, Math.min(1, completed / total)),
    }
  )
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
  const preparationStep =
    content._tag === "Preparation"
      ? content.progress.find((step) => step.status._tag === "Failed") ??
        content.progress.find((step) => step.status._tag === "Running") ??
        [...content.progress]
          .reverse()
          .find((step) => step.status._tag === "Completed") ??
        null
      : null
  const preparationProgress =
    preparationStep === null ? null : preparationFraction(preparationStep)
  const completedPreparation =
    content._tag === "Preparation"
      ? content.progress.filter((step) => step.status._tag === "Completed")
      : []
  return (
    <main className="flex min-h-screen justify-center overflow-auto bg-slate-50 dark:bg-slate-925 text-slate-900 dark:text-slate-200">
      <div className="box-border w-[min(980px,100%)] px-[clamp(18px,5vw,56px)] py-[clamp(28px,6vh,70px)]">
        <header className="flex max-w-[760px] items-start gap-4 [&_h1]:mb-[9px] [&_h1]:text-[clamp(24px,4vw,34px)] [&_h1]:tracking-[-.025em] [&_p]:text-sm [&_p]:leading-[1.55] [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
          <div className="grid size-[50px] shrink-0 place-items-center rounded-[13px] border border-blue-200 bg-blue-100 text-blue-700 dark:border-blue-800 dark:bg-blue-900 dark:text-blue-400">
            <Cpu size={26} />
          </div>
          <div>
            <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
              Local inference setup
            </span>
            <h1>Choose the model that powers Magnitude</h1>
            <p>
              Everything runs on this machine. Magnitude assessed these
              configurations against your actual hardware.
            </p>
          </div>
        </header>

        {hardware && (
          <div className="my-6 flex flex-wrap items-center gap-x-4 gap-y-[7px] rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 px-[13px] py-2.5 text-slate-600 dark:text-slate-400 text-[11px]">
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
          <div className="flex items-center gap-2 rounded-[7px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 px-3 py-2.5 text-slate-600 dark:text-slate-400 text-xs [&.danger]:border-red-300 [&.danger]:text-red-600 dark:[&.danger]:border-red-700 dark:[&.danger]:text-red-400 danger">
            <AlertTriangle size={16} />
            {notice}
          </div>
        )}
        {Result.isFailure(setup.hardware) && (
          <div className="flex items-center gap-2 rounded-[7px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 px-3 py-2.5 text-slate-600 dark:text-slate-400 text-xs [&.danger]:border-red-300 [&.danger]:text-red-600 dark:[&.danger]:border-red-700 dark:[&.danger]:text-red-400 danger">
            <AlertTriangle size={16} />
            Hardware details are unavailable.
          </div>
        )}

        {content._tag === "Preparation" && (
          <section
            aria-live="polite"
            className="mt-7 max-w-[720px] border-t border-slate-200 pt-6 dark:border-slate-800"
          >
            <div className="flex items-start gap-3.5">
              {content.discoveryFailure || preparationStep?.status._tag === "Failed" ? (
                <AlertTriangle
                  className="mt-0.5 shrink-0 text-red-600 dark:text-red-400"
                  size={20}
                />
              ) : (
                <Loader2
                  className="mt-0.5 shrink-0 animate-spin text-blue-700 motion-reduce:animate-none dark:text-blue-500"
                  size={20}
                />
              )}
              <div className="min-w-0 flex-1">
                <span className="block text-[10px] font-semibold uppercase tracking-[0.09em] text-slate-500">
                  Preparing local models
                </span>
                <h2 className="mt-1 text-[17px] font-semibold leading-6 text-slate-900 dark:text-slate-200">
                  {preparationStep === null
                    ? "Reading local inference state"
                    : preparationLabel(
                        preparationStep,
                        preparationStep.status._tag === "Completed"
                      )}
                </h2>
                <p className="mt-1 text-[12px] leading-5 text-slate-600 dark:text-slate-400">
                  {content.discoveryFailure?.message ??
                    (preparationStep === null
                      ? "Waiting for hardware and model inventory."
                      : preparationMetadata(preparationStep) ??
                        "Hardware detection and model assessment run locally.")}
                </p>
                {preparationStep !== null && preparationProgress !== null && (
                  <div className="mt-3 flex max-w-[480px] items-center gap-3">
                    <Progress
                      aria-label={preparationLabel(preparationStep, false)}
                      value={Math.round(preparationProgress * 100)}
                      className="min-w-0 flex-1"
                      trackClassName="h-1.5 rounded-full bg-slate-200 dark:bg-slate-800"
                      indicatorClassName="rounded-full bg-blue-700 duration-150 motion-reduce:transition-none dark:bg-blue-500"
                    />
                    <span className="w-9 text-right font-mono text-[11px] tabular-nums text-slate-500">
                      {Math.round(preparationProgress * 100)}%
                    </span>
                  </div>
                )}
                {completedPreparation.length > 0 &&
                  preparationStep?.status._tag !== "Completed" && (
                    <p className="mt-3 text-[11px] leading-5 text-slate-500">
                      {completedPreparation
                        .map((step) => preparationLabel(step, true))
                        .join(" · ")}
                    </p>
                  )}
              </div>
            </div>
          </section>
        )}

        {content._tag === "Closing" && (
          <section className="mt-[26px] flex items-center gap-[15px] rounded-[11px] border border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-5 text-blue-700 dark:text-blue-500 max-[760px]:flex-col max-[760px]:items-start [&>div:nth-child(2)]:min-w-0 [&>div:nth-child(2)]:flex-1 [&_h2]:mb-1 [&_h2]:text-[17px] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
            <Loader2 className="animate-spin" size={24} />
            <div>
              <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                Finishing
              </span>
              <h2>Saving setup</h2>
              <p>Preparing the workspace…</p>
            </div>
          </section>
        )}

        {operationModel && operation && (
          <section className="mt-[26px] flex items-center gap-[15px] rounded-[11px] border border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-5 text-blue-700 dark:text-blue-500 max-[760px]:flex-col max-[760px]:items-start [&>div:nth-child(2)]:min-w-0 [&>div:nth-child(2)]:flex-1 [&_h2]:mb-1 [&_h2]:text-[17px] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
            {operation._tag === "Loading" && operation.phase === "Failed" ? (
              <AlertTriangle size={24} />
            ) : (
              <Loader2
                className={
                  operation._tag === "Loading" && operation.phase === "Ready"
                    ? ""
                    : "animate-spin"
                }
                size={24}
              />
            )}
            <div>
              <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
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
                <Button variant="unstyled" size="unstyled"
                  type="button"
                  className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-blue-700 text-slate-50 hover:bg-blue-800 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400"
                  onClick={() => setup.select(operation.modelId)}
                >
                  Retry
                </Button>
              )}
              {operation._tag !== "Completing" &&
                !(
                  operation._tag === "Loading" && operation.phase === "Ready"
                ) && (
                  <Button variant="unstyled" size="unstyled"
                    type="button"
                    className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-200 border border-slate-300 dark:border-slate-750 hover:bg-slate-150 dark:hover:bg-slate-750"
                    onClick={setup.cancel}
                    disabled={"cancelling" in operation && operation.cancelling}
                  >
                    <X size={14} />
                    {"cancelling" in operation && operation.cancelling
                      ? "Cancelling…"
                      : "Choose another"}
                  </Button>
                )}
            </div>
          </section>
        )}

        {content._tag === "Chooser" && operation === null && (
          <section className="mt-[22px] grid grid-cols-2 gap-[11px] max-[760px]:grid-cols-1">
            {content.options.map((option) => {
              const modelId = option.model.modelId
              return (
                <Button variant="unstyled" size="unstyled"
                  type="button"
                  className="appearance-none rounded-[10px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 p-[17px] text-left cursor-pointer font-[inherit] hover:border-blue-400 hover:bg-slate-100 dark:hover:border-blue-600 dark:hover:bg-slate-800 [&_h3]:mb-[5px] [&_h3]:text-[15px] [&_h3]:leading-[1.3] [&_h3]:text-slate-900 dark:[&_h3]:text-slate-200 [&_p]:my-1.5 [&_p]:text-[12.5px] [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400"
                  key={option.id}
                  onClick={() =>
                    setup.select(modelId)
                  }
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
                </Button>
              )
            })}
            {content.options.length === 0 && (
              <div className="rounded-[10px] border border-dashed border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-[26px] text-center text-[13px] text-slate-500">
                No compatible model choices are available yet.
              </div>
            )}
          </section>
        )}

        <footer className="mt-6 flex items-center justify-between gap-4 text-[11px] text-slate-500 max-[760px]:flex-col max-[760px]:items-start">
          <span>Model files and inference stay on this machine.</span>
          {content._tag !== "Closing" && operation === null && (
            <Button variant="unstyled" size="unstyled"
              type="button"
              className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-transparent text-slate-600 dark:text-slate-400 !px-1 hover:text-slate-900 dark:hover:text-slate-200"
              onClick={setup.exit}
            >
              {state.exitKind === "Skip"
                ? "Continue without loading a model"
                : "Close"}
            </Button>
          )}
        </footer>
      </div>
    </main>
  )
}
