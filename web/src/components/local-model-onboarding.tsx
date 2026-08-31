import { Fragment, type ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { AlertTriangle, Cpu, HardDrive, Loader2, X } from "lucide-react"
import {
  formatLocalModelDisplayName,
  onboardingModelSetupNoticeMessage,
  rankedLocalModelOptions,
  LOCAL_MODEL_RANKING_SCALE_INTERVALS,
  LOCAL_MODEL_RANKING_SCALE_LABELS,
  LOCAL_MODEL_RANKING_SCALE_VALUES,
  localModelSpeculativeMethodLabel,
  localModelStorageBytes,
  localModelRankingScaleIndex,
  modelDownloadFailureMessage,
  targetPhysicalMemoryBytes,
  type useOnboardingModelSetup,
} from "@magnitudedev/client-common"
import {
  acquisitionFailure,
  acquisitionProgress,
  type LocalModel,
} from "@magnitudedev/sdk"
import {
  formatBytes,
  formatContext,
  modelContextLength,
  transferLabel,
  transferProgress,
} from "./local-inference-format"
type Setup = ReturnType<typeof useOnboardingModelSetup>

const ModelSummary = ({ model }: { model: LocalModel }): ReactNode => {
  const context = modelContextLength(model)
  const speculativeMethod = Option.getOrNull(localModelSpeculativeMethodLabel(model))
  const storageBytes = Option.getOrNull(localModelStorageBytes(model))
  return (
    <div className="model-meta-row">
      <span>
        <HardDrive size={13} />
        {storageBytes === null ? "Size pending" : formatBytes(storageBytes)}
      </span>
      {context !== null && <span>{formatContext(context)} context</span>}
      {speculativeMethod !== null && <span>{speculativeMethod} speculative decoding</span>}
    </div>
  )
}
export function LocalModelOnboarding({
  setup,
}: {
  readonly setup: Setup
}): ReactNode {
  const hardware = Result.isSuccess(setup.hardware) ? setup.hardware.value : null
  const state = Option.getOrNull(Result.value(setup.view))
  if (state === null || state._tag === "Closed") return null
  const notice = Option.match(state.notice, {
    onNone: () => null,
    onSome: onboardingModelSetupNoticeMessage,
  })
  const content = state.content
  const operation =
    content._tag === "Chooser" ? Option.getOrNull(content.operation) : null
  const operationModel = operation?.model ?? null
  const transfer = operationModel === null
    ? null
    : operationModel._tag === "Catalog"
      ? acquisitionProgress(operationModel.acquisitionState) ?? null
      : null
  const transferFailure = operationModel === null
    ? undefined
    : operationModel._tag === "Catalog"
      ? acquisitionFailure(operationModel.acquisitionState)
      : undefined
  const operationFailure =
    (operation?._tag === "Loading" && operation.status._tag === "Failed"
      ? operation.status.failure.message
      : null) ??
    (transferFailure !== undefined
      ? modelDownloadFailureMessage(transferFailure)
      : null)
  const rankedOptions = content._tag === "Chooser" && hardware !== null
      ? rankedLocalModelOptions(content.options, {
        fastToSmart: content.rankingControls.fastToSmart,
        memoryBudgetBytes: targetPhysicalMemoryBytes(hardware),
      }).map((option) => ({ ...option, id: `ranked:${option.id}` }))
    : []
  const installedOptions = content._tag === "Chooser"
    ? content.options.filter(({ kind }) => kind === "running" || kind === "stored")
    : []
  const chooserOptions = [...rankedOptions, ...installedOptions]
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
              <Loader2
                className="mt-0.5 shrink-0 animate-spin text-blue-700 motion-reduce:animate-none dark:text-blue-500"
                size={20}
              />
              <div className="min-w-0 flex-1">
                <span className="block text-[10px] font-semibold uppercase tracking-[0.09em] text-slate-500">
                  Preparing local models
                </span>
                <h2 className="mt-1 text-[17px] font-semibold leading-6 text-slate-900 dark:text-slate-200">
                  Reconciling local models
                </h2>
                <p className="mt-1 text-[12px] leading-5 text-slate-600 dark:text-slate-400">
                  Checking the model catalog and installed Hugging Face cache entries.
                </p>
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

        {content._tag === "Chooser" && hardware !== null && (
          <section className="mt-[22px] grid gap-4 rounded-[10px] border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-850">
            <label className="grid gap-2 text-xs font-semibold text-slate-700 dark:text-slate-300">
              <span className="grid grid-cols-5 font-normal text-slate-500 dark:text-slate-400">
                {LOCAL_MODEL_RANKING_SCALE_LABELS.map((label, index) => (
                  <span key={label} className={index === 0 ? "text-left" : index === LOCAL_MODEL_RANKING_SCALE_INTERVALS ? "text-right" : "text-center"}>{label}</span>
                ))}
              </span>
              <input
                aria-label="Fast to Smart"
                type="range"
                min={0}
                max={LOCAL_MODEL_RANKING_SCALE_INTERVALS}
                step={1}
                list="local-model-ranking-scale"
                disabled={operation !== null}
                value={localModelRankingScaleIndex(content.rankingControls.fastToSmart)}
                onChange={(event) => setup.setRankingControls({
                  ...content.rankingControls,
                  fastToSmart: LOCAL_MODEL_RANKING_SCALE_VALUES[Number(event.currentTarget.value)]!,
                })}
              />
              <datalist id="local-model-ranking-scale">
                {LOCAL_MODEL_RANKING_SCALE_LABELS.map((label, index) => <option key={label} value={index} label={label} />)}
              </datalist>
            </label>
          </section>
        )}

        {operationModel && operation && (
          <section className="mt-[26px] flex items-center gap-[15px] rounded-[11px] border border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-5 text-blue-700 dark:text-blue-500 max-[760px]:flex-col max-[760px]:items-start [&>div:nth-child(2)]:min-w-0 [&>div:nth-child(2)]:flex-1 [&_h2]:mb-1 [&_h2]:text-[17px] [&_h2]:text-slate-900 dark:[&_h2]:text-slate-200 [&_p]:text-xs [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
            {operation._tag === "Loading" && operation.status._tag === "Failed" ? (
              <AlertTriangle size={24} />
            ) : (
              <Loader2
                className={
                  operation._tag === "Loading" && operation.status._tag === "Ready"
                    ? ""
                    : "animate-spin"
                }
                size={24}
              />
            )}
            <div>
              <span className="block text-slate-500 font-sans text-[10px] font-[650] leading-[1.2] tracking-[.09em] uppercase mb-[5px]">
                {operation._tag === "Loading"
                  ? operation.status._tag
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
              {operation._tag === "Loading" && operation.status._tag === "Failed" && (
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
                  operation._tag === "Loading" && operation.status._tag === "Ready"
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
          <section className="mt-[22px]">
            <div className="grid grid-cols-2 gap-[11px] max-[760px]:grid-cols-1">
              {rankedOptions.length > 0 && (
                <h2 className="col-span-full text-[11px] font-semibold tracking-[.08em] text-slate-500">RECOMMENDED MODELS</h2>
              )}
              {chooserOptions.map((option) => {
                const modelId = option.model.modelId
                return (
                  <Fragment key={option.id}>
                    {option === installedOptions[0] && (
                      <h2 className="col-span-full mt-2 text-[11px] font-semibold tracking-[.08em] text-slate-500">ON THIS COMPUTER</h2>
                    )}
                    <Button variant="unstyled" size="unstyled"
                  type="button"
                  className="appearance-none rounded-[10px] border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-850 p-[17px] text-left cursor-pointer font-[inherit] hover:border-blue-400 hover:bg-slate-100 dark:hover:border-blue-600 dark:hover:bg-slate-800 [&_h3]:mb-[5px] [&_h3]:text-[15px] [&_h3]:leading-[1.3] [&_h3]:text-slate-900 dark:[&_h3]:text-slate-200 [&_p]:my-1.5 [&_p]:text-[12.5px] [&_p]:leading-normal [&_p]:text-slate-600 dark:[&_p]:text-slate-400"
                  onClick={() =>
                    setup.select(modelId)
                  }
                >
                  <div className="model-card-header">
                    <div>
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
                      {option.kind === "running" ? "Loaded" : option.kind === "stored" ? "Load" : "Download"}
                    </span>
                  </div>
                  <ModelSummary model={option.model} />
                    </Button>
                  </Fragment>
                )
              })}
              {chooserOptions.length === 0 && (
                <div className="rounded-[10px] border border-dashed border-slate-300 dark:border-slate-750 bg-white dark:bg-slate-850 p-[26px] text-center text-[13px] text-slate-500">
                  No compatible model choices are available yet.
                </div>
              )}
            </div>
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
