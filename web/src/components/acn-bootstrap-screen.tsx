import type { ReactNode } from "react"
import { AlertTriangle } from "lucide-react"
import { Option } from "effect"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Spinner } from "@/components/ui/spinner"
import type {
  AcnInstallationPhase,
  AcnLifecycleState,
  AcnStartingPhase,
} from "@magnitudedev/sdk"
import { MagnitudeMark } from "./magnitude-mark"

const INSTALLATION_PHASE_LABELS: Readonly<
  Record<AcnInstallationPhase, string>
> = {
  DownloadingDaemon: "Downloading daemon",
  DownloadingInferenceEngine: "Downloading inference engine",
  StartingMagnitude: "Starting Magnitude",
}

const STARTING_PHASE_LABELS: Readonly<
  Record<Extract<AcnStartingPhase, string>, string>
> = {
  PreparingAcn: "Preparing background server",
  WaitingForOwner: "Waiting for previous Magnitude process",
  ResolvingLocalInference: "Preparing local inference",
  LaunchingLocalInference: "Starting local inference",
  DiscoveringLocalModels: "Discovering local models",
}

const backendLabel = (
  backend: Extract<AcnStartingPhase, { readonly _tag: "PreparingBackend" }>[
    "backend"
  ]
): string =>
  ({
    Cpu: "CPU",
    Metal: "Metal",
    Cuda: "CUDA",
    Vulkan: "Vulkan",
  } as const)[backend._tag]

const startingPhaseLabel = (phase: AcnStartingPhase): string =>
  typeof phase === "string"
    ? STARTING_PHASE_LABELS[phase]
    : `Preparing ${backendLabel(phase.backend)} backend for ${phase.backend.hardwareLabel}`

const formatMebibytes = (bytes: number): string =>
  `${(bytes / (1024 * 1024)).toFixed(1)} MiB`

const failureTitle = (
  state: Extract<AcnLifecycleState, { readonly _tag: "Failed" }>
): string =>
  state.stage === "InstallDaemon" || state.stage === "PrepareLocalInference"
    ? "Magnitude failed to install"
    : "Magnitude failed to start"

export function AcnBootstrapScreen({
  state,
  onRetry,
  onQuit,
}: {
  readonly state: Exclude<AcnLifecycleState, { readonly _tag: "Ready" }>
  readonly onRetry: () => void
  readonly onQuit?: () => void
}): ReactNode {
  if (state._tag === "Failed") {
    return (
      <main className="flex min-h-screen items-center justify-center bg-slate-50 px-6 py-10 text-slate-900 dark:bg-slate-850 dark:text-slate-200">
        <section
          role="alert"
          aria-labelledby="acn-bootstrap-title"
          className="w-full max-w-[640px] text-center"
        >
          <MagnitudeMark className="mx-auto mb-6 h-auto w-[82px]" />
          <h1
            id="acn-bootstrap-title"
            className="text-[28px] font-semibold leading-tight tracking-[-0.025em]"
          >
            {failureTitle(state)}
          </h1>
          <div className="mx-auto mt-4 flex max-w-[540px] items-start justify-center gap-2.5">
            <AlertTriangle
              aria-hidden="true"
              className="mt-0.5 shrink-0 text-red-600 dark:text-red-400"
              size={18}
              strokeWidth={1.8}
            />
            <p className="text-left text-[14px] leading-5 text-slate-600 dark:text-slate-400">
              {state.message}
            </p>
          </div>
          <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
            {state.retryable && (
              <Button variant="unstyled" size="unstyled"
                type="button"
                className="min-h-8 rounded-[7px] bg-blue-700 px-3 text-xs font-semibold text-slate-50 hover:bg-blue-800 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400 dark:focus-visible:outline-blue-500"
                onClick={onRetry}
              >
                Retry
              </Button>
            )}
            {onQuit !== undefined && (
              <Button variant="unstyled" size="unstyled"
                type="button"
                className="min-h-8 rounded-[7px] border border-slate-300 bg-transparent px-3 text-xs font-semibold text-slate-600 hover:bg-slate-100 hover:text-slate-900 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:border-slate-750 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200 dark:focus-visible:outline-blue-500"
                onClick={onQuit}
              >
                Quit
              </Button>
            )}
          </div>
        </section>
      </main>
    )
  }

  const installing = state._tag === "Installing"
  const percentage = installing
    ? Math.floor(Math.max(0, Math.min(1, state.overallProgress)) * 100)
    : null
  const detail = installing
    && state.phase !== "StartingMagnitude"
    && state.detailIsExact
    && Option.isSome(state.detail)
    && state.detail.value.unit === "Bytes"
    ? `${formatMebibytes(state.detail.value.completed)} of ${formatMebibytes(
        state.detail.value.totalBytes
      )}`
    : null
  const phase = state._tag === "Checking"
    ? "Checking background server"
    : installing
      ? INSTALLATION_PHASE_LABELS[state.phase]
      : startingPhaseLabel(state.phase)

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-50 px-6 py-10 text-slate-900 dark:bg-slate-850 dark:text-slate-200">
      <section
        aria-labelledby="acn-bootstrap-title"
        aria-describedby="acn-bootstrap-phase"
        aria-live="polite"
        className="w-full max-w-[640px] text-center"
      >
        <MagnitudeMark className="mx-auto mb-6 h-auto w-[82px]" />
        <h1
          id="acn-bootstrap-title"
          className="text-[30px] font-semibold leading-tight tracking-[-0.025em]"
        >
          {installing ? "Installing Magnitude" : "Starting Magnitude"}
        </h1>
        {percentage === null ? (
          <div className="mt-5 flex items-center justify-center gap-2.5 text-[16px] leading-7 text-slate-600 dark:text-slate-300">
            <Spinner
              aria-hidden="true"
              aria-label={undefined}
              className="size-[17px] text-blue-700 motion-reduce:animate-none dark:text-blue-500"
            />
            <p id="acn-bootstrap-phase">{phase}</p>
          </div>
        ) : (
          <div className="mx-auto mt-7 w-full max-w-[520px] text-left">
            <div className="mb-2.5 flex items-baseline justify-between gap-5">
              <p
                id="acn-bootstrap-phase"
                className="text-[14px] leading-5 text-slate-600 dark:text-slate-400"
              >
                {phase}
              </p>
              <span className="shrink-0 font-mono text-[14px] font-medium tabular-nums text-slate-700 dark:text-slate-300">
                {percentage}%
              </span>
            </div>
            <div>
              <Progress
                aria-label="Magnitude installation progress"
                value={percentage}
                className="block w-full"
                trackClassName="h-1.5 rounded-full bg-slate-200 dark:bg-slate-800"
                indicatorClassName="rounded-full bg-blue-700 duration-150 motion-reduce:transition-none dark:bg-blue-500"
              />
            </div>
            {detail !== null && (
              <p className="mt-2.5 font-mono text-[12px] tabular-nums text-slate-500 dark:text-slate-500">
                {detail}
              </p>
            )}
          </div>
        )}
      </section>
    </main>
  )
}
