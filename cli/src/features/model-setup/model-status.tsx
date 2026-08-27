import { useCallback, useState, type ReactNode } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Option } from "effect"
import {
  formatLocalModelDisplayName,
  formatMemorySize,
  formatStorageSize,
  formatTransferRate,
  truncateToDisplayWidth,
  type OnboardingModelLoadStatus,
} from "@magnitudedev/client-common"
import { acquisitionProgress, type LocalModel, type ProviderModelId } from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import { ShimmerText } from "../../components/shimmer-text"
import { useTheme } from "../../hooks/use-theme"

const formatEta = (remainingBytes: number, bytesPerSecond: number): string => {
  const minutes = Math.max(1, Math.ceil(remainingBytes / bytesPerSecond / 60))
  return `about ${minutes} min remaining`
}

const formatDownloadSize = (bytes: number): string =>
  formatStorageSize(bytes, {
    minimumFractionDigits: { digits: 2, fromUnit: "GB" },
  })

const progressBar = (fraction: number, width: number): string => {
  const filled = Math.round(Math.max(0, Math.min(1, fraction)) * width)
  return `${"█".repeat(filled)}${"░".repeat(Math.max(0, width - filled))}`
}

const ModelOperationProgressBar = ({
  progress,
  width,
  tone = "active",
}: {
  readonly progress: Option.Option<number>
  readonly width: number
  readonly tone?: "active" | "failed"
}): ReactNode => {
  const theme = useTheme()
  if (tone === "failed") {
    return (
      <text style={{ width, fg: theme.status.failure }} wrapMode="none">
        {progressBar(1, width)}
      </text>
    )
  }
  const percentageLabel = Option.match(progress, {
    onNone: () => "0%",
    onSome: (fraction) => `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`,
  })
  const barWidth = Math.max(8, width - 6)
  return (
    <text style={{ width }} wrapMode="none">
      <span style={{ fg: theme.accent }}>
        {progressBar(Option.getOrElse(progress, () => 0), barWidth)}
      </span>
      <span style={{ fg: theme.text.supporting }}>{`  ${percentageLabel.padStart(4)}`}</span>
    </text>
  )
}

const ModelOperationStatusText = ({
  text,
  width,
}: {
  readonly text: string
  readonly width: number
}): ReactNode => {
  const theme = useTheme()
  return (
    <text style={{ width }} wrapMode="none">
      <ShimmerText
        text={text}
        baseColor={theme.text.supporting}
        highlightColor={theme.text.emphasized}
      />
    </text>
  )
}

type ConfirmationChoice = "yes" | "no"

export interface OnboardingModelDownloadOperation {
  readonly starting: boolean
  readonly cancelling: boolean
  readonly onCancel: () => void
}

export const ONBOARDING_MODEL_OPERATION_ROWS = 3

export function OnboardingModelConfiguringProgress({
  width,
}: {
  readonly width: number
}): ReactNode {
  return (
    <box style={{
      width,
      height: ONBOARDING_MODEL_OPERATION_ROWS,
      minHeight: ONBOARDING_MODEL_OPERATION_ROWS,
      maxHeight: ONBOARDING_MODEL_OPERATION_ROWS,
      flexDirection: "column",
      flexShrink: 0,
      overflow: "hidden",
    }}>
      <ModelOperationStatusText text="Configuring model…" width={width} />
    </box>
  )
}

export function OnboardingModelDownloadProgress({
  model,
  width,
  operation,
}: {
  readonly model: LocalModel
  readonly width: number
  readonly operation: OnboardingModelDownloadOperation
}): ReactNode {
  const theme = useTheme()
  const [confirmationModelId, setConfirmationModelId] = useState<ProviderModelId | null>(null)
  const [choice, setChoice] = useState<ConfirmationChoice>("yes")
  const [hovered, setHovered] = useState<string | null>(null)
  const starting = operation.starting
  const activeDownload = starting ? null : acquisitionProgress(model.acquisitionState) ?? null
  const downloading = activeDownload !== null
  const active = starting || downloading
  const cancelling = operation.cancelling
  const cancelable = active && !cancelling
  const confirming = cancelable && confirmationModelId === model.modelId
  const totalBytes = activeDownload?.totalBytes ?? model.downloadBytes
  const progress = activeDownload !== null
    ? Option.some(activeDownload.completedBytes / Math.max(1, activeDownload.totalBytes))
    : Option.none<number>()
  const rate = activeDownload === null ? null : Option.getOrNull(activeDownload.bytesPerSecond)
  const transferred = activeDownload === null
    ? formatDownloadSize(totalBytes)
    : `${formatDownloadSize(activeDownload.completedBytes)} / ${formatDownloadSize(activeDownload.totalBytes)}`
  const status = cancelling
    ? `Cancelling download · ${transferred}`
    : starting
      ? `Starting download · ${formatDownloadSize(totalBytes)}`
      : activeDownload?.stage === "verifying" || activeDownload?.stage === "publishing"
        ? `Verifying download · ${transferred}`
        : downloading
          ? rate === null || rate <= 0
            ? `Downloading · ${transferred} · Estimating…`
            : `Downloading · ${transferred} · ${formatTransferRate(rate)} · ${formatEta(activeDownload.totalBytes - activeDownload.completedBytes, rate)}`
          : null
  const declineCancellation = useCallback(() => {
    setConfirmationModelId(null)
    setChoice("yes")
  }, [])

  const confirmCancellation = useCallback(() => {
    if (!cancelable) return
    setConfirmationModelId(null)
    setChoice("yes")
    operation.onCancel()
  }, [cancelable, operation])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (cancelling) return
    if (!confirming) {
      if (key.name === "escape" && cancelable) {
        key.preventDefault()
        setChoice("yes")
        setConfirmationModelId(model.modelId)
      }
      return
    }
    if (key.name === "escape") {
      key.preventDefault()
      declineCancellation()
      return
    }
    if (key.name === "left" || key.name === "right") {
      key.preventDefault()
      setChoice((current) => current === "yes" ? "no" : "yes")
      return
    }
    if (key.name === "return" || key.name === "enter") {
      key.preventDefault()
      if (choice === "yes") confirmCancellation()
      else declineCancellation()
    }
  }, [model.modelId, cancelable, cancelling, choice, confirmCancellation, confirming, declineCancellation]))

  if (!active || status === null) return null

  const choiceButton = (value: ConfirmationChoice, label: string) => (
    <Button
      onClick={() => value === "yes" ? confirmCancellation() : declineCancellation()}
      onMouseOver={() => { setChoice(value); setHovered(value) }}
      onMouseOut={() => setHovered((current) => current === value ? null : current)}
    >
      <text style={{
        fg: value === "yes"
          ? choice === value || hovered === value ? theme.status.failure : theme.text.body
          : choice === value || hovered === value ? theme.accent : theme.text.body,
      }} attributes={choice === value ? TextAttributes.BOLD : TextAttributes.NONE}>
        {choice === value ? "› " : "  "}{label}
      </text>
    </Button>
  )

  return (
    <box style={{
      width,
      height: ONBOARDING_MODEL_OPERATION_ROWS,
      minHeight: ONBOARDING_MODEL_OPERATION_ROWS,
      maxHeight: ONBOARDING_MODEL_OPERATION_ROWS,
      flexDirection: "column",
      flexShrink: 0,
      overflow: "hidden",
    }}>
      <ModelOperationStatusText text={status} width={width} />
      <ModelOperationProgressBar progress={progress} width={width} />
      <box style={{
        width,
        height: 1,
        minHeight: 1,
        maxHeight: 1,
        flexDirection: "row",
        flexShrink: 0,
        overflow: "hidden",
      }}>
        {confirming ? (
          <>
            <text style={{ fg: theme.text.body, flexShrink: 0 }} wrapMode="none">
              Cancel download?
            </text>
            <box style={{ width: 2, flexShrink: 0 }} />
            {choiceButton("yes", "Yes")}
            <box style={{ width: 2, flexShrink: 0 }} />
            {choiceButton("no", "No")}
          </>
        ) : (
          <>
            {cancelable && (
              <Button
                onClick={() => {
                  setChoice("yes")
                  setConfirmationModelId(model.modelId)
                }}
                onMouseOver={() => setHovered("cancel")}
                onMouseOut={() => setHovered((current) => current === "cancel" ? null : current)}
              >
                <text style={{ fg: hovered === "cancel" ? theme.status.failure : theme.text.supporting }}>
                  Cancel (Esc)
                </text>
              </Button>
            )}
          </>
        )}
      </box>
    </box>
  )
}

const statusWithModel = (
  prefix: string,
  modelName: string,
  suffix: string,
  width: number,
): string => `${prefix}${truncateToDisplayWidth(
  modelName,
  Math.max(1, width - prefix.length - suffix.length),
)}${suffix}`

const loadingStatusLabel = (
  status: Exclude<OnboardingModelLoadStatus, { readonly _tag: "Failed" }>,
  modelName: string,
  width: number,
): string => {
  switch (status._tag) {
    case "Preparing": return statusWithModel("Preparing ", modelName, "…", width)
    case "Cancelling": return statusWithModel("Cancelling the load of ", modelName, "…", width)
    case "Stopping": return statusWithModel(
      "Unloading the current model before loading ",
      modelName,
      "…",
      width,
    )
    case "Ready": return statusWithModel("Finishing setup for ", modelName, "…", width)
    case "Loading": {
      switch (status.stage) {
        case "queued": return statusWithModel("Waiting to load ", modelName, "…", width)
        case "resolving": return statusWithModel("Resolving ", modelName, "…", width)
        case "unloading": return statusWithModel(
          "Unloading the current model before loading ",
          modelName,
          "…",
          width,
        )
        case "loading": return statusWithModel("Loading ", modelName, " into memory…", width)
        case "verifying": return statusWithModel("Verifying ", modelName, "…", width)
      }
    }
  }
}

const loadingFailureLabel = (
  status: Extract<OnboardingModelLoadStatus, { readonly _tag: "Failed" }>,
  modelName: string,
  width: number,
): string => "_tag" in status.failure && status.failure._tag === "LowMemory"
  ? statusWithModel(
      "Not enough memory for ",
      modelName,
      ` · Free at least ${formatMemorySize(status.failure.minimumAdditionalAvailableBytes, { rounding: "up" })}`,
      width,
    )
  : statusWithModel(
      "Couldn’t load ",
      modelName,
      ` · ${status.failure.message}`,
      width,
    )

export function OnboardingModelLoadProgress({
  model,
  status,
  width,
  onCancel,
  onRetry,
  onChooseAnother,
}: {
  readonly model: LocalModel
  readonly status: OnboardingModelLoadStatus
  readonly width: number
  readonly onCancel: () => void
  readonly onRetry: () => void
  readonly onChooseAnother: () => void
}): ReactNode {
  const theme = useTheme()
  const modelName = formatLocalModelDisplayName(model)
  const [hovered, setHovered] = useState<"retry" | "choose" | "cancel" | ConfirmationChoice | null>(null)
  const [confirmingCancellation, setConfirmingCancellation] = useState(false)
  const [cancellationChoice, setCancellationChoice] = useState<ConfirmationChoice>("yes")
  const [failureChoice, setFailureChoice] = useState<"retry" | "choose">("retry")
  const failureRetryable = status._tag === "Failed" && status.failure.retryable
  const selectedFailureChoice = failureRetryable ? failureChoice : "choose"
  const cancelable = status._tag !== "Cancelling"
    && status._tag !== "Ready"
    && status._tag !== "Failed"
  const declineCancellation = useCallback(() => {
    setConfirmingCancellation(false)
    setCancellationChoice("yes")
  }, [])
  const confirmCancellation = useCallback(() => {
    if (!cancelable) return
    setConfirmingCancellation(false)
    setCancellationChoice("yes")
    onCancel()
  }, [cancelable, onCancel])
  useKeyboard(useCallback((key: KeyEvent) => {
    if (status._tag === "Failed") {
      if ((key.name === "left" || key.name === "right") && failureRetryable) {
        key.preventDefault()
        setFailureChoice((current) => current === "retry" ? "choose" : "retry")
        return
      }
      if (key.name === "return" || key.name === "enter") {
        key.preventDefault()
        if (selectedFailureChoice === "retry") onRetry()
        else onChooseAnother()
      }
      return
    }
    if (!cancelable) return
    if (!confirmingCancellation) {
      if (key.name === "escape") {
        key.preventDefault()
        setCancellationChoice("yes")
        setConfirmingCancellation(true)
      }
      return
    }
    if (key.name === "escape") {
      key.preventDefault()
      declineCancellation()
      return
    }
    if (key.name === "left" || key.name === "right") {
      key.preventDefault()
      setCancellationChoice((current) => current === "yes" ? "no" : "yes")
      return
    }
    if (key.name === "return" || key.name === "enter") {
      key.preventDefault()
      if (cancellationChoice === "yes") confirmCancellation()
      else declineCancellation()
    }
  }, [
    cancelable,
    cancellationChoice,
    confirmCancellation,
    confirmingCancellation,
    declineCancellation,
    failureRetryable,
    onChooseAnother,
    onRetry,
    selectedFailureChoice,
    status._tag,
  ]))
  const progress = status._tag === "Loading"
    ? status.progress
    : status._tag === "Ready"
      ? Option.some(1)
      : Option.none<number>()
  const cancellationChoiceButton = (value: ConfirmationChoice, label: string) => (
    <Button
      onClick={() => value === "yes" ? confirmCancellation() : declineCancellation()}
      onMouseOver={() => setHovered(value)}
      onMouseOut={() => setHovered(null)}
    >
      <text style={{
        fg: value === "yes"
          ? cancellationChoice === value || hovered === value ? theme.status.failure : theme.text.body
          : cancellationChoice === value || hovered === value ? theme.accent : theme.text.body,
      }} attributes={cancellationChoice === value ? TextAttributes.BOLD : TextAttributes.NONE}>
        {cancellationChoice === value ? "› " : "  "}{label}
      </text>
    </Button>
  )
  return (
    <box style={{
      width,
      height: ONBOARDING_MODEL_OPERATION_ROWS,
      minHeight: ONBOARDING_MODEL_OPERATION_ROWS,
      maxHeight: ONBOARDING_MODEL_OPERATION_ROWS,
      flexDirection: "column",
      flexShrink: 0,
      overflow: "hidden",
    }}>
      {status._tag === "Failed"
        ? <text style={{ fg: theme.status.failure, width }} wrapMode="none">{loadingFailureLabel(status, modelName, width)}</text>
        : <ModelOperationStatusText text={loadingStatusLabel(status, modelName, width)} width={width} />}
      <ModelOperationProgressBar
        progress={progress}
        width={width}
        tone={status._tag === "Failed" ? "failed" : "active"}
      />
      <box style={{ width, height: 1, minHeight: 1, maxHeight: 1, flexDirection: "row", flexShrink: 0 }}>
        {status._tag === "Failed" ? (
          <>
            {status.failure.retryable && (
              <>
                <Button
                  onClick={onRetry}
                  onMouseOver={() => { setHovered("retry"); setFailureChoice("retry") }}
                  onMouseOut={() => setHovered(null)}
                >
                  <text
                    style={{ fg: selectedFailureChoice === "retry" || hovered === "retry" ? theme.accent : theme.text.body }}
                    attributes={selectedFailureChoice === "retry" ? TextAttributes.BOLD : TextAttributes.NONE}
                  >Retry loading</text>
                </Button>
                <box style={{ width: 2, flexShrink: 0 }} />
              </>
            )}
            <Button
              onClick={onChooseAnother}
              onMouseOver={() => { setHovered("choose"); setFailureChoice("choose") }}
              onMouseOut={() => setHovered(null)}
            >
              <text
                style={{ fg: selectedFailureChoice === "choose" || hovered === "choose" ? theme.accent : theme.text.body }}
                attributes={selectedFailureChoice === "choose" ? TextAttributes.BOLD : TextAttributes.NONE}
              >Choose another model</text>
            </Button>
          </>
        ) : confirmingCancellation ? (
          <>
            <text style={{ fg: theme.text.body, flexShrink: 0 }} wrapMode="none">
              Cancel loading?
            </text>
            <box style={{ width: 2, flexShrink: 0 }} />
            {cancellationChoiceButton("yes", "Yes")}
            <box style={{ width: 2, flexShrink: 0 }} />
            {cancellationChoiceButton("no", "No")}
          </>
        ) : cancelable ? (
          <Button
            onClick={() => {
              setCancellationChoice("yes")
              setConfirmingCancellation(true)
            }}
            onMouseOver={() => setHovered("cancel")}
            onMouseOut={() => setHovered(null)}
          >
            <text style={{ fg: hovered === "cancel" ? theme.status.failure : theme.text.supporting }}>
              Cancel (Esc)
            </text>
          </Button>
        ) : null}
      </box>
    </box>
  )
}
