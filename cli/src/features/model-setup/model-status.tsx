import { useCallback, useMemo, useState, type ReactNode } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Option } from "effect"
import {
  formatMemorySize,
  formatStorageSize,
  formatTransferRate,
  type OnboardingModelLoadStatus,
} from "@magnitudedev/client-common"
import { acquisitionProgress, type LocalModel, type ProviderModelId } from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import { ShimmerText } from "../../components/shimmer-text"
import { useTheme } from "../../hooks/use-theme"

const formatEta = (remainingBytes: number, bytesPerSecond: number): string => {
  const minutes = Math.max(1, Math.ceil(remainingBytes / bytesPerSecond / 60))
  return `about ${minutes} ${minutes === 1 ? "minute" : "minutes"} remaining`
}

const progressBar = (fraction: number, width: number): string => {
  const filled = Math.round(Math.max(0, Math.min(1, fraction)) * width)
  return `${"█".repeat(filled)}${"░".repeat(Math.max(0, width - filled))}`
}

const ModelOperationProgressBar = ({
  progress,
  width,
}: {
  readonly progress: Option.Option<number>
  readonly width: number
}): ReactNode => {
  const theme = useTheme()
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

export const ONBOARDING_MODEL_OPERATION_ROWS = 4

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
  const cancelable = activeDownload !== null && !cancelling
  const confirming = cancelable && confirmationModelId === model.modelId
  const totalBytes = activeDownload?.totalBytes ?? model.downloadBytes
  const progress = activeDownload !== null
    ? Option.some(activeDownload.completedBytes / Math.max(1, activeDownload.totalBytes))
    : Option.none<number>()
  const rate = activeDownload === null ? null : Option.getOrNull(activeDownload.bytesPerSecond)
  const transferDetail = useMemo(() => {
    if (starting || cancelling) return null
    if (activeDownload === null) return null
    if (activeDownload.stage === "verifying" || activeDownload.stage === "publishing") {
      return "Verifying download…"
    }
    const transferred = `${formatStorageSize(activeDownload.completedBytes)} / ${formatStorageSize(activeDownload.totalBytes)}`
    if (rate === null || rate <= 0) return `${transferred} · Estimating time remaining…`
    return `${transferred} · ${formatTransferRate(rate)} · ${formatEta(activeDownload.totalBytes - activeDownload.completedBytes, rate)}`
  }, [activeDownload, cancelling, rate, starting])
  const status = starting
    ? `Starting download (${formatStorageSize(totalBytes)})…`
    : cancelling
      ? "Cancelling download…"
      : downloading
        ? `Downloading (${formatStorageSize(totalBytes)})`
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
      <text style={{ fg: theme.text.supporting, width }} wrapMode="none">
        {transferDetail ?? ""}
      </text>
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
              Are you sure you want to cancel?
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

const loadingStatusLabel = (status: OnboardingModelLoadStatus): string => {
  switch (status._tag) {
    case "Preparing": return "Preparing model…"
    case "Stopping": return "Stopping current model…"
    case "Ready": return "Finishing setup…"
    case "Failed": return "Couldn’t load model"
    case "Loading": {
      switch (status.stage) {
        case "queued": return "Waiting to load model…"
        case "resolving": return "Preparing model…"
        case "unloading": return "Unloading current model…"
        case "loading": return "Loading model into memory…"
        case "verifying": return "Verifying model…"
      }
    }
  }
}

const loadingStatusDetail = (status: OnboardingModelLoadStatus): string => {
  if (status._tag === "Failed") {
    return "_tag" in status.failure && status.failure._tag === "LowMemory"
      ? `Not enough memory. Free ${formatMemorySize(status.failure.minimumAdditionalAvailableBytes, { rounding: "up" })} and try again.`
      : status.failure.message
  }
  if (status._tag !== "Loading") return ""
  switch (status.stage) {
    case "queued": return "Waiting for the model loader…"
    case "resolving": return "Resolving the model configuration…"
    case "unloading": return "Making room for the selected model…"
    case "loading": return "Loading model weights…"
    case "verifying": return "Checking that the model is ready…"
  }
}

export function OnboardingModelLoadProgress({
  status,
  width,
  onRetry,
  onChooseAnother,
}: {
  readonly status: OnboardingModelLoadStatus
  readonly width: number
  readonly onRetry: () => void
  readonly onChooseAnother: () => void
}): ReactNode {
  const theme = useTheme()
  const [hovered, setHovered] = useState<"retry" | "choose" | null>(null)
  const progress = status._tag === "Loading"
    ? status.progress
    : status._tag === "Ready"
      ? Option.some(1)
      : Option.none<number>()
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
        ? (
            <text style={{ fg: theme.status.failure, width }} wrapMode="none">
              {loadingStatusLabel(status)}
            </text>
          )
        : <ModelOperationStatusText text={loadingStatusLabel(status)} width={width} />}
      {status._tag === "Failed"
        ? <text style={{ fg: theme.text.supporting, width }} wrapMode="none">{loadingStatusDetail(status)}</text>
        : <ModelOperationProgressBar progress={progress} width={width} />}
      <text style={{ fg: theme.text.supporting, width }} wrapMode="none">
        {status._tag === "Failed" ? "" : loadingStatusDetail(status)}
      </text>
      <box style={{ width, height: 1, minHeight: 1, maxHeight: 1, flexDirection: "row", flexShrink: 0 }}>
        {status._tag === "Failed" && (
          <>
            <Button
              onClick={onRetry}
              onMouseOver={() => setHovered("retry")}
              onMouseOut={() => setHovered(null)}
            >
              <text style={{ fg: hovered === "retry" ? theme.accent : theme.text.body }}>Retry loading</text>
            </Button>
            <box style={{ width: 2, flexShrink: 0 }} />
            <Button
              onClick={onChooseAnother}
              onMouseOver={() => setHovered("choose")}
              onMouseOut={() => setHovered(null)}
            >
              <text style={{ fg: hovered === "choose" ? theme.accent : theme.text.body }}>Choose another model</text>
            </Button>
          </>
        )}
      </box>
    </box>
  )
}
