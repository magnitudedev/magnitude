import { useCallback, useEffect, useState, type ReactNode } from "react"
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
import { acquisitionProgress, type LocalModel } from "@magnitudedev/sdk"
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

const moveInlineChoice = <Choice extends string>(
  choices: readonly Choice[],
  selected: Choice,
  direction: -1 | 1,
): Choice => {
  const currentIndex = Math.max(0, choices.indexOf(selected))
  return choices[(currentIndex + direction + choices.length) % choices.length]!
}

const InlineChoiceButton = <Choice extends string>({
  value,
  label,
  selected,
  destructive = false,
  onSelect,
  onChoose,
}: {
  readonly value: Choice
  readonly label: string
  readonly selected: boolean
  readonly destructive?: boolean
  readonly onSelect: (value: Choice) => void
  readonly onChoose: () => void
}): ReactNode => {
  const theme = useTheme()
  return (
    <Button
      onClick={onChoose}
      onMouseOver={() => onSelect(value)}
    >
      <text
        style={{
          fg: selected
            ? destructive ? theme.status.failure : theme.accent
            : theme.text.body,
        }}
        attributes={selected ? TextAttributes.BOLD : TextAttributes.NONE}
      >
        {selected ? "› " : "  "}{label}
      </text>
    </Button>
  )
}

export interface OnboardingModelDownloadOperation {
  readonly starting: boolean
  readonly cancelling: boolean
  readonly onCancel: () => void
}

export type OnboardingModelOperationFooterOperation =
  | {
      readonly _tag: "Downloading"
      readonly cancelling: boolean
      readonly onCancel: () => void
    }
  | { readonly _tag: "Configuring" }
  | {
      readonly _tag: "Activating"
      readonly status: OnboardingModelLoadStatus
      readonly onCancel: () => void
      readonly onRetry: () => void
      readonly onChooseAnother: () => void
    }

export const onboardingModelOperationHint = (
  operation: OnboardingModelOperationFooterOperation,
): string => {
  switch (operation._tag) {
    case "Configuring": return "Ctrl+C to exit"
    case "Downloading": return operation.cancelling
      ? "Ctrl+C to exit"
      : "Esc cancel · Ctrl+C to exit"
    case "Activating": {
      switch (operation.status._tag) {
        case "Failed": return operation.status.failure.retryable
          ? "←/→ choose action · Enter confirm · Ctrl+C to exit"
          : "Enter choose another model · Ctrl+C to exit"
        case "Cancelling":
        case "Ready": return "Ctrl+C to exit"
        case "Preparing":
        case "Stopping":
        case "Loading": return "Esc cancel · Ctrl+C to exit"
      }
    }
  }
}

export function OnboardingModelOperationFooter({
  operation,
}: {
  readonly operation: OnboardingModelOperationFooterOperation
}): ReactNode {
  const theme = useTheme()
  const [confirmingCancellation, setConfirmingCancellation] = useState(false)
  const [cancellationChoice, setCancellationChoice] = useState<ConfirmationChoice>("yes")
  const [failureChoice, setFailureChoice] = useState<"retry" | "choose">("retry")
  const failureRetryable = operation._tag === "Activating"
    && operation.status._tag === "Failed"
    && operation.status.failure.retryable
  const selectedFailureChoice = failureRetryable ? failureChoice : "choose"
  const cancelable = operation._tag === "Downloading"
    ? !operation.cancelling
    : operation._tag === "Activating"
      && operation.status._tag !== "Cancelling"
      && operation.status._tag !== "Ready"
      && operation.status._tag !== "Failed"
  const confirming = confirmingCancellation && cancelable
  useEffect(() => {
    if (cancelable) return
    setConfirmingCancellation(false)
    setCancellationChoice("yes")
  }, [cancelable])
  const declineCancellation = useCallback(() => {
    setConfirmingCancellation(false)
    setCancellationChoice("yes")
  }, [])
  const confirmCancellation = useCallback(() => {
    if (!cancelable || operation._tag === "Configuring") return
    setConfirmingCancellation(false)
    setCancellationChoice("yes")
    operation.onCancel()
  }, [cancelable, operation])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (operation._tag === "Activating" && operation.status._tag === "Failed") {
      if ((key.name === "left" || key.name === "right") && failureRetryable) {
        key.preventDefault()
        setFailureChoice((current) => moveInlineChoice(
          ["retry", "choose"],
          current,
          key.name === "left" ? -1 : 1,
        ))
        return
      }
      if (key.name === "return" || key.name === "enter") {
        key.preventDefault()
        if (selectedFailureChoice === "retry") operation.onRetry()
        else operation.onChooseAnother()
      }
      return
    }
    if (confirming) {
      if (key.name === "escape") {
        key.preventDefault()
        declineCancellation()
        return
      }
      if (key.name === "left" || key.name === "right") {
        key.preventDefault()
        setCancellationChoice((current) => moveInlineChoice(
          ["yes", "no"],
          current,
          key.name === "left" ? -1 : 1,
        ))
        return
      }
      if (key.name === "return" || key.name === "enter") {
        key.preventDefault()
        if (cancellationChoice === "yes") confirmCancellation()
        else declineCancellation()
      }
      return
    }
    if (key.name === "escape" && cancelable) {
      key.preventDefault()
      setCancellationChoice("yes")
      setConfirmingCancellation(true)
    }
  }, [
    cancelable,
    cancellationChoice,
    confirmCancellation,
    confirming,
    declineCancellation,
    failureRetryable,
    operation,
    selectedFailureChoice,
  ]))

  if (operation._tag === "Activating" && operation.status._tag === "Failed") {
    return (
      <box style={{ flexDirection: "row" }}>
        {operation.status.failure.retryable && (
          <>
            <InlineChoiceButton
              value="retry"
              label="Retry loading"
              selected={selectedFailureChoice === "retry"}
              onSelect={setFailureChoice}
              onChoose={operation.onRetry}
            />
            <box style={{ width: 2, flexShrink: 0 }} />
          </>
        )}
        <InlineChoiceButton
          value="choose"
          label="Choose another model"
          selected={selectedFailureChoice === "choose"}
          onSelect={setFailureChoice}
          onChoose={operation.onChooseAnother}
        />
      </box>
    )
  }

  if (confirming) {
    return (
      <box style={{ flexDirection: "row" }}>
        <text style={{ fg: theme.text.body, flexShrink: 0 }} wrapMode="none">
          {operation._tag === "Downloading" ? "Cancel download?" : "Cancel loading?"}
        </text>
        <box style={{ width: 2, flexShrink: 0 }} />
        <InlineChoiceButton
          value="yes"
          label="Yes"
          selected={cancellationChoice === "yes"}
          destructive
          onSelect={setCancellationChoice}
          onChoose={confirmCancellation}
        />
        <box style={{ width: 2, flexShrink: 0 }} />
        <InlineChoiceButton
          value="no"
          label="No"
          selected={cancellationChoice === "no"}
          onSelect={setCancellationChoice}
          onChoose={declineCancellation}
        />
      </box>
    )
  }

  return (
    <text style={{ fg: theme.text.supporting }} wrapMode="none">
      {onboardingModelOperationHint(operation)}
    </text>
  )
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
  const starting = operation.starting
  const activeDownload = starting ? null : acquisitionProgress(model.acquisitionState) ?? null
  const downloading = activeDownload !== null
  const active = starting || downloading
  const cancelling = operation.cancelling
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
  if (!active || status === null) return null

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
}: {
  readonly model: LocalModel
  readonly status: OnboardingModelLoadStatus
  readonly width: number
}): ReactNode {
  const theme = useTheme()
  const modelName = formatLocalModelDisplayName(model)
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
        ? <text style={{ fg: theme.status.failure, width }} wrapMode="none">{loadingFailureLabel(status, modelName, width)}</text>
        : <ModelOperationStatusText text={loadingStatusLabel(status, modelName, width)} width={width} />}
      <ModelOperationProgressBar
        progress={progress}
        width={width}
        tone={status._tag === "Failed" ? "failed" : "active"}
      />
    </box>
  )
}
