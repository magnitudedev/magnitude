import { useCallback, useMemo, useState, type ReactNode } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Option } from "effect"
import {
  formatStorageSize,
  formatTransferRate,
} from "@magnitudedev/client-common"
import type { LocalModel, ModelDownloadId } from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import { useTheme } from "../../hooks/use-theme"

const formatEta = (remainingBytes: number, bytesPerSecond: number): string => {
  const minutes = Math.max(1, Math.ceil(remainingBytes / bytesPerSecond / 60))
  return `about ${minutes} ${minutes === 1 ? "minute" : "minutes"} remaining`
}

const progressBar = (fraction: number, width: number): string => {
  const filled = Math.round(Math.max(0, Math.min(1, fraction)) * width)
  return `${"█".repeat(filled)}${"░".repeat(Math.max(0, width - filled))}`
}

type ConfirmationChoice = "yes" | "no"

export interface OnboardingModelDownloadOperation {
  readonly starting: boolean
  readonly cancelling: boolean
  readonly onCancel: () => void
}

export const ONBOARDING_MODEL_DOWNLOAD_ROWS = 4

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
  const [confirmationDownloadId, setConfirmationDownloadId] = useState<ModelDownloadId | null>(null)
  const [choice, setChoice] = useState<ConfirmationChoice>("yes")
  const [hovered, setHovered] = useState<string | null>(null)
  const download = model.acquisitionState
  const starting = operation.starting
  const activeDownload = !starting && download._tag === "Downloading"
    ? download
    : null
  const downloading = activeDownload !== null
  const active = starting || downloading
  const cancelling = operation.cancelling
  const cancelable = activeDownload !== null && !cancelling
  const confirming = cancelable && confirmationDownloadId === activeDownload.downloadId
  const totalBytes = activeDownload?.totalBytes ?? model.downloadBytes
  const fraction = activeDownload !== null
    ? activeDownload.completedBytes / Math.max(1, activeDownload.totalBytes)
    : 0
  const percentageLabel = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`
  const barWidth = Math.max(8, width - percentageLabel.length - 2)
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
    setConfirmationDownloadId(null)
    setChoice("yes")
  }, [])

  const confirmCancellation = useCallback(() => {
    if (!cancelable) return
    setConfirmationDownloadId(null)
    setChoice("yes")
    operation.onCancel()
  }, [cancelable, operation])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (cancelling) return
    if (!confirming) {
      if (key.name === "escape" && cancelable) {
        key.preventDefault()
        setChoice("yes")
        setConfirmationDownloadId(activeDownload.downloadId)
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
  }, [activeDownload, cancelable, cancelling, choice, confirmCancellation, confirming, declineCancellation]))

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
      height: ONBOARDING_MODEL_DOWNLOAD_ROWS,
      minHeight: ONBOARDING_MODEL_DOWNLOAD_ROWS,
      maxHeight: ONBOARDING_MODEL_DOWNLOAD_ROWS,
      flexDirection: "column",
      flexShrink: 0,
      overflow: "hidden",
    }}>
      <text style={{ fg: theme.text.body, width }} wrapMode="none">
        {status}
      </text>
      <text style={{ width }} wrapMode="none">
        <span style={{ fg: theme.accent }}>{progressBar(fraction, barWidth)}</span>
        <span style={{ fg: theme.text.supporting }}>{`  ${percentageLabel}`}</span>
      </text>
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
                  setConfirmationDownloadId(activeDownload.downloadId)
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
