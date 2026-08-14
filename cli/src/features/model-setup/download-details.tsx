import { useCallback, useMemo, useState, type ReactNode } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Option } from "effect"
import type { LocalModel } from "@magnitudedev/sdk"
import { formatLocalModelDisplayName, truncateToDisplayWidth } from "@magnitudedev/client-common"
import { Button } from "../../components/button"
import { useTheme } from "../../hooks/use-theme"
import { formatDownloadBytes } from "../local-inference/view-model"

const MIB = 1024 ** 2

const formatRate = (bytesPerSecond: number): string => {
  const mebibytes = bytesPerSecond / MIB
  return `${mebibytes >= 10 ? Math.round(mebibytes) : mebibytes.toFixed(1)} MB/s`
}

const formatEta = (remainingBytes: number, bytesPerSecond: number): string => {
  const minutes = Math.max(1, Math.ceil(remainingBytes / bytesPerSecond / 60))
  return `about ${minutes} ${minutes === 1 ? "minute" : "minutes"} remaining`
}

const progressBar = (fraction: number, width: number): string => {
  const filled = Math.round(Math.max(0, Math.min(1, fraction)) * width)
  return `${"█".repeat(filled)}${"░".repeat(Math.max(0, width - filled))}`
}

type ConfirmationChoice = "yes" | "no"

interface DownloadDetailsOperation {
  readonly starting: boolean
  readonly cancelling: boolean
  readonly cancelError: string | null
  readonly onCancel: () => void
}

export function OnboardingModelDownloadDetails({
  model,
  width,
  height,
  operation,
}: {
  readonly model: LocalModel
  readonly width: number
  readonly height: number
  readonly operation: DownloadDetailsOperation
}): ReactNode {
  const theme = useTheme()
  const [confirming, setConfirming] = useState(false)
  const [choice, setChoice] = useState<ConfirmationChoice>("yes")
  const [hovered, setHovered] = useState<string | null>(null)
  const cancelling = operation.cancelling
  const starting = operation.starting
  const contentWidth = Math.max(1, width)
  const download = model.acquisitionState
  const downloading = !starting && download._tag === "Downloading"
  const cancelable = downloading
  const fraction = downloading
    ? download.completedBytes / Math.max(1, download.totalBytes)
    : 0
  const percentage = Math.round(Math.max(0, Math.min(1, fraction)) * 100)
  const percentageLabel = `${percentage}%`
  const barWidth = Math.max(8, contentWidth - percentageLabel.length - 2)
  const heading = `Downloading ${formatLocalModelDisplayName(model)}`
  const rate = downloading ? Option.getOrNull(download.bytesPerSecond) : null
  const detail = useMemo(() => {
    if (!downloading) return null
    if (download.stage === "verifying" || download.stage === "publishing") {
      return "Verifying download…"
    }
    if (rate === null || rate <= 0) return "Estimating time remaining…"
    return `${formatRate(rate)} · ${formatEta(download.totalBytes - download.completedBytes, rate)}`
  }, [download, downloading, rate])

  const declineCancellation = useCallback(() => {
    setConfirming(false)
    setChoice("yes")
  }, [])

  const confirmCancellation = useCallback(() => {
    if (cancelling) return
    operation.onCancel()
  }, [cancelling, operation])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (cancelling) {
      key.preventDefault()
      return
    }
    if (!confirming) {
      if (key.name === "escape" && cancelable) {
        key.preventDefault()
        setChoice("yes")
        setConfirming(true)
      }
      return
    }
    if (key.name === "escape") {
      key.preventDefault()
      declineCancellation()
      return
    }
    if (key.name === "left" || key.name === "right" || key.name === "up" || key.name === "down") {
      key.preventDefault()
      setChoice((current) => current === "yes" ? "no" : "yes")
      return
    }
    if (key.name === "return" || key.name === "enter") {
      key.preventDefault()
      if (choice === "yes") confirmCancellation()
      else declineCancellation()
    }
  }, [cancelable, cancelling, choice, confirmCancellation, confirming, declineCancellation, operation]))

  const choiceButton = (value: ConfirmationChoice, label: string) => (
    <Button
      onClick={() => value === "yes" ? confirmCancellation() : declineCancellation()}
      onMouseOver={() => { setChoice(value); setHovered(value) }}
      onMouseOut={() => setHovered((current) => current === value ? null : current)}
    >
      <text style={{
        fg: value === "yes"
          ? choice === value || hovered === value ? theme.error : theme.foreground
          : choice === value || hovered === value ? theme.primary : theme.foreground,
      }} attributes={choice === value ? TextAttributes.BOLD : TextAttributes.NONE}>
        {choice === value ? "› " : "  "}{label}
      </text>
    </Button>
  )

  return (
    <box style={{
      width: contentWidth,
      height,
      minHeight: height,
      maxHeight: height,
      flexShrink: 0,
      flexDirection: "column",
      overflow: "hidden",
    }}>
        <text
          style={{ fg: theme.foreground }}
          attributes={TextAttributes.BOLD}
          wrapMode="none"
        >
          {truncateToDisplayWidth(heading, contentWidth)}
        </text>
        <box style={{ height: 1 }} />
        <text wrapMode="none">
          <span style={{ fg: theme.primary }}>{progressBar(fraction, barWidth)}</span>
          <span style={{ fg: theme.muted }}>{`  ${percentageLabel}`}</span>
        </text>
        <box style={{ height: 1 }} />
        {downloading && (
          <text style={{ fg: theme.muted }}>
            {formatDownloadBytes(download.completedBytes)} / {formatDownloadBytes(download.totalBytes)}
          </text>
        )}
        {detail !== null && (
          <text style={{ fg: theme.muted }}>{detail}</text>
        )}
        <box style={{ height: 1 }} />
        <box style={{ height: 2, flexDirection: "column" }}>
          {cancelling ? (
            <text style={{ fg: theme.muted }}>Cancelling download…</text>
          ) : confirming ? (
            <>
              <text style={{ fg: theme.foreground }}>Are you sure you want to cancel?</text>
              <box style={{ flexDirection: "row", gap: 2 }}>
                {choiceButton("yes", "Yes")}
                {choiceButton("no", "No")}
              </box>
            </>
          ) : cancelable ? (
            <Button
              onClick={() => setConfirming(true)}
              onMouseOver={() => setHovered("cancel")}
              onMouseOut={() => setHovered((current) => current === "cancel" ? null : current)}
            >
              <text style={{ fg: hovered === "cancel" ? theme.error : theme.muted }}>Cancel (Esc)</text>
            </Button>
          ) : null}
          {operation.cancelError && (
            <text style={{ fg: theme.error }}>{operation.cancelError}</text>
          )}
        </box>
    </box>
  )
}
