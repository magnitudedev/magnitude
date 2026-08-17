import { Option } from "effect"
import type {
  ContextUsageDisplay,
  LocalModel,
  ModelSlot,
} from "@magnitudedev/sdk"

export const formatBytes = (bytes: number): string => {
  if (bytes <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"] as const
  const index = Math.min(
    units.length - 1,
    Math.floor(Math.log(bytes) / Math.log(1000))
  )
  const value = bytes / 1000 ** index
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`
}

export const formatContext = (tokens: number): string =>
  tokens >= 1_000_000
    ? `${(tokens / 1_000_000).toFixed(tokens % 1_000_000 === 0 ? 0 : 1)}M`
    : tokens >= 1_000
    ? `${Math.round(tokens / 1_000)}K`
    : String(tokens)

const formatContextTokens = (tokens: number): string =>
  tokens >= 1_000 ? `${Math.round(tokens / 1_000)}k` : `${Math.round(tokens)}`

export const formatFooterContextUsage = (
  context: ContextUsageDisplay | null,
  tokenCap: number | null | undefined
): string => {
  const tokenUsage =
    context && context.tokenEstimate > 0 ? context.tokenEstimate : null
  if (tokenUsage === null) {
    return tokenCap && tokenCap > 0
      ? `— / ${formatContextTokens(tokenCap)}`
      : "—"
  }
  const used = formatContextTokens(tokenUsage)
  return tokenCap && tokenCap > 0
    ? `${used} / ${formatContextTokens(tokenCap)} (${Math.round(
        (tokenUsage / tokenCap) * 100
      )}%)`
    : used
}

export const transferProgress = (transfer: {
  readonly completedBytes: number
  readonly totalBytes: number
}): number =>
  transfer.totalBytes > 0
    ? Math.max(
        0,
        Math.min(
          100,
          Math.round((transfer.completedBytes / transfer.totalBytes) * 100)
        )
      )
    : 0

export const transferLabel = (transfer: {
  readonly completedBytes: number
  readonly totalBytes: number
  readonly stage?: string
}): string =>
  `${transfer.stage ?? "Downloading"} · ${transferProgress(
    transfer
  )}% · ${formatBytes(transfer.completedBytes)} of ${formatBytes(
    transfer.totalBytes
  )}`

export const modelContextLength = (model: LocalModel): number | null =>
  model.servingState._tag === "Assessing"
    ? model.servingState.configuration.profile.contextLength
    : model.servingState._tag === "Assessed"
    ? model.servingState.configuration.profile.contextLength
    : model.servingState._tag === "Failed"
    ? Option.getOrNull(
        Option.map(
          model.servingState.configuration,
          ({ profile }) => profile.contextLength
        )
      )
    : null

export const slotStatus = (
  slot: ModelSlot
): { label: string; tone: string; detail: string | null } => {
  if (slot._tag === "Unassigned")
    return { label: "Not configured", tone: "neutral", detail: null }
  if (slot._tag === "ConfiguredRemote") {
    return {
      label: "Unsupported selection",
      tone: "warning",
      detail: "Choose a local model.",
    }
  }
  if (slot.availability._tag === "Unavailable") {
    return {
      label: "Unavailable",
      tone: "danger",
      detail: slot.availability.failure.message,
    }
  }
  if (slot.availability._tag === "Pending") {
    return { label: "Preparing", tone: "progress", detail: null }
  }
  switch (slot.residency._tag) {
    case "Unloaded":
      return { label: "Not loaded", tone: "neutral", detail: null }
    case "Requested":
      return { label: "Queued", tone: "progress", detail: "Waiting to load" }
    case "Loading":
      return {
        label: "Loading",
        tone: "progress",
        detail: `${slot.residency.stage}${Option.match(
          slot.residency.progress,
          {
            onNone: () => "",
            onSome: (progress) => ` · ${Math.round(progress * 100)}%`,
          }
        )}`,
      }
    case "Ready":
      return { label: "Ready", tone: "success", detail: null }
    case "Stopping":
      return {
        label: "Stopping",
        tone: "progress",
        detail: slot.residency.reason,
      }
    case "Failed":
      return {
        label: "Failed",
        tone: "danger",
        detail: slot.residency.failure.message,
      }
  }
}

export const intentLabel = (
  intent: "balanced" | "smartest" | "fastest" | "lightweight"
): string =>
  ({
    balanced: "Balanced",
    smartest: "Smartest",
    fastest: "Fastest",
    lightweight: "Lightweight",
  }[intent])
