import { Option } from "effect"
import type {
  LocalModelCatalogCandidate,
  LocalModelDownload,
  ModelSlot,
} from "@magnitudedev/sdk"

export const formatBytes = (bytes: number): string => {
  if (bytes <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"] as const
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1000)))
  const value = bytes / 1000 ** index
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`
}

export const formatContext = (tokens: number): string => tokens >= 1_000_000
  ? `${(tokens / 1_000_000).toFixed(tokens % 1_000_000 === 0 ? 0 : 1)}M`
  : tokens >= 1_000
    ? `${Math.round(tokens / 1_000)}K`
    : String(tokens)

export const downloadProgress = (download: LocalModelDownload): number | null =>
  download._tag === "Downloading" || download._tag === "Failed"
    || download._tag === "Cancelled" || download._tag === "NotDownloaded"
    ? download.totalBytes > 0
      ? Math.max(0, Math.min(100, Math.round(download.completedBytes / download.totalBytes * 100)))
      : 0
    : null

export const downloadLabel = (download: LocalModelDownload): string => {
  switch (download._tag) {
    case "NotDownloaded": return `${formatBytes(download.totalBytes)} download`
    case "Downloading": return `${download.stage} · ${downloadProgress(download)}%`
    case "Failed": return download.failure.message
    case "Cancelled": return "Download cancelled"
    case "Downloaded": return `${formatBytes(download.installedBytes)} installed`
  }
}

export const modelSpeedLabel = (candidate: LocalModelCatalogCandidate): string => {
  const lowerContext = Math.min(25_000, candidate.profile.contextLength)
  const upperContext = Math.min(75_000, candidate.profile.contextLength)
  const lower = candidate.performance.find(({ contextTokens }) => contextTokens === lowerContext)
  const upper = candidate.performance.find(({ contextTokens }) => contextTokens === upperContext)
  if (!lower || !upper) return "Speed unavailable"
  const slow = Math.round(Math.min(lower.estimatedTokensPerSecond, upper.estimatedTokensPerSecond))
  const fast = Math.round(Math.max(lower.estimatedTokensPerSecond, upper.estimatedTokensPerSecond))
  return slow === fast ? `~${slow} tok/s` : `~${slow}–${fast} tok/s`
}

export const slotStatus = (slot: ModelSlot): { label: string; tone: string; detail: string | null } => {
  if (slot._tag === "Unassigned") return { label: "Not configured", tone: "neutral", detail: null }
  if (slot._tag === "ConfiguredRemote") {
    return { label: "Unsupported selection", tone: "warning", detail: "Choose a local model." }
  }
  if (slot.availability._tag === "Unavailable") {
    return { label: "Unavailable", tone: "danger", detail: slot.availability.failure.message }
  }
  if (Option.isNone(slot.instance)) return { label: "Not loaded", tone: "neutral", detail: null }
  const lifecycle = slot.instance.value.lifecycle
  switch (lifecycle._tag) {
    case "Loading": return {
      label: "Loading",
      tone: "progress",
      detail: `${lifecycle.stage}${Option.match(lifecycle.progress, {
        onNone: () => "",
        onSome: (progress) => ` · ${Math.round(progress * 100)}%`,
      })}`,
    }
    case "Ready": return { label: "Ready", tone: "success", detail: null }
    case "Stopping": return { label: "Stopping", tone: "progress", detail: lifecycle.reason }
    case "Stopped": return { label: "Not loaded", tone: "neutral", detail: lifecycle.reason }
    case "Failed": return { label: "Failed", tone: "danger", detail: lifecycle.failure.message }
  }
}

export const intentLabel = (intent: "balanced" | "best_quality" | "fastest" | "lightweight"): string => ({
  balanced: "Balanced",
  best_quality: "Best quality",
  fastest: "Fastest",
  lightweight: "Lightweight",
})[intent]
