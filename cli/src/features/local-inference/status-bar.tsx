import { deriveHardwareMemoryView, type LocalInferenceView } from "@magnitudedev/client-common"
import { ProviderIdSchema } from "@magnitudedev/sdk"
import { TextAttributes } from "@opentui/core"
import { Button } from "../../components/button"
import { formatMemoryBytes } from "../../components/hardware-memory-domain"
import { StackedBar, type StackedBarSegment } from "../../components/stacked-bar"
import { useTheme } from "../../hooks/use-theme"
import { formatModelLoadProgress } from "./view-model"

interface LocalInferenceStatusBarProps {
  readonly state: LocalInferenceView
  readonly width: number
  readonly onOpenHardware: () => void
}

const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")
const compactMemory = (bytes: number): string => formatMemoryBytes(bytes).replace(".0 ", " ")

const compactBarSegments = (
  domains: ReturnType<typeof deriveHardwareMemoryView>["domains"],
  colors: { readonly fixed: string; readonly kv: string; readonly system: string },
): readonly StackedBarSegment[] | null => {
  const participating = domains.filter((domain) => domain.participatesInRuntime && domain.usedBytes !== null)
  const complete = participating.filter((domain) => domain.fixedBytes !== null
    && domain.kvCacheBytes !== null
    && domain.systemAndAppsBytes !== null)
  if (participating.length === 0) return null
  if (complete.length !== participating.length) return [{
    value: participating.reduce((sum, domain) => sum + (domain.usedBytes ?? 0), 0),
    color: colors.system,
  }]
  return [
    { value: complete.reduce((sum, domain) => sum + (domain.fixedBytes ?? 0), 0), color: colors.fixed },
    { value: complete.reduce((sum, domain) => sum + (domain.kvCacheBytes ?? 0), 0), color: colors.kv },
    { value: complete.reduce((sum, domain) => sum + (domain.systemAndAppsBytes ?? 0), 0), color: colors.system },
  ]
}

export const LocalInferenceStatusBar = ({ state, width, onOpenHardware }: LocalInferenceStatusBarProps) => {
  const theme = useTheme()
  const slots = [state.slots.slots.primary, state.slots.slots.secondary]
  const slot = slots.find((candidate) => candidate._tag !== "Unassigned"
    && candidate.selection.providerId === LOCAL_PROVIDER_ID)
  const activeModel = slot && slot._tag !== "Unassigned"
    ? state.models.models.find((model) => model.preparation._tag === "Available"
      && model.preparation.providerModelIds.includes(slot.selection.providerModelId))
    : undefined
  const downloadModel = state.models.models.find((model) =>
    model.download._tag === "Downloading" || model.download._tag === "Failed")
  const model = activeModel ?? downloadModel
  if (!slot && !model) return null
  const modelName = model?.displayName ?? "Local model"
  const status = slot
    ? slot._tag === "LoadingLocalModel" ? formatModelLoadProgress(slot.percentage)
      : slot._tag === "UnloadedLocalModel" ? "Unloaded"
        : slot._tag === "UnloadingLocalModel" ? "Unloading"
          : slot._tag === "Blocked" ? "Failed"
            : slot._tag === "Ready" ? "Ready"
              : "Unassigned"
    : model?.download._tag === "Downloading"
      ? `Downloading ${Math.round(model.download.completedBytes / Math.max(1, model.download.totalBytes) * 100)}%`
      : model?.download._tag === "Failed" ? "Download failed"
        : "Ready"
  const memoryView = deriveHardwareMemoryView(state.hardware, {
    fallbackToAccelerators: slot !== undefined,
  })
  const memory = memoryView.compact
  const barSegments = compactBarSegments(memoryView.domains, {
    fixed: theme.foreground,
    kv: theme.primary,
    system: theme.warning,
  })
  const barWidth = width >= 84 ? 16 : width >= 72 ? 12 : 0
  const showMemoryWord = width >= 62
  return (
    <box style={{ marginLeft: 1, marginRight: 1, flexDirection: "row", flexShrink: 0, borderStyle: "rounded", borderColor: theme.border, paddingLeft: 1, paddingRight: 1 }}>
      <text style={{ fg: theme.foreground, flexShrink: 1 }} attributes={TextAttributes.BOLD}>{modelName}</text>
      <text style={{ fg: theme.muted }}>  {status}</text>
      <box style={{ flexGrow: 1 }} />
      {memory && (
        <Button onClick={onOpenHardware} style={{ flexDirection: "row" }}>
          {barSegments && barWidth > 0 && <><StackedBar segments={barSegments} total={memory.totalBytes} width={barWidth} trackColor={theme.border} /><text> </text></>}
          <text style={{ fg: theme.link }}>{showMemoryWord ? "Memory " : ""}{compactMemory(memory.usedBytes)} / {compactMemory(memory.totalBytes)}</text>
        </Button>
      )}
    </box>
  )
}
