import { Fragment, memo, useCallback, useMemo, useState } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Atom, Result, useAtomMount } from "@effect-atom/atom-react"
import { Cause, Effect, Option } from "effect"
import {
  useLocalInferenceState,
  type LocalInferenceView,
} from "@magnitudedev/client-common"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ReasoningEffortSchema,
  type LocalModelId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import { useTheme } from "../../hooks/use-theme"
import { BOX_CHARS } from "../../utils/ui-constants"
import {
  buildLocalInferenceSelections,
  describeLocalHardware,
  formatBytes,
  formatContext,
  selectedInferenceIndex,
  selectionCapacityWarning,
  selectionMetadata,
  selectionTitle,
  type LocalInferenceSelection,
} from "./view-model"

interface LocalInferenceScreenProps {
  readonly management: boolean
  readonly onExit: () => void
  readonly onSkip: () => void
  readonly onConfigured: () => void
}

export const LOCAL_MODEL_SECTION_WIDTH = 72
const SECTION_LABEL_GAP = 2
const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")

type LocalSetupHoveredAction = "models-skip"
export const localModelSectionRule = (label: string): string =>
  "─".repeat(Math.max(0, LOCAL_MODEL_SECTION_WIDTH - label.length - SECTION_LABEL_GAP))

const recommendationBadge = (badge: "recommended" | "lighter" | "higher_fidelity" | "alternative"): string => {
  if (badge === "lighter") return "Smaller Model"
  if (badge === "higher_fidelity") return "Higher Fidelity Option"
  if (badge === "alternative") return "Alternative Option"
  return "Recommended"
}

type LocalInferenceController = ReturnType<typeof useLocalInferenceState>

export const LocalInferenceScreen = memo(function LocalInferenceScreen(props: LocalInferenceScreenProps) {
  const theme = useTheme()
  const local = useLocalInferenceState()
  const snapshot = Result.value(local.state)
  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.ctrl && key.name === "c" && !key.meta && !key.option) {
      key.preventDefault()
      props.onExit()
      return
    }
    if (Option.isNone(snapshot) && key.name === "escape") {
      key.preventDefault()
      props.onSkip()
    }
  }, [props, snapshot]))
  return Option.match(snapshot, {
    onNone: () => Result.isFailure(local.state) ? (
      <box style={{ height: "100%", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>
        <text style={{ fg: theme.error }}>Failed to inspect local inference.</text>
        <text style={{ fg: theme.muted }}>{Cause.pretty(local.state.cause)}</text>
      </box>
    ) : (
      <box style={{ height: "100%", alignItems: "center", justifyContent: "center" }}>
        <text style={{ fg: theme.muted }}>Inspecting local inference…</text>
      </box>
    ),
    onSome: (state) => <ReadyLocalInferenceScreen {...props} state={state} local={local} />,
  })
})

const ReadyLocalInferenceScreen = memo(function ReadyLocalInferenceScreen({
  state,
  local,
  onSkip,
  onConfigured,
}: LocalInferenceScreenProps & {
  readonly state: LocalInferenceView
  readonly local: LocalInferenceController
}) {
  const theme = useTheme()
  const [selectedId, setSelectedId] = useState<Option.Option<LocalModelId>>(Option.none())
  const [details, setDetails] = useState(false)
  const [hoveredAction, setHoveredAction] = useState<LocalSetupHoveredAction | null>(null)
  const [pendingActivationId, setPendingActivationId] = useState<Option.Option<LocalModelId>>(Option.none())
  const selections = useMemo(() => buildLocalInferenceSelections(state), [state])
  const selectedIndex = selectedInferenceIndex(selections, selectedId)
  const selected = selections[selectedIndex]
  const primarySlot = state.slots.slots.primary
  const activeBinding = primarySlot._tag === "Ready" && primarySlot.selection.providerId === LOCAL_PROVIDER_ID
    ? Option.some(primarySlot.selection)
    : Option.none<SlotSelection>()
  const pendingActivation = useMemo(() => Option.flatMap(pendingActivationId, (localModelId) =>
    Option.fromNullable(selections.find((selection) => selection.entry.model.localModelId === localModelId))),
  [pendingActivationId, selections])
  const mutationFailure = local.mutationFailure

  const activationCompletionAtom = useMemo(() => Atom.make(Effect.sync(() => {
    if (Option.isSome(pendingActivation)
      && Option.exists(activeBinding, (selection) =>
        selection.providerModelId === pendingActivation.value.entry.model.providerModelId)) {
      setPendingActivationId(Option.none())
      onConfigured()
    }
  })), [activeBinding, onConfigured, pendingActivation])
  useAtomMount(activationCompletionAtom)

  const selectionFor = useCallback((selection: LocalInferenceSelection): SlotSelection => {
    const reasoning = selection.entry.model.capabilities.reasoning
    return {
      providerId: LOCAL_PROVIDER_ID,
      providerModelId: selection.entry.model.providerModelId,
      reasoningEffort: Option.getOrElse(reasoning.defaultEffort, () => ReasoningEffortSchema.make("none")),
    }
  }, [])

  const confirmSelection = useCallback((selection: LocalInferenceSelection) => {
    const entry = selection.entry
    if (entry._tag === "Downloading") return
    if (entry._tag === "AvailableForDownload" || entry._tag === "DownloadFailed") {
      local.downloadModel(entry.model.localModelId)
      return
    }
    if (selection.kind === "running") {
      onConfigured()
      return
    }
    setPendingActivationId(Option.some(entry.model.localModelId))
    local.loadSlot(PRIMARY_SLOT_ID, selectionFor(selection))
  }, [local, onConfigured, selectionFor])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.ctrl && key.name === "c") return
    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      setSelectedId(Option.fromNullable(selections[Math.max(0, selectedIndex - 1)]?.entry.model.localModelId))
      return
    }
    if (key.name === "down" || key.name === "j" || key.name === "tab") {
      key.preventDefault()
      setSelectedId(Option.fromNullable(selections[Math.min(Math.max(0, selections.length - 1), selectedIndex + 1)]?.entry.model.localModelId))
      return
    }
    if (key.name === "d") { key.preventDefault(); setDetails((value) => !value); return }
    if (key.name === "r" && Option.isSome(activeBinding)) {
      key.preventDefault(); local.reloadSlot(PRIMARY_SLOT_ID); return
    }
    if (key.name === "u" && Option.isSome(activeBinding)) {
      key.preventDefault(); local.unloadSlot(PRIMARY_SLOT_ID); return
    }
    if (key.name === "delete" && selected?.kind === "stored" && selected.entry._tag === "Downloaded") {
      key.preventDefault(); local.deleteModel(selected.entry.model.localModelId); return
    }
    if (key.name === "c" && Option.isSome(activeBinding)) { key.preventDefault(); onConfigured(); return }
    if ((key.name === "return" || key.name === "enter") && selected) {
      key.preventDefault(); confirmSelection(selected); return
    }
    if (key.name === "escape") { key.preventDefault(); onSkip() }
  }, [activeBinding, confirmSelection, local, onConfigured, onSkip, selected, selectedIndex, selections]))

  const hardware = describeLocalHardware(state.hardware)
  const firstRunningIndex = selections.findIndex((selection) => selection.kind === "running")
  const firstStoredIndex = selections.findIndex((selection) => selection.kind === "stored")
  const firstRecommendationIndex = selections.findIndex((selection) => selection.kind === "recommendation")
  const hasExistingModels = firstRunningIndex >= 0 || firstStoredIndex >= 0
  return (
    <scrollbox key="local-models" scrollX={false} scrollbarOptions={{ visible: false }} verticalScrollbarOptions={{ visible: true, trackOptions: { width: 1 } }} style={{ height: "100%", rootOptions: { backgroundColor: "transparent" }, wrapperOptions: { border: false, backgroundColor: "transparent" }, contentOptions: { flexDirection: "column" } }}>
      <box style={{ flexDirection: "column", paddingLeft: 2, paddingRight: 2 }}>
        <box style={{ flexDirection: "column", paddingTop: 1, paddingBottom: 1 }}>
          <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>LOCAL MODEL SETUP</text>
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Choose what this machine should run</text>
        </box>
        <box style={{ flexDirection: "column", width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH, paddingBottom: 1 }}>
          <box style={{ flexDirection: "row", paddingBottom: 1 }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>HARDWARE DETECTED</text>
            <text style={{ fg: theme.border }}>{"  "}{localModelSectionRule("HARDWARE DETECTED")}</text>
          </box>
          <box style={{ paddingLeft: 1, paddingRight: 1, flexDirection: "column", width: "100%" }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{hardware.system.name}</text>
            {hardware.system.details.map((detail) => <text key={detail} style={{ fg: theme.muted }}>{detail}</text>)}
            {hardware.accelerators.map((accelerator) => (
              <box key={`${accelerator.name}:${accelerator.details}`} style={{ flexDirection: "column", paddingTop: 1 }}>
                <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{accelerator.name}</text>
                <text style={{ fg: theme.muted }}>{accelerator.details}</text>
              </box>
            ))}
            {hardware.accelerators.length === 0 && !state.hardware.memoryDomains.some((domain) => domain.kind === "UnifiedMemory") && (
              <text style={{ fg: theme.muted }}>CPU inference · No GPU detected</text>
            )}
          </box>
        </box>
        <box style={{ flexDirection: "column" }}>
          {state.inventory._tag === "Ready" && selections.length === 0
            ? <text style={{ fg: theme.warning }}>No curated model currently fits this configuration.</text>
            : selections.map((selection, index) => {
              const entry = selection.entry
              const recommendation = entry.model.recommendation
              const capacityWarning = selectionCapacityWarning(selection)
              const sectionLabel = index === firstRunningIndex ? "RUNNING NOW"
                : index === firstStoredIndex ? "DOWNLOADED"
                : index === firstRecommendationIndex ? (hasExistingModels ? "POSSIBLE DOWNLOADS" : "RECOMMENDED DOWNLOADS")
                : null
              const loading = primarySlot._tag === "LoadingLocalModel"
                && primarySlot.selection.providerModelId === entry.model.providerModelId
              return <Fragment key={entry.model.localModelId}>
                {sectionLabel && <box style={{ flexDirection: "row", paddingTop: index === 0 ? 0 : 1, paddingBottom: 1, width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH }}>
                  <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{sectionLabel}</text>
                  <text style={{ fg: theme.border }}>  {localModelSectionRule(sectionLabel)}</text>
                </box>}
                <Button id={`local-model-${index}`} onClick={() => confirmSelection(selection)} onMouseOver={() => setSelectedId(Option.some(entry.model.localModelId))} cursor={entry._tag === "Downloading" ? "default" : "pointer"} style={{ borderStyle: "single", customBorderChars: BOX_CHARS, borderColor: index === selectedIndex ? theme.primary : theme.border, paddingLeft: 1, paddingRight: 1, marginBottom: 1, flexDirection: "column", width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH }}>
                  <text style={{ fg: index === selectedIndex ? theme.primary : theme.foreground }} attributes={TextAttributes.BOLD}>
                    {index === selectedIndex ? "› " : "  "}{selectionTitle(selection)}
                    <span fg={theme.primary}>{selection.kind === "recommendation"
                      ? Option.match(recommendation, { onNone: () => "", onSome: ({ badge }) => `  ${recommendationBadge(badge)}` })
                      : selection.kind === "running" ? "  Already Running" : "  Already Downloaded"}</span>
                  </text>
                  <text style={{ fg: theme.muted }}>{selectionMetadata(selection)}</text>
                  {Option.isSome(recommendation) && <text style={{ fg: theme.muted }}>{recommendation.value.fidelityLabel}</text>}
                  {entry._tag === "Downloading" && <text style={{ fg: theme.primary }}>Downloading {entry.percentage}% · {formatBytes(entry.completedBytes)} / {formatBytes(entry.totalBytes)}</text>}
                  {entry._tag === "DownloadFailed" && <text style={{ fg: theme.error }}>Download failed · {entry.error.message}</text>}
                  {loading && <text style={{ fg: theme.primary }}>Loading {primarySlot.percentage}%</text>}
                  {capacityWarning && <text style={{ fg: theme.warning }}>{capacityWarning}</text>}
                </Button>
              </Fragment>
            })}
          {state.inventory._tag === "Loading" && <text style={{ fg: theme.primary }}>Calculating recommendations for this machine…</text>}
          {state.inventory._tag === "Failed" && <text style={{ fg: theme.warning }}>{state.inventory.error.message}</text>}
          {details && selected && Option.isSome(selected.entry.model.recommendation) && (() => {
            const recommendation = selected.entry.model.recommendation.value
            return <box style={{ flexDirection: "column", paddingLeft: 1 }}>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Exact artifact</text>
              <text style={{ fg: theme.muted }}>{recommendation.repository}@{recommendation.revision}</text>
              {recommendation.files.map((file) => <box key={file.path} style={{ flexDirection: "column" }}><text style={{ fg: theme.muted }}>{file.path} · SHA-256 {file.sha256}</text></box>)}
              <text style={{ fg: theme.primary }}>{recommendation.sourcePageUrl}</text>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Quantization fidelity</text>
              <text style={{ fg: theme.muted }}>{recommendation.fidelityEvidence}</text>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Fit</text>
              <text style={{ fg: theme.muted }}>Estimated {formatBytes(recommendation.estimatedRuntimeBytes)} runtime; {formatBytes(recommendation.fitMarginBytes)} stable headroom.</text>
            </box>
          })()}
        </box>
        {Option.isSome(mutationFailure) && <text style={{ fg: theme.error }}>{Cause.pretty(mutationFailure.value.cause)}</text>}
        <text style={{ fg: theme.muted, marginTop: 1 }}>↑/↓ choose · D details{selected?.kind === "recommendation" ? " · Enter download" : " · Enter use"}</text>
        <box style={{ paddingTop: 1, paddingBottom: 1, flexShrink: 0, flexDirection: "row" }}>
          <Button onClick={onSkip} onMouseOver={() => setHoveredAction("models-skip")} onMouseOut={() => setHoveredAction((current) => current === "models-skip" ? null : current)}>
            <box style={{ borderStyle: "single", borderColor: hoveredAction === "models-skip" ? theme.primary : theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
              <text style={{ fg: hoveredAction === "models-skip" ? theme.primary : theme.foreground }}>Skip for now (Esc)</text>
            </box>
          </Button>
        </box>
      </box>
    </scrollbox>
  )
})
