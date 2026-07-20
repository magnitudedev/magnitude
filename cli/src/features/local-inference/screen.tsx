import { Fragment, memo, useCallback, useMemo, useState } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Cause, Effect, Option } from "effect"
import { Atom, Result as AtomResult, useAtomMount } from "@effect-atom/atom-react"
import type {
  LocalInferenceState,
  LocalModelRecommendation,
} from "@magnitudedev/sdk"
import { useLocalInferenceState } from "@magnitudedev/client-common"
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

type LocalSetupHoveredAction = "models-skip"
export const localModelSectionRule = (label: string): string =>
  "─".repeat(Math.max(0, LOCAL_MODEL_SECTION_WIDTH - label.length - SECTION_LABEL_GAP))

const recommendationBadge = (badge: LocalModelRecommendation["badge"]): string => {
  if (badge === "lighter") return "Smaller Model"
  if (badge === "higher_fidelity") return "Higher Fidelity Option"
  if (badge === "alternative") return "Alternative Option"
  return "Recommended"
}

type LocalInferenceController = ReturnType<typeof useLocalInferenceState>

export const LocalInferenceScreen = memo(function LocalInferenceScreen(
  props: LocalInferenceScreenProps,
) {
  const theme = useTheme()
  const local = useLocalInferenceState()
  const snapshot = AtomResult.value(local.state)
  const ready = Option.isSome(snapshot)
  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.ctrl && key.name === "c" && !key.meta && !key.option) {
      key.preventDefault()
      props.onExit()
      return
    }
    if (!ready && key.name === "escape") {
      key.preventDefault()
      props.onSkip()
    }
  }, [props.onExit, props.onSkip, ready]))
  return Option.match(snapshot, {
    onNone: () => AtomResult.isFailure(local.state) ? (
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
  readonly state: LocalInferenceState
  readonly local: LocalInferenceController
}) {
  const theme = useTheme()
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [details, setDetails] = useState(false)
  const [hoveredAction, setHoveredAction] = useState<LocalSetupHoveredAction | null>(null)
  const [pendingActivationId, setPendingActivationId] = useState<string | null>(null)
  const selections = useMemo(() => buildLocalInferenceSelections(state), [state])
  const selectedIndex = selectedInferenceIndex(selections, selectedId)
  const selected = selections[selectedIndex]
  const error = local.mutationFailure
  const activeOperations = state.operations.filter((operation) => operation.status === "running")
  const visibleOperations = [
    ...activeOperations,
    ...state.operations.filter((operation) => operation.status === "failed").slice(-3),
  ]
  const selectedOperation = selected && activeOperations.find((operation) =>
    operation.target._tag === "configuration"
      ? operation.target.configurationId === selected.id
      : operation.target._tag === "model" && operation.target.selectionId === selected.id,
  )
  const restartRunning = activeOperations.some((operation) => operation.kind === "restart")

  const activationCompletionAtom = useMemo(
    () => Atom.make(Effect.sync(() => {
      if (
        pendingActivationId
        && state.activeBinding?.selectionId === pendingActivationId
      ) {
        setPendingActivationId(null)
        onConfigured()
      }
    })),
    [onConfigured, pendingActivationId, state.activeBinding?.selectionId],
  )
  useAtomMount(activationCompletionAtom)

  const confirmSelection = useCallback((selection: LocalInferenceSelection | undefined) => {
    if (!selection) return
    const hasActiveOperation = activeOperations.some((operation) =>
      operation.target._tag === "configuration"
        ? operation.target.configurationId === selection.id
        : operation.target._tag === "model" && operation.target.selectionId === selection.id,
    )
    if (hasActiveOperation) return
    if (selection.kind === "running") {
      if (state.activeBinding?.providerModelId === selection.choice.providerModelId) {
        onConfigured()
      } else {
        setPendingActivationId(selection.id)
        local.activateModel(selection.id)
      }
      return
    }
    if (selection.kind === "stored") {
      setPendingActivationId(selection.id)
      local.activateModel(selection.id)
      return
    }
    const stored = state.choices.some((choice) => choice.choiceId === selection.id && choice._tag === "Stored")
    if (stored) {
      setPendingActivationId(selection.id)
      local.activateModel(selection.id)
    } else {
      local.downloadModel(selection.id)
    }
  }, [activeOperations, local, onConfigured, state.activeBinding, state.choices])

  const confirmModel = useCallback(() => {
    confirmSelection(selected)
  }, [confirmSelection, selected])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.ctrl && key.name === "c") return
    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      setSelectedId(selections[Math.max(0, selectedIndex - 1)]?.id ?? null)
      return
    }
    if (key.name === "down" || key.name === "j" || key.name === "tab") {
      key.preventDefault()
      setSelectedId(selections[Math.min(Math.max(0, selections.length - 1), selectedIndex + 1)]?.id ?? null)
      return
    }
    if (key.name === "d") { key.preventDefault(); setDetails((value) => !value); return }
    if (key.name === "r" && state.activeBinding && !restartRunning) {
      key.preventDefault(); local.restart(); return
    }
    if (key.name === "u" && state.activeBinding && !local.pending.disable) {
      key.preventDefault(); local.disable(); return
    }
    if (key.name === "delete" && selected?.kind === "stored" && !selectedOperation && !local.pending.delete) {
      key.preventDefault(); local.deleteModel(selected.id); return
    }
    if (key.name === "c" && state.activeBinding) { key.preventDefault(); onConfigured(); return }
    if (key.name === "return" || key.name === "enter") { key.preventDefault(); confirmModel(); return }
    if (key.name === "escape") { key.preventDefault(); onSkip() }
  }, [confirmModel, local, onConfigured, onSkip, restartRunning, selected, selectedIndex, selectedOperation, selections, state.activeBinding]))

  const host = state.host._tag === "Available" ? state.host.profile : null
  const hardware = host ? describeLocalHardware(host) : null
  const firstRunningIndex = selections.findIndex((selection) => selection.kind === "running")
  const firstStoredIndex = selections.findIndex((selection) => selection.kind === "stored")
  const firstRecommendationIndex = selections.findIndex((selection) => selection.kind === "recommendation")
  const hasExistingModels = firstRunningIndex >= 0 || firstStoredIndex >= 0
  const recommendationsLoading = state.recommendationState._tag === "Loading"
  const recommendationsFailed = state.recommendationState._tag === "Failed"
    ? state.recommendationState.message
    : null
  return (
    <scrollbox
      key="local-models"
      scrollX={false}
      scrollbarOptions={{ visible: false }}
      verticalScrollbarOptions={{ visible: true, trackOptions: { width: 1 } }}
      style={{ height: "100%", rootOptions: { backgroundColor: "transparent" }, wrapperOptions: { border: false, backgroundColor: "transparent" }, contentOptions: { flexDirection: "column" } }}
    >
    <box style={{ flexDirection: "column", paddingLeft: 2, paddingRight: 2 }}>
      <box style={{ flexDirection: "column", paddingTop: 1, paddingBottom: 1 }}>
      <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>LOCAL MODEL SETUP</text>
      <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Choose what this machine should run</text>
      </box>
      <box style={{ flexDirection: "column", width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH, paddingBottom: 1 }}>
        <box style={{ flexDirection: "row", paddingBottom: 1 }}>
          <text style={{ fg: hardware ? theme.foreground : theme.warning }} attributes={TextAttributes.BOLD}>
            {hardware ? "HARDWARE DETECTED" : "HARDWARE DETECTION UNAVAILABLE"}
          </text>
          <text style={{ fg: theme.border }}>
            {"  "}{localModelSectionRule(hardware ? "HARDWARE DETECTED" : "HARDWARE DETECTION UNAVAILABLE")}
          </text>
        </box>
        {hardware ? (
          <box style={{ paddingLeft: 1, paddingRight: 1, flexDirection: "column", width: "100%" }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{hardware.system.name}</text>
            {hardware.system.details.map((detail) => (
              <text key={detail} style={{ fg: theme.muted }}>{detail}</text>
            ))}
            {hardware.accelerators.map((accelerator) => (
              <box key={`${accelerator.name}:${accelerator.details}`} style={{ flexDirection: "column", paddingTop: 1 }}>
                <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{accelerator.name}</text>
                <text style={{ fg: theme.muted }}>{accelerator.details}</text>
              </box>
            ))}
            {hardware.accelerators.length === 0 && !host?.memoryDomains.some((domain) => domain.kind === "unified_memory") && (
              <text style={{ fg: theme.muted }}>CPU inference · No GPU detected</text>
            )}
          </box>
        ) : (
          <text style={{ fg: theme.muted }}>{state.host._tag === "Unavailable" ? state.host.message : "Hardware information is unavailable."}</text>
        )}
      </box>
      <box style={{ flexDirection: "column" }}>
        {state.recommendationState._tag === "Ready" && selections.length === 0
          ? <text style={{ fg: theme.warning }}>No curated model currently fits this configuration.</text>
          : selections.map((selection, index) => {
            const capacityWarning = selectionCapacityWarning(selection)
            const operation = activeOperations.find((candidate) =>
              candidate.target._tag === "configuration"
                ? candidate.target.configurationId === selection.id
                : candidate.target._tag === "model" && candidate.target.selectionId === selection.id,
            )
            const operationProgress = operation?.progress && operation.progress.totalBytes > 0
              ? ` · ${Math.round(operation.progress.completedBytes / operation.progress.totalBytes * 100)}%`
              : ""
            const sectionLabel = index === firstRunningIndex ? "RUNNING NOW"
              : index === firstStoredIndex ? "DOWNLOADED"
              : index === firstRecommendationIndex ? (hasExistingModels ? "POSSIBLE DOWNLOADS" : "RECOMMENDED DOWNLOADS")
              : null
            return <Fragment key={selection.id}>
            {sectionLabel && <box style={{ flexDirection: "row", paddingTop: index === 0 ? 0 : 1, paddingBottom: 1, width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH }}>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{sectionLabel}</text>
              <text style={{ fg: theme.border }}>  {localModelSectionRule(sectionLabel)}</text>
            </box>}
            <Button
              id={`local-model-${index}`}
              onClick={() => confirmSelection(selection)}
              onMouseOver={() => setSelectedId(selection.id)}
              cursor={operation ? "default" : "pointer"}
              style={{ borderStyle: "single", customBorderChars: BOX_CHARS, borderColor: index === selectedIndex ? theme.primary : theme.border, paddingLeft: 1, paddingRight: 1, marginBottom: 1, flexDirection: "column", width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH }}
            >
              <text style={{ fg: index === selectedIndex ? theme.primary : theme.foreground }} attributes={TextAttributes.BOLD}>
                {index === selectedIndex ? "› " : "  "}{selectionTitle(selection)}
                <span fg={theme.primary}>{selection.kind === "recommendation" ? `  ${recommendationBadge(selection.recommendation.badge)}` : selection.kind === "running" ? "  Already Running" : "  Already Downloaded"}</span>
              </text>
              <text style={{ fg: theme.muted }}>{selectionMetadata(selection)}</text>
              {selection.kind === "recommendation" && <text style={{ fg: theme.muted }}>{selection.recommendation.quantization.fidelityLabel}</text>}
              {operation && <text style={{ fg: theme.primary }}>{operation.kind === "download" ? "Downloading" : "Activating"} · {operation.stage}{operationProgress}</text>}
              {capacityWarning && <text style={{ fg: theme.warning }}>{capacityWarning}</text>}
            </Button>
            </Fragment>
          })}
        {recommendationsLoading && (
          <box style={{ flexDirection: "column", width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH }}>
            <box style={{ flexDirection: "row", paddingBottom: 1 }}>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
                {hasExistingModels ? "POSSIBLE DOWNLOADS" : "RECOMMENDED DOWNLOADS"}
              </text>
            </box>
            <text style={{ fg: theme.primary }}>Calculating recommendations for this machine…</text>
          </box>
        )}
        {recommendationsFailed && <text style={{ fg: theme.warning }}>{recommendationsFailed}</text>}
        {details && selected?.kind === "recommendation" && (
          <box style={{ flexDirection: "column", paddingLeft: 1 }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Exact artifact</text>
            <text style={{ fg: theme.muted }}>{selected.recommendation.repo}@{selected.recommendation.revision}</text>
            {selected.recommendation.files.map((file) => <box key={file.path} style={{ flexDirection: "column" }}>
              <text style={{ fg: theme.muted }}>{file.path} · SHA-256 {file.sha256}</text>
            </box>)}
            <text style={{ fg: theme.primary }}>{selected.recommendation.sourcePageUrl}</text>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Quantization fidelity</text>
            <text style={{ fg: theme.muted }}>{selected.recommendation.quantization.fidelityEvidence}</text>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Fit</text>
            <text style={{ fg: theme.muted }}>
              Estimated {formatBytes(selected.recommendation.estimatedRuntimeBytes)} runtime at {formatContext(selected.recommendation.contextTokens)} context; {formatBytes(selected.recommendation.fitMarginBytes)} stable headroom. Model maximum {formatContext(selected.recommendation.modelMaximumContextTokens)}.
            </text>
          </box>
        )}
      </box>
      {visibleOperations.map((operation) => {
        const progress = operation.progress && operation.progress.totalBytes > 0
          ? ` · ${Math.round(operation.progress.completedBytes / operation.progress.totalBytes * 100)}% (${formatBytes(operation.progress.completedBytes)} / ${formatBytes(operation.progress.totalBytes)})`
          : ""
        const label = operation.kind === "download" ? "Downloading local model"
          : operation.kind === "activate" ? "Activating local model"
          : "Restarting local inference"
        return (
          <text key={operation.operationId} style={{ fg: operation.status === "failed" ? theme.error : theme.primary }}>
            {label} · {operation.status === "failed" ? operation.failure?.message ?? "Failed" : operation.stage}{progress}
          </text>
        )
      })}
      {Option.isSome(error) && <text style={{ fg: theme.error }}>{error.value}</text>}
      {state.warnings.map((warning) => <text key={warning.code} style={{ fg: theme.warning }}>{warning.message}</text>)}
      <text style={{ fg: theme.muted, marginTop: 1 }}>
        ↑/↓ choose · D details
        {selected?.kind === "recommendation"
          ? " · Enter download"
          : " · Enter use"}
      </text>
      <box style={{ paddingTop: 1, paddingBottom: 1, flexShrink: 0, flexDirection: "row" }}>
        <Button
          onClick={onSkip}
          onMouseOver={() => setHoveredAction("models-skip")}
          onMouseOut={() => setHoveredAction((current) => current === "models-skip" ? null : current)}
        >
          <box style={{ borderStyle: "single", borderColor: hoveredAction === "models-skip" ? theme.primary : theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
            <text style={{ fg: hoveredAction === "models-skip" ? theme.primary : theme.foreground }}>
              Skip for now (Esc)
            </text>
          </box>
        </Button>
      </box>
    </box>
    </scrollbox>
  )
})
