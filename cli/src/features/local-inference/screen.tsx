import {
  Fragment,
  memo,
  useCallback,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Result } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import {
  useLocalInferenceState,
  getAnimationTickSnapshot,
  subscribeNoop,
  subscribeAnimationTick,
  type LocalInferenceView,
} from "@magnitudedev/client-common"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ReasoningEffortSchema,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import { useTheme } from "../../hooks/use-theme"
import { BOX_CHARS } from "../../utils/ui-constants"
import {
  buildLocalInferenceSelections,
  formatModelLoadProgress,
  describeLocalHardware,
  formatBytes,
  formatContext,
  localInferenceProgressLines,
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
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const

type LocalSetupHoveredAction = "models-skip"
export const localModelSectionRule = (label: string): string =>
  "─".repeat(Math.max(0, LOCAL_MODEL_SECTION_WIDTH - label.length - SECTION_LABEL_GAP))

const recommendationIntent = (intent: "balanced" | "best_quality" | "fastest" | "lightweight"): string => {
  if (intent === "best_quality") return "Best Quality"
  if (intent === "fastest") return "Fastest"
  if (intent === "lightweight") return "Lightweight"
  return "Balanced"
}

type LocalInferenceController = ReturnType<typeof useLocalInferenceState>

export const LocalInferenceScreen = memo(function LocalInferenceScreen(props: LocalInferenceScreenProps) {
  const theme = useTheme()
  const local = useLocalInferenceState()
  const snapshot = Result.value(local.state)
  const [loadingStartedAt] = useState(() => Date.now())
  const trackingProgress = Option.match(snapshot, {
    onNone: () => true,
    onSome: (state) => state.models.recommendations.progress
      .some(({ status }) => status._tag === "Running"),
  })
  const animationTick = useSyncExternalStore(
    trackingProgress ? subscribeAnimationTick : subscribeNoop,
    getAnimationTickSnapshot,
    getAnimationTickSnapshot,
  )
  const spinnerFrame = SPINNER_FRAMES[animationTick % SPINNER_FRAMES.length]
  const nowMs = Date.now()
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
      <box style={{ height: "100%", flexDirection: "column", paddingLeft: 4, paddingTop: 2 }}>
        <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>LOCAL MODEL SETUP</text>
        <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Choose what this machine should run</text>
        <box style={{ flexDirection: "column", paddingTop: 2 }}>
          <text style={{ fg: theme.primary }}>
            {spinnerFrame} Detecting your hardware · {Math.max(0, Math.floor((nowMs - loadingStartedAt) / 1_000))}s
          </text>
          <text style={{ fg: theme.muted }}>○ Checking for downloaded models</text>
          <text style={{ fg: theme.muted }}>○ Loading curated Hugging Face model details</text>
          <text style={{ fg: theme.muted }}>○ Preparing model details</text>
          <text style={{ fg: theme.muted }}>○ Evaluating models for this machine</text>
          <text style={{ fg: theme.muted }}>○ Choosing recommendations</text>
        </box>
      </box>
    ),
    onSome: (state) => (
      <ReadyLocalInferenceScreen
        {...props}
        state={state}
        local={local}
        nowMs={nowMs}
        spinnerFrame={spinnerFrame}
      />
    ),
  })
})

const ReadyLocalInferenceScreen = memo(function ReadyLocalInferenceScreen({
  state,
  local,
  onSkip,
  onConfigured,
  nowMs,
  spinnerFrame,
}: LocalInferenceScreenProps & {
  readonly state: LocalInferenceView
  readonly local: LocalInferenceController
  readonly nowMs: number
  readonly spinnerFrame: string
}) {
  const theme = useTheme()
  const [selectedId, setSelectedId] = useState<Option.Option<string>>(Option.none())
  const [details, setDetails] = useState(false)
  const [hoveredAction, setHoveredAction] = useState<LocalSetupHoveredAction | null>(null)
  const selections = useMemo(() => buildLocalInferenceSelections(state), [state])
  const selectedIndex = selectedInferenceIndex(selections, selectedId)
  const selected = selections[selectedIndex]
  const primarySlot = state.slots.slots.primary
  const activeBinding = primarySlot._tag !== "Unassigned" && primarySlot.selection.providerId === LOCAL_PROVIDER_ID
    ? Option.some(primarySlot.selection)
    : Option.none<SlotSelection>()
  const mutationFailure = local.mutationFailure

  const selectionFor = useCallback((selection: LocalInferenceSelection): Option.Option<SlotSelection> =>
    Option.map(selection.providerModelId, (providerModelId) => ({
      providerId: LOCAL_PROVIDER_ID,
      providerModelId,
      reasoningEffort: Option.getOrElse(
        selection.reasoningEffort,
        () => ReasoningEffortSchema.make("none"),
      ),
    })), [])

  const confirmSelection = useCallback((selection: LocalInferenceSelection) => {
    if (selection.model.download._tag === "Downloading") return
    if (selection.model.download._tag === "Failed") {
      local.retryModelDownload(selection.model.id)
      return
    }
    if (Option.isSome(selection.recommendation)) {
      local.downloadRecommendedModel(selection.recommendation.value.id)
      return
    }
    if (selection.kind === "running") {
      onConfigured()
      return
    }
    Option.match(selectionFor(selection), {
      onNone: () => undefined,
      onSome: (slotSelection) => {
        local.assignSlot(PRIMARY_SLOT_ID, slotSelection)
        onConfigured()
      },
    })
  }, [local, onConfigured, selectionFor])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.ctrl && key.name === "c") return
    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      setSelectedId(Option.fromNullable(selections[Math.max(0, selectedIndex - 1)]?.id))
      return
    }
    if (key.name === "down" || key.name === "j" || key.name === "tab") {
      key.preventDefault()
      setSelectedId(Option.fromNullable(selections[Math.min(Math.max(0, selections.length - 1), selectedIndex + 1)]?.id))
      return
    }
    if (key.name === "d") { key.preventDefault(); setDetails((value) => !value); return }
    if (key.name === "l" && Option.isSome(activeBinding)
      && primarySlot._tag !== "Ready" && primarySlot._tag !== "LoadingLocalModel") {
      key.preventDefault(); local.loadModel(PRIMARY_SLOT_ID); return
    }
    if (key.name === "u" && primarySlot._tag === "Ready") {
      key.preventDefault(); local.unloadModel(PRIMARY_SLOT_ID); return
    }
    if (key.name === "x" && selected?.model.download._tag === "Downloading") {
      key.preventDefault(); local.cancelModelDownload(selected.model.id); return
    }
    if (key.name === "delete" && selected) {
      if (selected.model.download._tag === "Downloaded") {
        key.preventDefault(); local.deleteLocalModel(selected.model.id); return
      }
      if (selected.model.download._tag === "Failed") {
        key.preventDefault(); local.dismissModelDownloadFailure(selected.model.id); return
      }
    }
    if (key.name === "c" && Option.isSome(activeBinding)) { key.preventDefault(); onConfigured(); return }
    if ((key.name === "return" || key.name === "enter") && selected) {
      key.preventDefault(); confirmSelection(selected); return
    }
    if (key.name === "escape") { key.preventDefault(); onSkip() }
  }, [activeBinding, confirmSelection, local, onConfigured, onSkip, primarySlot._tag, selected, selectedIndex, selections]))

  const hardware = describeLocalHardware(state.hardware)
  const progress = localInferenceProgressLines(state.models.recommendations.progress, nowMs)
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
        {progress.length > 0 && (
          <box style={{
            flexDirection: "column",
            width: "100%",
            maxWidth: LOCAL_MODEL_SECTION_WIDTH,
            paddingBottom: 1,
          }}>
            <box style={{ flexDirection: "row", paddingBottom: 1 }}>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
                SETUP PROGRESS
              </text>
              <text style={{ fg: theme.border }}>
                {"  "}{localModelSectionRule("SETUP PROGRESS")}
              </text>
            </box>
            {progress.map((line) => (
              <text
                key={line.id}
                style={{ fg: line.state === "pending" ? theme.muted : theme.foreground }}
              >
                <span fg={line.state === "completed"
                  ? theme.success
                  : line.state === "running"
                    ? theme.primary
                    : line.state === "failed"
                      ? theme.error
                      : theme.muted}>
                  {line.state === "completed"
                    ? "✓ "
                    : line.state === "running"
                      ? `${spinnerFrame} `
                      : line.state === "failed"
                        ? "! "
                        : "○ "}
                </span>
                <span fg={line.state === "pending" ? theme.muted : theme.foreground}>
                  {line.label}
                </span>
                {line.metadata && (
                  <span fg={line.state === "failed" ? theme.error : theme.muted}>
                    {line.metadata}
                  </span>
                )}
              </text>
            ))}
          </box>
        )}
        <box style={{ flexDirection: "column" }}>
          {state.models.recommendations._tag === "Ready" && selections.length === 0
            ? <text style={{ fg: theme.warning }}>No curated model currently fits this configuration.</text>
            : selections.map((selection, index) => {
              const model = selection.model
              const recommendation = selection.recommendation
              const capacityWarning = selectionCapacityWarning(selection)
              const sectionLabel = index === firstRunningIndex ? "RUNNING NOW"
                : index === firstStoredIndex ? "DOWNLOADED"
                : index === firstRecommendationIndex ? (hasExistingModels ? "POSSIBLE DOWNLOADS" : "RECOMMENDED DOWNLOADS")
                : null
              const loading = primarySlot._tag === "LoadingLocalModel"
                && Option.exists(selection.providerModelId, (id) =>
                  primarySlot.selection.providerModelId === id)
              return <Fragment key={selection.id}>
                {sectionLabel && <box style={{ flexDirection: "row", paddingTop: index === 0 ? 0 : 1, paddingBottom: 1, width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH }}>
                  <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{sectionLabel}</text>
                  <text style={{ fg: theme.border }}>  {localModelSectionRule(sectionLabel)}</text>
                </box>}
                <Button id={`local-model-${index}`} onClick={() => confirmSelection(selection)} onMouseOver={() => setSelectedId(Option.some(selection.id))} cursor={model.download._tag === "Downloading" ? "default" : "pointer"} style={{ borderStyle: "single", customBorderChars: BOX_CHARS, borderColor: index === selectedIndex ? theme.primary : theme.border, paddingLeft: 1, paddingRight: 1, marginBottom: 1, flexDirection: "column", width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH }}>
                  <text style={{ fg: index === selectedIndex ? theme.primary : theme.foreground }} attributes={TextAttributes.BOLD}>
                    {index === selectedIndex ? "› " : "  "}{selectionTitle(selection)}
                    <span fg={theme.primary}>{selection.kind === "recommendation"
                      ? Option.match(recommendation, { onNone: () => "", onSome: ({ intent }) => `  ${recommendationIntent(intent)}` })
                      : selection.kind === "running" ? "  Already Running" : "  Already Downloaded"}</span>
                  </text>
                  <text style={{ fg: theme.muted }}>{selectionMetadata(selection)}</text>
                  {Option.isSome(recommendation) && <text style={{ fg: theme.muted }}>{recommendation.value.explanation}</text>}
                  {model.download._tag === "Downloading" && <text style={{ fg: theme.primary }}>Downloading {Math.round(model.download.completedBytes / Math.max(1, model.download.totalBytes) * 100)}% · {formatBytes(model.download.completedBytes)} / {formatBytes(model.download.totalBytes)}</text>}
                  {model.download._tag === "Failed" && <text style={{ fg: theme.error }}>Download failed · {model.download.failure.message}</text>}
                  {model.download._tag === "Downloaded" && model.preparation._tag === "Preparing" && (
                    <text style={{ fg: theme.primary }}>Choosing a serving profile for this machine…</text>
                  )}
                  {loading && <text style={{ fg: theme.primary }}>{formatModelLoadProgress(primarySlot.percentage)}</text>}
                  {capacityWarning && <text style={{ fg: theme.warning }}>{capacityWarning}</text>}
                </Button>
              </Fragment>
            })}
          {state.models.recommendations._tag === "Failed" && <text style={{ fg: theme.warning }}>{state.models.recommendations.failure.message}</text>}
          {details && selected && Option.isSome(selected.recommendation) && (() => {
            const recommendation = selected.recommendation.value
            return <box style={{ flexDirection: "column", paddingLeft: 1 }}>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Exact model files</text>
              {recommendation.sources.map(({ source, files }) => {
                const sourceLabel = source._tag === "HuggingFace"
                  ? `${source.repository}@${source.revision}`
                  : source.path
                return <box key={sourceLabel} style={{ flexDirection: "column" }}>
                  <text style={{ fg: theme.muted }}>{sourceLabel}</text>
                  {files.map((file) => <text key={`${sourceLabel}:${file.path}`} style={{ fg: theme.muted }}>{file.path} · SHA-256 {file.sha256}</text>)}
                  {source._tag === "HuggingFace" && <text style={{ fg: theme.primary }}>https://huggingface.co/{source.repository}/tree/{source.revision}</text>}
                </box>
              })}
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Quantization fidelity</text>
              {recommendation.qualityEvidence.map((evidence) => <text key={evidence} style={{ fg: theme.muted }}>{evidence}</text>)}
              <text style={{ fg: theme.muted }}>{recommendation.qualityScoreProvenance}</text>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Fit</text>
              <text style={{ fg: theme.muted }}>Estimated {formatBytes(recommendation.fit.requiredBytes)} runtime memory from {formatBytes(recommendation.fit.availableBytes)} available capacity.</text>
              {Option.isSome(recommendation.fit.estimatedTokensPerSecond) && <text style={{ fg: theme.muted }}>About {recommendation.fit.estimatedTokensPerSecond.value.toFixed(1)} tokens/sec at {formatContext(recommendation.profile.contextLength)} context.</text>}
            </box>
          })()}
        </box>
        {Option.isSome(mutationFailure) && <text style={{ fg: theme.error }}>{Cause.pretty(mutationFailure.value.cause)}</text>}
        <text style={{ fg: theme.muted, marginTop: 1 }}>
          ↑/↓ choose · D details
          {selected?.model.download._tag === "Downloading"
            ? " · X cancel"
            : selected?.model.download._tag === "Failed"
              ? " · Enter retry · Delete dismiss"
              : selected?.model.download._tag === "Downloaded"
                ? " · Enter use · Delete remove"
                : " · Enter download"}
        </text>
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
