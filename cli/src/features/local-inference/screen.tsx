import { Fragment, memo, useCallback, useMemo, useState } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Cause, Effect, Option } from "effect"
import { Atom, Result as AtomResult, useAtomMount } from "@effect-atom/atom-react"
import type {
  LocalInferenceState,
  LocalSessionConcurrency,
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
  readonly onBack: () => void
  readonly onSkip: () => void
  readonly onConfigured: () => void
}

export const LOCAL_MODEL_SECTION_WIDTH = 72
export const LOCAL_USAGE_SETUP_WIDTH = 88
const SECTION_LABEL_GAP = 2

export type LocalUsageFocusTarget = "one" | "up_to_three" | "continue"
type LocalSetupHoveredAction = "usage-skip" | "usage-recommendations" | "models-back" | "models-skip"
export const LOCAL_USAGE_FOCUS_ORDER: readonly LocalUsageFocusTarget[] = [
  "one", "up_to_three", "continue",
]
export const moveLocalUsageFocus = (currentIndex: number, direction: -1 | 1): number =>
  Math.max(0, Math.min(LOCAL_USAGE_FOCUS_ORDER.length - 1, currentIndex + direction))
export const localModelSectionRule = (label: string): string =>
  "─".repeat(Math.max(0, LOCAL_MODEL_SECTION_WIDTH - label.length - SECTION_LABEL_GAP))

const recommendationBadge = (badge: LocalInferenceState["recommendations"][number]["badge"]): string => {
  if (badge === "lighter") return "Smaller Model"
  if (badge === "higher_fidelity") return "Higher Fidelity Option"
  if (badge === "alternative") return "Alternative Option"
  return "Recommended"
}

const LocalUsageOption = memo(function LocalUsageOption({ focused, selected, title, description }: {
  readonly focused: boolean
  readonly selected: boolean
  readonly title: string
  readonly description: string
}) {
  const theme = useTheme()
  return (
    <box style={{ flexDirection: "column", flexShrink: 0 }}>
      <box style={{ flexDirection: "row" }}>
        <text style={{ fg: focused ? theme.primary : theme.muted }}>{focused ? "›" : " "}</text>
        <text style={{ fg: selected ? theme.primary : theme.muted }}>{selected ? " ● " : " ○ "}</text>
        <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{title}</text>
      </box>
      <text style={{ fg: theme.muted }}>    {description}</text>
    </box>
  )
})

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
  management,
  onBack,
  onSkip,
  onConfigured,
}: LocalInferenceScreenProps & {
  readonly state: LocalInferenceState
  readonly local: LocalInferenceController
}) {
  const theme = useTheme()
  const [requestedStage, setRequestedStage] = useState<"usage" | "models">("usage")
  const [concurrencyOverride, setConcurrencyOverride] = useState<LocalSessionConcurrency | null>(null)
  const concurrency = concurrencyOverride ?? state.usage?.sessionConcurrency ?? "one"
  const [usageRow, setUsageRow] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [details, setDetails] = useState(false)
  const [hoveredAction, setHoveredAction] = useState<LocalSetupHoveredAction | null>(null)
  const [pendingActivationId, setPendingActivationId] = useState<string | null>(null)
  const selections = useMemo(() => buildLocalInferenceSelections(state), [state])
  const selectedIndex = selectedInferenceIndex(selections, selectedId)
  const selected = selections[selectedIndex]
  const busy = local.mutationBusy
  const error = local.mutationFailure
  const usageReady = state.usage?.sessionConcurrency === concurrency
  const stage = requestedStage === "models" && usageReady ? "models" : "usage"

  const activationCompletionAtom = useMemo(
    () => Atom.make(Effect.sync(() => {
      if (
        pendingActivationId
        && state.activeBinding?.selectionId === pendingActivationId
        && !busy
      ) {
        setPendingActivationId(null)
        onConfigured()
      }
    })),
    [busy, onConfigured, pendingActivationId, state.activeBinding?.selectionId],
  )
  useAtomMount(activationCompletionAtom)

  const continueFromUsage = useCallback(() => {
    if (busy) return
    local.configureUsage({ sessionConcurrency: concurrency })
    setRequestedStage("models")
  }, [busy, concurrency, local])

  const confirmSelection = useCallback((selection: LocalInferenceSelection | undefined) => {
    if (busy) return
    if (!selection) return
    if (selection.kind === "running") {
      if (state.activeBinding?.providerModelId === selection.choice.providerModelId) {
        onConfigured()
      } else if (selection.choice._tag === "RunningExternal") {
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
    const stored = state.choices.some((choice) => choice.choiceId === selection.id && choice._tag === "StoredOwned")
    if (stored) {
      setPendingActivationId(selection.id)
      local.activateModel(selection.id)
    } else {
      local.downloadModel(selection.id)
    }
  }, [busy, local, onConfigured, state.activeBinding, state.choices])

  const confirmModel = useCallback(() => {
    confirmSelection(selected)
  }, [confirmSelection, selected])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.ctrl && key.name === "c") return
    if (stage === "usage") {
      if (key.name === "right") { key.preventDefault(); continueFromUsage(); return }
      if (key.name === "up" || key.name === "k") { key.preventDefault(); setUsageRow((row) => moveLocalUsageFocus(row, -1)); return }
      if (key.name === "down" || key.name === "j" || key.name === "tab") { key.preventDefault(); setUsageRow((row) => moveLocalUsageFocus(row, 1)); return }
      if (key.name === "space" || key.name === "return" || key.name === "enter") {
        key.preventDefault()
        const target = LOCAL_USAGE_FOCUS_ORDER[usageRow]
        if (target === "one" || target === "up_to_three") setConcurrencyOverride(target)
        else continueFromUsage()
        return
      }
      if (key.name === "escape") { key.preventDefault(); onSkip() }
      return
    }

    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      if (!busy) setSelectedId(selections[Math.max(0, selectedIndex - 1)]?.id ?? null)
      return
    }
    if (key.name === "down" || key.name === "j" || key.name === "tab") {
      key.preventDefault()
      if (!busy) setSelectedId(selections[Math.min(Math.max(0, selections.length - 1), selectedIndex + 1)]?.id ?? null)
      return
    }
    if (key.name === "d") { key.preventDefault(); setDetails((value) => !value); return }
    if (key.name === "r" && state.activeBinding?._tag === "Managed" && !busy) {
      key.preventDefault(); local.restart(); return
    }
    if (key.name === "u" && state.activeBinding && !busy) {
      key.preventDefault(); local.disable(); return
    }
    if (key.name === "delete" && selected?.kind === "stored" && selected.choice._tag === "StoredOwned" && !busy) {
      key.preventDefault(); local.deleteModel(selected.id); return
    }
    if (key.name === "c" && state.activeBinding && !busy) { key.preventDefault(); onConfigured(); return }
    if (key.name === "return" || key.name === "enter") { key.preventDefault(); confirmModel(); return }
    if (key.name === "left" || key.name === "backspace") { key.preventDefault(); if (!busy) setRequestedStage("usage"); return }
    if (key.name === "escape") { key.preventDefault(); if (!busy) onSkip() }
  }, [busy, confirmModel, continueFromUsage, local, onConfigured, onSkip, selected, selectedIndex, selections, stage, state.activeBinding, usageRow]))

  if (stage === "usage") {
    const focusTarget = LOCAL_USAGE_FOCUS_ORDER[usageRow]!
    const running = state.choices.find((choice) => choice._tag === "RunningExternal" || choice._tag === "RunningManaged")
    const runningMetadata = running
      ? [running.displayName, running.quantization?.format, running.contextTokens === undefined ? "Context unavailable" : `${formatContext(running.contextTokens)} context`, running._tag === "RunningManaged" ? "Managed by Magnitude" : "Running outside Magnitude"].filter(Boolean).join(" · ")
      : null
    return (
      <box key="local-usage" style={{ flexDirection: "column", height: "100%", paddingLeft: 2, paddingRight: 2 }}>
        <box style={{ flexDirection: "column", paddingTop: 1, flexShrink: 0, width: "100%", maxWidth: LOCAL_USAGE_SETUP_WIDTH }}>
          <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>LOCAL MODEL SETUP</text>
          <text style={{ fg: theme.foreground }}>Magnitude uses llama.cpp to run local models in the background.</text>
          <text style={{ fg: theme.muted }}>Tell us how many local sessions to reserve, and we'll recommend Hugging Face models that fit.</text>
          {running && <box style={{ flexDirection: "column", paddingTop: 1 }}>
            <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>● llama.cpp server detected</text>
            <text style={{ fg: theme.muted }}>  {runningMetadata}</text>
          </box>}
          <box style={{ flexDirection: "column", paddingTop: 1 }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>How many local coding sessions will you run at once?</text>
            <LocalUsageOption focused={focusTarget === "one"} selected={concurrency === "one"} title="One session" description="Reserve one local context window." />
            <LocalUsageOption focused={focusTarget === "up_to_three"} selected={concurrency === "up_to_three"} title="Multiple sessions" description="Reserve up to three local context windows. This may result in a smaller recommended local model." />
          </box>
          {busy && <text style={{ fg: theme.primary }}>Finding models for this setup…</text>}
          {Option.isSome(error) && <text style={{ fg: theme.error }}>{error.value}</text>}
          <text style={{ fg: theme.muted, marginTop: 1 }}>↑/↓ move · Enter select</text>
          <box style={{ flexDirection: "row", paddingTop: 1 }}>
            <Button
              onClick={() => { if (!busy) onSkip() }}
              onMouseOver={() => setHoveredAction("usage-skip")}
              onMouseOut={() => setHoveredAction((current) => current === "usage-skip" ? null : current)}
            >
              <box style={{ borderStyle: "single", borderColor: hoveredAction === "usage-skip" ? theme.primary : theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
                <text style={{ fg: hoveredAction === "usage-skip" ? theme.primary : theme.foreground }}>Skip for now (Esc)</text>
              </box>
            </Button>
            <text>  </text>
            <Button
              onClick={continueFromUsage}
              onMouseOver={() => setHoveredAction("usage-recommendations")}
              onMouseOut={() => setHoveredAction((current) => current === "usage-recommendations" ? null : current)}
            >
              <box style={{ borderStyle: "single", borderColor: focusTarget === "continue" || hoveredAction === "usage-recommendations" ? theme.primary : theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
                <text style={{ fg: focusTarget === "continue" || hoveredAction === "usage-recommendations" ? theme.primary : theme.foreground }}>See recommendations (→)</text>
              </box>
            </Button>
          </box>
        </box>
      </box>
    )
  }

  const host = state.host._tag === "Available" ? state.host.profile : null
  const hardware = host ? describeLocalHardware(host) : null
  const firstRunningIndex = selections.findIndex((selection) => selection.kind === "running")
  const firstStoredIndex = selections.findIndex((selection) => selection.kind === "stored")
  const firstRecommendationIndex = selections.findIndex((selection) => selection.kind === "recommendation")
  const hasExistingModels = firstRunningIndex >= 0 || firstStoredIndex >= 0
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
          <box style={{ borderStyle: "single", customBorderChars: BOX_CHARS, borderColor: theme.border, paddingLeft: 1, paddingRight: 1, flexDirection: "column", width: "100%" }}>
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
            {hardware.accelerators.length === 0 && !host?.memoryDomains.some((domain) => domain.kind === "unified_working_set") && (
              <text style={{ fg: theme.muted }}>CPU inference · No GPU detected by llama.cpp</text>
            )}
          </box>
        ) : (
          <text style={{ fg: theme.muted }}>{state.host._tag === "Unavailable" ? state.host.message : "Hardware information is unavailable."}</text>
        )}
      </box>
      <box style={{ flexDirection: "column" }}>
        {selections.length === 0
          ? <text style={{ fg: theme.warning }}>No curated model currently fits this configuration.</text>
          : selections.map((selection, index) => {
            const capacityWarning = selectionCapacityWarning(selection)
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
              onMouseOver={() => { if (!busy) setSelectedId(selection.id) }}
              cursor={busy ? "default" : "pointer"}
              style={{ borderStyle: "single", customBorderChars: BOX_CHARS, borderColor: index === selectedIndex ? theme.primary : theme.border, paddingLeft: 1, paddingRight: 1, marginBottom: 1, flexDirection: "column", width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH }}
            >
              <text style={{ fg: index === selectedIndex ? theme.primary : theme.foreground }} attributes={TextAttributes.BOLD}>
                {index === selectedIndex ? "› " : "  "}{selectionTitle(selection)}
                <span fg={theme.primary}>{selection.kind === "recommendation" ? `  ${recommendationBadge(selection.recommendation.badge)}` : selection.kind === "running" ? "  Already Running" : "  Already Downloaded"}</span>
              </text>
              <text style={{ fg: theme.muted }}>{selectionMetadata(selection)}</text>
              {selection.kind === "recommendation" && <text style={{ fg: theme.muted }}>{selection.recommendation.quantization.fidelityLabel}</text>}
              {capacityWarning && <text style={{ fg: theme.warning }}>{capacityWarning}</text>}
            </Button>
            </Fragment>
          })}
        {details && selected?.kind === "recommendation" && (
          <box style={{ flexDirection: "column", paddingLeft: 1 }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Exact artifact</text>
            <text style={{ fg: theme.muted }}>{selected.recommendation.repo}@{selected.recommendation.revision}</text>
            {selected.recommendation.files.map((file) => <box key={file.path} style={{ flexDirection: "column" }}>
              <text style={{ fg: theme.muted }}>{file.path} · SHA-256 {file.sha256}</text>
              <text style={{ fg: theme.primary }}>{file.downloadUrl}</text>
            </box>)}
            <text style={{ fg: theme.primary }}>{selected.recommendation.sourcePageUrl}</text>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Quantization fidelity</text>
            <text style={{ fg: theme.muted }}>{selected.recommendation.quantization.fidelityEvidence}</text>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Fit</text>
            <text style={{ fg: theme.muted }}>
              Estimated {formatBytes(selected.recommendation.estimatedRuntimeBytes)} runtime with {selected.recommendation.servingProfile.parallelSlots} × {formatContext(selected.recommendation.contextTokens)} context windows; {formatBytes(selected.recommendation.fitMarginBytes)} stable headroom. Model maximum {formatContext(selected.recommendation.modelMaximumContextTokens)} per window.
            </text>
          </box>
        )}
      </box>
      {busy && <text style={{ fg: theme.primary }}>Applying local model changes…</text>}
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
          onClick={() => { if (!busy) setRequestedStage("usage") }}
          onMouseOver={() => setHoveredAction("models-back")}
          onMouseOut={() => setHoveredAction((current) => current === "models-back" ? null : current)}
        >
          <box style={{ borderStyle: "single", borderColor: hoveredAction === "models-back" ? theme.primary : theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
            <text style={{ fg: hoveredAction === "models-back" ? theme.primary : theme.foreground }}>Back (←)</text>
          </box>
        </Button>
        <text>  </text>
        <Button
          onClick={() => { if (!busy) onSkip() }}
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
