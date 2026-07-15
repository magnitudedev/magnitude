import { Fragment, memo, useCallback, useMemo, useRef, useState } from "react"
import { TextAttributes, type KeyEvent, type ScrollBoxRenderable } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Cause, Effect } from "effect"
import { Atom, Result as AtomResult, useAtomMount } from "@effect-atom/atom-react"
import type {
  LocalInferenceState,
  LocalModelRole,
  LocalSessionConcurrency,
} from "@magnitudedev/sdk"
import { useLocalInferenceState } from "@magnitudedev/client-common"
import { Button } from "../../components/button"
import { useTheme } from "../../hooks/use-theme"
import { BOX_CHARS } from "../../utils/ui-constants"
import {
  buildLocalInferenceSelections,
  formatBytes,
  formatContext,
  selectionMetadata,
  selectionTitle,
} from "./view-model"

interface LocalInferenceScreenProps {
  readonly management: boolean
  readonly onBack: () => void
  readonly onSkip: () => void
  readonly onConfigured: () => void
}

export const LOCAL_MODEL_SECTION_WIDTH = 72
export const LOCAL_USAGE_SETUP_WIDTH = 88
const SECTION_LABEL_GAP = 2

export type LocalUsageFocusTarget = "main" | "subagent" | "one" | "up_to_three" | "continue"
export const LOCAL_USAGE_FOCUS_ORDER: readonly LocalUsageFocusTarget[] = [
  "main", "subagent", "one", "up_to_three", "continue",
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

const failureMessage = <A, E>(result: AtomResult.Result<A, E>): string | null =>
  AtomResult.isFailure(result) ? Cause.pretty(result.cause) : null

type LocalInferenceController = ReturnType<typeof useLocalInferenceState>

export const LocalInferenceScreen = memo(function LocalInferenceScreen(
  props: LocalInferenceScreenProps,
) {
  const theme = useTheme()
  const local = useLocalInferenceState()
  const ready = AtomResult.isSuccess(local.state)
  useKeyboard(useCallback((key: KeyEvent) => {
    if (!ready && key.name === "escape") {
      key.preventDefault()
      props.onSkip()
    }
  }, [props.onSkip, ready]))
  if (AtomResult.isInitial(local.state)) {
    return (
      <box style={{ height: "100%", alignItems: "center", justifyContent: "center" }}>
        <text style={{ fg: theme.muted }}>Inspecting local inference…</text>
      </box>
    )
  }
  if (AtomResult.isFailure(local.state)) {
    return (
      <box style={{ height: "100%", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>
        <text style={{ fg: theme.error }}>Failed to inspect local inference.</text>
        <text style={{ fg: theme.muted }}>{Cause.pretty(local.state.cause)}</text>
      </box>
    )
  }
  return <ReadyLocalInferenceScreen {...props} state={local.state.value} local={local} />
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
  const [requestedStage, setRequestedStage] = useState<"usage" | "models">(state.usage ? "models" : "usage")
  const [roleOverride, setRoleOverride] = useState<LocalModelRole | null>(null)
  const [concurrencyOverride, setConcurrencyOverride] = useState<LocalSessionConcurrency | null>(null)
  const role = roleOverride ?? state.usage?.localModelRole ?? "main"
  const concurrency = concurrencyOverride ?? state.usage?.sessionConcurrency ?? "one"
  const [usageRow, setUsageRow] = useState(role === "subagent" ? 1 : 0)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [details, setDetails] = useState(false)
  const [skipHovered, setSkipHovered] = useState(false)
  const [pendingActivationId, setPendingActivationId] = useState<string | null>(null)
  const scrollRef = useRef<ScrollBoxRenderable | null>(null)
  const selections = useMemo(() => buildLocalInferenceSelections(state), [state])
  const selected = selections[Math.min(selectedIndex, Math.max(0, selections.length - 1))]
  const mutationBusy = local.mutationResults.some(AtomResult.isWaiting)
  const busy = mutationBusy
  const error = local.mutationResults.map(failureMessage).find((message) => message !== null) ?? null
  const usageReady = state.usage?.localModelRole === role
    && state.usage.sessionConcurrency === concurrency
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
    local.configureUsage({ localModelRole: role, sessionConcurrency: concurrency })
    setRequestedStage("models")
  }, [busy, concurrency, local, role])

  const confirmModel = useCallback(() => {
    if (busy) return
    if (state.distribution._tag !== "Ready") {
      local.installDistribution()
      return
    }
    if (!selected) return
    if (selected.kind === "running") {
      if (state.activeBinding?.providerModelId === selected.choice.providerModelId) {
        onConfigured()
      } else if (selected.choice._tag === "RunningExternal") {
        setPendingActivationId(selected.id)
        local.activateModel(selected.id)
      }
      return
    }
    if (selected.kind === "stored") {
      setPendingActivationId(selected.id)
      local.activateModel(selected.id)
      return
    }
    const stored = state.choices.some((choice) => choice.choiceId === selected.id && choice._tag === "StoredOwned")
    if (stored) {
      setPendingActivationId(selected.id)
      local.activateModel(selected.id)
    } else {
      local.downloadModel(selected.id)
    }
  }, [busy, local, onConfigured, selected, state.activeBinding, state.choices, state.distribution._tag])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.ctrl && key.name === "c") return
    if (stage === "usage") {
      if (key.name === "up" || key.name === "k") { key.preventDefault(); setUsageRow((row) => moveLocalUsageFocus(row, -1)); return }
      if (key.name === "down" || key.name === "j" || key.name === "tab") { key.preventDefault(); setUsageRow((row) => moveLocalUsageFocus(row, 1)); return }
      if (key.name === "space" || key.name === "return" || key.name === "enter") {
        key.preventDefault()
        const target = LOCAL_USAGE_FOCUS_ORDER[usageRow]
        if (target === "main" || target === "subagent") setRoleOverride(target)
        else if (target === "one" || target === "up_to_three") setConcurrencyOverride(target)
        else continueFromUsage()
        return
      }
      if (key.name === "escape") { key.preventDefault(); onSkip() }
      return
    }

    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      if (!busy) setSelectedIndex((index) => Math.max(0, index - 1))
      return
    }
    if (key.name === "down" || key.name === "j" || key.name === "tab") {
      key.preventDefault()
      if (!busy) setSelectedIndex((index) => Math.min(Math.max(0, selections.length - 1), index + 1))
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
  }, [busy, concurrency, confirmModel, continueFromUsage, local, onConfigured, onSkip, role, selected, selections.length, stage, state.activeBinding, usageRow]))

  if (stage === "usage") {
    const focusTarget = LOCAL_USAGE_FOCUS_ORDER[usageRow]!
    const running = state.choices.find((choice) => choice._tag === "RunningExternal" || choice._tag === "RunningManaged")
    const runningMetadata = running
      ? [running.displayName, running.quantization?.format, `${formatContext(running.contextTokens)} context`, running._tag === "RunningManaged" ? "Managed by Magnitude" : "Running outside Magnitude"].filter(Boolean).join(" · ")
      : null
    return (
      <box style={{ flexDirection: "column", height: "100%", paddingLeft: 2, paddingRight: 2 }}>
        <box style={{ flexDirection: "column", paddingTop: 1, flexShrink: 0, width: "100%", maxWidth: LOCAL_USAGE_SETUP_WIDTH }}>
          <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>LOCAL MODEL SETUP</text>
          <text style={{ fg: theme.foreground }}>Magnitude uses llama.cpp to run local models in the background.</text>
          <text style={{ fg: theme.muted }}>Answer two questions and we'll recommend Hugging Face models that fit your setup.</text>
          {running && <box style={{ flexDirection: "column", paddingTop: 1 }}>
            <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>● llama.cpp server detected</text>
            <text style={{ fg: theme.muted }}>  {runningMetadata}</text>
          </box>}
          <box style={{ flexDirection: "column", paddingTop: running ? 1 : 0 }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>How do you plan to use local models?</text>
            <LocalUsageOption focused={focusTarget === "main"} selected={role === "main"} title="As my main agent" description="One larger context window per active session." />
            <LocalUsageOption focused={focusTarget === "subagent"} selected={role === "subagent"} title="For local subagents" description="Uses a cloud main agent and reserves three context windows for local subagents." />
          </box>
          <box style={{ flexDirection: "column", paddingTop: 1 }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>How many Magnitude sessions will you run at once?</text>
            <LocalUsageOption focused={focusTarget === "one"} selected={concurrency === "one"} title="One session" description="Reserve one set of context windows." />
            <LocalUsageOption focused={focusTarget === "up_to_three"} selected={concurrency === "up_to_three"} title="Multiple sessions" description="Reserve up to three sets of context windows. This may result in a smaller recommended local model." />
          </box>
          {busy && <text style={{ fg: theme.primary }}>Finding models for this setup…</text>}
          {error && <text style={{ fg: theme.error }}>{error}</text>}
        </box>
        <box style={{ flexGrow: 1 }} />
        <box style={{ paddingTop: 1, paddingBottom: 1, flexDirection: "row", justifyContent: "space-between", width: "100%", maxWidth: LOCAL_USAGE_SETUP_WIDTH }}>
          <text style={{ fg: theme.muted }}>↑/↓ move · Enter select</text>
          <box style={{ flexDirection: "row" }}>
            <Button onClick={() => { if (!busy) onSkip() }}><box style={{ borderStyle: "single", borderColor: theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}><text style={{ fg: theme.foreground }}>Skip for now (Esc)</text></box></Button>
            <text>  </text>
            <Button onClick={continueFromUsage}><box style={{ borderStyle: "single", borderColor: focusTarget === "continue" ? theme.primary : theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}><text style={{ fg: focusTarget === "continue" ? theme.primary : theme.foreground }}>See recommendations</text></box></Button>
          </box>
        </box>
      </box>
    )
  }

  const host = state.host._tag === "Available" ? state.host.profile : null
  const firstRunningIndex = selections.findIndex((selection) => selection.kind === "running")
  const firstStoredIndex = selections.findIndex((selection) => selection.kind === "stored")
  const firstRecommendationIndex = selections.findIndex((selection) => selection.kind === "recommendation")
  const hasExistingModels = firstRunningIndex >= 0 || firstStoredIndex >= 0
  return (
    <box style={{ flexDirection: "column", height: "100%", paddingLeft: 2, paddingRight: 2 }}>
      <box style={{ flexDirection: "column", paddingTop: 1, paddingBottom: 1 }}>
      <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>LOCAL MODEL SETUP</text>
      <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Choose what this machine should run</text>
      <text style={{ fg: theme.muted }}>
        {host ? `${formatBytes(host.systemMemoryBytes)} total system memory` : "System memory unavailable"}
      </text>
      </box>
      {state.distribution._tag !== "Ready" && (
        <box style={{ borderStyle: "single", customBorderChars: BOX_CHARS, borderColor: theme.warning, paddingLeft: 1, paddingRight: 1 }}>
          <text style={{ fg: theme.warning }}>Install llama.cpp to download or run a recommended model.</text>
        </box>
      )}
      <scrollbox
        ref={scrollRef}
        scrollX={false}
        scrollbarOptions={{ visible: false }}
        verticalScrollbarOptions={{ visible: true, trackOptions: { width: 1 } }}
        style={{ flexGrow: 1, rootOptions: { flexGrow: 1, backgroundColor: "transparent" }, wrapperOptions: { border: false, backgroundColor: "transparent" }, contentOptions: { flexDirection: "column" } }}
      >
        {selections.length === 0
          ? <text style={{ fg: theme.warning }}>No curated model currently fits this configuration.</text>
          : selections.map((selection, index) => {
            const sectionLabel = index === firstRunningIndex ? "RUNNING NOW"
              : index === firstStoredIndex ? "DOWNLOADED"
              : index === firstRecommendationIndex ? (hasExistingModels ? "POSSIBLE DOWNLOADS" : "RECOMMENDED DOWNLOADS")
              : null
            return <Fragment key={selection.id}>
            {sectionLabel && <box style={{ flexDirection: "row", paddingTop: index === 0 ? 0 : 1, paddingBottom: 1, width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH }}>
              <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{sectionLabel}</text>
              <text style={{ fg: theme.border }}>  {localModelSectionRule(sectionLabel)}</text>
            </box>}
            <box
              id={`local-model-${index}`}
              style={{ borderStyle: "single", customBorderChars: BOX_CHARS, borderColor: index === selectedIndex ? theme.primary : theme.border, paddingLeft: 1, paddingRight: 1, marginBottom: 1, flexDirection: "column", width: "100%", maxWidth: LOCAL_MODEL_SECTION_WIDTH }}
            >
              <text style={{ fg: index === selectedIndex ? theme.primary : theme.foreground }} attributes={TextAttributes.BOLD}>
                {index === selectedIndex ? "› " : "  "}{selectionTitle(selection)}
                <span fg={theme.primary}>{selection.kind === "recommendation" ? `  ${recommendationBadge(selection.recommendation.badge)}` : selection.kind === "running" ? "  Already Running" : "  Already Downloaded"}</span>
              </text>
              <text style={{ fg: theme.muted }}>{selectionMetadata(selection)}</text>
              {selection.kind === "recommendation" && <text style={{ fg: theme.muted }}>{selection.recommendation.quantization.fidelityLabel}</text>}
            </box>
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
      </scrollbox>
      {busy && <text style={{ fg: theme.primary }}>Applying local model changes…</text>}
      {error && <text style={{ fg: theme.error }}>{error}</text>}
      {state.warnings.map((warning) => <text key={warning.code} style={{ fg: theme.warning }}>{warning.message}</text>)}
      <box style={{ paddingTop: 1, paddingBottom: 1, flexShrink: 0, flexDirection: "row", justifyContent: "space-between" }}>
        <box style={{ flexDirection: "row" }}>
          <Button onClick={() => { if (!busy) setRequestedStage("usage") }}>
            <box style={{ borderStyle: "single", borderColor: theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
              <text style={{ fg: theme.foreground }}>Back (←)</text>
            </box>
          </Button>
          <text style={{ fg: theme.muted }}>
            {"  "}↑/↓ choose · D details
            {selected?.kind === "recommendation"
              ? state.distribution._tag === "Ready" ? " · Enter download" : " · Enter install"
              : " · Enter use"}
          </text>
        </box>
        <Button
          onClick={() => { if (!busy) onSkip() }}
          onMouseOver={() => setSkipHovered(true)}
          onMouseOut={() => setSkipHovered(false)}
        >
          <box style={{ borderStyle: "single", borderColor: skipHovered ? theme.primary : theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
            <text style={{ fg: skipHovered ? theme.primary : theme.foreground }}>
              Skip for now (Esc)
            </text>
          </box>
        </Button>
      </box>
    </box>
  )
})
