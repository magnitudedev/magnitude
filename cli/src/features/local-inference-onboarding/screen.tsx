import { Fragment, memo, useCallback, useMemo, useRef, useState } from "react"
import { TextAttributes, type KeyEvent, type ScrollBoxRenderable } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import type {
  LocalInferenceOnboardingSnapshot,
  LocalModelRole,
  LocalSessionConcurrency,
} from "@magnitudedev/sdk"
import { useTheme } from "../../hooks/use-theme"
import { Button } from "../../components/button"
import { BOX_CHARS } from "../../utils/ui-constants"
import { MagnitudeLoginScreen } from "../app-shell/login"
import {
  buildLocalInferenceSelections,
  formatBytes,
  formatContext,
  selectionFidelity,
  selectionMetadata,
  selectionTitle,
} from "./view-model"
import { useLocalInferenceOnboarding } from "./use-local-inference-onboarding"

export type ModelSetupStep = "local" | "cloud"
export type ModelSetupMode = "onboarding" | "management"

interface ModelSetupProps {
  readonly snapshot: LocalInferenceOnboardingSnapshot
  readonly onExit: () => void
  readonly onComplete: () => void
  readonly initialStep?: ModelSetupStep
  readonly mode?: ModelSetupMode
}

interface LocalViewProps {
  readonly snapshot: LocalInferenceOnboardingSnapshot
  readonly onExit: () => void
  readonly onConfigured: () => void
  readonly onSkip: () => void
  readonly onBack: () => void
  readonly controller: ReturnType<typeof useLocalInferenceOnboarding>
}

interface ModelSetupViewProps extends ModelSetupProps {
  readonly controller: ReturnType<typeof useLocalInferenceOnboarding>
}

export const LOCAL_MODEL_SECTION_WIDTH = 72
export const LOCAL_USAGE_SETUP_WIDTH = 88
const SECTION_LABEL_GAP = 2

export type LocalUsageFocusTarget =
  | "main"
  | "subagent"
  | "one"
  | "up_to_three"
  | "continue"

export const LOCAL_USAGE_FOCUS_ORDER: readonly LocalUsageFocusTarget[] = [
  "main",
  "subagent",
  "one",
  "up_to_three",
  "continue",
]

export const moveLocalUsageFocus = (
  currentIndex: number,
  direction: -1 | 1,
): number => Math.max(0, Math.min(LOCAL_USAGE_FOCUS_ORDER.length - 1, currentIndex + direction))

export const localModelSectionRule = (label: string): string =>
  "─".repeat(Math.max(0, LOCAL_MODEL_SECTION_WIDTH - label.length - SECTION_LABEL_GAP))

const recommendationBadge = (
  recommendation: LocalInferenceOnboardingSnapshot["recommendations"][number],
): string => {
  if (recommendation.badge === "lighter") return "Smaller Model"
  if (recommendation.badge === "higher_fidelity") return "Higher Fidelity Option"
  if (recommendation.badge === "alternative") return "Alternative Option"
  return "Recommended"
}

export const finishModelSetup = async (
  mode: ModelSetupMode,
  completeOnboarding: () => Promise<boolean>,
  onComplete: () => void,
): Promise<boolean> => {
  if (mode === "onboarding" && !(await completeOnboarding())) return false
  onComplete()
  return true
}

export const connectCloudAndFinish = async (
  key: string,
  configureCloud: (key: string) => Promise<void>,
  finish: () => Promise<boolean>,
): Promise<boolean> => {
  await configureCloud(key)
  return finish()
}

export const PreparingLocalInferenceScreen = memo(function PreparingLocalInferenceScreen() {
  const theme = useTheme()
  return (
    <box style={{ height: "100%", alignItems: "center", justifyContent: "center" }}>
      <box style={{ flexDirection: "column", alignItems: "center" }}>
        <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>Preparing local inference</text>
        <text style={{ fg: theme.muted }}>Finding models for this machine…</text>
      </box>
    </box>
  )
})

export const ModelSetupOnboardingScreen = memo(function ModelSetupOnboardingScreen(props: ModelSetupProps) {
  const controller = useLocalInferenceOnboarding()
  return <ModelSetupOnboardingView {...props} controller={controller} />
})

export const ModelSetupOnboardingView = memo(function ModelSetupOnboardingView({
  snapshot,
  onExit,
  onComplete,
  initialStep = "local",
  mode = "onboarding",
  controller,
}: ModelSetupViewProps) {
  const [step, setStep] = useState<ModelSetupStep>(initialStep)
  const [localStage, setLocalStage] = useState<"usage" | "models">("usage")
  const [localModelRole, setLocalModelRole] = useState<LocalModelRole>(
    snapshot.usage.selection?.localModelRole ?? "main",
  )
  const [sessionConcurrency, setSessionConcurrency] = useState<LocalSessionConcurrency>(
    snapshot.usage.selection?.sessionConcurrency ?? "one",
  )
  const effectiveSnapshot = controller.snapshot ?? snapshot

  const finish = useCallback(
    () => finishModelSetup(mode, controller.completeOnboarding, onComplete),
    [controller.completeOnboarding, mode, onComplete],
  )

  if (step === "cloud") {
    return (
      <MagnitudeLoginScreen
        onSubmit={async (key) => {
          await connectCloudAndFinish(key, controller.configureCloud, finish)
        }}
        onSkip={async () => {
          await finish()
        }}
        onExit={onExit}
        onBack={() => setStep("local")}
        busy={controller.busy}
        error={controller.error}
      />
    )
  }

  const finishLocal = () => {
    if (mode === "management") {
      onComplete()
      return
    }
    if (controller.cloudKeyAlreadySet) {
      void finish()
      return
    }
    setStep("cloud")
  }

  if (localStage === "usage") {
    return (
      <LocalUsageSetupView
        snapshot={effectiveSnapshot}
        localModelRole={localModelRole}
        sessionConcurrency={sessionConcurrency}
        onSelectRole={setLocalModelRole}
        onSelectConcurrency={setSessionConcurrency}
        onContinue={() => {
          void controller.configureUsage({ localModelRole, sessionConcurrency }).then((result) => {
            if (result) setLocalStage("models")
          })
        }}
        onExit={onExit}
        onSkip={finishLocal}
        busy={controller.busy}
        error={controller.error}
      />
    )
  }

  const effectiveUsage = effectiveSnapshot.usage.selection
  if (
    controller.snapshotLoading
    || effectiveUsage?.localModelRole !== localModelRole
    || effectiveUsage?.sessionConcurrency !== sessionConcurrency
  ) {
    return <PreparingLocalInferenceScreen />
  }

  return (
    <LocalInferenceOnboardingView
      snapshot={effectiveSnapshot}
      onExit={onExit}
      onConfigured={finishLocal}
      onSkip={finishLocal}
      onBack={() => setLocalStage("usage")}
      controller={controller}
    />
  )
})

interface LocalUsageSetupViewProps {
  readonly snapshot: LocalInferenceOnboardingSnapshot
  readonly localModelRole: LocalModelRole
  readonly sessionConcurrency: LocalSessionConcurrency
  readonly onSelectRole: (role: LocalModelRole) => void
  readonly onSelectConcurrency: (concurrency: LocalSessionConcurrency) => void
  readonly onContinue: () => void
  readonly onSkip: () => void
  readonly onExit: () => void
  readonly busy: boolean
  readonly error?: string | null
}

interface LocalUsageOptionProps {
  readonly focused: boolean
  readonly selected: boolean
  readonly title: string
  readonly description: string
}

const LocalUsageOption = memo(function LocalUsageOption({
  focused,
  selected,
  title,
  description,
}: LocalUsageOptionProps) {
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

export const LocalUsageSetupView = memo(function LocalUsageSetupView({
  snapshot,
  localModelRole,
  sessionConcurrency,
  onSelectRole,
  onSelectConcurrency,
  onContinue,
  onSkip,
  onExit,
  busy,
  error,
}: LocalUsageSetupViewProps) {
  const theme = useTheme()
  const initialFocus = localModelRole === "subagent" ? 1 : 0
  const [focusIndex, setFocusIndex] = useState(initialFocus)
  const [skipHovered, setSkipHovered] = useState(false)
  const [continueHovered, setContinueHovered] = useState(false)
  const focusTarget = LOCAL_USAGE_FOCUS_ORDER[focusIndex]!
  const running = snapshot.running[0]

  const selectFocused = useCallback(() => {
    if (busy) return
    if (focusTarget === "main" || focusTarget === "subagent") {
      onSelectRole(focusTarget)
      return
    }
    if (focusTarget === "one" || focusTarget === "up_to_three") {
      onSelectConcurrency(focusTarget)
      return
    }
    onContinue()
  }, [busy, focusTarget, onContinue, onSelectConcurrency, onSelectRole])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      if (!busy) setFocusIndex((index) => moveLocalUsageFocus(index, -1))
      return
    }
    if (key.name === "down" || key.name === "j" || key.name === "tab") {
      key.preventDefault()
      if (!busy) setFocusIndex((index) => moveLocalUsageFocus(index, 1))
      return
    }
    if (key.name === "return" || key.name === "enter" || key.name === "space") {
      key.preventDefault()
      selectFocused()
      return
    }
    if (key.name === "escape") {
      key.preventDefault()
      if (!busy) onSkip()
      return
    }
    if (key.ctrl && key.name === "c") {
      key.preventDefault()
      onExit()
    }
  }, [busy, onExit, onSkip, selectFocused]))

  const runningMetadata = running
    ? [
      running.displayName,
      running.quantization?.format,
      `${formatContext(running.contextTokens)} context`,
      running.managed ? "Managed by Magnitude" : "Running outside Magnitude",
    ].filter((value): value is string => Boolean(value)).join(" · ")
    : null

  return (
    <box style={{ flexDirection: "column", height: "100%", paddingLeft: 2, paddingRight: 2 }}>
      <box style={{ flexDirection: "column", paddingTop: 1, flexShrink: 0, width: "100%", maxWidth: LOCAL_USAGE_SETUP_WIDTH }}>
        <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>LOCAL MODEL SETUP</text>
        <text style={{ fg: theme.foreground }}>Magnitude uses llama.cpp to run local models in the background.</text>
        <text style={{ fg: theme.muted }}>Answer two questions and we'll recommend Hugging Face models that fit your setup.</text>

        {running && (
          <box style={{ flexDirection: "column", paddingTop: 1, flexShrink: 0 }}>
            <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>● llama.cpp server detected</text>
            <text style={{ fg: theme.muted }}>  {runningMetadata}</text>
          </box>
        )}

        {/*
         * TODO(llamacpp-runtime-ownership, CTO-owned): The final bridge must
         * report external-versus-managed ownership, compatibility with the
         * selected serving profile, and whether an external process can be
         * safely restarted and adopted. Compatible external servers are reused
         * automatically. When a restart is required, show this confirmation:
         *
         *   Restart llama.cpp to apply this setup?
         *   Magnitude needs to restart your running llama.cpp server with
         *   different context settings. It will be briefly unavailable, then
         *   Magnitude will manage it in the background.
         *
         * Never restart from the CLI directly. If the bridge cannot restart it
         * safely, ask the user to stop it manually before starting the managed
         * server. Do not silently run two memory-heavy servers at once.
         */}

        <box style={{ flexDirection: "column", flexShrink: 0, paddingTop: running ? 1 : 0 }}>
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>How do you plan to use local models?</text>
          <LocalUsageOption
            focused={focusTarget === "main"}
            selected={localModelRole === "main"}
            title="As my main agent"
            description="One larger context window per active session."
          />
          <LocalUsageOption
            focused={focusTarget === "subagent"}
            selected={localModelRole === "subagent"}
            title="For local subagents"
            description="Uses a cloud main agent and reserves three context windows for local subagents."
          />
        </box>

        <box style={{ flexDirection: "column", flexShrink: 0, paddingTop: 1 }}>
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>How many Magnitude sessions will you run at once?</text>
          <LocalUsageOption
            focused={focusTarget === "one"}
            selected={sessionConcurrency === "one"}
            title="One session"
            description="Reserve one set of context windows."
          />
          <LocalUsageOption
            focused={focusTarget === "up_to_three"}
            selected={sessionConcurrency === "up_to_three"}
            title="Multiple sessions"
            description="Reserve up to three sets of context windows. This may result in a smaller recommended local model."
          />
        </box>

        {busy && <text style={{ fg: theme.primary }}>Finding models for this setup…</text>}
        {error && <text style={{ fg: theme.error }}>{error}</text>}
      </box>

      <box style={{ flexGrow: 1 }} />

      <box style={{ paddingTop: 1, paddingBottom: 1, flexShrink: 0, flexDirection: "row", justifyContent: "space-between", width: "100%", maxWidth: LOCAL_USAGE_SETUP_WIDTH }}>
        <text style={{ fg: theme.muted }}>↑/↓ move · Enter select</text>
        <box style={{ flexDirection: "row" }}>
          <Button
            onClick={() => { if (!busy) onSkip() }}
            onMouseOver={() => setSkipHovered(true)}
            onMouseOut={() => setSkipHovered(false)}
          >
            <box style={{ borderStyle: "single", borderColor: skipHovered ? theme.primary : theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
              <text style={{ fg: skipHovered ? theme.primary : theme.foreground }}>Skip for now (Esc)</text>
            </box>
          </Button>
          <text>  </text>
          <Button
            onClick={() => { if (!busy) onContinue() }}
            onMouseOver={() => setContinueHovered(true)}
            onMouseOut={() => setContinueHovered(false)}
          >
            <box style={{ borderStyle: "single", borderColor: focusTarget === "continue" || continueHovered ? theme.primary : theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
              <text style={{ fg: focusTarget === "continue" || continueHovered ? theme.primary : theme.foreground }}>See recommendations</text>
            </box>
          </Button>
        </box>
      </box>
    </box>
  )
})

export const LocalInferenceOnboardingView = memo(function LocalInferenceOnboardingView({
  snapshot,
  onExit,
  onConfigured,
  onSkip,
  onBack,
  controller,
}: LocalViewProps) {
  const theme = useTheme()
  const selections = useMemo(() => buildLocalInferenceSelections(snapshot), [snapshot])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [details, setDetails] = useState(false)
  const [skipHovered, setSkipHovered] = useState(false)
  const choicesScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const selected = selections[Math.min(selectedIndex, Math.max(0, selections.length - 1))]
  const firstRunningIndex = selections.findIndex((selection) => selection.kind === "running")
  const firstDownloadedIndex = selections.findIndex((selection) => selection.kind === "downloaded")
  const firstRecommendationIndex = selections.findIndex((selection) => selection.kind === "recommendation")
  const hasExistingModels = firstRunningIndex >= 0 || firstDownloadedIndex >= 0

  const confirm = useCallback(() => {
    if (!selected || controller.busy) return
    if (selected.kind === "recommendation") {
      if (!snapshot.runtime.canDownload) return
      if (controller.progress?.status === "ready") {
        const downloadedSelection = controller.progress.selectionId ?? controller.downloadConfigurationId
        if (!downloadedSelection) return
        void controller.activate(downloadedSelection).then((completed) => {
          if (completed) onConfigured()
        })
      } else if (!controller.operationId) {
        void controller.startDownload(selected.id)
      }
      return
    }
    if (!snapshot.runtime.canActivate) return
    void controller.activate(selected.id).then((completed) => {
      if (completed) onConfigured()
    })
  }, [controller, onConfigured, selected, snapshot.runtime.canActivate, snapshot.runtime.canDownload])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      if (controller.operationId) return
      setSelectedIndex((index) => {
        const next = Math.max(0, index - 1)
        queueMicrotask(() => choicesScrollRef.current?.scrollChildIntoView(`local-model-choice-${next}`))
        return next
      })
      setDetails(false)
      return
    }
    if (key.name === "down" || key.name === "j" || key.name === "tab") {
      key.preventDefault()
      if (controller.operationId) return
      setSelectedIndex((index) => {
        const next = Math.min(selections.length - 1, index + 1)
        queueMicrotask(() => choicesScrollRef.current?.scrollChildIntoView(`local-model-choice-${next}`))
        return next
      })
      setDetails(false)
      return
    }
    if (key.name === "d") {
      key.preventDefault()
      setDetails((value) => !value)
      return
    }
    if (key.name === "x" && controller.operationId) {
      key.preventDefault()
      void controller.cancelDownload()
      return
    }
    if (key.name === "return" || key.name === "enter") {
      key.preventDefault()
      confirm()
      return
    }
    if (key.name === "backspace" || key.name === "left") {
      key.preventDefault()
      if (!controller.operationId) onBack()
      return
    }
    if (key.ctrl && key.name === "c") {
      key.preventDefault()
      onExit()
      return
    }
    if (key.name === "escape") {
      key.preventDefault()
      if (details) setDetails(false)
      else if (!controller.operationId) onSkip()
    }
  }, [confirm, controller, details, onBack, onExit, onSkip, selections.length]))

  const totalMemory = snapshot.capabilities?.system.totalMemoryBytes
  const acceleratorNames = snapshot.capabilities?.accelerators.map((item) =>
    `${item.backend}: ${item.description}${item.capacityBytes ? ` (${formatBytes(item.capacityBytes)} total)` : ""}`,
  ) ?? []
  const progressPercent = controller.progress && controller.progress.totalBytes > 0
    ? Math.min(100, Math.floor(controller.progress.completedBytes / controller.progress.totalBytes * 100))
    : 0

  return (
    <box style={{ flexDirection: "column", height: "100%", paddingLeft: 2, paddingRight: 2 }}>
      <box style={{ flexDirection: "column", paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>LOCAL MODEL SETUP</text>
        <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Choose what this machine should run</text>
        <text style={{ fg: theme.muted }}>
          {totalMemory ? `${formatBytes(totalMemory)} total system memory` : "System memory unavailable"}
          {acceleratorNames.length > 0 ? ` · ${acceleratorNames.join(", ")}` : ""}
        </text>
      </box>

      {/*
       * TODO(llamacpp-binary-bootstrap-integration, CTO-owned): When the final
       * runtime contract distinguishes a missing binary, render its Install
       * action and package-reported progress here. The CLI must invoke the ACN
       * adapter only; it must never detect, download, unpack, or chmod a
       * llama.cpp binary itself.
       */}
      {snapshot.runtime.status !== "ready" && (
        <box style={{ borderStyle: "single", borderColor: theme.warning, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1, marginBottom: 1 }}>
          <text style={{ fg: theme.warning }}>Install llama.cpp to download or run a recommended model.</text>
        </box>
      )}

      <scrollbox
        ref={choicesScrollRef}
        scrollX={false}
        scrollbarOptions={{ visible: false }}
        verticalScrollbarOptions={{ visible: true, trackOptions: { width: 1 } }}
        style={{
          flexGrow: 1,
          rootOptions: { flexGrow: 1, backgroundColor: "transparent" },
          wrapperOptions: { border: false, backgroundColor: "transparent" },
          contentOptions: { flexDirection: "column" },
        }}
      >
        {selections.length === 0 ? (
          <text style={{ fg: theme.warning }}>No recommended model fits the detected hardware.</text>
        ) : selections.map((selection, index) => {
          const isSelected = index === selectedIndex
          const recommendation = selection.kind === "recommendation" ? selection.recommendation : null
          const sectionLabel = index === firstRunningIndex
            ? "RUNNING NOW"
            : index === firstDownloadedIndex
              ? "DOWNLOADED"
              : index === firstRecommendationIndex
                ? hasExistingModels ? "POSSIBLE DOWNLOADS" : "RECOMMENDED DOWNLOADS"
                : null
          return (
            <Fragment key={selection.id}>
              {sectionLabel && (
                <box style={{
                  flexDirection: "row",
                  paddingTop: index === 0 ? 0 : 1,
                  paddingBottom: 1,
                  flexShrink: 0,
                  width: "100%",
                  maxWidth: LOCAL_MODEL_SECTION_WIDTH,
                }}>
                  <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{sectionLabel}</text>
                  <text style={{ fg: theme.border }}>  {localModelSectionRule(sectionLabel)}</text>
                </box>
              )}
              <box
                id={`local-model-choice-${index}`}
                style={{
                  borderStyle: "single",
                  customBorderChars: BOX_CHARS,
                  borderColor: isSelected ? theme.primary : theme.border,
                  paddingLeft: 1,
                  paddingRight: 1,
                  marginBottom: 1,
                  flexDirection: "column",
                  flexShrink: 0,
                  width: "100%",
                  maxWidth: LOCAL_MODEL_SECTION_WIDTH,
                }}
              >
                <box style={{ flexDirection: "row" }}>
                  <text style={{ fg: isSelected ? theme.primary : theme.muted }}>{isSelected ? "› " : "  "}</text>
                  <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{selectionTitle(selection)}</text>
                  {recommendation && (
                    <text style={{ fg: theme.primary }}>  {recommendationBadge(recommendation)}</text>
                  )}
                  {!recommendation && (
                    <text style={{ fg: theme.primary }}>
                      {selection.kind === "running" ? "  Already Running" : "  Already Downloaded"}
                    </text>
                  )}
                </box>
                <text style={{ fg: theme.muted }}>{selectionMetadata(selection)}</text>
                {selectionFidelity(selection) && (
                  <text style={{ fg: theme.muted }}>{selectionFidelity(selection)}</text>
                )}
              </box>
            </Fragment>
          )
        })}

        {details && selected?.kind === "recommendation" && (
          <box style={{ flexDirection: "column", paddingLeft: 1 }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Exact artifact</text>
            <text style={{ fg: theme.muted }}>{selected.recommendation.repo}@{selected.recommendation.revision}</text>
            {selected.recommendation.files.map((file) => (
              <box key={file.path} style={{ flexDirection: "column" }}>
                <text style={{ fg: theme.muted }}>{file.path} · SHA-256 {file.sha256}</text>
                <text style={{ fg: theme.primary }}>{file.downloadUrl}</text>
              </box>
            ))}
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

      {controller.progress && (
        <box style={{ flexDirection: "column", paddingBottom: 1 }}>
          <text style={{ fg: theme.primary }}>
            {controller.progress.status} · {progressPercent}% · {formatBytes(controller.progress.completedBytes)} / {formatBytes(controller.progress.totalBytes)}
          </text>
          {controller.progress.currentFile && <text style={{ fg: theme.muted }}>{controller.progress.currentFile}</text>}
        </box>
      )}
      {controller.error && <text style={{ fg: theme.error }}>{controller.error}</text>}
      {snapshot.warnings.map((warning) => <text key={warning.code} style={{ fg: theme.warning }}>{warning.message}</text>)}

      <box style={{ paddingTop: 1, paddingBottom: 1, flexShrink: 0, flexDirection: "row", justifyContent: "space-between" }}>
        <box style={{ flexDirection: "row" }}>
          <Button onClick={() => { if (!controller.operationId) onBack() }}>
            <box style={{ borderStyle: "single", borderColor: theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
              <text style={{ fg: theme.foreground }}>Back (←)</text>
            </box>
          </Button>
          <text style={{ fg: theme.muted }}>
            {"  "}↑/↓ choose · D details
            {selected?.kind === "recommendation"
              ? controller.progress?.status === "ready"
                ? " · Enter start"
                : snapshot.runtime.canDownload ? " · Enter download" : " · Unavailable"
              : snapshot.runtime.canActivate ? " · Enter use" : " · Unavailable"}
            {controller.operationId ? " · X cancel" : ""}
          </text>
        </box>
        <Button
          onClick={() => {
            if (!controller.operationId) onSkip()
          }}
          onMouseOver={() => setSkipHovered(true)}
          onMouseOut={() => setSkipHovered(false)}
        >
          <box style={{
            borderStyle: "single",
            borderColor: skipHovered ? theme.primary : theme.border,
            customBorderChars: BOX_CHARS,
            paddingLeft: 1,
            paddingRight: 1,
          }}>
            <text style={{ fg: skipHovered ? theme.primary : theme.foreground }}>
              {controller.operationId ? "Cancel download before skipping" : "Skip for now (Esc)"}
            </text>
          </box>
        </Button>
      </box>
    </box>
  )
})
