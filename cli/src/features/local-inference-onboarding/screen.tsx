import { Fragment, memo, useCallback, useMemo, useRef, useState } from "react"
import { TextAttributes, type KeyEvent, type ScrollBoxRenderable } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import type { LocalInferenceOnboardingSnapshot } from "@magnitudedev/sdk"
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
  readonly controller: ReturnType<typeof useLocalInferenceOnboarding>
}

interface ModelSetupViewProps extends ModelSetupProps {
  readonly controller: ReturnType<typeof useLocalInferenceOnboarding>
}

const quantRank = (bitsClass: string): number => ({
  q4: 4,
  mxfp4: 4,
  q5: 5,
  q6: 6,
  q8: 8,
  fp8: 8,
})[bitsClass] ?? 0

const recommendationBadge = (
  recommendation: LocalInferenceOnboardingSnapshot["recommendations"][number],
  bestExistingQuantRank: number,
): string => {
  if (recommendation.badge === "lighter") return "Lighter Weight Model"
  if (recommendation.badge === "higher_fidelity") return "Higher Fidelity Option"
  if (bestExistingQuantRank > 0) {
    return quantRank(recommendation.quantization.bitsClass) > bestExistingQuantRank
      ? "Higher Fidelity Option"
      : "Alternative Option"
  }
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
  return (
    <LocalInferenceOnboardingView
      snapshot={snapshot}
      onExit={onExit}
      onConfigured={finishLocal}
      onSkip={finishLocal}
      controller={controller}
    />
  )
})

export const LocalInferenceOnboardingView = memo(function LocalInferenceOnboardingView({
  snapshot,
  onExit,
  onConfigured,
  onSkip,
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
  const bestExistingQuantRank = Math.max(
    0,
    ...[...snapshot.running, ...snapshot.downloaded]
      .filter((choice) => choice.compatible)
      .map((choice) => quantRank(choice.quantization?.bitsClass ?? "other")),
  )

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
  }, [confirm, controller, details, onExit, onSkip, selections.length]))

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
                <box style={{ flexDirection: "row", paddingTop: index === 0 ? 0 : 1, paddingBottom: 1, flexShrink: 0 }}>
                  <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{sectionLabel}</text>
                  <text style={{ fg: theme.border }}>  {'─'.repeat(48)}</text>
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
                }}
              >
                <box style={{ flexDirection: "row" }}>
                  <text style={{ fg: isSelected ? theme.primary : theme.muted }}>{isSelected ? "› " : "  "}</text>
                  <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{selectionTitle(selection)}</text>
                  {recommendation && (
                    <text style={{ fg: theme.primary }}>  {recommendationBadge(recommendation, bestExistingQuantRank)}</text>
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
              Estimated {formatBytes(selected.recommendation.estimatedRuntimeBytes)} runtime at {formatContext(selected.recommendation.contextTokens)} context; {formatBytes(selected.recommendation.fitMarginBytes)} stable headroom. Model maximum {formatContext(selected.recommendation.modelMaximumContextTokens)}.
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
        <text style={{ fg: theme.muted }}>
          ↑/↓ · D details
          {selected?.kind === "recommendation"
            ? controller.progress?.status === "ready"
              ? " · Enter start"
              : snapshot.runtime.canDownload ? " · Enter download" : " · Download unavailable"
            : snapshot.runtime.canActivate ? " · Enter use" : " · Model unavailable"}
          {controller.operationId ? " · X cancel" : ""} · Ctrl+C
        </text>
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
