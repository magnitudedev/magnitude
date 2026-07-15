import { memo, useCallback, useMemo, useState } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { useSettingsState } from "@magnitudedev/client-common"
import { Atom, useAtomMount, useAtomValue } from "@effect-atom/atom-react"
import { Effect } from "effect"
import { authSourceAtom } from "../../state/cli-atoms"
import { useTheme } from "../../hooks/use-theme"
import { MagnitudeLoginScreen } from "../app-shell/login"
import { LocalInferenceScreen } from "../local-inference"

export type ModelSetupStep = "local" | "cloud"
export type ModelSetupMode = "onboarding" | "management"
export type ModelSetupSurface = "local-inference" | "cloud-provider"

export function resolveModelSetupSurface(
  _mode: ModelSetupMode,
  step: ModelSetupStep,
): ModelSetupSurface {
  return step === "cloud" ? "cloud-provider" : "local-inference"
}

interface ModelSetupScreenProps {
  readonly onExit: () => void
  readonly onComplete?: () => void
  readonly initialStep?: ModelSetupStep
  readonly mode?: ModelSetupMode
  readonly completing?: boolean
  readonly completionError?: string | null
}

export const PreparingModelSetupScreen = memo(function PreparingModelSetupScreen() {
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

function CloudSetup({ onFinish, onBack, onExit, completing, completionError }: {
  readonly onFinish: () => void
  readonly onBack: () => void
  readonly onExit: () => void
  readonly completing: boolean
  readonly completionError: string | null
}) {
  const settings = useSettingsState()
  const authSource = useAtomValue(authSourceAtom)
  const [submitted, setSubmitted] = useState(false)
  const cloudConfigured = settings.keyAlreadySet
    || authSource.source === "env"
    || authSource.source === "env-local"

  const completionAtom = useMemo(
    () => Atom.make(Effect.sync(() => {
      if (submitted && cloudConfigured && !settings.saving) {
        setSubmitted(false)
        onFinish()
      }
    })),
    [cloudConfigured, onFinish, settings.saving, submitted],
  )
  useAtomMount(completionAtom)

  if (cloudConfigured) {
    return <CloudConfigured onFinish={onFinish} onBack={onBack} onExit={onExit} />
  }
  return (
    <MagnitudeLoginScreen
      onSubmit={(key) => {
        setSubmitted(true)
        settings.saveApiKey(key)
      }}
      onSkip={onFinish}
      onExit={onExit}
      onBack={onBack}
      busy={settings.saving || completing}
      error={settings.saveError ?? completionError}
    />
  )
}

function CloudConfigured({ onFinish, onBack, onExit }: {
  readonly onFinish: () => void
  readonly onBack: () => void
  readonly onExit: () => void
}) {
  const theme = useTheme()
  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === "return" || key.name === "enter") { key.preventDefault(); onFinish(); return }
    if (key.name === "left" || key.name === "backspace") { key.preventDefault(); onBack(); return }
    if (key.ctrl && key.name === "c") { key.preventDefault(); onExit() }
  }, [onBack, onExit, onFinish]))
  return (
    <box style={{ height: "100%", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
      <text style={{ fg: theme.success }} attributes={TextAttributes.BOLD}>Magnitude Cloud is configured.</text>
      <text style={{ fg: theme.muted }}>Press Enter to finish model setup or ← to configure local inference.</text>
    </box>
  )
}

export const ModelSetupScreen = memo(function ModelSetupScreen({
  onExit,
  onComplete,
  initialStep = "local",
  mode = "onboarding",
  completing = false,
  completionError = null,
}: ModelSetupScreenProps) {
  const [step, setStep] = useState<ModelSetupStep>(initialStep)
  const settings = useSettingsState()
  const authSource = useAtomValue(authSourceAtom)
  const cloudConfigured = settings.keyAlreadySet
    || authSource.source === "env"
    || authSource.source === "env-local"

  const finish = useCallback(() => {
    (onComplete ?? onExit)()
  }, [onComplete, onExit])

  const finishLocal = useCallback(() => {
    if (mode === "management") onExit()
    else if (cloudConfigured) finish()
    else setStep("cloud")
  }, [cloudConfigured, finish, mode, onExit])

  if (resolveModelSetupSurface(mode, step) === "cloud-provider") {
    return (
      <CloudSetup
        onFinish={finish}
        onExit={onExit}
        onBack={() => mode === "management" ? onExit() : setStep("local")}
        completing={completing}
        completionError={completionError}
      />
    )
  }

  return (
    <LocalInferenceScreen
      management={mode === "management"}
      onBack={() => mode === "management" ? onExit() : setStep("cloud")}
      onSkip={finishLocal}
      onConfigured={finishLocal}
    />
  )
})
