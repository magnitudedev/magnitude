import { memo, useCallback } from "react"
import { TextAttributes } from "@opentui/core"
import { useTheme } from "../../hooks/use-theme"
import { LocalInferenceScreen } from "../local-inference"

interface ModelSetupScreenProps {
  readonly onExit: () => void
  readonly onComplete?: () => void
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

export const ModelSetupScreen = memo(function ModelSetupScreen({
  onExit,
  onComplete,
}: ModelSetupScreenProps) {
  const finish = useCallback(() => {
    (onComplete ?? onExit)()
  }, [onComplete, onExit])

  return (
    <LocalInferenceScreen
      onExit={onExit}
      onSkip={finish}
      onConfigured={finish}
    />
  )
})
