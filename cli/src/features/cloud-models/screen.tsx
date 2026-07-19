import { memo, useCallback, useMemo, useState } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import { Atom, useAtomMount, useAtomValue } from "@effect-atom/atom-react"
import { Effect } from "effect"
import { useSettingsState } from "@magnitudedev/client-common"
import { authSourceAtom } from "../../state/cli-atoms"
import { Button } from "../../components/button"
import { useTheme } from "../../hooks/use-theme"
import { BOX_CHARS } from "../../utils/ui-constants"
import { CloudModelsConnectScreen } from "./connect"

interface CloudModelsScreenProps {
  readonly onExit: () => void
}

export const CloudModelsScreen = memo(function CloudModelsScreen({
  onExit,
}: CloudModelsScreenProps) {
  const settings = useSettingsState()
  const authSource = useAtomValue(authSourceAtom)
  const [submitted, setSubmitted] = useState(false)
  const cloudConfigured = settings.keyAlreadySet || authSource.source === "env"

  const completionAtom = useMemo(
    () => Atom.make(Effect.sync(() => {
      if (submitted && cloudConfigured && !settings.saving) {
        setSubmitted(false)
        onExit()
      }
    })),
    [cloudConfigured, onExit, settings.saving, submitted],
  )
  useAtomMount(completionAtom)

  if (cloudConfigured) {
    return <CloudConfiguredScreen onExit={onExit} />
  }

  return (
    <CloudModelsConnectScreen
      onSubmit={(key) => {
        setSubmitted(true)
        settings.saveApiKey(key)
      }}
      onExit={onExit}
      busy={settings.saving}
      error={settings.saveError}
    />
  )
})

const CloudConfiguredScreen = memo(function CloudConfiguredScreen({
  onExit,
}: CloudModelsScreenProps) {
  const theme = useTheme()
  const [closeHovered, setCloseHovered] = useState(false)

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === "return" || key.name === "enter" || key.name === "escape") {
      key.preventDefault()
      onExit()
    }
  }, [onExit]))

  return (
    <box style={{ height: "100%", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
      <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>CLOUD MODELS</text>
      <text style={{ fg: theme.success }} attributes={TextAttributes.BOLD}>Magnitude Cloud is configured.</text>
      <text style={{ fg: theme.muted }}>Cloud models are ready.</text>
      <box style={{ paddingTop: 1 }}>
        <Button
          onClick={onExit}
          onMouseOver={() => setCloseHovered(true)}
          onMouseOut={() => setCloseHovered(false)}
        >
          <box style={{
            borderStyle: "single",
            borderColor: closeHovered ? theme.primary : theme.border,
            customBorderChars: BOX_CHARS,
            paddingLeft: 1,
            paddingRight: 1,
          }}>
            <text style={{ fg: closeHovered ? theme.primary : theme.foreground }}>Close (Esc)</text>
          </box>
        </Button>
      </box>
    </box>
  )
})
