import { memo, useCallback, useState } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import type { UpdateAction } from "@magnitudedev/sdk"
import {
  updateCommandString,
  updateReleaseNotesUrl,
} from "@magnitudedev/sdk"
import { useTheme } from "../../hooks/use-theme"

export type UpdatePromptOutcome =
  | { readonly _tag: "Update" }
  | { readonly _tag: "Skip" }
  | { readonly _tag: "Dismiss" }

type UpdateSelection = UpdatePromptOutcome["_tag"]

const selections: ReadonlyArray<UpdateSelection> = [
  "Update",
  "Skip",
  "Dismiss",
]

export const adjacentUpdateSelection = (
  selection: UpdateSelection,
  direction: -1 | 1,
): UpdateSelection => {
  const index = selections.indexOf(selection)
  return selections[(index + direction + selections.length) % selections.length]!
}

const selectionForNumber = (name: string): UpdateSelection | null => {
  if (name === "1") return "Update"
  if (name === "2") return "Skip"
  if (name === "3") return "Dismiss"
  return null
}

const labelFor = (
  selection: UpdateSelection,
  command: string,
): string => {
  if (selection === "Update") return `Update now (runs \`${command}\`)`
  if (selection === "Skip") return "Skip"
  return "Skip until next version"
}

export const UpdatePrompt = memo(function UpdatePrompt({
  currentVersion,
  latestVersion,
  action,
  onSelect,
}: {
  readonly currentVersion: string
  readonly latestVersion: string
  readonly action: UpdateAction
  readonly onSelect: (outcome: UpdatePromptOutcome) => void
}) {
  const theme = useTheme()
  const [highlighted, setHighlighted] = useState<UpdateSelection>("Update")
  const command = updateCommandString(action)

  const choose = useCallback((selection: UpdateSelection) => {
    onSelect({ _tag: selection })
  }, [onSelect])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.defaultPrevented) return
    if (key.ctrl && (key.name === "c" || key.name === "d")) {
      key.preventDefault()
      choose("Skip")
      return
    }
    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      setHighlighted((current) => adjacentUpdateSelection(current, -1))
      return
    }
    if (key.name === "down" || key.name === "j") {
      key.preventDefault()
      setHighlighted((current) => adjacentUpdateSelection(current, 1))
      return
    }
    if (key.name === "escape") {
      key.preventDefault()
      choose("Skip")
      return
    }
    const numbered = selectionForNumber(key.name)
    if (numbered !== null) {
      key.preventDefault()
      choose(numbered)
      return
    }
    if (key.name === "return" || key.name === "enter") {
      key.preventDefault()
      choose(highlighted)
    }
  }, [choose, highlighted]))

  return (
    <box
      style={{
        width: "100%",
        height: "100%",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <box
        style={{
          width: "90%",
          maxWidth: 88,
          flexDirection: "column",
          paddingLeft: 2,
          paddingRight: 2,
        }}
      >
        <text style={{ fg: theme.text.body }}>
          <span fg={theme.accent} attributes={TextAttributes.BOLD}>Update available! </span>
          <span>{currentVersion}</span>
          <span fg={theme.text.supporting}> -&gt; </span>
          <span>{latestVersion}</span>
        </text>
        <box style={{ height: 1 }} />
        <text style={{ fg: theme.text.supporting }}>
          Release notes:{" "}
          <span attributes={TextAttributes.UNDERLINE}>
            {updateReleaseNotesUrl(latestVersion)}
          </span>
        </text>
        <box style={{ height: 1 }} />
        {selections.map((selection, index) => (
          <text
            key={selection}
            style={{
              fg: highlighted === selection ? theme.accent : theme.text.body,
            }}
            attributes={highlighted === selection
              ? TextAttributes.BOLD
              : TextAttributes.NONE}
          >
            {highlighted === selection ? "› " : "  "}
            {index + 1}. {labelFor(selection, command)}
          </text>
        ))}
        <box style={{ height: 1 }} />
        <text style={{ fg: theme.text.supporting }}>Press Enter to continue</text>
      </box>
    </box>
  )
})
