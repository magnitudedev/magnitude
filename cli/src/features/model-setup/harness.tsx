import { useCallback, useMemo, useState, type ReactNode } from "react"
import { TextAttributes, type KeyEvent } from "@opentui/core"
import { useKeyboard } from "@opentui/react"
import {
  formatLocalModelDisplayName,
  type HarnessDestination,
  type HarnessId,
} from "@magnitudedev/client-common"
import type { LocalModel } from "@magnitudedev/sdk"
import { Button } from "../../components/button"
import { useTheme } from "../../hooks/use-theme"
import { SetupFrame } from "./setup-frame"

type HarnessChooserItem =
  | { readonly _tag: "Harness"; readonly harness: HarnessId }
  | { readonly _tag: "LaunchOnStartup" }
  | { readonly _tag: "InstallSkill" }

export function HarnessChooser({
  width,
  additionalRows = 0,
  model,
  destinations,
  applying,
  onContinue,
}: {
  readonly width: number
  readonly additionalRows?: number
  readonly model: LocalModel
  readonly destinations: ReadonlyArray<HarnessDestination>
  readonly applying: HarnessId | null
  readonly onContinue: (input: { harness: HarnessId; launchOnStartup: boolean; installSkill: boolean }) => void
}): ReactNode {
  const theme = useTheme()
  const selectable = useMemo(() => destinations.filter((row) => row.selectable), [destinations])
  const [selected, setSelected] = useState<HarnessId>(selectable[0]?.id ?? destinations[0].id)
  const [launchOnStartup, setLaunchOnStartup] = useState(true)
  const [installSkill, setInstallSkill] = useState(true)
  const items = useMemo<ReadonlyArray<HarnessChooserItem>>(() => [
    ...selectable.map((row) => ({ _tag: "Harness" as const, harness: row.id })),
    { _tag: "LaunchOnStartup" },
    { _tag: "InstallSkill" },
  ], [selectable])
  const [focusedIndex, setFocusedIndex] = useState(0)
  const focusedItem = items[focusedIndex] ?? items[0]
  const selectedRow = destinations.find((row) => row.id === selected) ?? destinations[0]
  const locked = applying !== null
  const submit = useCallback((harness: HarnessId = selectedRow.id) => {
    const row = destinations.find((destination) => destination.id === harness)
    if (!locked && row?.selectable) onContinue({ harness, launchOnStartup, installSkill })
  }, [destinations, installSkill, launchOnStartup, locked, onContinue, selectedRow.id])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (locked) return
    if (key.name === "up" || key.name === "k") {
      key.preventDefault()
      const nextIndex = Math.max(0, focusedIndex - 1)
      setFocusedIndex(nextIndex)
      const next = items[nextIndex]
      if (next?._tag === "Harness") setSelected(next.harness)
      return
    }
    if (key.name === "down" || key.name === "j" || key.name === "tab") {
      key.preventDefault()
      const nextIndex = Math.min(items.length - 1, focusedIndex + 1)
      setFocusedIndex(nextIndex)
      const next = items[nextIndex]
      if (next?._tag === "Harness") setSelected(next.harness)
      return
    }
    if (key.name === "space") {
      key.preventDefault()
      if (focusedItem?._tag === "LaunchOnStartup") setLaunchOnStartup((value) => !value)
      else if (focusedItem?._tag === "InstallSkill") setInstallSkill((value) => !value)
      return
    }
    if (key.name === "return" || key.name === "enter") {
      key.preventDefault()
      submit()
    }
  }, [focusedIndex, focusedItem, items, locked, submit]))

  const focusItem = useCallback((item: HarnessChooserItem) => {
    const index = items.findIndex((candidate) => candidate._tag === item._tag
      && (candidate._tag !== "Harness" || item._tag !== "Harness" || candidate.harness === item.harness))
    if (index >= 0) setFocusedIndex(index)
  }, [items])
  const chooseHarness = useCallback((row: HarnessDestination) => {
    if (!row.selectable || locked) return
    setSelected(row.id)
    focusItem({ _tag: "Harness", harness: row.id })
    submit(row.id)
  }, [focusItem, locked, submit])

  return (
    <SetupFrame
      width={width}
      stage="harness"
      additionalRows={additionalRows}
    >
      <text style={{ fg: theme.text.body }} attributes={TextAttributes.BOLD}>
        Where do you want to use {formatLocalModelDisplayName(model)}?
      </text>
      <box style={{ height: 1 }} />
      <box style={{ flexDirection: "column" }}>
        {destinations.map((row) => {
          const item = { _tag: "Harness" as const, harness: row.id }
          const focused = focusedItem?._tag === "Harness" && focusedItem.harness === row.id
          return (
            <Button key={row.id} cursor={row.selectable && !locked ? "pointer" : "default"} onMouseOver={() => { if (row.selectable && !locked) { setSelected(row.id); focusItem(item) } }} onClick={() => chooseHarness(row)}>
              <text style={{ fg: !row.selectable ? theme.text.disabled : focused ? theme.accent : theme.text.body }} attributes={focused ? TextAttributes.BOLD : TextAttributes.NONE}>
                {focused ? "› " : "  "}{row.name}
                {row.note ? <span fg={theme.accent}>{`  ${row.note}`}</span> : null}
                {row.note ? null : <span fg={!row.selectable ? theme.text.disabled : theme.text.supporting}>{`  ${row.availability}`}</span>}
              </text>
            </Button>
          )
        })}
      </box>
      <box style={{ height: 1 }} />
      <Button onMouseOver={() => focusItem({ _tag: "LaunchOnStartup" })} onClick={() => { focusItem({ _tag: "LaunchOnStartup" }); setLaunchOnStartup((value) => !value) }}>
        <text style={{ fg: focusedItem?._tag === "LaunchOnStartup" ? theme.accent : theme.text.body }}>{focusedItem?._tag === "LaunchOnStartup" ? "› " : "  "}{launchOnStartup ? "[x]" : "[ ]"} Launch Magnitude service on startup</text>
      </Button>
      <Button onMouseOver={() => focusItem({ _tag: "InstallSkill" })} onClick={() => { focusItem({ _tag: "InstallSkill" }); setInstallSkill((value) => !value) }}>
        <text style={{ fg: focusedItem?._tag === "InstallSkill" ? theme.accent : theme.text.body }}>{focusedItem?._tag === "InstallSkill" ? "› " : "  "}{installSkill ? "[x]" : "[ ]"} Install Magnitude skill to help agents manage local models</text>
      </Button>
      <box style={{ height: 1, minHeight: 1, flexShrink: 0 }} />
      <text style={{ fg: theme.text.guidance }} wrapMode="none">You can change harness connections later with magnitude connections.</text>
      <text style={{ fg: theme.text.supporting }} wrapMode="none">↑/↓ choose · Space toggle · Enter continue · Ctrl+C to exit</text>
    </SetupFrame>
  )
}
