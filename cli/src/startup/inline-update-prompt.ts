import * as Terminal from "@effect/platform/Terminal"
import {
  updateCommandString,
  type UpdateAction,
} from "@magnitudedev/release"
import ansis from "ansis"
import { Effect } from "effect"
import type { CliTheme } from "../types/theme-system"
import { updateReleaseNotesUrl } from "../features/update/updater"

export type InlineUpdatePromptOutcome =
  | { readonly _tag: "Update" }
  | { readonly _tag: "Skip" }
  | { readonly _tag: "Dismiss" }

type UpdateSelection = InlineUpdatePromptOutcome["_tag"]

const selections: ReadonlyArray<UpdateSelection> = [
  "Update",
  "Skip",
  "Dismiss",
]

export const adjacentInlineUpdateSelection = (
  selection: UpdateSelection,
  direction: -1 | 1,
): UpdateSelection => {
  const index = selections.indexOf(selection)
  return selections[(index + direction + selections.length) % selections.length]!
}

const labelFor = (selection: UpdateSelection, command: string): string => {
  if (selection === "Update") return `Update now (runs \`${command}\`)`
  if (selection === "Skip") return "Skip"
  return "Skip until next version"
}

const hyperlink = (url: string, label: string): string =>
  `\u001b]8;;${url}\u0007${label}\u001b]8;;\u0007`

export const renderInlineUpdatePrompt = (
  currentVersion: string,
  latestVersion: string,
  action: UpdateAction,
  highlighted: UpdateSelection,
  theme: CliTheme,
  color: boolean,
): ReadonlyArray<string> => {
  const style = color
    ? {
        accent: ansis.hex(theme.accent).bold,
        body: ansis.hex(theme.text.body),
        supporting: ansis.hex(theme.text.supporting),
        link: ansis.hex(theme.link).underline,
      }
    : {
        accent: (value: string) => value,
        body: (value: string) => value,
        supporting: (value: string) => value,
        link: (value: string) => value,
      }
  const command = updateCommandString(action)
  const releaseNotes = updateReleaseNotesUrl(latestVersion)
  return [
    `${style.accent("Update available!")} ${style.body(currentVersion)} ${style.supporting("→")} ${style.body(latestVersion)}`,
    "",
    `${style.supporting("Release notes:")} ${color ? hyperlink(releaseNotes, style.link(releaseNotes)) : releaseNotes}`,
    "",
    ...selections.map((selection, index) => {
      const selected = highlighted === selection
      const row = `${selected ? "›" : " "} ${index + 1}. ${labelFor(selection, command)}`
      return selected ? style.accent(row) : style.body(row)
    }),
    "",
    style.supporting("Press enter to continue"),
  ]
}

const clearRenderedLines = (lineCount: number): string =>
  `\u001b[${lineCount}A\r\u001b[0J`

const completionCopy = (selection: UpdateSelection): string => {
  if (selection === "Update") return "Updating now"
  if (selection === "Dismiss") return "Skipped until a newer version is available"
  return "Skipped for this launch"
}

export const runInlineUpdatePrompt = (
  input: {
    readonly currentVersion: string
    readonly latestVersion: string
    readonly action: UpdateAction
    readonly theme: CliTheme
  },
): Effect.Effect<InlineUpdatePromptOutcome, never, Terminal.Terminal> => Effect.scoped(
  Effect.gen(function* () {
    const terminal = yield* Terminal.Terminal
    const tty = yield* terminal.isTTY
    if (!tty) return { _tag: "Skip" } as const

    const mailbox = yield* terminal.readInput
    let highlighted: UpdateSelection = "Update"
    let rendered = renderInlineUpdatePrompt(
      input.currentVersion,
      input.latestVersion,
      input.action,
      highlighted,
      input.theme,
      true,
    )
    yield* terminal.display(`${rendered.join("\n")}\n`).pipe(Effect.orDie)

    while (true) {
      const event = yield* mailbox.take.pipe(
        Effect.catchAll(() => Effect.succeed({
          input: undefined,
          key: { name: "escape", ctrl: false, meta: false, shift: false },
        } as const)),
      )
      let selected: UpdateSelection | null = null
      if (event.key.ctrl && event.key.name === "c") {
        yield* Effect.sync(() => process.kill(process.pid, "SIGINT"))
        return yield* Effect.never
      } else if (event.key.name === "up") {
        highlighted = adjacentInlineUpdateSelection(highlighted, -1)
      } else if (event.key.name === "down") {
        highlighted = adjacentInlineUpdateSelection(highlighted, 1)
      } else if (event.key.name === "escape") {
        selected = "Skip"
      } else if (event.key.name === "return" || event.key.name === "enter") {
        selected = highlighted
      } else {
        continue
      }

      if (selected !== null) {
        yield* terminal.display(clearRenderedLines(rendered.length)).pipe(Effect.orDie)
        yield* terminal.display(
          `Update available! ${input.currentVersion} → ${input.latestVersion}\n${completionCopy(selected)}\n`,
        ).pipe(Effect.orDie)
        return { _tag: selected }
      }

      const next = renderInlineUpdatePrompt(
        input.currentVersion,
        input.latestVersion,
        input.action,
        highlighted,
        input.theme,
        true,
      )
      yield* terminal.display(
        `${clearRenderedLines(rendered.length)}${next.join("\n")}\n`,
      ).pipe(Effect.orDie)
      rendered = next
    }
  }),
)
