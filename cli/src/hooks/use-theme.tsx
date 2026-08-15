import { Atom, useAtomValue } from "@effect-atom/atom-react"
import type { CliTheme, TerminalAppearance } from "../types/theme-system"
import { fallbackTerminalAppearances, resolveCliTheme } from "../utils/theme"

export const terminalAppearanceAtom = Atom.keepAlive(
  Atom.make<TerminalAppearance>(fallbackTerminalAppearances.dark),
)

export const themeAtom = Atom.make((get): CliTheme =>
  resolveCliTheme(get(terminalAppearanceAtom)),
)

export function useTheme(): CliTheme {
  return useAtomValue(themeAtom)
}
