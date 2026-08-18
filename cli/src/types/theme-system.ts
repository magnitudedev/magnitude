import type { SyntaxColors } from "@magnitudedev/client-common"

export type ThemeMode = "dark" | "light"
export type MarkdownHeadingLevel = 1 | 2 | 3 | 4 | 5 | 6

export interface TerminalAppearance {
  readonly mode: ThemeMode
  readonly defaultBackground: string | null
}

export interface CliTheme {
  readonly text: {
    readonly body: string
    readonly emphasized: string
    readonly detail: string
    readonly guidance: string
    readonly supporting: string
    readonly metadata: string
    readonly placeholder: string
    readonly disabled: string
  }
  readonly background: {
    readonly canvas: string
    readonly terminal: string
    readonly surface: string
    readonly input: string
    readonly menu: string
    readonly alternateRow: string
    readonly selected: string
    readonly hovered: string
    readonly focused: string
    readonly diffAdded: string
    readonly diffRemoved: string
    readonly diffContext: string
  }
  readonly border: {
    readonly subtle: string
    readonly standard: string
    readonly strong: string
  }
  readonly accent: string
  readonly highlightAccent: string
  readonly link: string
  readonly activityPulse: readonly string[]
  readonly neutralPulse: readonly string[]
  readonly planAccent: string
  readonly bashAccent: string
  readonly status: {
    readonly information: string
    readonly progress: string
    readonly success: string
    readonly achievement: string
    readonly changeAdded: string
    readonly warning: string
    readonly failure: string
    readonly interrupted: string
    readonly terminated: string
    readonly inactive: string
  }
  readonly markdown: {
    readonly text: string
    readonly heading: Readonly<Record<MarkdownHeadingLevel, string>>
    readonly link: string
    readonly inlineCode: string
    readonly codeText: string
    readonly codeBackground: string
    readonly codeBorder: string
    readonly codeHeader: string
    readonly listMarker: string
    readonly blockquoteText: string
    readonly blockquoteBorder: string
    readonly divider: string
  }
  readonly syntax: SyntaxColors
}
