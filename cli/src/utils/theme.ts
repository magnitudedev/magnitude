import {
  blue,
  green,
  orange,
  red,
  slate,
  violet,
  type MarkdownPalette,
  type SyntaxColors,
} from "@magnitudedev/client-common"
import type {
  CliTheme,
  MarkdownHeadingLevel,
  TerminalAppearance,
  ThemeMode,
} from "../types/theme-system"

interface Rgb {
  readonly r: number
  readonly g: number
  readonly b: number
}

const DARK_BACKGROUND = "#111827"
const LIGHT_BACKGROUND = "#ffffff"

export const parseHexColor = (value: string): Rgb | null => {
  const cleaned = value.trim().replace(/^#/, "")
  if (!/^[0-9a-f]+$/i.test(cleaned)) return null
  if (cleaned.length === 3) {
    return {
      r: Number.parseInt(cleaned[0]! + cleaned[0]!, 16),
      g: Number.parseInt(cleaned[1]! + cleaned[1]!, 16),
      b: Number.parseInt(cleaned[2]! + cleaned[2]!, 16),
    }
  }
  if (cleaned.length === 6) {
    return {
      r: Number.parseInt(cleaned.slice(0, 2), 16),
      g: Number.parseInt(cleaned.slice(2, 4), 16),
      b: Number.parseInt(cleaned.slice(4, 6), 16),
    }
  }
  if (cleaned.length === 12) {
    return {
      r: Number.parseInt(cleaned.slice(0, 2), 16),
      g: Number.parseInt(cleaned.slice(4, 6), 16),
      b: Number.parseInt(cleaned.slice(8, 10), 16),
    }
  }
  return null
}

const linearChannel = (value: number): number => {
  const channel = value / 255
  return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4
}

export const relativeLuminance = (value: string): number | null => {
  const color = parseHexColor(value)
  if (!color) return null
  return 0.2126 * linearChannel(color.r)
    + 0.7152 * linearChannel(color.g)
    + 0.0722 * linearChannel(color.b)
}

export const contrastRatio = (foreground: string, background: string): number => {
  const foregroundLuminance = relativeLuminance(foreground)
  const backgroundLuminance = relativeLuminance(background)
  if (foregroundLuminance === null || backgroundLuminance === null) return 1
  const lighter = Math.max(foregroundLuminance, backgroundLuminance)
  const darker = Math.min(foregroundLuminance, backgroundLuminance)
  return (lighter + 0.05) / (darker + 0.05)
}

export const isLightBackground = (value: string): boolean =>
  (relativeLuminance(value) ?? 0) > 0.179

const DARK_ACTIVITY_PULSE = [
  blue[50], blue[100], blue[200], blue[300], blue[400], blue[500], blue[600], blue[700], blue[800], blue[900],
  blue[800], blue[700], blue[600], blue[500], blue[400], blue[300], blue[200], blue[100], blue[50],
] as const

const LIGHT_ACTIVITY_PULSE = [
  blue[900], blue[800], blue[700], blue[600], blue[500], blue[600], blue[700], blue[800], blue[900],
] as const

const channelHex = (value: number): string => Math.round(value).toString(16).padStart(2, "0")

const interpolate = (from: string, to: string, progress: number): string => {
  const start = parseHexColor(from)!
  const end = parseHexColor(to)!
  return `#${channelHex(start.r + (end.r - start.r) * progress)}${channelHex(start.g + (end.g - start.g) * progress)}${channelHex(start.b + (end.b - start.b) * progress)}`
}

const pulse = (from: string, to: string): readonly string[] =>
  Array.from({ length: 101 }, (_, index) => interpolate(from, to, index / 100))

const DARK_NEUTRAL_PULSE = pulse(slate[500], slate[200])
const LIGHT_NEUTRAL_PULSE = pulse(slate[500], slate[800])

export const resolveCliTheme = (appearance: TerminalAppearance): CliTheme => {
  const light = appearance.mode === "light"
  const backgroundFallback = light ? LIGHT_BACKGROUND : DARK_BACKGROUND
  const terminalBackground = parseHexColor(appearance.defaultBackground ?? "")
    ? appearance.defaultBackground!
    : backgroundFallback
  const body = light ? slate[900] : slate[100]
  const emphasized = light ? "#000000" : "#ffffff"
  const supporting = light ? slate[600] : slate[400]
  const metadata = light ? slate[600] : slate[400]
  const disabled = light ? slate[600] : slate[400]
  const accent = light ? blue[700] : blue[500]
  const link = light ? blue[700] : blue[400]
  const planAccent = light ? violet[600] : violet[300]
  const bashAccent = light ? orange[700] : orange[400]
  const information = light ? blue[700] : blue[500]
  const success = light ? green[700] : green[600]
  const warning = light ? violet[600] : violet[300]
  const failure = light ? red[600] : red[400]

  const syntax = {
    default: body,
    keyword: light ? violet[600] : violet[300],
    string: light ? green[700] : green[400],
    number: light ? blue[700] : blue[300],
    comment: slate[500],
    function: light ? blue[700] : blue[400],
    variable: light ? slate[700] : slate[200],
    type: light ? green[700] : green[400],
    operator: light ? blue[700] : slate[400],
    property: light ? blue[700] : slate[200],
    punctuation: light ? slate[500] : slate[400],
    literal: light ? blue[700] : blue[300],
  } satisfies SyntaxColors

  return {
    text: {
      body,
      emphasized,
      detail: light ? slate[700] : slate[300],
      guidance: light ? slate[700] : slate[200],
      supporting,
      metadata,
      placeholder: supporting,
      disabled,
    },
    background: {
      canvas: "transparent",
      terminal: terminalBackground,
      surface: light ? slate[100] : slate[900],
      input: light ? slate[150] : "#232f41",
      menu: light ? slate[150] : "#232f41",
      alternateRow: light ? slate[200] : slate[750],
      selected: light ? slate[150] : slate[800],
      hovered: light ? slate[200] : slate[750],
      focused: light ? slate[100] : slate[700],
      diffAdded: light ? "#e6f5ee" : "#122b22",
      diffRemoved: light ? "#f5e6e6" : "#2c1919",
      diffContext: "transparent",
    },
    border: {
      subtle: light ? slate[500] : slate[400],
      standard: light ? slate[300] : slate[600],
      strong: slate[500],
    },
    accent,
    highlightAccent: light ? violet[700] : violet[200],
    link,
    activityPulse: light ? LIGHT_ACTIVITY_PULSE : DARK_ACTIVITY_PULSE,
    neutralPulse: light ? LIGHT_NEUTRAL_PULSE : DARK_NEUTRAL_PULSE,
    planAccent,
    bashAccent,
    status: {
      information,
      progress: accent,
      success,
      achievement: light ? green[700] : green[300],
      changeAdded: light ? green[700] : green[500],
      warning,
      failure,
      interrupted: failure,
      terminated: light ? red[600] : red[500],
      inactive: slate[600],
    },
    markdown: {
      text: body,
      heading: {
        1: light ? blue[700] : blue[400],
        2: light ? blue[700] : blue[400],
        3: light ? blue[700] : blue[400],
        4: light ? blue[700] : blue[400],
        5: light ? blue[700] : blue[400],
        6: light ? blue[700] : blue[400],
      },
      link,
      inlineCode: light ? green[700] : green[400],
      codeText: body,
      codeBackground: "transparent",
      codeBorder: light ? slate[300] : slate[400],
      codeHeader: slate[500],
      listMarker: light ? slate[500] : slate[400],
      blockquoteText: light ? slate[700] : slate[200],
      blockquoteBorder: light ? slate[300] : slate[700],
      divider: light ? slate[200] : slate[800],
    },
    syntax,
  }
}

export const fallbackTerminalAppearances: Readonly<Record<ThemeMode, TerminalAppearance>> = {
  dark: {
    mode: "dark",
    defaultBackground: DARK_BACKGROUND,
  },
  light: {
    mode: "light",
    defaultBackground: LIGHT_BACKGROUND,
  },
}

export const defaultCliThemes: Readonly<Record<ThemeMode, CliTheme>> = {
  dark: resolveCliTheme(fallbackTerminalAppearances.dark),
  light: resolveCliTheme(fallbackTerminalAppearances.light),
}

export const buildMarkdownColorPalette = (theme: CliTheme): MarkdownPalette => ({
  inlineCodeFg: theme.markdown.inlineCode,
  codeBackground: theme.markdown.codeBackground,
  codeBorderColor: theme.markdown.codeBorder,
  codeHeaderFg: theme.markdown.codeHeader,
  headingFg: { ...theme.markdown.heading },
  listBulletFg: theme.markdown.listMarker,
  blockquoteBorderFg: theme.markdown.blockquoteBorder,
  blockquoteTextFg: theme.markdown.blockquoteText,
  dividerFg: theme.markdown.divider,
  codeTextFg: theme.markdown.codeText,
  codeMonochrome: false,
  linkFg: theme.markdown.link,
  syntax: theme.syntax,
})
