import { describe, expect, it } from "vitest"
import { blue, green, orange, red, slate, violet } from "@magnitudedev/client-common"
import type { TerminalAppearance } from "../types/theme-system"
import { contrastRatio, resolveCliTheme } from "./theme"

const appearance = (
  mode: TerminalAppearance["mode"],
  background: string,
): TerminalAppearance => ({
  mode,
  defaultBackground: background,
})

describe("CLI theme resolution", () => {
  it("preserves the established dark CLI palette exactly", () => {
    const theme = resolveCliTheme(appearance("dark", "#101010"))

    expect(theme.text).toEqual({
      body: slate[100],
      emphasized: "#ffffff",
      detail: slate[300],
      guidance: slate[200],
      supporting: slate[400],
      metadata: slate[400],
      placeholder: slate[400],
      disabled: slate[400],
    })
    expect(theme.accent).toBe(blue[500])
    expect(theme.highlightAccent).toBe(violet[200])
    expect(theme.link).toBe(blue[400])
    expect(theme.planAccent).toBe(violet[300])
    expect(theme.bashAccent).toBe(orange[400])
    expect(theme.status).toEqual({
      information: blue[500],
      progress: blue[500],
      success: green[600],
      achievement: green[300],
      changeAdded: green[500],
      warning: violet[300],
      failure: red[400],
      interrupted: red[400],
      terminated: red[500],
      inactive: slate[600],
    })
    expect(theme.activityPulse).toEqual([
      blue[50], blue[100], blue[200], blue[300], blue[400], blue[500], blue[600], blue[700], blue[800], blue[900],
      blue[800], blue[700], blue[600], blue[500], blue[400], blue[300], blue[200], blue[100], blue[50],
    ])
    expect(theme.background).toMatchObject({
      canvas: "transparent",
      surface: slate[900],
      input: "#232f41",
      menu: "#232f41",
      alternateRow: slate[750],
      selected: slate[800],
      hovered: slate[750],
      focused: slate[700],
      diffAdded: "#122b22",
      diffRemoved: "#2c1919",
    })
    expect(theme.border).toEqual({ subtle: slate[400], standard: slate[600], strong: slate[500] })
    expect(theme.markdown).toMatchObject({
      heading: { 1: blue[400], 2: blue[400], 3: blue[400], 4: blue[400], 5: blue[400], 6: blue[400] },
      inlineCode: green[400],
      codeText: slate[100],
      codeBorder: slate[400],
      codeHeader: slate[500],
      listMarker: slate[400],
      blockquoteText: slate[200],
      blockquoteBorder: slate[700],
      divider: slate[800],
    })
  })

  it.each([
    appearance("dark", "#101010"),
    appearance("light", "#faf7f0"),
  ])("keeps the text hierarchy readable against $mode terminal backgrounds", (terminal) => {
    const theme = resolveCliTheme(terminal)

    expect(theme.background.terminal).toBe(terminal.defaultBackground)
    expect(contrastRatio(theme.text.body, theme.background.terminal)).toBeGreaterThanOrEqual(7)
    expect(contrastRatio(theme.text.supporting, theme.background.terminal)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(theme.text.metadata, theme.background.terminal)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(theme.text.disabled, theme.background.terminal)).toBeGreaterThanOrEqual(3)
  })

  it("keeps Magnitude semantic colors readable when the terminal ANSI palette is unreadable", () => {
    const palette = Array<string | null>(16).fill(null)
    palette[1] = "#fffafa"
    palette[2] = "#f8fff8"
    palette[3] = "#fffff0"
    palette[4] = "#f8fbff"
    palette[6] = "#f8ffff"
    const contaminatedAppearance = {
      ...appearance("light", "#ffffff"),
      ansiPalette: palette,
    }
    const theme = resolveCliTheme(contaminatedAppearance)

    for (const color of [theme.accent, theme.link, theme.status.success, theme.status.warning, theme.status.failure]) {
      expect(contrastRatio(color, theme.background.terminal)).toBeGreaterThanOrEqual(4.5)
    }
  })

  it("never substitutes user-defined ANSI hues for Magnitude brand colors", () => {
    const palette = Array<string | null>(16).fill(null)
    palette[4] = "#ff00ff"
    palette[5] = "#ff69b4"
    palette[6] = "#a020f0"

    const contaminatedAppearance = {
      ...appearance("dark", "#101010"),
      ansiPalette: palette,
    }
    const theme = resolveCliTheme(contaminatedAppearance)

    expect(theme.accent).toBe(blue[500])
    expect(theme.link).toBe(blue[400])
    expect(theme.planAccent).toBe(violet[300])
    expect(theme.accent).not.toBe(palette[6])
    expect(theme.planAccent).not.toBe(palette[5])
  })

  it("keeps specialized vocabulary limited to genuinely distinct visual roles", () => {
    const theme = resolveCliTheme(appearance("dark", "#101010"))

    expect(theme.status.progress).toBe(theme.accent)
    expect(theme.markdown.link).toBe(theme.link)
    expect(theme.markdown.heading[1]).toBe(theme.link)
    expect(theme.activityPulse.length).toBeGreaterThan(2)
  })
})
