import {
  blue,
  green,
  indigo,
  orange,
  red,
  rose,
  slate,
  violet,
} from "@magnitudedev/client-common"

const palette = {
  blue,
  slate,
  green,
  rose,
  violet,
  indigo,
  orange,
  red,
} as const

export const paletteCssVars = (): Readonly<Record<string, string>> =>
  Object.fromEntries(
    Object.entries(palette).flatMap(([family, shades]) =>
      Object.entries(shades).map(([shade, value]) => [
        `--magnitude-${family}-${shade}`,
        value,
      ])
    )
  )

export const injectPaletteCssVars = (): void => {
  const root = document.documentElement
  for (const [name, value] of Object.entries(paletteCssVars())) {
    root.style.setProperty(name, value)
  }
}
