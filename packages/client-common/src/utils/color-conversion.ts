export type RGBColor = { r: number; g: number; b: number }
export type HSLColor = { h: number; s: number; l: number }

export const clampRange = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))

export const normalizeHexColor = (hex: string): string | null => {
  const trimmed = hex.trim()
  const withoutHash = trimmed.startsWith('#') ? trimmed.slice(1) : trimmed
  if (withoutHash.length === 3) {
    return withoutHash
      .split('')
      .map((char) => char + char)
      .join('')
  }
  if (withoutHash.length === 6) {
    return withoutHash
  }
  return null
}

export const parseHexColorToRgb = (hex: string): RGBColor | null => {
  const normalized = normalizeHexColor(hex)
  if (!normalized) return null
  const r = parseInt(normalized.slice(0, 2), 16) / 255
  const g = parseInt(normalized.slice(2, 4), 16) / 255
  const b = parseInt(normalized.slice(4, 6), 16) / 255
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return null
  return { r, g, b }
}

export const convertRgbToHsl = (r: number, g: number, b: number): HSLColor => {
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2

  let h = 0
  let s = 0

  if (max !== min) {
    const d = max - min
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0)
        break
      case g:
        h = (b - r) / d + 2
        break
      default:
        h = (r - g) / d + 4
    }
    h /= 6
  }

  return { h, s, l }
}

export const resolveHueToRgbComponent = (
  p: number,
  q: number,
  t: number,
): number => {
  let temp = t
  if (temp < 0) temp += 1
  if (temp > 1) temp -= 1
  if (temp < 1 / 6) return p + (q - p) * 6 * temp
  if (temp < 1 / 2) return q
  if (temp < 2 / 3) return p + (q - p) * (2 / 3 - temp) * 6
  return p
}

export const convertHslToRgb = (h: number, s: number, l: number): RGBColor => {
  if (s === 0) {
    return { r: l, g: l, b: l }
  }

  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q

  return {
    r: resolveHueToRgbComponent(p, q, h + 1 / 3),
    g: resolveHueToRgbComponent(p, q, h),
    b: resolveHueToRgbComponent(p, q, h - 1 / 3),
  }
}

export const formatRgbToHex = (r: number, g: number, b: number): string => {
  const toHex = (value: number) =>
    Math.round(clampRange(value, 0, 1) * 255)
      .toString(16)
      .padStart(2, '0')
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}